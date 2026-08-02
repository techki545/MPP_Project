"""Resumable orchestration for local ingestion and paid embedding stages."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, ClassVar

from filelock import FileLock, Timeout as FileLockTimeout

from .chunker import chunk_pages
from .deduplicator import DocumentLookup, hash_file, match_pdf
from .evidence_classifier import classify_evidence
from .errors import KnowledgeBaseError
from .metadata_loader import (
    CsvDecodeDiagnostics,
    load_csv_document_items,
    load_csv_documents,
)
from .models import EmbeddingItem, FileRecord
from .pdf_parser import PDFParser, ParsedPDF, rapidocr_ocr_factory
from .sqlite_store import SQLiteStore


@dataclass(frozen=True)
class InspectionReport:
    source_files: int
    metadata_records: int
    pdf_files: int
    xlsx_files: int
    csv_files: int
    zip_files: int
    duplicate_pdf_files: int
    total_bytes: int
    decode_replacement_count: int = 0
    errors: tuple[dict[str, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_files": self.source_files,
            "metadata_records": self.metadata_records,
            "pdf_files": self.pdf_files,
            "xlsx_files": self.xlsx_files,
            "csv_files": self.csv_files,
            "zip_files": self.zip_files,
            "duplicate_pdf_files": self.duplicate_pdf_files,
            "total_bytes": self.total_bytes,
            "decode_replacement_count": self.decode_replacement_count,
            "errors": [dict(error) for error in self.errors],
        }


@dataclass(frozen=True)
class EmbeddingRunResult:
    completed_embeddings: int
    skipped_embeddings: int
    failed_embeddings: int
    pending_embeddings: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "completed_embeddings": self.completed_embeddings,
            "skipped_embeddings": self.skipped_embeddings,
            "failed_embeddings": self.failed_embeddings,
            "pending_embeddings": self.pending_embeddings,
        }


@dataclass(frozen=True)
class BuildRunResult:
    stage_counters: dict[str, dict[str, int]]
    final_state: str
    pending_embedding_count: int
    error_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_counters": {
                stage: dict(counters) for stage, counters in self.stage_counters.items()
            },
            "final_state": self.final_state,
            "pending_embedding_count": self.pending_embedding_count,
            "error_count": self.error_count,
        }


@dataclass(frozen=True)
class BuildItem:
    record_id: str
    input_hash: str
    run: Callable[[], Mapping[str, int] | int | None] = field(repr=False)


@dataclass(frozen=True)
class BuildPipeline:
    LOCAL_STAGES: ClassVar[tuple[str, ...]] = (
        "metadata",
        "pdf_inventory",
        "pdf_match",
        "parse",
        "chunk",
        "lexical",
    )

    stages: Mapping[str, Callable[[], Iterable[BuildItem]]]
    algorithm_version: str

    def __post_init__(self) -> None:
        unknown = set(self.stages) - set(self.LOCAL_STAGES)
        if unknown or not self.algorithm_version.strip():
            raise ValueError("Build pipeline configuration is invalid")

    def items(self, stage: str) -> Iterable[BuildItem]:
        supplier = self.stages.get(stage)
        return () if supplier is None else supplier()


class Indexer:
    JOB_ID = "knowledge-base-build"

    def __init__(
        self,
        *,
        store: SQLiteStore,
        embedding_client: Any | None,
        vector_store: Any | None,
        lock_path: Path,
        model_name: str = "default",
        embedding_batch_size: int = 32,
    ) -> None:
        if not model_name.strip() or embedding_batch_size < 1:
            raise KnowledgeBaseError(
                "indexer_config_invalid", "Knowledge base indexer configuration is invalid"
            )
        self.store = store
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.lock_path = Path(lock_path)
        self.model_name = model_name.strip()
        self.embedding_batch_size = embedding_batch_size

    @staticmethod
    def inspect(source_dir: Path) -> InspectionReport:
        source = Path(source_dir)
        if not source.is_dir():
            raise KnowledgeBaseError(
                "source_not_found", "Knowledge base source directory does not exist"
            )

        files = sorted(path for path in source.rglob("*") if path.is_file())
        suffix_counts = Counter(path.suffix.lower() for path in files)
        diagnostics: list[CsvDecodeDiagnostics] = []
        metadata_records = 0
        errors: list[dict[str, str]] = []
        for csv_path in (path for path in files if path.suffix.lower() == ".csv"):
            try:
                metadata_records += sum(
                    1
                    for _ in load_csv_documents(csv_path, diagnostics=diagnostics)
                )
            except Exception:
                errors.append(
                    {
                        "stage": "metadata",
                        "record_id": csv_path.name,
                        "code": "metadata_read_failed",
                        "message": "Metadata CSV could not be inspected",
                    }
                )

        pdf_hashes: Counter[str] = Counter()
        for pdf_path in (path for path in files if path.suffix.lower() == ".pdf"):
            try:
                pdf_hashes[hash_file(pdf_path)] += 1
            except OSError:
                errors.append(
                    {
                        "stage": "pdf_inventory",
                        "record_id": pdf_path.name,
                        "code": "pdf_hash_failed",
                        "message": "PDF could not be hashed",
                    }
                )

        return InspectionReport(
            source_files=len(files),
            metadata_records=metadata_records,
            pdf_files=suffix_counts[".pdf"],
            xlsx_files=suffix_counts[".xlsx"],
            csv_files=suffix_counts[".csv"],
            zip_files=suffix_counts[".zip"],
            duplicate_pdf_files=sum(count - 1 for count in pdf_hashes.values()),
            total_bytes=sum(path.stat().st_size for path in files),
            decode_replacement_count=sum(item.replacement_count for item in diagnostics),
            errors=tuple(errors),
        )

    def request_pause(self) -> dict[str, Any]:
        current = self.store.get_job(self.JOB_ID)
        progress = dict(current["progress"]) if current else {}
        self.store.set_job_state(self.JOB_ID, "pause_requested", progress)
        return {"job_id": self.JOB_ID, "state": "pause_requested", "progress": progress}

    def build(
        self,
        pipeline: BuildPipeline,
        *,
        document_limit: int | None = None,
        embedding_limit: int | None = None,
        confirm_embedding_cost: bool = False,
    ) -> BuildRunResult:
        self._validate_optional_limit(document_limit, "document_limit_invalid")
        self._validate_optional_limit(embedding_limit, "embedding_limit_invalid")
        return self._with_lock(
            lambda: self._build_unlocked(
                pipeline,
                document_limit=document_limit,
                embedding_limit=embedding_limit,
                confirm_embedding_cost=confirm_embedding_cost,
            )
        )

    def embed_pending(
        self,
        *,
        confirm_embedding_cost: bool,
        embedding_limit: int | None = None,
    ) -> EmbeddingRunResult:
        self._validate_optional_limit(embedding_limit, "embedding_limit_invalid")
        return self._with_lock(
            lambda: self._embed_pending_unlocked(
                confirm_embedding_cost=confirm_embedding_cost,
                embedding_limit=embedding_limit,
            )
        )

    def _build_unlocked(
        self,
        pipeline: BuildPipeline,
        *,
        document_limit: int | None,
        embedding_limit: int | None,
        confirm_embedding_cost: bool,
    ) -> BuildRunResult:
        stage_counters: dict[str, dict[str, int]] = {}
        self.store.set_job_state(self.JOB_ID, "running", {"stage_counters": {}})

        for stage in BuildPipeline.LOCAL_STAGES:
            counters: Counter[str] = Counter(processed=0, skipped=0, failed=0)
            stage_counters[stage] = counters  # type: ignore[assignment]
            for item_index, item in enumerate(pipeline.items(stage)):
                if document_limit is not None and item_index >= document_limit:
                    break
                checkpoint = self.store.get_build_checkpoint(
                    stage, item.record_id, pipeline.algorithm_version
                )
                if (
                    checkpoint is not None
                    and checkpoint["status"] == "completed"
                    and checkpoint["input_hash"] == item.input_hash
                ):
                    counters["skipped"] += 1
                    continue
                try:
                    outcome = item.run()
                    item_counters = self._normalize_item_counters(outcome)
                    counters["processed"] += 1
                    counters.update(item_counters)
                    self.store.set_build_checkpoint(
                        stage=stage,
                        record_id=item.record_id,
                        input_hash=item.input_hash,
                        algorithm_version=pipeline.algorithm_version,
                        status="completed",
                        counters=item_counters,
                    )
                except Exception as error:
                    code = (
                        error.code
                        if isinstance(error, KnowledgeBaseError)
                        else f"{stage}_failed"
                    )
                    counters["failed"] += 1
                    self.store.record_error(
                        stage, item.record_id, code, "Build item could not be processed"
                    )
                    self.store.set_build_checkpoint(
                        stage=stage,
                        record_id=item.record_id,
                        input_hash=item.input_hash,
                        algorithm_version=pipeline.algorithm_version,
                        status="failed",
                        error_code=code,
                        error_message="Build item could not be processed",
                    )

                if self._pause_requested():
                    result = self._build_result(stage_counters, "paused")
                    self.store.set_job_state(self.JOB_ID, "paused", result.as_dict())
                    return result
                self.store.set_job_state(
                    self.JOB_ID,
                    "running",
                    {"stage": stage, "stage_counters": self._plain_counters(stage_counters)},
                )

        embedding = self._embed_pending_unlocked(
            confirm_embedding_cost=confirm_embedding_cost,
            embedding_limit=embedding_limit,
        )
        stage_counters["embedding"] = {
            "processed": embedding.completed_embeddings,
            "skipped": embedding.skipped_embeddings,
            "failed": embedding.failed_embeddings,
        }
        current_failures = sum(
            counters.get("failed", 0) for counters in stage_counters.values()
        )
        if self._pause_requested():
            final_state = "paused"
        elif embedding.pending_embeddings:
            final_state = "embedding_pending"
        elif current_failures:
            final_state = "completed_with_errors"
        else:
            final_state = "completed"
        result = self._build_result(stage_counters, final_state)
        self.store.set_job_state(self.JOB_ID, final_state, result.as_dict())
        return result

    def _embed_pending_unlocked(
        self, *, confirm_embedding_cost: bool, embedding_limit: int | None
    ) -> EmbeddingRunResult:
        total_candidates = self.store.count_embedding_candidates()
        all_pending = self.store.list_pending_embedding_items(self.model_name)
        skipped = total_candidates - len(all_pending)
        if not confirm_embedding_cost:
            return EmbeddingRunResult(0, skipped, 0, len(all_pending))
        if self.embedding_client is None or self.vector_store is None:
            raise KnowledgeBaseError(
                "embedding_config_missing", "Embedding dependencies are not configured"
            )

        selected = all_pending if embedding_limit is None else all_pending[:embedding_limit]
        completed = 0
        failed = 0
        for start in range(0, len(selected), self.embedding_batch_size):
            batch = selected[start : start + self.embedding_batch_size]
            try:
                vectors = self._vectors_for_batch(batch)
                metadata_items = []
                fulltext_items = []
                for item, vector in zip(batch, vectors):
                    vector_item = (
                        item.record_id,
                        vector,
                        self._embedding_payload(item),
                    )
                    if item.kind == "metadata":
                        metadata_items.append(vector_item)
                    else:
                        fulltext_items.append(vector_item)
                if metadata_items:
                    self.vector_store.upsert_metadata(metadata_items)
                if fulltext_items:
                    self.vector_store.upsert_fulltext(fulltext_items)
            except KnowledgeBaseError:
                raise
            except Exception:
                failed += len(batch)
                for item in batch:
                    self.store.record_error(
                        "vector",
                        item.record_id,
                        "vector_write_failed",
                        "Embedding vector could not be indexed",
                    )
                continue

            for item in batch:
                if self.store.mark_embedding_indexed(
                    item.record_id,
                    item.kind,
                    self.model_name,
                    item.content_hash,
                ):
                    completed += 1
                else:
                    failed += 1
            if self._pause_requested():
                break

        pending = len(self.store.list_pending_embedding_items(self.model_name))
        return EmbeddingRunResult(completed, skipped, failed, pending)

    def _vectors_for_batch(self, batch: list[EmbeddingItem]) -> list[list[float]]:
        vectors: list[list[float] | None] = [None] * len(batch)
        missing_positions: list[int] = []
        missing_texts: list[str] = []
        for index, item in enumerate(batch):
            cached = self.store.get_cached_embedding(item.content_hash, self.model_name)
            if cached is None:
                missing_positions.append(index)
                missing_texts.append(item.text)
            else:
                vectors[index] = cached

        if missing_texts:
            generated = self.embedding_client.embed(missing_texts)
            if len(generated) != len(missing_positions):
                raise KnowledgeBaseError(
                    "embedding_response_invalid", "Embedding response is invalid"
                )
            for position, vector in zip(missing_positions, generated):
                item = batch[position]
                self.store.put_cached_embedding(item.content_hash, self.model_name, vector)
                vectors[position] = list(vector)
        if any(vector is None for vector in vectors):
            raise KnowledgeBaseError(
                "embedding_response_invalid", "Embedding response is invalid"
            )
        return [vector for vector in vectors if vector is not None]

    def _embedding_payload(self, item: EmbeddingItem) -> dict[str, Any]:
        document = self.store.get_document(item.document_id)
        if document is None:
            raise KnowledgeBaseError(
                "embedding_record_not_found", "Embedding record was not found"
            )
        payload: dict[str, Any] = {
            "document_id": document.document_id,
            "title": document.title,
            "abstract": document.abstract,
            "year": document.year,
            "evidence_type": document.evidence_type,
            "has_fulltext": document.has_fulltext,
            "text": item.text,
        }
        if item.kind == "chunk":
            chunk = self.store.get_chunk(item.record_id)
            if chunk is None:
                raise KnowledgeBaseError(
                    "embedding_record_not_found", "Embedding record was not found"
                )
            payload.update(
                {
                    "file_id": chunk.file_id,
                    "section": chunk.section,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "quality": chunk.quality,
                    "is_ocr": chunk.is_ocr,
                }
            )
        return payload

    def _pause_requested(self) -> bool:
        job = self.store.get_job(self.JOB_ID)
        return job is not None and job["state"] == "pause_requested"

    def _build_result(
        self, stage_counters: Mapping[str, Mapping[str, int]], state: str
    ) -> BuildRunResult:
        return BuildRunResult(
            stage_counters=self._plain_counters(stage_counters),
            final_state=state,
            pending_embedding_count=len(
                self.store.list_pending_embedding_items(self.model_name)
            ),
            error_count=len(self.store.list_errors()),
        )

    def _with_lock(self, operation: Callable[[], Any]) -> Any:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(str(self.lock_path), timeout=0):
                return operation()
        except FileLockTimeout:
            raise KnowledgeBaseError(
                "build_already_running", "A knowledge base build is already running"
            ) from None

    @staticmethod
    def _normalize_item_counters(
        outcome: Mapping[str, int] | int | None,
    ) -> dict[str, int]:
        if outcome is None:
            return {}
        if isinstance(outcome, int) and not isinstance(outcome, bool):
            return {"items": outcome}
        if not isinstance(outcome, Mapping):
            raise ValueError("Build item result is invalid")
        counters: dict[str, int] = {}
        for key, value in outcome.items():
            if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("Build item result is invalid")
            counters[key] = value
        return counters

    @staticmethod
    def _validate_optional_limit(value: int | None, code: str) -> None:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise KnowledgeBaseError(code, "Build limit must be a positive integer")

    @staticmethod
    def _plain_counters(
        values: Mapping[str, Mapping[str, int]],
    ) -> dict[str, dict[str, int]]:
        return {stage: dict(counters) for stage, counters in values.items()}


def build_default_pipeline(
    source_dir: Path,
    *,
    store: SQLiteStore,
    ocr: Callable[[bytes], str] | None = None,
) -> BuildPipeline:
    """Compose the local CSV/PDF pipeline without contacting a model provider."""

    source = Path(source_dir)
    if not source.is_dir():
        raise KnowledgeBaseError(
            "source_not_found", "Knowledge base source directory does not exist"
        )
    csv_paths = tuple(sorted(source.rglob("*.csv")))
    pdf_paths = tuple(sorted(source.rglob("*.pdf")))
    if not csv_paths:
        raise KnowledgeBaseError(
            "metadata_missing", "No metadata CSV was found in the source directory"
        )

    parser = PDFParser(ocr=ocr if ocr is not None else rapidocr_ocr_factory())
    parsed_cache: dict[str, ParsedPDF] = {}
    pdf_hash_cache: dict[Path, str] = {}
    metadata_fingerprint = _combined_file_hash(csv_paths)
    lookup_cache: dict[str, DocumentLookup] = {}

    def metadata_items() -> Iterable[BuildItem]:
        imported_rows = 0
        for csv_path in csv_paths:
            for _, document in load_csv_document_items(csv_path, diagnostics=[]):
                imported_rows += 1
                assessment = classify_evidence(document.title, document.abstract)
                document = replace(
                    document,
                    evidence_type=assessment.evidence_type,
                    classification_confidence=assessment.confidence,
                    classification_basis=assessment.basis,
                )
                record_id = f"{csv_path.name}:{document.source_row}:{document.source_id}"
                input_hash = sha256(
                    json.dumps(
                        asdict(document),
                        ensure_ascii=False,
                        sort_keys=True,
                        default=list,
                    ).encode("utf-8")
                ).hexdigest()

                def import_document(document=document) -> dict[str, int]:
                    store.upsert_document(document)
                    return {"documents": 1}

                yield BuildItem(record_id, input_hash, import_document)

        if store.metadata_reimport_required():
            def finalize_metadata(row_count=imported_rows) -> dict[str, int]:
                store.clear_metadata_reimport_required(
                    imported_row_count=row_count,
                    expected_row_count=row_count,
                )
                return {"metadata_reimports": 1}

            yield BuildItem(
                "__metadata_reimport_finalize__",
                metadata_fingerprint,
                finalize_metadata,
            )

    def inventory_items() -> Iterable[BuildItem]:
        for path in pdf_paths:
            digest = pdf_digest(path)
            file_id = _file_id(path)

            def inventory(path=path, digest=digest, file_id=file_id) -> dict[str, int]:
                store.upsert_file(
                    FileRecord(
                        file_id=file_id,
                        path=str(path),
                        sha256=digest,
                        size_bytes=path.stat().st_size,
                        document_id=None,
                        match_method="",
                        match_confidence=0.0,
                        status="inventoried",
                    )
                )
                return {"pdf_files": 1}

            yield BuildItem(file_id, digest, inventory)

    def document_lookup() -> DocumentLookup:
        lookup = lookup_cache.get("metadata")
        if lookup is None:
            items = (
                item
                for csv_path in csv_paths
                for item in load_csv_document_items(csv_path, diagnostics=[])
            )
            lookup = DocumentLookup.from_documents(items)
            lookup_cache["metadata"] = lookup
        return lookup

    def match_items() -> Iterable[BuildItem]:
        lookup = document_lookup()
        for path in pdf_paths:
            digest = pdf_digest(path)
            file_id = _file_id(path)
            input_hash = sha256(
                f"{digest}|{metadata_fingerprint}|pdf-match-v1".encode("utf-8")
            ).hexdigest()

            def match_one(
                path=path, digest=digest, file_id=file_id, lookup=lookup
            ) -> dict[str, int]:
                result = match_pdf(path, lookup)
                document_id = result.document_id
                if document_id is not None and store.get_document(document_id) is None:
                    document_id = None
                    status = "unmatched"
                    method = "metadata_limited"
                    confidence = 0.0
                else:
                    status = "matched" if document_id is not None else result.method
                    method = result.method
                    confidence = result.confidence
                existing = store.get_file(file_id)
                stored_status = status
                if (
                    status == "matched"
                    and existing is not None
                    and existing.sha256 == digest
                    and existing.document_id == document_id
                    and existing.status in {"parsed", "partial"}
                ):
                    stored_status = existing.status
                store.upsert_file(
                    FileRecord(
                        file_id=file_id,
                        path=str(path),
                        sha256=digest,
                        size_bytes=path.stat().st_size,
                        document_id=document_id,
                        match_method=method,
                        match_confidence=confidence,
                        status=stored_status,
                        error_code="" if status == "matched" else f"pdf_{status}",
                        error_message="" if status == "matched" else "PDF metadata match was not unique",
                    )
                )
                return {status: 1}

            yield BuildItem(file_id, input_hash, match_one)

    def parse_items() -> Iterable[BuildItem]:
        for file_record in store.list_files(("matched", "parsed", "partial", "parse_failed")):
            input_hash = sha256(
                f"{file_record.sha256}|{file_record.document_id}|pdf-parser-v1".encode("utf-8")
            ).hexdigest()

            def parse_one(file_record=file_record) -> dict[str, int]:
                parsed = parser.parse(Path(file_record.path))
                parsed_cache[file_record.file_id] = parsed
                if parsed.status == "failed":
                    store.upsert_file(
                        replace(
                            file_record,
                            status="parse_failed",
                            error_code=parsed.error_code,
                            error_message=parsed.error_message,
                        )
                    )
                    raise KnowledgeBaseError(
                        parsed.error_code or "pdf_parse_failed",
                        "PDF could not be parsed",
                    )
                store.upsert_file(
                    replace(
                        file_record,
                        status=parsed.status,
                        error_code=parsed.error_code,
                        error_message=parsed.error_message,
                    )
                )
                return {
                    "parsed_pdfs": 1,
                    "pages": len(parsed.pages),
                    "ocr_pages": sum(page.used_ocr for page in parsed.pages),
                }

            yield BuildItem(file_record.file_id, input_hash, parse_one)

    def chunk_items() -> Iterable[BuildItem]:
        for file_record in store.list_files(("parsed", "partial")):
            input_hash = sha256(
                f"{file_record.sha256}|{file_record.document_id}|chunker-v1".encode("utf-8")
            ).hexdigest()

            def chunk_one(file_record=file_record) -> dict[str, int]:
                parsed = parsed_cache.get(file_record.file_id)
                if parsed is None:
                    parsed = parser.parse(Path(file_record.path))
                if parsed.status == "failed" or file_record.document_id is None:
                    raise KnowledgeBaseError(
                        parsed.error_code or "pdf_chunk_unavailable",
                        "PDF text is unavailable for chunking",
                    )
                chunks = chunk_pages(
                    file_record.document_id,
                    file_record.file_id,
                    parsed.pages,
                )
                store.delete_chunks_for_file(file_record.file_id)
                for chunk in chunks:
                    store.upsert_chunk(chunk)
                store.update_document_fulltext(
                    file_record.document_id,
                    has_fulltext=bool(chunks),
                    status=parsed.status if chunks else "low_information",
                )
                return {"chunks": len(chunks)}

            yield BuildItem(file_record.file_id, input_hash, chunk_one)

    def lexical_items() -> Iterable[BuildItem]:
        for file_record in store.list_files(("parsed", "partial")):
            yield BuildItem(
                file_record.file_id,
                sha256(f"{file_record.sha256}|fts-v1".encode("utf-8")).hexdigest(),
                lambda: {"fts_documents": 1},
            )

    def pdf_digest(path: Path) -> str:
        digest = pdf_hash_cache.get(path)
        if digest is None:
            digest = hash_file(path)
            pdf_hash_cache[path] = digest
        return digest

    return BuildPipeline(
        stages={
            "metadata": metadata_items,
            "pdf_inventory": inventory_items,
            "pdf_match": match_items,
            "parse": parse_items,
            "chunk": chunk_items,
            "lexical": lexical_items,
        },
        algorithm_version="mpp-local-v1",
    )


def _file_id(path: Path) -> str:
    normalized = path.resolve().as_posix().casefold()
    return "file-" + sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _combined_file_hash(paths: Iterable[Path]) -> str:
    digest = sha256()
    for path in paths:
        digest.update(path.resolve().as_posix().casefold().encode("utf-8"))
        digest.update(hash_file(path).encode("ascii"))
    return digest.hexdigest()
