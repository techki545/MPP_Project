"""Application service and production evidence-synthesis pipeline."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any, Mapping

from .chat_client import ChatClient
from .config import Settings
from .document_graph import build_document_graph
from .embedding_client import EmbeddingClient
from .errors import KnowledgeBaseError
from .evidence_classifier import EvidenceAssessment, classify_evidence
from .graph_builder import LocalGraphBuilder
from .indexer import (
    BuildPipeline,
    Indexer,
    build_default_pipeline,
    validate_embedding_cost_gate,
)
from .models import SearchFilters
from .reporter import (
    ChatEvidenceRefiner,
    EvidenceBundle,
    EvidenceSource,
    GroundedClaimExtractor,
    GroundedReporter,
    ValidatedClaims,
    extractive_fallback_claims,
)
from .retriever import HybridRetriever
from .sqlite_store import SQLiteStore
from .vector_store import LocalVectorStore


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_PATH_KEYS = {"path", "pdf_path", "local_path", "absolute_path"}


class KnowledgeBaseService:
    REQUIRED_QUERY_KEYS = (
        "question",
        "mode",
        "degraded_reason",
        "filters",
        "reasoning_steps",
        "answer_markdown",
        "model_used",
        "model_name",
        "model_error",
        "summary",
        "sources",
        "graph",
        "retrieval_stats",
    )

    def __init__(self, settings: Settings, store: Any, pipeline: Any) -> None:
        self.settings = settings
        self.store = store
        self.pipeline = pipeline

    def query(
        self,
        question: str,
        filters: SearchFilters,
        *,
        model_config: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        clean_question = " ".join(str(question or "").split())
        if not clean_question:
            raise KnowledgeBaseError("empty_question", "Clinical question is required")
        if len(clean_question) > 2000:
            raise KnowledgeBaseError(
                "question_too_long", "Clinical question exceeds 2000 characters"
            )
        if not isinstance(filters, SearchFilters):
            raise KnowledgeBaseError("invalid_filters", "Search filters are invalid")
        if (
            filters.year_from is not None
            and filters.year_to is not None
            and filters.year_from > filters.year_to
        ):
            raise KnowledgeBaseError("invalid_year_range", "Year range is invalid")

        if model_config is None:
            raw_result = self.pipeline.run(clean_question, filters)
        else:
            chat_client = ChatClient(
                model_config["base_url"],
                model_config["api_key"],
                model_config["model_name"],
            )
            raw_result = self.pipeline.run(
                clean_question, filters, chat_client=chat_client
            )
        if not isinstance(raw_result, Mapping):
            raise KnowledgeBaseError(
                "query_pipeline_invalid", "Query pipeline response is invalid"
            )
        result = dict(raw_result)
        sources, citation_map = _number_deduplicated_sources(result.get("sources", []))
        reasoning_steps = _rewrite_reasoning_steps(
            result.get("reasoning_steps", result.get("analysis_steps", [])), citation_map
        )
        answer = _rewrite_citations(
            str(
                result.get(
                    "answer_markdown", result.get("final_answer_markdown", "")
                )
            ),
            citation_map,
        )
        stable = {
            "question": clean_question,
            "mode": str(result.get("mode", "keyword")),
            "degraded_reason": str(result.get("degraded_reason", "")),
            "filters": _filters_dict(filters),
            "reasoning_steps": reasoning_steps,
            "answer_markdown": answer,
            "model_used": bool(result.get("model_used", False)),
            "model_name": str(result.get("model_name", self.settings.chat_model)),
            "model_error": result.get("model_error"),
            "summary": dict(result.get("summary", {})),
            "sources": sources,
            "graph": dict(result.get("graph", {"nodes": [], "edges": []})),
            "retrieval_stats": dict(result.get("retrieval_stats", {})),
        }
        for key, value in result.items():
            if key not in stable and key not in {"analysis_steps", "final_answer_markdown"}:
                stable[key] = value
        return stable

    def document_detail(self, document_id: str) -> dict[str, Any]:
        clean_id = _validate_identifier(document_id, "document_id_invalid")
        raw = self.store.document_detail(clean_id)
        if raw is None:
            raise KnowledgeBaseError("document_not_found", "Document was not found")
        detail = _strip_path_fields(dict(raw))
        try:
            self.resolve_pdf(clean_id)
            detail["pdf_available"] = True
        except KnowledgeBaseError:
            detail["pdf_available"] = False
        return detail

    def resolve_pdf(self, document_id: str) -> Path:
        clean_id = _validate_identifier(document_id, "document_id_invalid")
        candidates: list[Path] = []
        indexed_paths = getattr(self.store, "indexed_pdf_paths", None)
        if callable(indexed_paths):
            candidates.extend(Path(value) for value in indexed_paths(clean_id))
        if not candidates:
            raw = self.store.document_detail(clean_id)
            if raw is None:
                raise KnowledgeBaseError("document_not_found", "Document was not found")
            raw_path = raw.get("pdf_path") if isinstance(raw, Mapping) else None
            if raw_path:
                candidates.append(Path(str(raw_path)))

        source_root = self.settings.source_dir.resolve()
        for candidate in candidates:
            resolved = candidate.resolve()
            if (
                resolved.suffix.casefold() == ".pdf"
                and resolved.is_file()
                and resolved.is_relative_to(source_root)
            ):
                return resolved
        raise KnowledgeBaseError("pdf_unavailable", "Indexed PDF is unavailable")

    def knowledge_base_status(self) -> dict[str, Any]:
        statistics = getattr(self.store, "statistics", None)
        counts = dict(statistics()) if callable(statistics) else {}
        readiness = getattr(self.store, "knowledge_base_ready", None)
        ready = (
            bool(readiness())
            if callable(readiness)
            else bool(counts.get("metadata_records", 0))
        )
        probe_status = getattr(self.store, "embedding_probe_completed", None)
        if callable(probe_status):
            counts["embedding_probe_completed"] = bool(
                probe_status(self.settings.embedding_model)
            )
        build_status = getattr(self.store, "build_status", None)
        if callable(build_status):
            job = build_status()
            progress = dict(job.get("progress", {})) if job else {}
            counts["build_state"] = str(job.get("state", "idle")) if job else "idle"
            counts["embedding_pending"] = int(
                progress.get("pending_embedding_count", 0) or 0
            )
        counts["status"] = "ready" if ready else "not_built"
        return counts

    def health(self) -> dict[str, Any]:
        status = self.knowledge_base_status()
        return {
            "status": status["status"],
            "documents": status.get("metadata_records", 0),
            **status,
        }

    def model_status(self) -> dict[str, Any]:
        configured = bool(
            self.settings.api_base
            and self.settings.api_key
            and self.settings.chat_model
        )
        return {
            "configured": configured,
            "chat_model": self.settings.chat_model,
            "embedding_model": self.settings.embedding_model,
            "api_base_configured": bool(self.settings.api_base),
        }


class SQLiteDocumentStore:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def statistics(self) -> dict[str, int]:
        return self.store.statistics()

    def knowledge_base_ready(self) -> bool:
        return self.store.local_index_ready()

    def embedding_probe_completed(self, model_name: str) -> bool:
        return self.store.embedding_probe_completed(model_name)

    def build_status(self) -> dict[str, Any] | None:
        return self.store.get_job(Indexer.JOB_ID)

    def document_detail(self, document_id: str) -> dict[str, Any] | None:
        document = self.store.get_document(document_id)
        if document is None:
            return None
        chunks = self.store.list_chunks_for_document(document_id, limit=8)
        files = [
            item for item in self.store.list_files() if item.document_id == document_id
        ]
        pdf_paths = [item.path for item in files if Path(item.path).suffix.casefold() == ".pdf"]
        quality = _best_quality(tuple(chunk.quality for chunk in chunks))
        return {
            "document_id": document.document_id,
            "title": document.title,
            "authors": list(document.authors),
            "year": document.year,
            "journal": document.journal,
            "doi": document.doi,
            "url": document.url,
            "abstract": document.abstract,
            "language": document.language,
            "evidence_type": document.evidence_type,
            "classification_confidence": document.classification_confidence,
            "classification_basis": document.classification_basis,
            "quality": quality,
            "has_fulltext": document.has_fulltext,
            "fulltext_status": document.fulltext_status,
            "document_snippets": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "section": chunk.section,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "quality": chunk.quality,
                    "is_ocr": chunk.is_ocr,
                }
                for chunk in chunks
            ],
            "pdf_path": pdf_paths[0] if pdf_paths else "",
        }

    def indexed_pdf_paths(self, document_id: str) -> tuple[str, ...]:
        return tuple(
            item.path
            for item in self.store.list_files()
            if item.document_id == document_id
            and Path(item.path).suffix.casefold() == ".pdf"
        )


def _chat_components(
    chat_client: Any,
) -> tuple[ChatEvidenceRefiner, GroundedClaimExtractor, GroundedReporter]:
    return (
        ChatEvidenceRefiner(chat_client),
        GroundedClaimExtractor(chat_client),
        GroundedReporter(chat_client),
    )


class ProductionQueryPipeline:
    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        document_store: SQLiteDocumentStore,
        evidence_refiner: ChatEvidenceRefiner,
        claim_extractor: GroundedClaimExtractor,
        graph_builder: LocalGraphBuilder,
        reporter: GroundedReporter,
    ) -> None:
        self.retriever = retriever
        self.document_store = document_store
        self.evidence_refiner = evidence_refiner
        self.claim_extractor = claim_extractor
        self.graph_builder = graph_builder
        self.reporter = reporter

    def run(
        self,
        question: str,
        filters: SearchFilters,
        *,
        chat_client: Any | None = None,
    ) -> dict[str, Any]:
        evidence_refiner = self.evidence_refiner
        claim_extractor = self.claim_extractor
        reporter = self.reporter
        if chat_client is not None:
            (
                evidence_refiner,
                claim_extractor,
                reporter,
            ) = _chat_components(chat_client)
        retrieval = self.retriever.search(question, filters)
        sources = tuple(
            self._evidence_source(
                index, item, evidence_refiner=evidence_refiner
            )
            for index, item in enumerate(retrieval.documents, start=1)
        )
        empty_bundle = EvidenceBundle(sources=sources, graph={"nodes": [], "edges": []})
        claim_error: str | None = None
        try:
            validated = claim_extractor.extract(question, empty_bundle)
            model_claim_count = len(validated.claims)
            fallback = extractive_fallback_claims(sources)
            accepted_documents = {claim.document_id for claim in validated.claims}
            supplemental_claims = tuple(
                claim
                for claim in fallback.claims
                if claim.document_id not in accepted_documents
            )
            if supplemental_claims or fallback.audit:
                validated = ValidatedClaims(
                    validated.claims + supplemental_claims,
                    validated.audit + fallback.audit,
                )
            if model_claim_count == 0:
                claim_error = "claim_validation_empty"
        except KnowledgeBaseError as error:
            claim_error = error.code
            validated = extractive_fallback_claims(sources)
        except Exception:
            claim_error = "claim_extraction_failed"
            validated = extractive_fallback_claims(sources)
        claim_graph = self.graph_builder.build(question, validated.claims).as_dict()
        graph = build_document_graph(
            sources,
            validated.claims,
            claim_graph,
            limit=10,
        )
        bundle = EvidenceBundle(
            sources=sources,
            graph=graph,
            claims=validated.claims,
        )
        report = reporter.generate_with_fallback(question, bundle)
        relation_counts = Counter(edge["relation"] for edge in graph["edges"])
        model_error = report.model_error or claim_error
        return {
            "question": question,
            "mode": retrieval.mode,
            "degraded_reason": retrieval.degraded_reason,
            "filters": _filters_dict(filters),
            "reasoning_steps": list(report.analysis_steps),
            "answer_markdown": report.final_answer_markdown,
            "model_used": report.model_used and claim_error is None,
            "model_name": report.model_name,
            "model_error": model_error,
            "summary": {
                "evidence_count": len(sources),
                "relation_count": len(graph["edges"]),
                "support_count": relation_counts["supports"],
                "update_count": relation_counts["updates"],
                "supplement_count": relation_counts["supplements"],
                "confirm_count": relation_counts["confirms"],
                "conflict_count": relation_counts["conflicts"],
                "caution_count": relation_counts["cautions"],
            },
            "sources": [source.as_dict() for source in sources],
            "graph": graph,
            "retrieval_stats": {
                "candidate_count": retrieval.candidate_count,
                "claim_count": len(validated.claims),
                "discarded_claim_count": len(validated.audit),
            },
            "data_source": "knowledge_base",
        }

    def _evidence_source(
        self,
        number: int,
        item: Any,
        *,
        evidence_refiner: ChatEvidenceRefiner,
    ) -> EvidenceSource:
        detail = self.document_store.document_detail(item.document_id) or {}
        title = str(detail.get("title") or item.payload.get("title") or item.document_id)
        abstract = str(detail.get("abstract") or item.payload.get("abstract") or "")
        assessment = EvidenceAssessment(
            evidence_type=str(
                detail.get("evidence_type") or item.payload.get("evidence_type") or "unknown"
            ),
            confidence=float(detail.get("classification_confidence") or 0.0),
            quality=str(detail.get("quality") or item.payload.get("quality") or "unknown"),
            basis=str(detail.get("classification_basis") or "retrieved_payload"),
            quality_signals=(),
        )
        if assessment.evidence_type == "unknown" and not detail:
            assessment = classify_evidence(title, abstract)
        refined = evidence_refiner.refine(title, abstract, assessment)
        fulltext_hits = [
            hit for hit in item.supporting_hits if "fulltext" in hit.source
        ]
        selected_hits = fulltext_hits or list(item.supporting_hits[:1])
        chunk_ids = tuple(hit.record_id for hit in selected_hits)
        snippets = tuple(hit.text for hit in selected_hits if hit.text.strip())
        page_ranges = tuple(
            _page_range(hit.payload.get("page_start"), hit.payload.get("page_end"))
            for hit in selected_hits
        )
        return EvidenceSource(
            source_number=number,
            document_id=item.document_id,
            title=title,
            evidence_type=refined.evidence_type,
            year=_optional_int(detail.get("year", item.payload.get("year"))),
            chunk_ids=chunk_ids,
            snippets=snippets,
            page_ranges=page_ranges,
            fulltext=bool(fulltext_hits),
            abstract=abstract,
            classification_confidence=refined.confidence,
            classification_basis=refined.basis,
            quality=refined.quality,
            journal=str(detail.get("journal", "")),
            doi=str(detail.get("doi", "")),
        )


class _UnavailableChatClient:
    def __init__(self, model: str) -> None:
        self.model = model or "not-configured"

    def complete_json(self, system_prompt: str, payload: Mapping[str, Any]) -> dict:
        raise KnowledgeBaseError("model_config_missing", "Chat model is not configured")


def create_production_service(
    settings: Settings, store: SQLiteStore | None = None
) -> KnowledgeBaseService:
    sqlite_store = store or SQLiteStore(settings.sqlite_path)
    sqlite_store.initialize()
    document_store = SQLiteDocumentStore(sqlite_store)

    embedding_client: Any = None
    vector_store: Any = None
    if settings.api_base and settings.api_key and settings.embedding_model:
        embedding_client = EmbeddingClient(
            base_url=settings.api_base,
            api_key=settings.api_key,
            model=settings.embedding_model,
        )
        try:
            candidate = LocalVectorStore(settings.qdrant_dir)
            candidate.load_existing(model_name=settings.embedding_model)
            vector_store = candidate
        except KnowledgeBaseError:
            if "candidate" in locals():
                candidate.close()
            vector_store = None

    if settings.api_base and settings.api_key and settings.chat_model:
        chat_client: Any = ChatClient(
            settings.api_base,
            settings.api_key,
            settings.chat_model,
        )
    else:
        chat_client = _UnavailableChatClient(settings.chat_model)
    retriever = HybridRetriever(embedding_client, vector_store, sqlite_store)
    pipeline = ProductionQueryPipeline(
        retriever=retriever,
        document_store=document_store,
        evidence_refiner=ChatEvidenceRefiner(chat_client),
        claim_extractor=GroundedClaimExtractor(chat_client),
        graph_builder=LocalGraphBuilder(),
        reporter=GroundedReporter(chat_client),
    )
    return KnowledgeBaseService(settings, document_store, pipeline)


class ProductionBuildRunner:
    """Bind the generic job interface to the resumable production indexer."""

    def __init__(self, settings: Settings, store: SQLiteStore) -> None:
        self.settings = settings
        self.store = store

    def build(
        self,
        *,
        should_pause: Any,
        confirm_embedding_cost: bool,
        document_limit: int | None = None,
        embedding_limit: int | None = None,
        confirm_full_embedding_cost: bool = False,
        stage: str | None = None,
    ) -> Any:
        embedding_client: EmbeddingClient | None = None
        vector_store: LocalVectorStore | None = None
        validate_embedding_cost_gate(
            self.store,
            model_name=self.settings.embedding_model,
            confirm_embedding_cost=confirm_embedding_cost,
            confirm_full_embedding_cost=confirm_full_embedding_cost,
            embedding_limit=embedding_limit,
        )
        if confirm_embedding_cost:
            self.settings.require_embedding_access()
            embedding_client = EmbeddingClient(
                base_url=self.settings.api_base,
                api_key=self.settings.api_key,
                model=self.settings.embedding_model,
            )
            dimension = embedding_client.probe()
            vector_store = LocalVectorStore(self.settings.qdrant_dir)
            vector_store.ensure_collections(
                dimension=dimension,
                model_name=self.settings.embedding_model,
            )
        try:
            indexer = Indexer(
                store=self.store,
                embedding_client=embedding_client,
                vector_store=vector_store,
                lock_path=self.settings.data_dir / "build.lock",
                model_name=self.settings.embedding_model,
                embedding_batch_size=self.settings.embedding_batch_size,
            )
            if stage in {"embedding", "vector"}:
                pipeline = BuildPipeline(stages={}, algorithm_version="mpp-local-v1")
            else:
                pipeline = build_default_pipeline(self.settings.source_dir, store=self.store)
            if stage is not None:
                allowed = {*BuildPipeline.LOCAL_STAGES, "embedding", "vector"}
                if stage not in allowed:
                    raise KnowledgeBaseError("invalid_retry_stage", "Retry stage is invalid")
                if stage in BuildPipeline.LOCAL_STAGES:
                    pipeline = BuildPipeline(
                        stages={stage: pipeline.stages[stage]},
                        algorithm_version=pipeline.algorithm_version,
                    )
            result = indexer.build(
                pipeline,
                document_limit=document_limit,
                embedding_limit=embedding_limit,
                confirm_embedding_cost=confirm_embedding_cost,
                should_pause=should_pause,
            )
            if (
                confirm_embedding_cost
                and not confirm_full_embedding_cost
                and embedding_limit is not None
                and result.stage_counters.get("embedding", {}).get("processed", 0) > 0
                and result.stage_counters.get("embedding", {}).get("failed", 0) == 0
            ):
                self.store.mark_embedding_probe_completed(
                    self.settings.embedding_model
                )
            return result
        finally:
            if vector_store is not None:
                vector_store.close()


def _number_deduplicated_sources(
    raw_sources: object,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    if not isinstance(raw_sources, list):
        return [], {}
    sources: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    citation_map: dict[int, int] = {}
    for fallback_number, raw in enumerate(raw_sources, start=1):
        if not isinstance(raw, Mapping):
            continue
        document_id = str(raw.get("document_id", "")).strip()
        if not document_id:
            continue
        old_number = raw.get("source_number", fallback_number)
        old_number = old_number if isinstance(old_number, int) else fallback_number
        if document_id in seen:
            citation_map[old_number] = seen[document_id]
            continue
        new_number = len(sources) + 1
        seen[document_id] = new_number
        citation_map[old_number] = new_number
        source = dict(raw)
        source["source_number"] = new_number
        sources.append(source)
    return sources, citation_map


def _rewrite_reasoning_steps(raw_steps: object, citation_map: dict[int, int]) -> list[dict]:
    if not isinstance(raw_steps, (list, tuple)):
        return []
    steps: list[dict] = []
    for raw in raw_steps:
        if not isinstance(raw, Mapping):
            continue
        step = dict(raw)
        step["title"] = _rewrite_citations(str(step.get("title", "")), citation_map)
        step["body"] = _rewrite_citations(str(step.get("body", "")), citation_map)
        source_ids = step.get("source_ids", [])
        if isinstance(source_ids, list):
            step["source_ids"] = sorted(
                {
                    citation_map[value]
                    for value in source_ids
                    if isinstance(value, int) and value in citation_map
                }
            )
        steps.append(step)
    return steps


def _rewrite_citations(text: str, citation_map: dict[int, int]) -> str:
    return _CITATION_PATTERN.sub(
        lambda match: f"[{citation_map.get(int(match.group(1)), int(match.group(1)))}]",
        text,
    )


def _filters_dict(filters: SearchFilters) -> dict[str, Any]:
    return {
        "evidence_types": sorted(filters.evidence_types),
        "year_from": filters.year_from,
        "year_to": filters.year_to,
        "fulltext_only": filters.fulltext_only,
    }


def _validate_identifier(value: object, code: str) -> str:
    clean = str(value or "").strip()
    if (
        not clean
        or len(clean) > 256
        or ".." in clean
        or "/" in clean
        or "\\" in clean
        or any(ord(char) < 32 or ord(char) == 127 for char in clean)
    ):
        raise KnowledgeBaseError(code, "Identifier is invalid")
    return clean


def _best_quality(values: tuple[str, ...]) -> str:
    order = {"unknown": 0, "very_low": 1, "low": 2, "moderate": 3, "high": 4}
    return max(values, key=lambda value: order.get(value, 0), default="unknown")


def _page_range(start: object, end: object) -> str:
    if isinstance(start, int) and isinstance(end, int):
        return str(start) if start == end else f"{start}-{end}"
    return ""


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _strip_path_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_path_fields(item)
            for key, item in value.items()
            if str(key).casefold() not in _PATH_KEYS
        }
    if isinstance(value, list):
        return [_strip_path_fields(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_path_fields(item) for item in value)
    return value
