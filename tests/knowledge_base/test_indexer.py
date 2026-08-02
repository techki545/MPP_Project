from __future__ import annotations

from pathlib import Path

import pytest
from filelock import FileLock
import pymupdf

from knowledge_base.errors import KnowledgeBaseError
from knowledge_base.indexer import (
    BuildItem,
    BuildPipeline,
    Indexer,
    build_default_pipeline,
    validate_embedding_cost_gate,
)
from knowledge_base.models import ChunkRecord, DocumentRecord, FileRecord
from knowledge_base.sqlite_store import SQLiteStore


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts):
        values = list(texts)
        self.calls.append(values)
        return [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(values)]


class FakeVectorStore:
    def __init__(self, *, fail_once: bool = False) -> None:
        self.fail_once = fail_once
        self.metadata_batches = []
        self.fulltext_batches = []

    def upsert_metadata(self, items):
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("vector write failed")
        self.metadata_batches.append(list(items))

    def upsert_fulltext(self, items):
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("vector write failed")
        self.fulltext_batches.append(list(items))


def _seed_store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "manifest.sqlite3")
    store.initialize()
    document = DocumentRecord(
        document_id="doc-1",
        source_row=2,
        title="Steroid trial",
        normalized_title="steroidtrial",
        authors=("Zhang",),
        year=2025,
        journal="Journal",
        doi="",
        normalized_doi="",
        abstract="Low-dose treatment",
        language="en",
    )
    store.upsert_document(document)
    store.upsert_file(
        FileRecord(
            file_id="file-1",
            path=str(tmp_path / "trial.pdf"),
            sha256="pdf-hash",
            size_bytes=100,
            document_id="doc-1",
            match_method="source_id",
            match_confidence=1.0,
            status="matched",
        )
    )
    store.upsert_chunk(
        ChunkRecord(
            chunk_id="chunk-1",
            document_id="doc-1",
            file_id="file-1",
            section="Results",
            page_start=1,
            page_end=1,
            text="Low dose result",
            token_count=3,
            is_ocr=False,
            quality="extracted",
            content_hash="chunk-hash",
        )
    )
    return store


def _indexer(
    tmp_path: Path,
    store: SQLiteStore,
    embedder: FakeEmbeddingClient | None = None,
    vectors: FakeVectorStore | None = None,
) -> Indexer:
    return Indexer(
        store=store,
        embedding_client=embedder,
        vector_store=vectors,
        lock_path=tmp_path / "build.lock",
        model_name="test-model",
        embedding_batch_size=2,
    )


def test_embedding_requires_confirmation_and_reports_pending_count(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    embedder = FakeEmbeddingClient()
    vectors = FakeVectorStore()
    indexer = _indexer(tmp_path, store, embedder, vectors)

    result = indexer.embed_pending(confirm_embedding_cost=False)

    assert result.completed_embeddings == 0
    assert result.failed_embeddings == 0
    assert result.pending_embeddings == 2
    assert embedder.calls == []
    assert vectors.metadata_batches == []


def test_indexer_resumes_without_reembedding_completed_items(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    embedder = FakeEmbeddingClient()
    vectors = FakeVectorStore()
    indexer = _indexer(tmp_path, store, embedder, vectors)

    first = indexer.embed_pending(confirm_embedding_cost=True)
    second = indexer.embed_pending(confirm_embedding_cost=True)

    assert first.completed_embeddings == 2
    assert first.skipped_embeddings == 0
    assert second.completed_embeddings == 0
    assert second.skipped_embeddings == 2
    assert second.pending_embeddings == 0
    assert sum(len(call) for call in embedder.calls) == 2


def test_cached_embeddings_survive_vector_write_failure(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    embedder = FakeEmbeddingClient()
    vectors = FakeVectorStore(fail_once=True)
    indexer = _indexer(tmp_path, store, embedder, vectors)

    first = indexer.embed_pending(confirm_embedding_cost=True)
    calls_after_failure = sum(len(call) for call in embedder.calls)
    second = indexer.embed_pending(confirm_embedding_cost=True)

    assert first.failed_embeddings == 2
    assert first.pending_embeddings == 2
    assert second.completed_embeddings == 2
    assert second.pending_embeddings == 0
    assert sum(len(call) for call in embedder.calls) == calls_after_failure


def test_build_pipeline_checkpoints_each_item_and_resumes(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "manifest.sqlite3")
    store.initialize()
    calls: list[str] = []

    def item(stage: str) -> BuildItem:
        return BuildItem(
            record_id=f"{stage}-1",
            input_hash=f"{stage}-hash",
            run=lambda: calls.append(stage) or {stage: 1},
        )

    pipeline = BuildPipeline(
        stages={stage: (lambda stage=stage: (item(stage),)) for stage in BuildPipeline.LOCAL_STAGES},
        algorithm_version="test-v1",
    )
    indexer = _indexer(tmp_path, store)

    first = indexer.build(pipeline, confirm_embedding_cost=False)
    second = indexer.build(pipeline, confirm_embedding_cost=False)

    assert calls == list(BuildPipeline.LOCAL_STAGES)
    assert first.final_state == "completed"
    assert second.final_state == "completed"
    assert all(second.stage_counters[stage]["skipped"] == 1 for stage in BuildPipeline.LOCAL_STAGES)


def test_pause_request_is_observed_after_completed_item(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "manifest.sqlite3")
    store.initialize()
    calls: list[str] = []
    holder: dict[str, Indexer] = {}

    def first_action():
        calls.append("first")
        holder["indexer"].request_pause()
        return {"documents": 1}

    pipeline = BuildPipeline(
        stages={
            "metadata": lambda: (
                BuildItem("one", "hash-one", first_action),
                BuildItem("two", "hash-two", lambda: calls.append("second") or {}),
            )
        },
        algorithm_version="test-v1",
    )
    indexer = _indexer(tmp_path, store)
    holder["indexer"] = indexer

    result = indexer.build(pipeline, confirm_embedding_cost=False)

    assert result.final_state == "paused"
    assert calls == ["first"]
    assert store.get_job(Indexer.JOB_ID)["state"] == "paused"


def test_pause_request_wins_race_with_running_progress_update(tmp_path: Path) -> None:
    class PauseRacingStore(SQLiteStore):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.inject_pause = True

        def update_job_running_progress(self, job_id, progress):
            if self.inject_pause:
                self.inject_pause = False
                self.set_job_state(job_id, "pause_requested", {"completed": 1})
            return super().update_job_running_progress(job_id, progress)

    store = PauseRacingStore(tmp_path / "manifest.sqlite3")
    store.initialize()
    pipeline = BuildPipeline(
        stages={
            "metadata": lambda: (
                BuildItem("one", "hash-one", lambda: {"documents": 1}),
            )
        },
        algorithm_version="test-v1",
    )

    result = _indexer(tmp_path, store).build(
        pipeline, confirm_embedding_cost=False
    )

    assert result.final_state == "paused"
    assert store.get_job(Indexer.JOB_ID)["state"] == "paused"


def test_pause_request_is_observed_after_embedding_batch(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    vectors = FakeVectorStore()
    holder: dict[str, Indexer] = {}

    class PausingEmbedder(FakeEmbeddingClient):
        def embed(self, texts):
            result = super().embed(texts)
            holder["indexer"].request_pause()
            return result

    indexer = Indexer(
        store=store,
        embedding_client=PausingEmbedder(),
        vector_store=vectors,
        lock_path=tmp_path / "build.lock",
        model_name="test-model",
        embedding_batch_size=1,
    )
    holder["indexer"] = indexer

    result = indexer.build(
        BuildPipeline(stages={}, algorithm_version="test-v1"),
        confirm_embedding_cost=True,
    )

    assert result.final_state == "paused"
    assert result.pending_embedding_count == 1
    assert store.get_job(Indexer.JOB_ID)["state"] == "paused"


def test_current_stage_failure_is_not_reported_as_completed(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "manifest.sqlite3")
    store.initialize()
    pipeline = BuildPipeline(
        stages={
            "metadata": lambda: (
                BuildItem(
                    "bad-row",
                    "bad-hash",
                    lambda: (_ for _ in ()).throw(ValueError("bad row")),
                ),
            )
        },
        algorithm_version="test-v1",
    )

    result = _indexer(tmp_path, store).build(
        pipeline, confirm_embedding_cost=False
    )

    assert result.final_state == "completed_with_errors"
    assert result.stage_counters["metadata"]["failed"] == 1


def test_build_rejects_a_concurrent_writer(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "manifest.sqlite3")
    store.initialize()
    indexer = _indexer(tmp_path, store)
    pipeline = BuildPipeline(stages={}, algorithm_version="test-v1")

    with FileLock(str(tmp_path / "build.lock"), timeout=0):
        with pytest.raises(KnowledgeBaseError) as caught:
            indexer.build(pipeline, confirm_embedding_cost=False)

    assert caught.value.code == "build_already_running"


def test_inspect_is_read_only_and_counts_duplicate_pdf_hashes(tmp_path: Path) -> None:
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "metadata.csv").write_text(
        "ID,Title,Abstract Note\n1,Trial,Abstract\n2,Other,\n", encoding="utf-8"
    )
    (source / "flags.xlsx").write_bytes(b"xlsx")
    (source / "archive.zip").write_bytes(b"zip")
    (source / "a.pdf").write_bytes(b"same")
    (source / "b.pdf").write_bytes(b"same")
    before = sorted(path.relative_to(source) for path in source.rglob("*"))

    report = Indexer.inspect(source)

    after = sorted(path.relative_to(source) for path in source.rglob("*"))
    assert report.metadata_records == 2
    assert report.pdf_files == 2
    assert report.xlsx_files == 1
    assert report.csv_files == 1
    assert report.zip_files == 1
    assert report.duplicate_pdf_files == 1
    assert before == after


def test_result_records_expose_stable_json_dicts(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    result = _indexer(tmp_path, store).embed_pending(confirm_embedding_cost=False)

    assert result.as_dict() == {
        "completed_embeddings": 0,
        "skipped_embeddings": 0,
        "failed_embeddings": 0,
        "pending_embeddings": 2,
    }


def test_default_pipeline_builds_local_metadata_pdf_chunks_and_fts(tmp_path: Path) -> None:
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "metadata.csv").write_text(
        "ID,Author,Publication Year,Title,Publication Title,DOI,Url,Abstract Note,Language\n"
        "1,Zhang,2025,Randomized Controlled Trial of Steroid,Journal,,,Randomized treatment evidence,en\n",
        encoding="utf-8",
    )
    pdf_path = source / "1-Steroid Trial.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text(
        (72, 72),
        "Methods randomized treatment evidence with enough characters for extraction. "
        "Results low dose treatment improved recovery without severe events.",
    )
    pdf.save(pdf_path)
    pdf.close()

    store = SQLiteStore(tmp_path / "manifest.sqlite3")
    store.initialize()
    indexer = _indexer(tmp_path, store)
    pipeline = build_default_pipeline(source, store=store, ocr=lambda _: "")

    result = indexer.build(pipeline, confirm_embedding_cost=False)

    assert result.final_state == "embedding_pending"
    assert result.pending_embedding_count >= 2
    assert result.stage_counters["metadata"]["documents"] == 1
    assert result.stage_counters["pdf_match"]["matched"] == 1
    assert result.stage_counters["chunk"]["chunks"] >= 1
    assert store.search_fulltext("randomized treatment", limit=5)
    document = store.search_metadata("Randomized Controlled Trial", limit=1)[0]
    assert document.payload["evidence_type"] == "randomized_controlled_trial"


def test_metadata_change_relinks_existing_pdf_and_rebuilds_chunks(tmp_path: Path) -> None:
    source = tmp_path / "corpus"
    source.mkdir()
    metadata = source / "metadata.csv"
    metadata.write_text(
        "ID,Author,Publication Year,Title,Publication Title,DOI,Url,Abstract Note,Language\n"
        "1,Zhang,2025,Original Trial,Journal,,,Randomized evidence,en\n",
        encoding="utf-8",
    )
    pdf_path = source / "1-Trial.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text(
        (72, 72),
        "Methods randomized evidence with enough characters for normal extraction. "
        "Results low dose treatment improved recovery without severe events.",
    )
    pdf.save(pdf_path)
    pdf.close()
    store = SQLiteStore(tmp_path / "manifest.sqlite3")
    store.initialize()
    indexer = _indexer(tmp_path, store)

    indexer.build(
        build_default_pipeline(source, store=store, ocr=lambda _: ""),
        confirm_embedding_cost=False,
    )
    original_document_id = store.search_metadata("Original Trial", limit=1)[0].document_id
    metadata.write_text(
        "ID,Author,Publication Year,Title,Publication Title,DOI,Url,Abstract Note,Language\n"
        "1,Zhang,2025,Updated Trial,Journal,,,Randomized evidence,en\n",
        encoding="utf-8",
    )

    indexer.build(
        build_default_pipeline(source, store=store, ocr=lambda _: ""),
        confirm_embedding_cost=False,
    )

    updated_document_id = store.search_metadata("Updated Trial", limit=1)[0].document_id
    fulltext_ids = {
        hit.document_id for hit in store.search_fulltext("randomized evidence", limit=10)
    }
    assert updated_document_id != original_document_id
    assert updated_document_id in fulltext_ids
    assert original_document_id not in fulltext_ids


def test_default_pipeline_hashes_each_pdf_once_per_run(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "corpus"
    _write_source = (
        "ID,Author,Publication Year,Title,Publication Title,DOI,Url,Abstract Note,Language\n"
        "1,Zhang,2025,Trial,Journal,,,Evidence,en\n"
    )
    source.mkdir()
    (source / "metadata.csv").write_text(_write_source, encoding="utf-8")
    (source / "1-Trial.pdf").write_bytes(b"not parsed because the build is limited")
    store = SQLiteStore(tmp_path / "manifest.sqlite3")
    store.initialize()
    import knowledge_base.indexer as indexer_module

    original_hash_file = indexer_module.hash_file
    calls: list[Path] = []

    def counting_hash(path: Path, *args, **kwargs):
        calls.append(Path(path))
        return original_hash_file(path, *args, **kwargs)

    monkeypatch.setattr(indexer_module, "hash_file", counting_hash)
    pipeline = build_default_pipeline(source, store=store, ocr=lambda _: "")

    _indexer(tmp_path, store).build(
        pipeline, document_limit=1, confirm_embedding_cost=False
    )

    assert sum(path.suffix.lower() == ".pdf" for path in calls) == 1


def test_default_pipeline_marks_duplicate_pdf_and_chunks_only_canonical_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "metadata.csv").write_text(
        "ID,Author,Publication Year,Title,Publication Title,DOI,Url,Abstract Note,Language\n"
        "1,Zhang,2025,Steroid Trial,Journal,,,Randomized treatment evidence,en\n",
        encoding="utf-8",
    )
    canonical = source / "1-Steroid Trial.pdf"
    duplicate = source / "1-Steroid Trial copy.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text(
        (72, 72),
        "Methods randomized treatment evidence with enough characters for extraction. "
        "Results low dose treatment improved recovery without severe events.",
    )
    pdf.save(canonical)
    pdf.close()
    duplicate.write_bytes(canonical.read_bytes())
    store = SQLiteStore(tmp_path / "manifest.sqlite3")
    store.initialize()

    result = _indexer(tmp_path, store).build(
        build_default_pipeline(source, store=store, ocr=lambda _: ""),
        confirm_embedding_cost=False,
    )

    files = store.list_files()
    assert len(files) == 2
    assert sum(item.status == "duplicate" for item in files) == 1
    assert result.stage_counters["pdf_match"]["duplicate"] == 1
    parsed = [item for item in files if item.status in {"parsed", "partial"}]
    assert len(parsed) == 1
    document_id = parsed[0].document_id
    assert document_id is not None
    chunks = store.list_chunks_for_document(document_id, limit=100)
    assert chunks
    assert {chunk.file_id for chunk in chunks} == {parsed[0].file_id}


def test_full_embedding_gate_requires_a_successful_bounded_probe(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "manifest.sqlite3")
    store.initialize()

    with pytest.raises(KnowledgeBaseError) as missing_probe:
        validate_embedding_cost_gate(
            store,
            model_name="embedding-model",
            confirm_embedding_cost=True,
            confirm_full_embedding_cost=True,
            embedding_limit=None,
        )
    assert missing_probe.value.code == "embedding_probe_required"

    store.mark_embedding_probe_completed("embedding-model")
    validate_embedding_cost_gate(
        store,
        model_name="embedding-model",
        confirm_embedding_cost=True,
        confirm_full_embedding_cost=True,
        embedding_limit=None,
    )


def test_bounded_embedding_probe_covers_metadata_and_fulltext(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    for index in (2, 3):
        store.upsert_document(
            DocumentRecord(
                document_id=f"doc-{index}",
                source_row=index,
                title=f"Metadata study {index}",
                normalized_title=f"metadatastudy{index}",
                authors=("Zhang",),
                year=2025,
                journal="Journal",
                doi="",
                normalized_doi="",
                abstract="Metadata-only evidence",
                language="en",
            )
        )
    embedder = FakeEmbeddingClient()
    vectors = FakeVectorStore()

    result = _indexer(tmp_path, store, embedder, vectors).embed_pending(
        confirm_embedding_cost=True,
        embedding_limit=2,
    )

    assert result.completed_embeddings == 2
    assert sum(len(batch) for batch in vectors.metadata_batches) == 1
    assert sum(len(batch) for batch in vectors.fulltext_batches) == 1
