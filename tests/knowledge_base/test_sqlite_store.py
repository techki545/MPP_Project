import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sqlite3
import struct
import zlib

import pytest

from knowledge_base.errors import KnowledgeBaseError
from knowledge_base.models import ChunkRecord, DocumentRecord, FileRecord, SearchFilters
import knowledge_base.sqlite_store as sqlite_store_module
from knowledge_base.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    result = SQLiteStore(tmp_path / "nested" / "manifest.sqlite3")
    result.initialize()
    return result


@pytest.fixture
def document() -> DocumentRecord:
    return DocumentRecord(
        document_id="doc-1",
        source_row=7,
        title="儿童重症支原体肺炎糖皮质激素治疗",
        normalized_title="儿童重症支原体肺炎糖皮质激素治疗",
        authors=("张三", "李四"),
        year=2024,
        journal="儿科医学",
        doi="10.1000/example",
        normalized_doi="10.1000/example",
        abstract="比较低剂量和高剂量甲泼尼龙。",
        language="zh",
        evidence_type="randomized_trial",
        classification_confidence=0.9,
        classification_basis="title",
        has_fulltext=True,
        fulltext_status="available",
    )


@pytest.fixture
def file_record() -> FileRecord:
    return FileRecord(
        file_id="file-1",
        path="articles/doc-1.pdf",
        sha256="a" * 64,
        size_bytes=1234,
        document_id="doc-1",
        match_method="doi",
        match_confidence=1.0,
        status="indexed",
    )


@pytest.fixture
def chunk() -> ChunkRecord:
    return ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        file_id="file-1",
        section="results",
        page_start=2,
        page_end=3,
        text="低剂量组与高剂量组疗效相近。",
        token_count=8,
        is_ocr=False,
        quality="high",
        content_hash="chunk-content-hash",
    )


def insert_document_graph(
    store: SQLiteStore,
    document: DocumentRecord,
    file_record: FileRecord,
    chunk: ChunkRecord,
) -> None:
    store.upsert_document(document)
    store.upsert_file(file_record)
    store.upsert_chunk(chunk)


def test_document_file_chunk_round_trip_and_chinese_fts_search(
    store: SQLiteStore,
    document: DocumentRecord,
    file_record: FileRecord,
    chunk: ChunkRecord,
):
    insert_document_graph(store, document, file_record, chunk)

    assert store.get_document("doc-1") == document
    assert [hit.record_id for hit in store.search_metadata("低剂量 高剂量", 5)] == [
        "doc-1"
    ]
    hits = store.search_fulltext("低剂量 高剂量", 5)
    assert [(hit.record_id, hit.document_id) for hit in hits] == [("chunk-1", "doc-1")]
    assert hits[0].text == chunk.text


def test_shared_records_are_immutable(document: DocumentRecord):
    with pytest.raises(FrozenInstanceError):
        document.title = "changed"  # type: ignore[misc]


def test_initialize_reports_schema_version_and_wal(store: SQLiteStore):
    assert store.schema_version() == 1
    assert store.journal_mode().lower() == "wal"


def test_embedding_cache_round_trip_is_model_scoped(store: SQLiteStore):
    vector = [0.25, -1.5, 3.75]

    store.put_cached_embedding("content-hash", "model-a", vector)

    assert store.get_cached_embedding("content-hash", "model-a") == pytest.approx(vector)
    assert store.get_cached_embedding("content-hash", "model-b") is None


def test_embedding_cache_uses_little_endian_float32(store: SQLiteStore):
    vector = [0.25, -1.5]
    store.put_cached_embedding("content-hash", "model-a", vector)

    with sqlite3.connect(store.path) as connection:
        blob = connection.execute(
            """
            SELECT vector_blob FROM embedding_cache
            WHERE content_hash = ? AND model_name = ?
            """,
            ("content-hash", "model-a"),
        ).fetchone()[0]

    assert zlib.decompress(blob) == struct.pack("<2f", *vector)


@pytest.mark.parametrize(
    ("content_hash", "dimension", "blob"),
    [
        ("invalid-zlib", 1, b"not-zlib"),
        ("invalid-byte-length", 1, zlib.compress(b"\x00\x00\x00")),
        ("mismatched-dimension", 1, zlib.compress(struct.pack("<2f", 1.0, 2.0))),
    ],
)
def test_corrupt_embedding_cache_raises_stable_error(
    store: SQLiteStore, content_hash: str, dimension: int, blob: bytes
):
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            INSERT INTO embedding_cache(content_hash, model_name, dimension, vector_blob)
            VALUES (?, ?, ?, ?)
            """,
            (content_hash, "model-a", dimension, blob),
        )

    with pytest.raises(KnowledgeBaseError) as exc_info:
        store.get_cached_embedding(content_hash, "model-a")

    assert exc_info.value.code == "embedding_cache_corrupt"


def test_sqlite_store_source_is_python_310_compatible():
    source = Path(sqlite_store_module.__file__).read_text(encoding="utf-8")

    ast.parse(source, feature_version=(3, 10))


def test_pending_embedding_items_are_scoped_to_model(
    store: SQLiteStore,
    document: DocumentRecord,
    file_record: FileRecord,
    chunk: ChunkRecord,
):
    insert_document_graph(store, document, file_record, chunk)

    pending = store.list_pending_embedding_items("model-a")
    assert {(item.record_id, item.kind) for item in pending} == {
        ("doc-1", "metadata"),
        ("chunk-1", "chunk"),
    }
    assert all(item.content_hash for item in pending)

    for item in pending:
        assert store.mark_embedding_indexed(
            item.record_id, item.kind, "model-a", item.content_hash
        )

    assert store.list_pending_embedding_items("model-a") == []
    assert {(item.record_id, item.kind) for item in store.list_pending_embedding_items("model-b")} == {
        ("doc-1", "metadata"),
        ("chunk-1", "chunk"),
    }


def test_changed_metadata_becomes_pending_after_being_indexed(
    store: SQLiteStore,
    document: DocumentRecord,
    file_record: FileRecord,
    chunk: ChunkRecord,
):
    insert_document_graph(store, document, file_record, chunk)
    original_item = next(
        item
        for item in store.list_pending_embedding_items("model-a")
        if item.record_id == "doc-1"
    )
    assert store.mark_embedding_indexed(
        "doc-1", "metadata", "model-a", original_item.content_hash
    )

    updated_document = replace(document, abstract="更新后的甲泼尼龙剂量比较。")
    store.upsert_document(updated_document)

    pending = store.list_pending_embedding_items("model-a")
    metadata_item = next(item for item in pending if item.record_id == "doc-1")
    assert metadata_item.kind == "metadata"
    assert metadata_item.content_hash != original_item.content_hash


def test_changed_chunk_becomes_pending_after_being_indexed(
    store: SQLiteStore,
    document: DocumentRecord,
    file_record: FileRecord,
    chunk: ChunkRecord,
):
    insert_document_graph(store, document, file_record, chunk)
    original_item = next(
        item
        for item in store.list_pending_embedding_items("model-a")
        if item.record_id == "chunk-1"
    )
    assert store.mark_embedding_indexed(
        "chunk-1", "chunk", "model-a", original_item.content_hash
    )

    updated_chunk = replace(
        chunk,
        text="更新后的低剂量组与高剂量组疗效比较。",
        content_hash="updated-chunk-content-hash",
    )
    store.upsert_chunk(updated_chunk)

    pending = store.list_pending_embedding_items("model-a")
    chunk_item = next(item for item in pending if item.record_id == "chunk-1")
    assert chunk_item.kind == "chunk"
    assert chunk_item.content_hash == "updated-chunk-content-hash"


def test_stale_metadata_item_is_not_marked_current(
    store: SQLiteStore,
    document: DocumentRecord,
    file_record: FileRecord,
    chunk: ChunkRecord,
):
    insert_document_graph(store, document, file_record, chunk)
    item = next(
        item
        for item in store.list_pending_embedding_items("model-a")
        if item.record_id == "doc-1"
    )
    store.upsert_document(replace(document, abstract="更新后的元数据内容。"))

    assert not store.mark_embedding_indexed(
        item.record_id, item.kind, "model-a", item.content_hash
    )
    assert any(
        pending.record_id == "doc-1"
        for pending in store.list_pending_embedding_items("model-a")
    )


def test_stale_chunk_item_is_not_marked_current(
    store: SQLiteStore,
    document: DocumentRecord,
    file_record: FileRecord,
    chunk: ChunkRecord,
):
    insert_document_graph(store, document, file_record, chunk)
    item = next(
        item
        for item in store.list_pending_embedding_items("model-a")
        if item.record_id == "chunk-1"
    )
    store.upsert_chunk(
        replace(
            chunk,
            text="更新后的正文内容。",
            content_hash="stale-chunk-content-hash",
        )
    )

    assert not store.mark_embedding_indexed(
        item.record_id, item.kind, "model-a", item.content_hash
    )
    assert any(
        pending.record_id == "chunk-1"
        for pending in store.list_pending_embedding_items("model-a")
    )


def test_document_fts_replaces_stale_terms_on_upsert(
    store: SQLiteStore,
    document: DocumentRecord,
):
    store.upsert_document(replace(document, title="legacymetadata", abstract=""))
    store.upsert_document(replace(document, title="freshmetadata", abstract=""))

    assert store.search_metadata("legacymetadata", 5) == []
    assert [hit.record_id for hit in store.search_metadata("freshmetadata", 5)] == ["doc-1"]


def test_chunk_fts_replaces_stale_terms_on_upsert(
    store: SQLiteStore,
    document: DocumentRecord,
    file_record: FileRecord,
    chunk: ChunkRecord,
):
    insert_document_graph(
        store,
        document,
        file_record,
        replace(chunk, text="legacychunk", content_hash="original-hash"),
    )
    store.upsert_chunk(
        replace(chunk, text="freshchunk", content_hash="replacement-hash")
    )

    assert store.search_fulltext("legacychunk", 5) == []
    assert [hit.record_id for hit in store.search_fulltext("freshchunk", 5)] == ["chunk-1"]


def test_job_progress_and_error_records_round_trip(store: SQLiteStore):
    progress = {"completed": 3, "total": 10, "phase": "extract"}
    store.set_job_state("job-1", "running", progress)
    store.record_error("extract", "chunk-1", "pdf_failed", "Could not read PDF")

    assert store.get_job("job-1") == {
        "job_id": "job-1",
        "state": "running",
        "progress": progress,
    }
    assert store.list_errors() == [
        {
            "stage": "extract",
            "record_id": "chunk-1",
            "code": "pdf_failed",
            "message": "Could not read PDF",
        }
    ]
    assert store.list_errors(stage="other") == []


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_search_limit_raises_stable_domain_error(
    store: SQLiteStore, limit: int
):
    with pytest.raises(KnowledgeBaseError) as exc_info:
        store.search_metadata("query", limit)

    assert exc_info.value.code == "invalid_search_limit"


@pytest.mark.parametrize("limit", [None, "5"])
def test_invalid_search_limit_raises_stable_domain_error(
    store: SQLiteStore, limit: object
):
    with pytest.raises(KnowledgeBaseError) as exc_info:
        store.search_fulltext("query", limit)  # type: ignore[arg-type]

    assert exc_info.value.code == "invalid_search_limit"


def test_search_filters_apply_to_metadata(
    store: SQLiteStore,
    document: DocumentRecord,
    file_record: FileRecord,
    chunk: ChunkRecord,
):
    insert_document_graph(store, document, file_record, chunk)

    assert store.search_metadata(
        "低剂量", 5, SearchFilters(evidence_types=frozenset({"systematic_review"}))
    ) == []
    assert store.search_metadata("低剂量", 5, SearchFilters(year_from=2025)) == []
    assert [hit.record_id for hit in store.search_metadata("低剂量", 5, SearchFilters(fulltext_only=True))] == [
        "doc-1"
    ]
    assert store.search_fulltext(
        "低剂量", 5, SearchFilters(evidence_types=frozenset({"systematic_review"}))
    ) == []
    assert store.search_fulltext("低剂量", 5, SearchFilters(year_to=2023)) == []
    store.upsert_document(replace(document, has_fulltext=False))
    assert store.search_metadata("低剂量", 5, SearchFilters(fulltext_only=True)) == []
