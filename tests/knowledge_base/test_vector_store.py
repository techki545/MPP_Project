from pathlib import Path
from uuid import UUID

import pytest

from knowledge_base.errors import KnowledgeBaseError
from knowledge_base.vector_store import LocalVectorStore


def test_local_qdrant_upsert_and_query_returns_ranked_payload_hit() -> None:
    store = LocalVectorStore(":memory:")
    try:
        store.ensure_collections(dimension=3, model_name="test-model")
        store.upsert_metadata(
            [("document / arbitrary id", [1.0, 0.0, 0.0], {"document_id": "doc-1", "title": "RCT", "text": "abstract text"})]
        )

        hits = store.query_metadata([1.0, 0.0, 0.0], limit=5)

        assert len(hits) == 1
        assert hits[0].record_id == "document / arbitrary id"
        assert hits[0].document_id == "doc-1"
        assert hits[0].source == "metadata"
        assert hits[0].rank == 1
        assert hits[0].score == pytest.approx(1.0)
        assert hits[0].text == "abstract text"
        assert hits[0].payload["title"] == "RCT"
        assert isinstance(store.point_id_for("document / arbitrary id"), UUID)
    finally:
        store.close()


def test_collections_have_cosine_manifest_and_are_separate() -> None:
    store = LocalVectorStore(":memory:")
    try:
        store.ensure_collections(dimension=2, model_name="model-a")
        metadata_info = store.client.get_collection("mpp_metadata")
        fulltext_info = store.client.get_collection("mpp_fulltext")

        assert metadata_info.config.params.vectors.size == 2
        assert metadata_info.config.params.vectors.distance.value == "Cosine"
        assert metadata_info.config.metadata == {"embedding_model": "model-a", "schema_version": 1}
        assert fulltext_info.config.metadata == metadata_info.config.metadata

        store.upsert_metadata([("meta", [1, 0], {"document_id": "doc-meta", "text": "metadata"})])
        store.upsert_fulltext([("chunk", [1, 0], {"document_id": "doc-full", "text": "full text"})])
        assert [hit.document_id for hit in store.query_metadata([1, 0], limit=5)] == ["doc-meta"]
        assert [hit.document_id for hit in store.query_fulltext([1, 0], limit=5)] == ["doc-full"]
    finally:
        store.close()


def test_existing_dimension_or_model_manifest_mismatch_is_rejected() -> None:
    store = LocalVectorStore(":memory:")
    try:
        store.ensure_collections(dimension=3, model_name="model-a")
        with pytest.raises(KnowledgeBaseError) as dimension_error:
            store.ensure_collections(dimension=4, model_name="model-a")
        assert dimension_error.value.code == "vector_collection_mismatch"

        with pytest.raises(KnowledgeBaseError) as model_error:
            store.ensure_collections(dimension=3, model_name="model-b")
        assert model_error.value.code == "vector_collection_mismatch"
    finally:
        store.close()


def test_reopen_persistent_store_preserves_collection_manifest_and_points(tmp_path: Path) -> None:
    path = tmp_path / "qdrant"
    first = LocalVectorStore(path)
    first.ensure_collections(dimension=2, model_name="model-a")
    first.upsert_metadata([("doc", [0, 1], {"document_id": "doc", "text": "persisted"})])
    first.close()

    second = LocalVectorStore(path)
    try:
        second.ensure_collections(dimension=2, model_name="model-a")
        assert [hit.record_id for hit in second.query_metadata([0, 1], limit=1)] == ["doc"]
    finally:
        second.close()


@pytest.mark.parametrize("vector", [[1.0], [1.0, float("inf")], [True, 0.0]])
def test_upsert_validates_vector_before_qdrant(vector) -> None:
    store = LocalVectorStore(":memory:")
    try:
        store.ensure_collections(dimension=2, model_name="model-a")
        with pytest.raises(KnowledgeBaseError) as exc_info:
            store.upsert_metadata([("doc", vector, {"document_id": "doc"})])
        assert exc_info.value.code == "vector_invalid"
    finally:
        store.close()


def test_empty_batches_are_noops_and_query_limit_must_be_positive() -> None:
    store = LocalVectorStore(":memory:")
    try:
        store.ensure_collections(dimension=2, model_name="model-a")
        store.upsert_metadata([])
        store.upsert_fulltext([])
        with pytest.raises(KnowledgeBaseError) as exc_info:
            store.query_metadata([1.0, 0.0], limit=0)
        assert exc_info.value.code == "vector_query_invalid"
    finally:
        store.close()


def test_upsert_is_idempotent_and_query_can_use_a_qdrant_filter() -> None:
    from qdrant_client.http import models

    store = LocalVectorStore(":memory:")
    try:
        store.ensure_collections(dimension=2, model_name="model-a")
        item = ("doc", [1.0, 0.0], {"document_id": "doc", "text": "v1", "year": 2024})
        store.upsert_metadata([item])
        store.upsert_metadata([("doc", [1.0, 0.0], {"document_id": "doc", "text": "v2", "year": 2025})])
        query_filter = models.Filter(must=[models.FieldCondition(key="year", match=models.MatchValue(value=2025))])
        hits = store.query_metadata([1.0, 0.0], limit=5, query_filter=query_filter)
        assert [(hit.record_id, hit.text) for hit in hits] == [("doc", "v2")]
    finally:
        store.close()
