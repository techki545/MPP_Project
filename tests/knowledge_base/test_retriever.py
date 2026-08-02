from __future__ import annotations

from dataclasses import replace

import pytest

from knowledge_base.errors import KnowledgeBaseError
from knowledge_base.models import RankedHit, SearchFilters
from knowledge_base.retriever import HybridRetriever, build_query_context


def hit(
    record_id: str,
    document_id: str,
    source: str,
    rank: int,
    *,
    score: float = 0.9,
    text: str = "evidence",
    evidence_type: str = "unknown",
    quality: str = "unknown",
    year: int | None = 2024,
    has_fulltext: bool = False,
) -> RankedHit:
    return RankedHit(
        record_id=record_id,
        document_id=document_id,
        source=source,
        rank=rank,
        score=score,
        text=text,
        payload={
            "document_id": document_id,
            "evidence_type": evidence_type,
            "quality": quality,
            "year": year,
            "has_fulltext": has_fulltext,
        },
    )


class FakeEmbeddingClient:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(
        self,
        metadata: list[RankedHit] | None = None,
        fulltext: list[RankedHit] | None = None,
    ) -> None:
        self.metadata = metadata or []
        self.fulltext = fulltext or []
        self.calls: list[tuple[str, int, object]] = []

    def query_metadata(self, vector, *, limit, query_filter=None):
        self.calls.append(("metadata", limit, query_filter))
        return self.metadata

    def query_fulltext(self, vector, *, limit, query_filter=None):
        self.calls.append(("fulltext", limit, query_filter))
        return self.fulltext


class FakeLexicalStore:
    def __init__(
        self,
        metadata: list[RankedHit] | None = None,
        fulltext: list[RankedHit] | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self.metadata = metadata or []
        self.fulltext = fulltext or []
        self.fail = fail
        self.calls: list[tuple[str, str, int, SearchFilters]] = []

    def search_metadata(self, query, limit, filters=None):
        if self.fail:
            raise RuntimeError("fts unavailable")
        self.calls.append(("metadata", query, limit, filters))
        return self.metadata

    def search_fulltext(self, query, limit, filters=None):
        if self.fail:
            raise RuntimeError("fts unavailable")
        self.calls.append(("fulltext", query, limit, filters))
        return self.fulltext


def test_hybrid_retrieval_uses_four_lists_and_deduplicates_documents() -> None:
    embedder = FakeEmbeddingClient()
    vectors = FakeVectorStore(
        metadata=[hit("doc-a", "doc-a", "metadata", 1, has_fulltext=True)],
        fulltext=[hit("chunk-a", "doc-a", "fulltext", 1, has_fulltext=True)],
    )
    lexical = FakeLexicalStore(
        metadata=[hit("doc-b", "doc-b", "metadata", 1)],
        fulltext=[hit("chunk-a", "doc-a", "fulltext", 1, has_fulltext=True)],
    )

    result = HybridRetriever(embedder, vectors, lexical).search(
        "low dose steroid", SearchFilters()
    )

    assert result.mode == "hybrid"
    assert result.degraded_reason == ""
    assert [item.document_id for item in result.documents] == ["doc-a", "doc-b"]
    assert [call[1] for call in vectors.calls] == [40, 80]
    assert [call[2] for call in lexical.calls] == [40, 80]
    assert embedder.calls == [["low dose steroid"]]
    assert {item.source for item in result.documents[0].supporting_hits} == {
        "vector_metadata",
        "vector_fulltext",
        "fts_fulltext",
    }


def test_embedding_failure_returns_explicit_keyword_mode() -> None:
    embedder = FakeEmbeddingClient(fail=True)
    lexical = FakeLexicalStore(
        metadata=[hit("doc-b", "doc-b", "metadata", 1)],
        fulltext=[],
    )

    result = HybridRetriever(
        embedder,
        FakeVectorStore(metadata=[hit("doc-a", "doc-a", "metadata", 1)]),
        lexical,
    ).search("low dose steroid", SearchFilters())

    assert result.mode == "keyword"
    assert result.degraded_reason == "query_embedding_unavailable"
    assert [item.document_id for item in result.documents] == ["doc-b"]


def test_both_embedding_and_fts_failure_raises_retrieval_unavailable() -> None:
    with pytest.raises(KnowledgeBaseError) as captured:
        HybridRetriever(
            FakeEmbeddingClient(fail=True),
            FakeVectorStore(),
            FakeLexicalStore(fail=True),
        ).search("steroid", SearchFilters())

    assert captured.value.code == "retrieval_unavailable"


def test_query_context_detects_known_pico_terms_and_preserves_unknown_terms() -> None:
    context = build_query_context("SMPP children low-dose steroid versus pulse dose for fever")

    assert "children" in context.population_terms
    assert "steroid" in context.intervention_terms
    assert "pulse dose" in context.comparator_terms
    assert "fever" in context.outcome_terms
    assert "smpp" in context.normalized_fts_query


def test_invalid_year_range_is_rejected_before_any_search() -> None:
    embedder = FakeEmbeddingClient()

    with pytest.raises(KnowledgeBaseError) as captured:
        HybridRetriever(embedder, FakeVectorStore(), FakeLexicalStore()).search(
            "steroid", SearchFilters(year_from=2025, year_to=2020)
        )

    assert captured.value.code == "invalid_year_range"
    assert embedder.calls == []


def test_filters_are_passed_to_all_lists_and_applied_before_reranking() -> None:
    allowed = hit(
        "doc-rct",
        "doc-rct",
        "metadata",
        2,
        evidence_type="randomized_controlled_trial",
        year=2024,
        has_fulltext=True,
    )
    excluded = hit(
        "doc-old",
        "doc-old",
        "metadata",
        1,
        evidence_type="case_report",
        year=2010,
    )
    filters = SearchFilters(
        evidence_types=frozenset({"randomized_controlled_trial"}),
        year_from=2020,
        year_to=2025,
        fulltext_only=True,
    )
    vectors = FakeVectorStore(metadata=[excluded, allowed])
    lexical = FakeLexicalStore(metadata=[excluded, allowed])

    result = HybridRetriever(FakeEmbeddingClient(), vectors, lexical).search(
        "steroid", filters
    )

    assert [item.document_id for item in result.documents] == ["doc-rct"]
    assert all(call[2] is not None for call in vectors.calls)
    assert all(call[3] == filters for call in lexical.calls)
    assert result.query_context.year_from == 2020
    assert result.query_context.year_to == 2025
    assert result.query_context.evidence_types == (
        "randomized_controlled_trial",
    )
    assert result.query_context.fulltext_only is True


def test_reranking_uses_exact_weights_and_year_normalization() -> None:
    newest = hit(
        "doc-new",
        "doc-new",
        "metadata",
        1,
        evidence_type="guideline",
        quality="high",
        year=2025,
        has_fulltext=True,
    )
    oldest = hit(
        "doc-old",
        "doc-old",
        "metadata",
        2,
        evidence_type="unknown",
        quality="unknown",
        year=2015,
    )
    result = HybridRetriever(
        FakeEmbeddingClient(),
        FakeVectorStore(metadata=[newest, oldest]),
        FakeLexicalStore(),
    ).search("steroid", SearchFilters())

    expected_new = 0.65 + 0.15 + 0.10 + 0.05 + 0.05 * 0.4
    assert result.documents[0].document_id == "doc-new"
    assert result.documents[0].final_score == pytest.approx(expected_new)
    assert result.documents[1].final_score < result.documents[0].final_score


def test_at_most_three_fulltext_chunks_are_retained_per_document() -> None:
    vector_chunks = [
        hit(f"v-{index}", "doc-a", "fulltext", index, has_fulltext=True)
        for index in range(1, 5)
    ]
    fts_chunks = [
        hit(f"f-{index}", "doc-a", "fulltext", index, has_fulltext=True)
        for index in range(1, 5)
    ]
    result = HybridRetriever(
        FakeEmbeddingClient(),
        FakeVectorStore(fulltext=vector_chunks),
        FakeLexicalStore(fulltext=fts_chunks),
    ).search("steroid", SearchFilters())

    supporting = result.documents[0].supporting_hits
    assert len([item for item in supporting if "fulltext" in item.source]) == 3


def test_candidate_pool_is_bounded_and_safety_evidence_is_retained() -> None:
    all_hits = [
        hit(
            f"doc-{index:03d}",
            f"doc-{index:03d}",
            "metadata",
            index,
            evidence_type="observational_study",
            quality="moderate",
        )
        for index in range(1, 241)
    ]
    safety = replace(
        all_hits[4],
        text="hypertension adverse events and safety",
        payload={**all_hits[4].payload, "evidence_type": "case_report"},
    )
    all_hits[4] = safety

    result = HybridRetriever(
        FakeEmbeddingClient(),
        FakeVectorStore(metadata=all_hits[:40], fulltext=all_hits[40:120]),
        FakeLexicalStore(metadata=all_hits[120:160], fulltext=all_hits[160:]),
    ).search("steroid safety", SearchFilters())

    assert result.candidate_count == 200
    assert len(result.documents) == 15
    assert "doc-005" in {item.document_id for item in result.documents}
