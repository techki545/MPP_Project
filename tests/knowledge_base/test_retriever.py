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


class FakeE5EmbeddingClient(FakeEmbeddingClient):
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.calls.append([f"query::{text}" for text in texts])
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


def test_hybrid_retrieval_prefers_query_specific_embedding_method() -> None:
    embedder = FakeE5EmbeddingClient()
    evidence = hit("doc-a", "doc-a", "metadata", 1)

    HybridRetriever(
        embedder,
        FakeVectorStore(metadata=[evidence]),
        FakeLexicalStore(metadata=[evidence]),
    ).search("clinical question", SearchFilters())

    assert embedder.calls == [["query::clinical question"]]


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


def test_broad_chinese_treatment_question_is_typed_and_decomposed() -> None:
    context = build_query_context("重症支原体肺炎儿童要吃什么药？")

    assert context.question_type == "treatment"
    assert "重症支原体肺炎" in context.condition_terms
    assert "儿童" in context.population_terms
    assert len(context.retrieval_queries) == 3
    assert "抗菌药物" in context.retrieval_queries[1]
    assert "糖皮质激素" in context.retrieval_queries[2]
    assert context.as_dict()["pico"]["population"]


def test_broad_duration_question_decomposes_component_specific_courses() -> None:
    context = build_query_context("重症支原体肺炎儿童治疗周期是多长时间？")

    assert context.question_type == "treatment"
    assert len(context.retrieval_queries) == 3
    assert "阿奇霉素" in context.retrieval_queries[1]
    assert "疗程" in context.retrieval_queries[1]
    assert "甲泼尼龙" in context.retrieval_queries[2]
    assert "疗程" in context.retrieval_queries[2]


def test_broad_medication_wording_is_typed_and_decomposed() -> None:
    context = build_query_context("重症支原体肺炎儿童要用哪些药？")

    assert context.question_type == "treatment"
    assert "抗菌药物" in context.retrieval_queries[1]
    assert "糖皮质激素" in context.retrieval_queries[2]


@pytest.mark.parametrize(
    ("question", "question_type", "expected_focus"),
    [
        ("如何诊断儿童重症支原体肺炎？", "diagnosis", "诊断标准"),
        ("儿童支原体肺炎为什么会发展成重症？", "etiology", "危险因素"),
        ("阿奇霉素治疗儿童SMPP有哪些不良反应？", "safety", "QT间期"),
    ],
)
def test_general_clinical_intents_add_matching_focus_queries(
    question: str, question_type: str, expected_focus: str
) -> None:
    context = build_query_context(question)

    assert context.question_type == question_type
    assert len(context.retrieval_queries) == 2
    assert expected_focus in context.retrieval_queries[1]


def test_broad_treatment_question_runs_decomposed_vector_queries() -> None:
    embedder = FakeEmbeddingClient()
    evidence = hit("doc-a", "doc-a", "metadata", 1)
    vectors = FakeVectorStore(metadata=[evidence])
    lexical = FakeLexicalStore(metadata=[evidence])

    result = HybridRetriever(
        embedder,
        vectors,
        lexical,
    ).search("重症支原体肺炎儿童要吃什么药？", SearchFilters())

    assert len(embedder.calls[0]) == 3
    assert len(vectors.calls) == 6
    assert len(lexical.calls) == 6
    assert [item.document_id for item in result.documents] == ["doc-a"]


@pytest.mark.parametrize(
    ("question", "expected_focus"),
    [
        ("儿童SMPP何时开始糖皮质激素治疗更合适？", "early corticosteroid"),
        ("大环内酯耐药MPP应如何识别并调整治疗？", "macrolide resistant"),
        ("何时可考虑多西环素或米诺环素？", "doxycycline minocycline"),
        ("MPP合并肺栓塞时抗凝治疗如何决策？", "pulmonary embolism"),
        ("儿童难治性MPP如何早期诊断？", "refractory early diagnosis"),
        ("儿童SMPP达到重症标准的危险因素是什么？", "severe diagnostic criteria"),
        ("SMPP后如何随访闭塞性细支气管炎？", "bronchiolitis obliterans"),
    ],
)
def test_domain_focus_questions_add_concise_bilingual_queries(
    question: str, expected_focus: str
) -> None:
    context = build_query_context(question)

    assert len(context.retrieval_queries) == 2
    assert expected_focus in context.retrieval_queries[1]


def test_focused_query_runs_both_vector_and_lexical_retrieval() -> None:
    embedder = FakeEmbeddingClient()
    evidence = hit("doc-a", "doc-a", "metadata", 1)
    vectors = FakeVectorStore(metadata=[evidence])
    lexical = FakeLexicalStore(metadata=[evidence])

    HybridRetriever(embedder, vectors, lexical).search(
        "儿童SMPP何时开始糖皮质激素治疗更合适？", SearchFilters()
    )

    assert len(embedder.calls[0]) == 2
    assert len(vectors.calls) == 4
    assert len(lexical.calls) == 4
    assert {call[0] for call in lexical.calls} == {"metadata", "fulltext"}


@pytest.mark.parametrize(
    "question",
    [
        "大环内酯耐药MPP应如何识别并调整治疗？",
        "何时可考虑多西环素或米诺环素？",
        "何时需要支气管镜肺泡灌洗？",
        "何时可考虑左氧氟沙星，其获益和风险是什么？",
    ],
)
def test_mixed_intent_treatment_questions_prioritize_the_decision(question: str) -> None:
    assert build_query_context(question).question_type == "treatment"


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


def test_smpp_reranking_labels_direct_and_rmpp_extrapolated_evidence() -> None:
    direct = hit("direct", "direct", "metadata", 2, text="重症支原体肺炎 糖皮质激素")
    indirect = hit("indirect", "indirect", "metadata", 1, text="难治性支原体肺炎 糖皮质激素")
    result = HybridRetriever(
        FakeEmbeddingClient(),
        FakeVectorStore(metadata=[indirect, direct]),
        FakeLexicalStore(),
    ).search("重症支原体肺炎儿童是否使用糖皮质激素？", SearchFilters())

    by_id = {item.document_id: item for item in result.documents}
    assert by_id["direct"].payload["population_applicability"] == "direct"
    assert by_id["indirect"].payload["population_applicability"] == "indirect_rmpp"
    assert by_id["direct"].payload["population_applicability_score"] == 1.0
    assert by_id["direct"].final_score > by_id["indirect"].final_score


def test_question_aware_reranking_penalizes_wrong_clinical_focus() -> None:
    focused = hit("focused", "focused", "metadata", 2, text="重症支原体肺炎儿童运动与康复训练")
    wrong_focus = hit("wrong", "wrong", "metadata", 1, text="重症支原体肺炎儿童糖皮质激素治疗")
    result = HybridRetriever(
        FakeEmbeddingClient(),
        FakeVectorStore(metadata=[wrong_focus, focused]),
        FakeLexicalStore(),
    ).search("重症支原体肺炎儿童可以运动吗？", SearchFilters())

    by_id = {item.document_id: item for item in result.documents}
    assert by_id["focused"].payload["question_relevance_score"] == 1.0
    assert by_id["wrong"].payload["question_relevance_score"] == 0.0
    assert by_id["focused"].final_score > by_id["wrong"].final_score


def test_local_vector_similarity_reranks_post_fusion_candidates() -> None:
    stronger_semantics = hit(
        "strong",
        "strong",
        "metadata",
        2,
        score=0.95,
        text="重症支原体肺炎儿童糖皮质激素治疗",
    )
    weaker_semantics = hit(
        "weak",
        "weak",
        "metadata",
        1,
        score=0.55,
        text="重症支原体肺炎儿童糖皮质激素治疗",
    )
    result = HybridRetriever(
        FakeEmbeddingClient(),
        FakeVectorStore(metadata=[weaker_semantics, stronger_semantics]),
        FakeLexicalStore(),
    ).search("重症支原体肺炎儿童是否使用糖皮质激素？", SearchFilters())

    by_id = {item.document_id: item for item in result.documents}
    assert by_id["strong"].payload["semantic_similarity_score"] == 0.95
    assert by_id["weak"].payload["semantic_similarity_score"] == 0.55
    assert result.documents[0].document_id == "strong"


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


def test_pediatric_query_excludes_explicit_adult_only_evidence() -> None:
    adult = replace(
        hit("doc-adult", "doc-adult", "metadata", 1),
        text="Treatment outcomes in adults with Mycoplasma pneumoniae pneumonia",
        payload={
            "document_id": "doc-adult",
            "title": "MPP treatment in adult patients",
            "abstract": "A cohort of adults aged 40 years.",
            "evidence_type": "observational_study",
            "year": 2024,
        },
    )
    pediatric = replace(
        hit("doc-child", "doc-child", "metadata", 2),
        text="Treatment outcomes in children with Mycoplasma pneumoniae pneumonia",
        payload={
            "document_id": "doc-child",
            "title": "MPP treatment in children",
            "abstract": "A pediatric cohort.",
            "evidence_type": "observational_study",
            "year": 2024,
        },
    )

    result = HybridRetriever(
        FakeEmbeddingClient(),
        FakeVectorStore(metadata=[adult, pediatric]),
        FakeLexicalStore(metadata=[adult, pediatric]),
    ).search("儿童肺炎支原体肺炎如何治疗？", SearchFilters())

    assert [item.document_id for item in result.documents] == ["doc-child"]


def test_pediatric_query_retains_evidence_with_unspecified_population() -> None:
    unspecified = replace(
        hit("doc-unknown", "doc-unknown", "metadata", 1),
        text="Mycoplasma pneumoniae treatment guideline",
    )

    result = HybridRetriever(
        FakeEmbeddingClient(),
        FakeVectorStore(metadata=[unspecified]),
        FakeLexicalStore(),
    ).search("SMPP儿童如何治疗？", SearchFilters())

    assert [item.document_id for item in result.documents] == ["doc-unknown"]
