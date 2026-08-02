from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from knowledge_base.errors import KnowledgeBaseError
from knowledge_base.graph_builder import LocalGraphBuilder
from knowledge_base.models import SearchFilters
from knowledge_base.reporter import (
    ChatEvidenceRefiner,
    EvidenceBundle,
    EvidenceSource,
    GroundedClaimExtractor,
    GroundedReporter,
)
from knowledge_base.service import ProductionQueryPipeline


class _TimeoutChatClient:
    model = "deepseek-chat"

    def complete_json(self, system_prompt: str, payload: dict[str, Any]) -> dict:
        raise KnowledgeBaseError("chat_timeout", "Chat request timed out")


class _Retriever:
    def __init__(self, documents: tuple[SimpleNamespace, ...]) -> None:
        self.documents = documents

    def search(self, question: str, filters: SearchFilters) -> SimpleNamespace:
        return SimpleNamespace(
            documents=self.documents,
            mode="keyword",
            degraded_reason="vector_unavailable",
            candidate_count=len(self.documents),
        )


class _DocumentStore:
    def __init__(self, details: dict[str, dict[str, Any]]) -> None:
        self.details = details

    def document_detail(self, document_id: str) -> dict[str, Any]:
        return self.details[document_id]


def _document(
    number: int,
    *,
    title: str,
    evidence_type: str,
    year: int,
    snippet: str,
) -> tuple[SimpleNamespace, dict[str, Any]]:
    document_id = f"doc-{number}"
    hit = SimpleNamespace(
        source="fulltext",
        record_id=f"chunk-{number}",
        text=snippet,
        payload={"page_start": number, "page_end": number},
    )
    item = SimpleNamespace(
        document_id=document_id,
        payload={},
        supporting_hits=(hit,),
    )
    detail = {
        "title": title,
        "abstract": snippet,
        "evidence_type": evidence_type,
        "classification_confidence": 1.0,
        "classification_basis": "metadata",
        "quality": "moderate",
        "year": year,
    }
    return item, detail


def test_timeout_still_returns_extractable_synthesis_and_relation_graph() -> None:
    records = (
        _document(
            1,
            title="儿童肺炎诊疗指南",
            evidence_type="guideline",
            year=2023,
            snippet="指南讨论了重症儿童糖皮质激素治疗的适用范围和证据限制。",
        ),
        _document(
            2,
            title="糖皮质激素治疗系统综述",
            evidence_type="systematic_review",
            year=2024,
            snippet="系统综述汇总了糖皮质激素联合抗菌治疗的随机试验证据。",
        ),
        _document(
            3,
            title="不同剂量甲泼尼龙随机试验",
            evidence_type="randomized_controlled_trial",
            year=2025,
            snippet="随机试验比较了不同剂量方案的长期肺部结局和不良事件。",
        ),
    )
    documents = tuple(record[0] for record in records)
    details = {
        document.document_id: detail
        for document, detail in records
    }
    chat_client = _TimeoutChatClient()
    pipeline = ProductionQueryPipeline(
        retriever=_Retriever(documents),
        document_store=_DocumentStore(details),
        evidence_refiner=ChatEvidenceRefiner(chat_client),
        claim_extractor=GroundedClaimExtractor(chat_client),
        graph_builder=LocalGraphBuilder(),
        reporter=GroundedReporter(chat_client),
    )

    result = pipeline.run(
        "重症支原体肺炎儿童是否应常规使用糖皮质激素？",
        SearchFilters(),
    )

    assert result["model_error"] == "chat_timeout"
    assert result["retrieval_stats"]["claim_count"] == 3
    assert len(result["graph"]["edges"]) >= 3
    assert len(result["reasoning_steps"]) == 6
    assert [step["stage_key"] for step in result["reasoning_steps"]] == [
        "inventory",
        "guidelines",
        "systematic_reviews",
        "randomized_trials",
        "lower_level_evidence",
        "synthesis",
    ]
    assert result["answer_markdown"].startswith("## 综合回答")
    assert "### 证据链" in result["answer_markdown"]
    assert "### 时间更新" in result["answer_markdown"]
    assert "### 安全性与适用边界" in result["answer_markdown"]
    assert "### 证据缺口" in result["answer_markdown"]
    assert "[1]" in result["answer_markdown"]


class _PayloadCaptureClient:
    model = "deepseek-v4-flash"

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.payloads: list[dict[str, Any]] = []

    def complete_json(self, system_prompt: str, payload: dict[str, Any]) -> dict:
        self.payloads.append(payload)
        return self.response


def _large_bundle(source_count: int = 12) -> EvidenceBundle:
    sources = tuple(
        EvidenceSource(
            source_number=number,
            document_id=f"large-doc-{number}",
            title=f"Evidence source {number}",
            evidence_type="observational_study",
            year=2020 + number % 6,
            chunk_ids=(f"large-chunk-{number}",),
            snippets=((f"Grounded evidence sentence {number}. " * 220).strip(),),
            page_ranges=(str(number),),
            fulltext=True,
            abstract=(f"Abstract {number}. " * 150).strip(),
        )
        for number in range(1, source_count + 1)
    )
    return EvidenceBundle(sources=sources, graph={"nodes": [], "edges": []})


def test_claim_extraction_sends_a_bounded_evidence_payload() -> None:
    client = _PayloadCaptureClient({"claims": []})

    GroundedClaimExtractor(client).extract("Clinical question", _large_bundle())

    sent_sources = client.payloads[0]["sources"]
    assert len(sent_sources) == 8
    assert all(sum(map(len, source["snippets"])) <= 1600 for source in sent_sources)
    assert all(len(source["abstract"]) <= 1200 for source in sent_sources)


def test_report_generation_sends_a_bounded_evidence_payload() -> None:
    client = _PayloadCaptureClient({})
    bundle = _large_bundle()

    GroundedReporter(client).generate_with_fallback("Clinical question", bundle)

    sent_sources = client.payloads[0]["evidence_bundle"]["sources"]
    assert len(sent_sources) == 8
    assert all(sum(map(len, source["snippets"])) <= 1600 for source in sent_sources)
    assert all(len(source["abstract"]) <= 1200 for source in sent_sources)


def test_claim_validation_allows_unreported_optional_descriptors() -> None:
    bundle = _large_bundle(source_count=1)
    source = bundle.sources[0]
    quote = source.snippets[0]
    client = _PayloadCaptureClient(
        {
            "claims": [
                {
                    "source_number": 1,
                    "source_chunk_ids": ["large-chunk-1"],
                    "population": "",
                    "intervention": "",
                    "comparator": "",
                    "design": "",
                    "sample_size": "",
                    "dose": "",
                    "outcome": "",
                    "direction": "uncertain",
                    "effect_measures": [],
                    "safety_signal": False,
                    "limitations": [],
                    "statement": quote,
                    "source_quote": quote,
                }
            ]
        }
    )

    validated = GroundedClaimExtractor(client).extract("Clinical question", bundle)

    assert len(validated.claims) == 1
    assert validated.audit == ()


def test_empty_model_claims_fall_back_to_extractive_claims() -> None:
    document, detail = _document(
        1,
        title="儿童肺炎诊疗指南",
        evidence_type="guideline",
        year=2023,
        snippet="指南讨论了重症儿童糖皮质激素治疗的适用范围和证据限制。",
    )
    pipeline = ProductionQueryPipeline(
        retriever=_Retriever((document,)),
        document_store=_DocumentStore({document.document_id: detail}),
        evidence_refiner=ChatEvidenceRefiner(_TimeoutChatClient()),
        claim_extractor=GroundedClaimExtractor(_PayloadCaptureClient({"claims": []})),
        graph_builder=LocalGraphBuilder(),
        reporter=GroundedReporter(_TimeoutChatClient()),
    )

    result = pipeline.run("Clinical question", SearchFilters())

    assert result["retrieval_stats"]["claim_count"] == 1
    assert len(result["graph"]["edges"]) == 1


def test_claim_extractor_resolves_quote_id_to_exact_source_text() -> None:
    bundle = _large_bundle(source_count=1)
    client = _PayloadCaptureClient(
        {
            "claims": [
                {
                    "source_number": 1,
                    "source_chunk_ids": ["model-invented-chunk"],
                    "source_quote_id": "quote-1-1",
                    "population": "",
                    "intervention": "",
                    "comparator": "",
                    "design": "",
                    "sample_size": "",
                    "dose": "",
                    "outcome": "",
                    "direction": "uncertain",
                    "effect_measures": [],
                    "safety_signal": True,
                    "limitations": [],
                    "statement": "A model-written paraphrase that must not be trusted.",
                }
            ]
        }
    )

    validated = GroundedClaimExtractor(client).extract("Clinical question", bundle)

    assert len(validated.claims) == 1
    claim = validated.claims[0]
    assert claim.source_quote == "Grounded evidence sentence 1"
    assert claim.statement == claim.source_quote
    assert claim.source_chunk_ids == ("large-chunk-1",)
    assert claim.safety_signal is False
    sent_option = client.payloads[0]["sources"][0]["quote_options"][0]
    assert sent_option == {
        "quote_id": "quote-1-1",
        "source_chunk_id": "large-chunk-1",
        "text": "Grounded evidence sentence 1",
    }


def test_quote_id_claim_keeps_only_closed_reasoning_labels() -> None:
    bundle = _large_bundle(source_count=1)
    client = _PayloadCaptureClient(
        {
            "claims": [
                {
                    "source_number": 1,
                    "source_quote_id": "quote-1-1",
                    "clinical_aspect": "effectiveness",
                    "evidence_role": "core",
                    "population": "invented adult population",
                    "intervention": "invented treatment",
                    "comparator": "invented placebo",
                    "design": "invented trial design",
                    "sample_size": "99999",
                    "dose": "999 mg/kg/day",
                    "outcome": "invented mortality benefit",
                    "follow_up": "99 years",
                    "direction": "supports",
                    "effect_measures": ["RR 0.01"],
                    "safety_signal": True,
                    "limitations": ["invented limitation"],
                }
            ]
        }
    )

    validated = GroundedClaimExtractor(client).extract("Clinical question", bundle)

    assert len(validated.claims) == 1
    claim = validated.claims[0]
    assert claim.statement == "Grounded evidence sentence 1"
    assert claim.clinical_aspect == "effectiveness"
    assert claim.direction == "supports"
    assert claim.evidence_role == "core"
    assert claim.population == ""
    assert claim.intervention == ""
    assert claim.dose == ""
    assert claim.effect_measures == ()
    assert claim.limitations == ()
    assert claim.safety_signal is False


def test_quote_id_claim_downgrades_unknown_reasoning_labels() -> None:
    bundle = _large_bundle(source_count=1)
    client = _PayloadCaptureClient(
        {
            "claims": [
                {
                    "source_number": 1,
                    "source_quote_id": "quote-1-1",
                    "clinical_aspect": "invented aspect",
                    "direction": "strongly proves",
                    "evidence_role": "definitive",
                }
            ]
        }
    )

    validated = GroundedClaimExtractor(client).extract("Clinical question", bundle)

    assert len(validated.claims) == 1
    claim = validated.claims[0]
    assert claim.clinical_aspect == "other"
    assert claim.direction == "uncertain"
    assert claim.evidence_role == "supplement"


def test_partial_model_claims_are_completed_with_extractive_claims() -> None:
    first, first_detail = _document(
        1,
        title="First evidence source",
        evidence_type="guideline",
        year=2023,
        snippet="First grounded evidence sentence for the clinical question.",
    )
    second, second_detail = _document(
        2,
        title="Second evidence source",
        evidence_type="systematic_review",
        year=2024,
        snippet="Second grounded evidence sentence for the clinical question.",
    )
    claim_client = _PayloadCaptureClient(
        {
            "claims": [
                {
                    "source_number": 1,
                    "source_quote_id": "quote-1-1",
                    "population": "",
                    "intervention": "",
                    "comparator": "",
                    "design": "",
                    "sample_size": "",
                    "dose": "",
                    "outcome": "",
                    "direction": "uncertain",
                    "effect_measures": [],
                    "limitations": [],
                    "statement": "ignored",
                }
            ]
        }
    )
    pipeline = ProductionQueryPipeline(
        retriever=_Retriever((first, second)),
        document_store=_DocumentStore(
            {first.document_id: first_detail, second.document_id: second_detail}
        ),
        evidence_refiner=ChatEvidenceRefiner(_TimeoutChatClient()),
        claim_extractor=GroundedClaimExtractor(claim_client),
        graph_builder=LocalGraphBuilder(),
        reporter=GroundedReporter(_TimeoutChatClient()),
    )

    result = pipeline.run("Clinical question", SearchFilters())

    assert result["retrieval_stats"]["claim_count"] == 2
    assert len(result["graph"]["edges"]) == 2
