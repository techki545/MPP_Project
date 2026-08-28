from __future__ import annotations

from pathlib import Path

import pytest

import knowledge_base.service as service_module
from knowledge_base.config import Settings
from knowledge_base.errors import KnowledgeBaseError
from knowledge_base.graph_builder import LocalGraphBuilder
from knowledge_base.models import RankedHit
from knowledge_base.models import SearchFilters
from knowledge_base.reporter import (
    ChatEvidenceRefiner,
    GroundedClaimExtractor,
    GroundedReporter,
)
from knowledge_base.retriever import (
    QueryContext,
    RetrievedDocument,
    RetrievalResult,
)
from knowledge_base.service import KnowledgeBaseService, ProductionQueryPipeline


class FakeQueryPipeline:
    def run(self, question, filters):
        return {
            "question": question,
            "mode": "hybrid",
            "degraded_reason": "",
            "filters": {},
            "reasoning_steps": [
                {"title": "证据盘点", "body": "一项RCT。[7]", "source_ids": [7]}
            ],
            "answer_markdown": "综合回答。[7]",
            "model_used": True,
            "model_name": "test-model",
            "model_error": None,
            "summary": {"evidence_count": 1},
            "sources": [
                {"source_number": 7, "document_id": "doc-1", "title": "Trial"},
                {"source_number": 8, "document_id": "doc-1", "title": "Duplicate"},
            ],
            "graph": {"nodes": [{"node_id": "doc-1"}], "edges": []},
            "retrieval_stats": {"candidate_count": 1},
        }


class FakeDocumentStore:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    def document_detail(self, document_id):
        if document_id != "doc-1":
            return None
        return {
            "document_id": document_id,
            "title": "Trial",
            "pdf_path": str(self.pdf_path),
            "local_path": str(self.pdf_path),
            "matched_snippets": [{"text": "Low dose result", "page_start": 3}],
        }

    def statistics(self):
        return {"metadata_records": 1, "pdf_files": 1, "matched_pdf_files": 1}


def make_service(
    tmp_path: Path, *, pdf_outside: bool = False, pipeline=None
) -> KnowledgeBaseService:
    source = tmp_path / "corpus"
    source.mkdir()
    pdf_path = (tmp_path if pdf_outside else source) / "trial.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")
    settings = Settings.from_mapping(
        {
            "MPP_KB_SOURCE": str(source),
            "MPP_KB_DATA": str(tmp_path / "index"),
            "MPP_API_BASE": "https://provider.example/v1",
            "MPP_CHAT_MODEL": "chat-model",
        },
        project_root=tmp_path,
    )
    return KnowledgeBaseService(
        settings,
        FakeDocumentStore(pdf_path),
        pipeline if pipeline is not None else FakeQueryPipeline(),
    )


class SequencedChat:
    def __init__(self, model: str, *responses: dict) -> None:
        self.model = model
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, system_prompt, payload):
        self.calls.append((system_prompt, payload))
        return self.responses.pop(0)


def make_grounded_pipeline(
    chat_client, *, include_unselected_document: bool = False
) -> ProductionQueryPipeline:
    text = (
        "Randomized controlled trial in SMPP children. Low dose "
        "methylprednisolone with antibiotics was supported for fever duration."
    )
    hit = RankedHit(
        record_id="chunk-1",
        document_id="doc-1",
        source="vector_fulltext",
        rank=1,
        score=0.9,
        text=text,
        payload={"page_start": 3, "page_end": 3},
    )
    document = RetrievedDocument(
        document_id="doc-1",
        fused_score=1.0,
        final_score=0.9,
        supporting_hits=(hit,),
        payload={"title": "Randomized controlled trial", "year": 2025},
    )
    extra_hit = RankedHit(
        record_id="chunk-2",
        document_id="doc-2",
        source="keyword_fulltext",
        rank=2,
        score=0.4,
        text="A risk prediction nomogram was developed for hospitalized children.",
        payload={"page_start": 2, "page_end": 2},
    )
    extra_document = RetrievedDocument(
        document_id="doc-2",
        fused_score=0.5,
        final_score=0.4,
        supporting_hits=(extra_hit,),
        payload={"title": "Risk prediction nomogram", "year": 2024},
    )
    documents = (document, extra_document) if include_unselected_document else (document,)
    context = QueryContext(
        raw_question="question",
        normalized_fts_query="question",
        population_terms=(),
        intervention_terms=(),
        comparator_terms=(),
        outcome_terms=(),
        safety_focused=False,
        year_from=None,
        year_to=None,
        evidence_types=(),
        fulltext_only=False,
    )

    class Retriever:
        def search(self, question, filters):
            return RetrievalResult("hybrid", "", documents, len(documents), context)

    class Documents:
        def document_detail(self, document_id):
            if document_id == "doc-2":
                return {
                    "title": "Risk prediction nomogram",
                    "abstract": "Prediction model abstract",
                    "year": 2024,
                    "evidence_type": "observational_study",
                    "classification_confidence": 0.9,
                    "quality": "moderate",
                }
            return {
                "title": "Randomized controlled trial",
                "abstract": "Trial abstract",
                "year": 2025,
                "evidence_type": "randomized_controlled_trial",
                "classification_confidence": 0.9,
                "quality": "high",
            }

    return ProductionQueryPipeline(
        retriever=Retriever(),
        document_store=Documents(),
        evidence_refiner=ChatEvidenceRefiner(chat_client),
        claim_extractor=GroundedClaimExtractor(chat_client),
        graph_builder=LocalGraphBuilder(),
        reporter=GroundedReporter(chat_client),
    )


def grounded_chat_responses() -> tuple[dict, dict]:
    return (
        {
            "claims": [
                {
                    "source_number": 1,
                    "source_chunk_ids": ["chunk-1"],
                    "population": "SMPP children",
                    "intervention": "methylprednisolone",
                    "comparator": "antibiotics",
                    "design": "randomized controlled trial",
                    "sample_size": "",
                    "dose": "low dose",
                    "outcome": "fever duration",
                    "direction": "supports",
                    "effect_measures": [],
                    "safety_signal": False,
                    "limitations": [],
                    "statement": "Low dose was supported.",
                    "source_quote": (
                        "Low dose methylprednisolone with antibiotics was "
                        "supported for fever duration"
                    ),
                }
            ]
        },
        grounded_report_response("Low dose was supported."),
    )


def grounded_report_response(statement: str) -> dict:
    stages = (
        ("inventory", "第一步：检索并盘点证据库存"),
        ("guidelines", "第二步：优先查看指南"),
        ("systematic_reviews", "第三步：查阅系统综述，检验指南结论"),
        ("randomized_trials", "第四步：聚焦关键随机对照试验"),
        ("lower_level_evidence", "第五步：用下级证据补充安全性与边界"),
        ("synthesis", "第六步：检查一致性并形成综合判断"),
    )
    final_answer = "\n\n".join(
        (
            "## 综合回答",
            f"{statement}[1]",
            f"### 证据链\n{statement}[1]",
            f"### 时间更新\n{statement}[1]",
            f"### 安全性与适用边界\n{statement}[1]",
            f"### 证据缺口\n{statement}[1]",
        )
    )
    return {
        "analysis_steps": [
            {
                "stage_key": stage_key,
                "title": title,
                "body": f"{statement}[1]",
                "source_ids": [1],
            }
            for stage_key, title in stages
        ],
        "final_answer_markdown": final_answer,
    }


def test_query_returns_two_part_report_and_numbers_after_deduplication(
    tmp_path: Path,
) -> None:
    result = make_service(tmp_path).query("SMPP儿童是否使用糖皮质激素？", SearchFilters())

    assert result["mode"] == "hybrid"
    assert result["reasoning_steps"]
    assert result["answer_markdown"].endswith("[1]")
    assert result["sources"] == [
        {"source_number": 1, "document_id": "doc-1", "title": "Trial"}
    ]
    assert result["graph"]["nodes"]


@pytest.mark.parametrize("question", ["", " ", "x" * 2001])
def test_query_validates_question_length(tmp_path: Path, question: str) -> None:
    with pytest.raises(KnowledgeBaseError) as captured:
        make_service(tmp_path).query(question, SearchFilters())

    assert captured.value.code in {"empty_question", "question_too_long"}


def test_document_detail_never_exposes_absolute_paths(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    detail = service.document_detail("doc-1")

    assert "local_path" not in detail
    assert "pdf_path" not in detail
    assert detail["pdf_available"] is True
    assert service.resolve_pdf("doc-1").name == "trial.pdf"


def test_pdf_outside_configured_corpus_is_rejected(tmp_path: Path) -> None:
    service = make_service(tmp_path, pdf_outside=True)

    assert service.document_detail("doc-1")["pdf_available"] is False
    with pytest.raises(KnowledgeBaseError) as captured:
        service.resolve_pdf("doc-1")
    assert captured.value.code == "pdf_unavailable"


def test_model_status_never_exposes_api_key(tmp_path: Path) -> None:
    status = make_service(tmp_path).model_status()

    assert status["configured"] is False
    assert status["chat_model"] == "chat-model"
    assert "api_key" not in status


def test_query_model_config_creates_one_ephemeral_client_without_leaking_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = []
    received = []

    class RecordingChatClient:
        def __init__(self, base_url, api_key, model):
            self.model = model
            self.config = (base_url, api_key, model)
            created.append(self)

    class RequestPipeline:
        def run(self, question, filters, *, chat_client):
            received.append(chat_client)
            result = FakeQueryPipeline().run(question, filters)
            result["model_name"] = chat_client.model
            return result

    monkeypatch.setattr(service_module, "ChatClient", RecordingChatClient)
    pipeline = RequestPipeline()
    service = make_service(tmp_path, pipeline=pipeline)
    settings_before = vars(service.settings).copy()
    model_config = {
        "base_url": "https://override.example/v1",
        "api_key": "request-only-secret-token",
        "model_name": "request-model",
    }

    result = service.query(
        "clinical question", SearchFilters(), model_config=model_config
    )

    assert len(created) == 1
    assert created[0].config == (
        "https://override.example/v1",
        "request-only-secret-token",
        "request-model",
    )
    assert received == [created[0]]
    assert result["model_name"] == "request-model"
    assert vars(service.settings) == settings_before
    assert vars(pipeline) == {}
    assert "request-only-secret-token" not in repr(result)
    assert "request-only-secret-token" not in repr(service)
    assert "request-only-secret-token" not in repr(created[0])


def test_query_without_model_config_supports_legacy_pipeline_and_no_new_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnexpectedChatClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("request ChatClient must not be constructed")

    monkeypatch.setattr(service_module, "ChatClient", UnexpectedChatClient)

    result = make_service(tmp_path).query("clinical question", SearchFilters())

    assert result["model_name"] == "test-model"


def test_chat_components_share_the_same_ephemeral_client() -> None:
    chat_client = object()

    evidence_refiner, claim_extractor, reporter = service_module._chat_components(
        chat_client
    )

    assert isinstance(evidence_refiner, ChatEvidenceRefiner)
    assert isinstance(claim_extractor, GroundedClaimExtractor)
    assert isinstance(reporter, GroundedReporter)
    assert evidence_refiner.chat_client is chat_client
    assert claim_extractor.chat_client is chat_client
    assert reporter.chat_client is chat_client


def test_production_pipeline_request_chat_replaces_all_default_chat_components(
) -> None:
    default_chat = SequencedChat("server-model")
    request_chat = SequencedChat("request-model", *grounded_chat_responses())
    pipeline = make_grounded_pipeline(default_chat)
    default_components = (
        pipeline.evidence_refiner,
        pipeline.claim_extractor,
        pipeline.reporter,
    )

    result = pipeline.run(
        "question", SearchFilters(), chat_client=request_chat
    )

    assert len(request_chat.calls) == 2
    assert default_chat.calls == []
    assert result["model_used"] is True
    assert result["model_name"] == "request-model"
    assert (
        pipeline.evidence_refiner,
        pipeline.claim_extractor,
        pipeline.reporter,
    ) == default_components


def test_model_selected_claims_are_not_supplemented_with_unselected_documents(
) -> None:
    chat = SequencedChat("request-model", *grounded_chat_responses())
    pipeline = make_grounded_pipeline(chat, include_unselected_document=True)

    result = pipeline.run("Should steroids be used?", SearchFilters())

    assert result["retrieval_stats"]["claim_count"] == 1
    assert "Risk prediction nomogram" not in result["answer_markdown"]


def test_sequential_model_overrides_use_distinct_unchanged_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = []
    received = []

    class RecordingChatClient:
        def __init__(self, base_url, api_key, model):
            self.model = model
            self.config = (base_url, api_key, model)
            created.append(self)

    class RequestPipeline:
        def run(self, question, filters, *, chat_client):
            received.append(chat_client)
            result = FakeQueryPipeline().run(question, filters)
            result["model_name"] = chat_client.model
            return result

    monkeypatch.setattr(service_module, "ChatClient", RecordingChatClient)
    pipeline = RequestPipeline()
    service = make_service(tmp_path, pipeline=pipeline)
    settings_before = vars(service.settings).copy()
    first_config = {
        "base_url": "https://first.example/v1",
        "api_key": "first-placeholder-key",
        "model_name": "first-model",
    }
    second_config = {
        "base_url": "https://second.example/v1",
        "api_key": "second-placeholder-key",
        "model_name": "second-model",
    }

    first_result = service.query(
        "first question", SearchFilters(), model_config=first_config
    )
    first_client_config = created[0].config
    second_result = service.query(
        "second question", SearchFilters(), model_config=second_config
    )

    assert len(created) == 2
    assert received == created
    assert created[0] is not created[1]
    assert created[0].config == first_client_config == (
        "https://first.example/v1",
        "first-placeholder-key",
        "first-model",
    )
    assert created[1].config == (
        "https://second.example/v1",
        "second-placeholder-key",
        "second-model",
    )
    assert first_result["model_name"] == "first-model"
    assert second_result["model_name"] == "second-model"
    assert vars(service.settings) == settings_before
    assert vars(pipeline) == {}


def test_knowledge_base_is_not_ready_until_local_index_is_complete(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.store.knowledge_base_ready = lambda: False

    assert service.knowledge_base_status()["status"] == "not_built"

    service.store.knowledge_base_ready = lambda: True
    assert service.knowledge_base_status()["status"] == "ready"


def test_production_pipeline_runs_retrieval_claim_graph_and_report_in_order() -> None:
    hit = RankedHit(
        record_id="chunk-1",
        document_id="doc-1",
        source="vector_fulltext",
        rank=1,
        score=0.9,
        text=(
            "Randomized controlled trial in SMPP children. Low dose "
            "methylprednisolone with antibiotics was supported for fever duration."
        ),
        payload={"page_start": 3, "page_end": 3},
    )
    document = RetrievedDocument(
        document_id="doc-1",
        fused_score=1.0,
        final_score=0.9,
        supporting_hits=(hit,),
        payload={"title": "Randomized controlled trial", "year": 2025},
    )
    context = QueryContext(
        raw_question="question",
        normalized_fts_query="question",
        population_terms=(),
        intervention_terms=(),
        comparator_terms=(),
        outcome_terms=(),
        safety_focused=False,
        year_from=None,
        year_to=None,
        evidence_types=(),
        fulltext_only=False,
    )

    class Retriever:
        def search(self, question, filters):
            return RetrievalResult("hybrid", "", (document,), 1, context)

    class Documents:
        def document_detail(self, document_id):
            return {
                "title": "Randomized controlled trial",
                "abstract": "Trial abstract",
                "year": 2025,
                "evidence_type": "randomized_controlled_trial",
                "classification_confidence": 0.9,
                "quality": "high",
            }

    class Chat:
        model = "test-model"

        def __init__(self):
            self.calls = 0

        def complete_json(self, system_prompt, payload):
            self.calls += 1
            if self.calls == 1:
                return {
                    "claims": [
                        {
                            "source_number": 1,
                            "source_chunk_ids": ["chunk-1"],
                            "population": "SMPP children",
                            "intervention": "methylprednisolone",
                            "comparator": "antibiotics",
                            "design": "randomized controlled trial",
                            "sample_size": "",
                            "dose": "low dose",
                            "outcome": "fever duration",
                            "direction": "supports",
                            "effect_measures": [],
                            "safety_signal": False,
                            "limitations": [],
                            "statement": "Low dose was supported.",
                        "source_quote": "Low dose methylprednisolone with antibiotics was supported for fever duration",
                        }
                    ]
                }
            return grounded_report_response("Low dose was supported.")

    chat = Chat()
    pipeline = ProductionQueryPipeline(
        retriever=Retriever(),
        document_store=Documents(),
        evidence_refiner=ChatEvidenceRefiner(chat),
        claim_extractor=GroundedClaimExtractor(chat),
        graph_builder=LocalGraphBuilder(),
        reporter=GroundedReporter(chat),
    )

    result = pipeline.run("question", SearchFilters())

    assert result["model_used"] is True
    assert result["sources"][0]["chunk_ids"] == ["chunk-1"]
    assert len(result["reasoning_steps"]) == 6
    assert {node["node_type"] for node in result["graph"]["nodes"]} == {"document"}
    assert "confirm_count" in result["summary"]
    assert "caution_count" in result["summary"]
    assert result["retrieval_stats"]["claim_count"] == 1


def test_metadata_only_hit_can_supply_a_grounded_graph_claim() -> None:
    text = "Randomized trial in 40 children. Low dose reduced fever duration."
    hit = RankedHit(
        record_id="doc-1",
        document_id="doc-1",
        source="fts_metadata",
        rank=1,
        score=0.9,
        text=text,
        payload={"title": "Randomized trial", "year": 2025},
    )
    document = RetrievedDocument(
        document_id="doc-1",
        fused_score=1.0,
        final_score=0.9,
        supporting_hits=(hit,),
        payload={"title": "Randomized trial", "year": 2025},
    )
    context = QueryContext(
        raw_question="question",
        normalized_fts_query="question",
        population_terms=("children",),
        intervention_terms=(),
        comparator_terms=(),
        outcome_terms=(),
        safety_focused=False,
        year_from=None,
        year_to=None,
        evidence_types=(),
        fulltext_only=False,
    )

    class Retriever:
        def search(self, question, filters):
            return RetrievalResult("keyword", "query_embedding_unavailable", (document,), 1, context)

    class Documents:
        def document_detail(self, document_id):
            return {
                "title": "Randomized trial",
                "abstract": text,
                "year": 2025,
                "evidence_type": "randomized_controlled_trial",
                "classification_confidence": 0.9,
                "quality": "moderate",
            }

    class Chat:
        model = "test-model"

        def __init__(self):
            self.calls = 0

        def complete_json(self, system_prompt, payload):
            self.calls += 1
            if self.calls == 1:
                return {
                    "claims": [
                        {
                            "source_number": 1,
                            "source_chunk_ids": ["doc-1"],
                            "population": "children",
                            "intervention": "low dose",
                            "comparator": "",
                            "design": "randomized trial",
                            "sample_size": "40",
                            "dose": "",
                            "outcome": "fever duration",
                            "direction": "supports",
                            "effect_measures": ["reduced fever duration"],
                            "safety_signal": False,
                            "limitations": [],
                            "statement": "Low dose reduced fever duration.",
                            "source_quote": text,
                        }
                    ]
                }
            return grounded_report_response("Low dose reduced fever duration.")

    chat = Chat()
    pipeline = ProductionQueryPipeline(
        retriever=Retriever(),
        document_store=Documents(),
        evidence_refiner=ChatEvidenceRefiner(chat),
        claim_extractor=GroundedClaimExtractor(chat),
        graph_builder=LocalGraphBuilder(),
        reporter=GroundedReporter(chat),
    )

    result = pipeline.run("question", SearchFilters())

    assert result["sources"][0]["fulltext"] is False
    assert result["sources"][0]["chunk_ids"] == ["doc-1"]
    assert result["retrieval_stats"]["claim_count"] == 1
    assert {node["node_type"] for node in result["graph"]["nodes"]} == {"document"}
