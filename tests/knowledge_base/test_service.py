from __future__ import annotations

from pathlib import Path

import pytest

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


def make_service(tmp_path: Path, *, pdf_outside: bool = False) -> KnowledgeBaseService:
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
        settings, FakeDocumentStore(pdf_path), FakeQueryPipeline()
    )


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


def test_production_pipeline_runs_retrieval_claim_graph_and_report_in_order() -> None:
    hit = RankedHit(
        record_id="chunk-1",
        document_id="doc-1",
        source="vector_fulltext",
        rank=1,
        score=0.9,
        text="Low dose was supported for fever duration.",
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
                            "source_quote": "Low dose was supported for fever duration",
                        }
                    ]
                }
            return {
                "analysis_steps": [
                    {"title": "Evidence", "body": "One RCT.[1]", "source_ids": [1]}
                ],
                "final_answer_markdown": "Low dose is supported.[1]",
            }

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
    assert any(node["node_type"] == "claim" for node in result["graph"]["nodes"])
    assert result["retrieval_stats"]["claim_count"] == 1
