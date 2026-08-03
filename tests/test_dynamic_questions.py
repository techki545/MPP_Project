from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from knowledge_base.errors import KnowledgeBaseError
from knowledge_base.graph_builder import LocalGraphBuilder
from knowledge_base.models import SearchFilters
from knowledge_base.reporter import (
    ChatEvidenceRefiner,
    GroundedClaimExtractor,
    GroundedReporter,
)
from knowledge_base.service import ProductionQueryPipeline


class TimeoutChat:
    model = "test-timeout-model"

    def complete_json(self, system_prompt: str, payload: dict[str, Any]) -> dict:
        raise KnowledgeBaseError("chat_timeout", "controlled timeout")


class Retriever:
    def __init__(self, documents: tuple[SimpleNamespace, ...]) -> None:
        self.documents = documents

    def search(self, question: str, filters: SearchFilters) -> SimpleNamespace:
        return SimpleNamespace(
            documents=self.documents,
            mode="keyword",
            degraded_reason="vector_unavailable",
            candidate_count=len(self.documents),
        )


class DocumentStore:
    def __init__(self, details: dict[str, dict[str, Any]]) -> None:
        self.details = details

    def document_detail(self, document_id: str) -> dict[str, Any]:
        return self.details[document_id]


def record(
    document_id: str,
    title: str,
    evidence_type: str,
    year: int,
    excerpt: str,
) -> tuple[SimpleNamespace, dict[str, Any]]:
    hit = SimpleNamespace(
        source="fulltext",
        record_id=f"chunk-{document_id}",
        text=excerpt,
        payload={"page_start": 1, "page_end": 1},
    )
    document = SimpleNamespace(
        document_id=document_id,
        payload={},
        supporting_hits=(hit,),
    )
    detail = {
        "title": title,
        "abstract": excerpt,
        "evidence_type": evidence_type,
        "classification_confidence": 1.0,
        "classification_basis": "fixture",
        "quality": "moderate",
        "year": year,
    }
    return document, detail


def pipeline_from(
    records: tuple[tuple[SimpleNamespace, dict[str, Any]], ...]
) -> ProductionQueryPipeline:
    chat = TimeoutChat()
    documents = tuple(item[0] for item in records)
    details = {item[0].document_id: item[1] for item in records}
    return ProductionQueryPipeline(
        retriever=Retriever(documents),
        document_store=DocumentStore(details),
        evidence_refiner=ChatEvidenceRefiner(chat),
        claim_extractor=GroundedClaimExtractor(chat),
        graph_builder=LocalGraphBuilder(),
        reporter=GroundedReporter(chat),
    )


def test_distinct_questions_produce_distinct_reports_and_graphs() -> None:
    treatment_pipeline = pipeline_from(
        (
            record(
                "treatment-guide",
                "Treatment guideline",
                "guideline",
                2023,
                "The guideline describes treatment indications and evidence limits.",
            ),
            record(
                "treatment-trial",
                "Treatment trial",
                "randomized_controlled_trial",
                2025,
                "The trial compares treatment strategies and clinical outcomes.",
            ),
        )
    )
    prognosis_pipeline = pipeline_from(
        (
            record(
                "prognosis-review",
                "Prognostic factor review",
                "systematic_review",
                2024,
                "The review summarizes predictors of refractory disease.",
            ),
            record(
                "prognosis-cohort",
                "Prognostic cohort",
                "observational_study",
                2025,
                "The cohort reports findings associated with refractory disease.",
            ),
        )
    )

    treatment = treatment_pipeline.run(
        "Should the treatment be used?", SearchFilters()
    )
    prognosis = prognosis_pipeline.run(
        "Which findings predict refractory disease?", SearchFilters()
    )

    assert treatment["answer_markdown"] != prognosis["answer_markdown"]
    assert {node["node_id"] for node in treatment["graph"]["nodes"]} != {
        node["node_id"] for node in prognosis["graph"]["nodes"]
    }
    assert len(treatment["reasoning_steps"]) == 6
    assert len(prognosis["reasoning_steps"]) == 6
    assert "未检索到指南" not in treatment["reasoning_steps"][1]["body"]
    assert "未检索到指南" in prognosis["reasoning_steps"][1]["body"]
