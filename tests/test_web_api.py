from __future__ import annotations

from pathlib import Path

import pytest

from graph_rag_core import load_default_graph
from web_api import APIError, GraphRAGWebService


class FakeKnowledgeService:
    def health(self):
        return {"status": "ready", "documents": 1}

    def knowledge_base_status(self):
        return {"status": "ready", "metadata_records": 1}

    def model_status(self):
        return {"configured": True, "chat_model": "test-model"}

    def query(self, question, filters):
        return {
            "question": question,
            "mode": "hybrid",
            "answer_markdown": "answer",
            "reasoning_steps": [],
            "sources": [],
            "received_filters": filters,
        }

    def document_detail(self, document_id):
        return {"document_id": document_id, "title": "Trial"}


def make_web_service(knowledge_service=None) -> GraphRAGWebService:
    root = Path(__file__).resolve().parents[1]
    return GraphRAGWebService(
        load_default_graph(str(root)), knowledge_service=knowledge_service
    )


def test_health_reports_demo_and_knowledge_base_components() -> None:
    health = make_web_service(FakeKnowledgeService()).health()

    assert health["status"] in {"ok", "degraded"}
    assert "knowledge_base" in health
    assert "model" in health


def test_query_uses_backend_model_configuration_and_builds_filters() -> None:
    service = make_web_service(FakeKnowledgeService())

    result = service.query(
        {
            "question": "clinical question",
            "evidence_types": ["guideline"],
            "year_from": 2020,
            "year_to": 2025,
            "fulltext_only": True,
        }
    )

    filters = result.pop("received_filters")
    assert result["answer_markdown"] == "answer"
    assert filters.evidence_types == frozenset({"guideline"})
    assert filters.year_from == 2020
    assert filters.fulltext_only is True


def test_browser_supplied_model_configuration_is_rejected() -> None:
    with pytest.raises(APIError) as captured:
        make_web_service(FakeKnowledgeService()).query(
            {"question": "clinical question", "model": {"api_key": "secret"}}
        )

    assert captured.value.code == "client_model_config_forbidden"


def test_not_built_uses_explicit_demo_fallback() -> None:
    result = make_web_service().query({"question": "steroid", "evidence_types": []})

    assert result["data_source"] == "demo"
    assert result["warning"]
    assert all(source["data_source"] == "demo" for source in result["sources"])


class FakeJobManager:
    def __init__(self):
        self.calls = []

    def start_build(self, *, confirm_embedding_cost, options):
        self.calls.append((confirm_embedding_cost, dict(options)))

        class Record:
            def as_dict(self):
                return {"job_id": "job-1", "state": "queued", "progress": {}}

        return Record()


def test_web_build_requires_a_bounded_embedding_probe_before_full_confirmation() -> None:
    manager = FakeJobManager()
    service = GraphRAGWebService(
        load_default_graph(str(Path(__file__).resolve().parents[1])),
        knowledge_service=FakeKnowledgeService(),
        job_manager=manager,
    )

    with pytest.raises(APIError) as missing_limit:
        service.start_build({"confirm_embedding_cost": True})
    assert missing_limit.value.code == "embedding_probe_limit_required"

    service.start_build(
        {"confirm_embedding_cost": True, "embedding_limit": 32}
    )
    assert manager.calls[-1] == (True, {"embedding_limit": 32})

    service.start_build(
        {
            "confirm_embedding_cost": True,
            "confirm_full_embedding_cost": True,
        }
    )
    assert manager.calls[-1] == (
        True,
        {"confirm_full_embedding_cost": True},
    )
