from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_rag_core import load_default_graph
from web_api import APIError, GraphRAGWebService


class FakeKnowledgeService:
    def __init__(self):
        self.received_model_config = None

    def health(self):
        return {"status": "ready", "documents": 1}

    def knowledge_base_status(self):
        return {"status": "ready", "metadata_records": 1}

    def model_status(self):
        return {"configured": True, "chat_model": "test-model"}

    def query(self, question, filters, *, model_config=None):
        self.received_model_config = model_config
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


def test_query_without_model_config_uses_server_configuration_and_builds_filters() -> None:
    knowledge_service = FakeKnowledgeService()
    service = make_web_service(knowledge_service)

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
    assert knowledge_service.received_model_config is None


def test_query_trims_normalizes_and_forwards_complete_model_config() -> None:
    knowledge_service = FakeKnowledgeService()
    service = make_web_service(knowledge_service)

    result = service.query(
        {
            "question": "clinical question",
            "model_config": {
                "base_url": "  https://provider.example/v1///  ",
                "api_key": "  secret-token  ",
                "model_name": "  chat-model  ",
            },
        }
    )

    assert knowledge_service.received_model_config == {
        "base_url": "https://provider.example/v1",
        "api_key": "secret-token",
        "model_name": "chat-model",
    }
    assert "secret-token" not in repr(result)


def _valid_model_config(**overrides):
    config = {
        "base_url": "https://provider.example/v1",
        "api_key": "secret-token",
        "model_name": "chat-model",
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    "model_config",
    [
        None,
        [],
        {},
        {"base_url": "https://provider.example/v1"},
        {"api_key": "secret-token"},
        {"model_name": "chat-model"},
        _valid_model_config(extra="value"),
        _valid_model_config(base_url=123),
        _valid_model_config(api_key=123),
        _valid_model_config(model_name=123),
        _valid_model_config(base_url="x" * 2049),
        _valid_model_config(api_key="x" * 4097),
        _valid_model_config(model_name="x" * 257),
        _valid_model_config(base_url="http://provider.example/v1"),
        _valid_model_config(base_url="https://user:password@provider.example/v1"),
        _valid_model_config(base_url="https://provider.example/v1?mode=chat"),
        _valid_model_config(base_url="https://provider.example/v1#chat"),
        _valid_model_config(base_url="https:///v1"),
        _valid_model_config(base_url="ftp://provider.example/v1"),
    ],
    ids=[
        "null",
        "list",
        "empty",
        "base-url-only",
        "api-key-only",
        "model-name-only",
        "unknown-key",
        "non-string-base-url",
        "non-string-api-key",
        "non-string-model-name",
        "overlong-base-url",
        "overlong-api-key",
        "overlong-model-name",
        "public-http",
        "url-credentials",
        "query-string",
        "fragment",
        "missing-hostname",
        "unsupported-scheme",
    ],
)
def test_query_rejects_invalid_model_config(model_config) -> None:
    with pytest.raises(APIError) as captured:
        make_web_service(FakeKnowledgeService()).query(
            {"question": "clinical question", "model_config": model_config}
        )

    assert captured.value.code == "invalid_model_config"


def test_invalid_model_config_error_never_serializes_api_key() -> None:
    unique_api_key = "unique-api-key-must-not-leak"

    with pytest.raises(APIError) as captured:
        make_web_service(FakeKnowledgeService()).query(
            {
                "question": "clinical question",
                "model_config": _valid_model_config(
                    base_url="http://provider.example/v1",
                    api_key=unique_api_key,
                ),
            }
        )

    serialized = json.dumps(captured.value.as_dict(), ensure_ascii=False)
    assert captured.value.code == "invalid_model_config"
    assert unique_api_key not in serialized


def test_query_accepts_public_https_model_config() -> None:
    knowledge_service = FakeKnowledgeService()

    make_web_service(knowledge_service).query(
        {
            "question": "clinical question",
            "model_config": _valid_model_config(
                base_url="https://provider.example/v1/"
            ),
        }
    )

    assert knowledge_service.received_model_config["base_url"] == (
        "https://provider.example/v1"
    )


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost:8000/v1/", "http://localhost:8000/v1"),
        ("http://127.0.0.1:8000/v1//", "http://127.0.0.1:8000/v1"),
        ("http://[::1]:8000/v1///", "http://[::1]:8000/v1"),
    ],
)
def test_query_accepts_http_only_for_loopback_urls(base_url, expected) -> None:
    knowledge_service = FakeKnowledgeService()

    make_web_service(knowledge_service).query(
        {
            "question": "clinical question",
            "model_config": _valid_model_config(base_url=base_url),
        }
    )

    assert knowledge_service.received_model_config["base_url"] == expected


@pytest.mark.parametrize("field", ["model", "api_key", "base_url", "model_name"])
def test_query_still_rejects_legacy_top_level_model_configuration(field) -> None:
    with pytest.raises(APIError) as captured:
        make_web_service(FakeKnowledgeService()).query(
            {"question": "clinical question", field: "legacy-value"}
        )

    assert captured.value.code == "client_model_config_forbidden"


def test_analyze_uses_model_config_field_instead_of_legacy_model_field() -> None:
    knowledge_service = FakeKnowledgeService()
    model_config = _valid_model_config()

    make_web_service(knowledge_service).analyze(
        "clinical question", model_config=model_config
    )

    assert knowledge_service.received_model_config == model_config


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


def test_web_build_still_rejects_browser_model_config() -> None:
    service = GraphRAGWebService(
        load_default_graph(str(Path(__file__).resolve().parents[1])),
        knowledge_service=FakeKnowledgeService(),
        job_manager=FakeJobManager(),
    )

    with pytest.raises(APIError) as captured:
        service.start_build({"model_config": _valid_model_config()})

    assert captured.value.code == "client_model_config_forbidden"


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
