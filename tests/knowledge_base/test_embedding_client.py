from urllib.error import HTTPError

import pytest

from knowledge_base.embedding_client import EmbeddingClient
from knowledge_base.errors import KnowledgeBaseError


def make_client(transport, *, sleep=None, random_value=None, base_url="https://provider.example/v1"):
    return EmbeddingClient(
        base_url=base_url,
        api_key="test-secret-key",
        model="test-embedding-model",
        transport=transport,
        sleep=sleep or (lambda seconds: None),
        random_value=random_value or (lambda: 0.0),
    )


def test_probe_uses_normalized_embeddings_endpoint_and_discovers_dimension() -> None:
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        return {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}

    client = make_client(transport, base_url="https://provider.example/v1/embeddings/")

    assert client.probe() == 3
    assert calls == [
        (
            "https://provider.example/v1/embeddings",
            {"Authorization": "Bearer test-secret-key", "Content-Type": "application/json"},
            {"model": "test-embedding-model", "input": ["MPP embedding probe"]},
            30.0,
        )
    ]


def test_embed_sorts_indexed_response_and_converts_numeric_values_to_floats() -> None:
    def transport(url, headers, payload, timeout):
        return {
            "data": [
                {"index": 1, "embedding": [2, 3.0]},
                {"index": 0, "embedding": [0, 1]},
            ]
        }

    assert make_client(transport).embed(["first", "second"]) == [
        [0.0, 1.0],
        [2.0, 3.0],
    ]


@pytest.mark.parametrize(
    "response",
    [
        {"data": []},
        {"data": [{"index": 0, "embedding": []}]},
        {"data": [{"index": 0, "embedding": [1, float("nan")]}]},
        {"data": [{"index": 0, "embedding": [True, 1]}]},
        {"data": [{"index": 0, "embedding": [1, 2]}, {"index": 1, "embedding": [1]}]},
        {"data": [{"index": 2, "embedding": [1, 2]}]},
    ],
)
def test_embed_rejects_invalid_embedding_schema_without_retry(response) -> None:
    calls = 0

    def transport(url, headers, payload, timeout):
        nonlocal calls
        calls += 1
        return response

    with pytest.raises(KnowledgeBaseError) as exc_info:
        make_client(transport).embed(["private source text"])

    assert exc_info.value.code == "embedding_response_invalid"
    assert calls == 1
    assert "private source text" not in str(exc_info.value)


def test_empty_batch_returns_without_transport_and_blank_text_is_rejected() -> None:
    def transport(url, headers, payload, timeout):
        raise AssertionError("transport must not be called")

    client = make_client(transport)
    assert client.embed([]) == []
    with pytest.raises(KnowledgeBaseError) as exc_info:
        client.embed(["  "])
    assert exc_info.value.code == "embedding_input_invalid"


def test_retries_rate_limit_with_exponential_backoff_and_jitter() -> None:
    attempts = 0
    delays = []

    def transport(url, headers, payload, timeout):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise HTTPError(url, 429, "rate limited", {}, None)
        return {"data": [{"index": 0, "embedding": [1, 2]}]}

    result = make_client(
        transport,
        sleep=delays.append,
        random_value=lambda: 0.25,
    ).embed(["text"])

    assert result == [[1.0, 2.0]]
    assert attempts == 3
    assert delays == [1.25, 2.25]


@pytest.mark.parametrize(
    "status, expected_code",
    [(401, "embedding_auth_failed"), (403, "embedding_auth_failed"), (400, "embedding_request_rejected")],
)
def test_non_retryable_http_errors_do_not_leak_sensitive_values(status, expected_code) -> None:
    calls = 0

    def transport(url, headers, payload, timeout):
        nonlocal calls
        calls += 1
        raise HTTPError(url, status, "server said test-secret-key private source text", {}, None)

    with pytest.raises(KnowledgeBaseError) as exc_info:
        make_client(transport).embed(["private source text"])

    assert calls == 1
    assert exc_info.value.code == expected_code
    rendered = repr(exc_info.value)
    assert "test-secret-key" not in rendered
    assert "private source text" not in rendered
    assert "Authorization" not in rendered


def test_network_failure_retries_four_attempts_then_returns_safe_error() -> None:
    attempts = 0
    delays = []

    def transport(url, headers, payload, timeout):
        nonlocal attempts
        attempts += 1
        raise TimeoutError("test-secret-key private source text")

    with pytest.raises(KnowledgeBaseError) as exc_info:
        make_client(transport, sleep=delays.append, random_value=lambda: 0.0).embed(["private source text"])

    assert attempts == 4
    assert delays == [1.0, 2.0, 4.0]
    assert exc_info.value.code == "embedding_transport_failed"
    assert "test-secret-key" not in repr(exc_info.value)
    assert "private source text" not in repr(exc_info.value)


def test_probe_rejects_empty_vector() -> None:
    def transport(url, headers, payload, timeout):
        return {"data": [{"index": 0, "embedding": []}]}

    with pytest.raises(KnowledgeBaseError) as exc_info:
        make_client(transport).probe()
    assert exc_info.value.code == "embedding_response_invalid"


def test_default_transport_masks_malformed_upstream_response(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return b"test-secret-key private source text"

    monkeypatch.setattr("knowledge_base.embedding_client.urlopen", lambda request, timeout: FakeResponse())

    with pytest.raises(KnowledgeBaseError) as exc_info:
        EmbeddingClient._default_transport("https://provider.example/v1/embeddings", {}, {}, 1.0)

    assert exc_info.value.code == "embedding_response_invalid"
    assert "test-secret-key" not in repr(exc_info.value)
