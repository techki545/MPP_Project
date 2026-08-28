from __future__ import annotations

from io import BytesIO
import json
import socket
from urllib.error import HTTPError, URLError

import pytest

from knowledge_base.chat_client import ChatClient
from knowledge_base.errors import KnowledgeBaseError


def test_chat_client_reads_fixed_server_configuration() -> None:
    captured = {}

    def transport(url, headers, payload, timeout):
        captured.update(url=url, headers=headers, payload=payload, timeout=timeout)
        return {"choices": [{"message": {"content": '{"status":"ok"}'}}]}

    client = ChatClient(
        "https://provider.example/v1", "secret", "chat-model", transport=transport
    )

    assert client.complete_json("system", {"task": "probe"}) == {"status": "ok"}
    assert captured["url"] == "https://provider.example/v1/chat/completions"
    assert captured["payload"]["model"] == "chat-model"
    assert captured["payload"]["max_tokens"] == 6144
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["timeout"] == 120.0
    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_response_format_is_retried_only_when_provider_explicitly_rejects_it() -> None:
    payloads = []

    def transport(url, headers, payload, timeout):
        payloads.append(payload)
        if len(payloads) == 1:
            raise HTTPError(
                url,
                400,
                "bad request",
                {},
                BytesIO(b'{"error":"response_format is not supported"}'),
            )
        return {"choices": [{"message": {"content": "```json\n{\"ok\": true}\n```"}}]}

    result = ChatClient(
        "https://provider.example/v1/chat/completions",
        "secret",
        "chat-model",
        transport=transport,
    ).complete_json("system", {"task": "probe"})

    assert result == {"ok": True}
    assert len(payloads) == 2
    assert "response_format" in payloads[0]
    assert "response_format" not in payloads[1]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "chat_auth_failed"),
        (403, "chat_auth_failed"),
        (404, "chat_model_not_found"),
        (408, "chat_timeout"),
        (409, "chat_unavailable"),
        (429, "chat_unavailable"),
        (500, "chat_unavailable"),
    ],
)
def test_http_failures_are_stably_mapped_without_exposing_provider_body(
    status: int, expected: str
) -> None:
    base_url = "https://private-provider-url.example/v1"
    api_key = "unique-api-key-value"

    def transport(url, headers, payload, timeout):
        raise HTTPError(
            url,
            status,
            "unique-provider-exception-text",
            {},
            BytesIO(b"unique provider response body"),
        )

    with pytest.raises(KnowledgeBaseError) as captured:
        ChatClient(
            base_url, api_key, "chat-model", transport=transport
        ).complete_json("system", {"task": "probe"})

    assert captured.value.code == expected
    assert captured.value.details == {}
    exposed = f"{captured.value!s} {captured.value!r}"
    for secret in (
        base_url,
        api_key,
        "unique-provider-exception-text",
        "unique provider response body",
    ):
        assert secret not in exposed


@pytest.mark.parametrize(
    ("transport_error", "expected"),
    [
        (TimeoutError("unique direct timeout text"), "chat_timeout"),
        (socket.timeout("unique socket timeout text"), "chat_timeout"),
        (URLError(socket.timeout("unique nested timeout text")), "chat_timeout"),
        (URLError("unique network failure text"), "chat_unavailable"),
    ],
    ids=["direct-timeout", "socket-timeout", "urlerror-timeout", "urlerror-network"],
)
def test_transport_failures_are_stably_mapped_without_exposing_exception_text(
    transport_error: BaseException, expected: str
) -> None:
    def transport(url, headers, payload, timeout):
        raise transport_error

    with pytest.raises(KnowledgeBaseError) as captured:
        ChatClient(
            "https://unique-network-url.example/v1",
            "unique-network-api-key",
            "chat-model",
            transport=transport,
        ).complete_json("system", {"task": "probe"})

    assert captured.value.code == expected
    assert captured.value.details == {}
    exposed = f"{captured.value!s} {captured.value!r}"
    for secret in (
        "unique-network-url.example",
        "unique-network-api-key",
        str(transport_error),
    ):
        assert secret not in exposed


def test_generic_bad_request_is_not_retried() -> None:
    calls = 0

    def transport(url, headers, payload, timeout):
        nonlocal calls
        calls += 1
        raise HTTPError(url, 400, "bad", {}, BytesIO(b'{"error":"invalid model"}'))

    with pytest.raises(KnowledgeBaseError) as captured:
        ChatClient(
            "https://provider.example/v1", "secret", "chat-model", transport=transport
        ).complete_json("system", {"task": "probe"})

    assert captured.value.code == "chat_request_rejected"
    assert calls == 1


def test_malformed_chat_content_is_rejected() -> None:
    client = ChatClient(
        "https://provider.example/v1",
        "secret",
        "chat-model",
        transport=lambda *args: {"choices": [{"message": {"content": "not json"}}]},
    )

    with pytest.raises(KnowledgeBaseError) as captured:
        client.complete_json("system", {"task": "probe"})

    assert captured.value.code == "chat_response_invalid"


def test_transient_rate_limit_is_retried_with_bounded_backoff() -> None:
    calls = 0
    delays = []

    def transport(url, headers, payload, timeout):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise HTTPError(url, 429, "rate limited", {}, BytesIO(b"busy"))
        return {"choices": [{"message": {"content": '{"status":"ok"}'}}]}

    client = ChatClient(
        "https://provider.example/v1",
        "secret",
        "chat-model",
        transport=transport,
        sleep=delays.append,
    )

    assert client.complete_json("system", {"task": "probe"}) == {"status": "ok"}
    assert calls == 3
    assert delays == [1.0, 2.0]


def test_transient_failure_does_not_poison_later_pipeline_calls() -> None:
    calls = 0

    def transport(url, headers, payload, timeout):
        nonlocal calls
        calls += 1
        if calls <= 4:
            raise HTTPError(url, 500, "temporary", {}, BytesIO(b"busy"))
        return {"choices": [{"message": {"content": '{"status":"ok"}'}}]}

    client = ChatClient(
        "https://provider.example/v1",
        "secret",
        "chat-model",
        transport=transport,
        sleep=lambda seconds: None,
    )

    with pytest.raises(KnowledgeBaseError) as captured:
        client.complete_json("system", {"task": "first"})
    assert captured.value.code == "chat_unavailable"
    assert client.complete_json("system", {"task": "second"}) == {"status": "ok"}
