from __future__ import annotations

from typing import Any

import pytest

from knowledge_base.chat_client import ChatClient
from knowledge_base.errors import KnowledgeBaseError


def test_chat_client_stops_reopening_transport_after_timeout() -> None:
    calls: list[dict[str, Any]] = []

    def timed_out_transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        calls.append(payload)
        raise TimeoutError("provider did not respond")

    client = ChatClient(
        "https://models.example.test/v1",
        "test-key",
        "test-model",
        transport=timed_out_transport,
    )

    for _ in range(2):
        with pytest.raises(KnowledgeBaseError) as captured:
            client.complete_json("Return JSON.", {"question": "test"})
        assert captured.value.code == "chat_timeout"

    assert len(calls) == 1


def test_chat_client_explicitly_requests_json_in_every_system_message() -> None:
    captured: dict[str, Any] = {}

    def successful_transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        captured.update(payload)
        return {"choices": [{"message": {"content": '{"status":"ok"}'}}]}

    client = ChatClient(
        "https://api.deepseek.com",
        "test-key",
        "deepseek-v4-flash",
        transport=successful_transport,
    )

    assert client.complete_json("Classify this evidence type.", {"title": "Trial"}) == {
        "status": "ok"
    }

    system_message = captured["messages"][0]["content"]
    assert "json" in system_message.casefold()
    assert captured["max_tokens"] == 6144
    assert captured["thinking"] == {"type": "disabled"}
