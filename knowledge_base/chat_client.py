"""Server-side OpenAI-compatible JSON chat client."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from MycoplasmaPneumonia.llm_client import extract_json_object

from .errors import KnowledgeBaseError


Transport = Callable[[str, dict[str, str], dict[str, Any], float], Mapping[str, Any]]


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 60.0,
        transport: Transport | None = None,
    ) -> None:
        self._endpoint = self._normalize_endpoint(base_url)
        self._api_key = str(api_key).strip()
        self.model = str(model).strip()
        self._timeout = float(timeout)
        self._transport = transport or self._default_transport
        if not self._api_key or not self.model or self._timeout <= 0:
            raise KnowledgeBaseError(
                "chat_config_invalid", "Chat client configuration is invalid"
            )

    def complete_json(
        self, system_prompt: str, user_payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise KnowledgeBaseError("chat_input_invalid", "Chat input is invalid")
        if not isinstance(user_payload, Mapping):
            raise KnowledgeBaseError("chat_input_invalid", "Chat input is invalid")

        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        dict(user_payload), ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ],
        }
        try:
            response = self._request(payload)
        except HTTPError as error:
            if error.code == 400 and self._explicitly_rejects_response_format(error):
                fallback_payload = dict(payload)
                fallback_payload.pop("response_format", None)
                try:
                    response = self._request(fallback_payload)
                except (HTTPError, URLError, TimeoutError, socket.timeout, OSError) as retry_error:
                    self._raise_transport_error(retry_error)
            else:
                self._raise_transport_error(error)
        except (URLError, TimeoutError, socket.timeout, OSError) as error:
            self._raise_transport_error(error)

        return self._parse_response(response)

    def _request(self, payload: dict[str, Any]) -> Mapping[str, Any]:
        return self._transport(
            self._endpoint,
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            payload,
            self._timeout,
        )

    @staticmethod
    def _normalize_endpoint(base_url: str) -> str:
        normalized = str(base_url).strip().rstrip("/")
        if not normalized:
            raise KnowledgeBaseError(
                "chat_config_invalid", "Chat client configuration is invalid"
            )
        return (
            normalized
            if normalized.endswith("/chat/completions")
            else f"{normalized}/chat/completions"
        )

    @staticmethod
    def _explicitly_rejects_response_format(error: HTTPError) -> bool:
        try:
            body = error.read(8192).decode("utf-8", errors="replace").casefold()
        except Exception:
            return False
        return "response_format" in body and any(
            marker in body
            for marker in (
                "not supported",
                "unsupported",
                "unknown",
                "unrecognized",
                "not allowed",
            )
        )

    @staticmethod
    def _raise_transport_error(error: BaseException) -> None:
        if isinstance(error, HTTPError):
            if error.code in (401, 403):
                code = "chat_auth_failed"
                message = "Chat authentication failed"
            elif error.code == 400:
                code = "chat_request_rejected"
                message = "Chat request was rejected"
            elif error.code in (408, 409, 429) or 500 <= error.code <= 599:
                code = "chat_unavailable"
                message = "Chat service is temporarily unavailable"
            else:
                code = "chat_request_rejected"
                message = "Chat request was rejected"
        else:
            code = "chat_unavailable"
            message = "Chat service is temporarily unavailable"
        raise KnowledgeBaseError(code, message) from None

    @staticmethod
    def _parse_response(response: Mapping[str, Any]) -> dict[str, Any]:
        try:
            choices = response["choices"]
            content = choices[0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError
            parsed = extract_json_object(content)
            if not isinstance(parsed, dict):
                raise TypeError
            return parsed
        except Exception:
            raise KnowledgeBaseError(
                "chat_response_invalid", "Chat response is invalid"
            ) from None

    @staticmethod
    def _default_transport(
        url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
    ) -> Mapping[str, Any]:
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            try:
                parsed = json.loads(response.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise KnowledgeBaseError(
                    "chat_response_invalid", "Chat response is invalid"
                ) from None
        if not isinstance(parsed, Mapping):
            raise KnowledgeBaseError(
                "chat_response_invalid", "Chat response is invalid"
            )
        return parsed
