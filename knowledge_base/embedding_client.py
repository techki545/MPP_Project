"""OpenAI-compatible embedding client with safe retry behavior."""

from __future__ import annotations

import json
import math
import random as random_module
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import KnowledgeBaseError

Transport = Callable[[str, dict[str, str], dict[str, Any], float], Mapping[str, Any]]


class EmbeddingClient:
    """Request embeddings without exposing provider responses to callers."""

    _MAX_ATTEMPTS = 4
    _RETRY_DELAYS = (1.0, 2.0, 4.0)

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        transport: Transport | None = None,
        sleep: Callable[[float], None] | None = None,
        random_value: Callable[[], float] | None = None,
    ) -> None:
        self._endpoint = self._normalize_endpoint(base_url)
        self._api_key = api_key.strip()
        self._model = model.strip()
        self._timeout = float(timeout)
        self._transport = transport or self._default_transport
        self._sleep = sleep or time.sleep
        self._random_value = random_value or random_module.random
        if not self._api_key or not self._model or self._timeout <= 0:
            raise KnowledgeBaseError(
                "embedding_config_invalid", "Embedding client configuration is invalid"
            )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if isinstance(texts, (str, bytes)):
            raise KnowledgeBaseError("embedding_input_invalid", "Embedding inputs are invalid")
        values = list(texts)
        if not values:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in values):
            raise KnowledgeBaseError("embedding_input_invalid", "Embedding inputs are invalid")

        payload = {"model": self._model, "input": values}
        for attempt in range(self._MAX_ATTEMPTS):
            try:
                response = self._transport(
                    self._endpoint,
                    {
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    payload,
                    self._timeout,
                )
                return self._parse_vectors(response, len(values))
            except (HTTPError, URLError, TimeoutError, socket.timeout) as exc:
                code = self._map_transport_failure(exc)
                if code not in {"embedding_rate_limited", "embedding_transport_failed"}:
                    raise KnowledgeBaseError(
                        code,
                        "Embedding authentication failed"
                        if code == "embedding_auth_failed"
                        else "Embedding request was rejected",
                    ) from None
                if attempt == self._MAX_ATTEMPTS - 1:
                    raise KnowledgeBaseError(
                        code, "Embedding service is temporarily unavailable"
                    ) from None
                self._sleep(self._RETRY_DELAYS[attempt] + self._jitter())

        raise AssertionError("embedding retry loop exhausted unexpectedly")

    def probe(self) -> int:
        vectors = self.embed(["MPP embedding probe"])
        if not vectors or not vectors[0]:
            raise KnowledgeBaseError(
                "embedding_response_invalid", "Embedding response is invalid"
            )
        return len(vectors[0])

    @staticmethod
    def _normalize_endpoint(base_url: str) -> str:
        normalized = base_url.strip().rstrip("/")
        if not normalized:
            raise KnowledgeBaseError(
                "embedding_config_invalid", "Embedding client configuration is invalid"
            )
        return (
            normalized
            if normalized.endswith("/embeddings")
            else f"{normalized}/embeddings"
        )

    def _jitter(self) -> float:
        value = self._random_value()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0.0
        return min(0.25, max(0.0, float(value)))

    @staticmethod
    def _map_transport_failure(exc: BaseException) -> str:
        if isinstance(exc, HTTPError):
            if exc.code in (401, 403):
                return "embedding_auth_failed"
            if exc.code == 429:
                return "embedding_rate_limited"
            if 500 <= exc.code <= 599:
                return "embedding_transport_failed"
            return "embedding_request_rejected"
        return "embedding_transport_failed"

    @staticmethod
    def _parse_vectors(response: Mapping[str, Any], expected_count: int) -> list[list[float]]:
        try:
            items = response["data"]
        except (KeyError, TypeError):
            raise KnowledgeBaseError(
                "embedding_response_invalid", "Embedding response is invalid"
            ) from None
        if not isinstance(items, list) or len(items) != expected_count:
            raise KnowledgeBaseError(
                "embedding_response_invalid", "Embedding response is invalid"
            )

        parsed: list[tuple[int, list[float]]] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise KnowledgeBaseError(
                    "embedding_response_invalid", "Embedding response is invalid"
                )
            index = item.get("index")
            vector = item.get("embedding")
            if isinstance(index, bool) or not isinstance(index, int) or not isinstance(vector, list):
                raise KnowledgeBaseError(
                    "embedding_response_invalid", "Embedding response is invalid"
                )
            parsed_vector: list[float] = []
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise KnowledgeBaseError(
                        "embedding_response_invalid", "Embedding response is invalid"
                    )
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise KnowledgeBaseError(
                        "embedding_response_invalid", "Embedding response is invalid"
                    )
                parsed_vector.append(numeric)
            if not parsed_vector:
                raise KnowledgeBaseError(
                    "embedding_response_invalid", "Embedding response is invalid"
                )
            parsed.append((index, parsed_vector))

        parsed.sort(key=lambda item: item[0])
        if [item[0] for item in parsed] != list(range(expected_count)):
            raise KnowledgeBaseError(
                "embedding_response_invalid", "Embedding response is invalid"
            )
        dimensions = {len(vector) for _, vector in parsed}
        if len(dimensions) != 1:
            raise KnowledgeBaseError(
                "embedding_response_invalid", "Embedding response is invalid"
            )
        return [vector for _, vector in parsed]

    @staticmethod
    def _default_transport(
        url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
    ) -> Mapping[str, Any]:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            try:
                return json.loads(response.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise KnowledgeBaseError(
                    "embedding_response_invalid", "Embedding response is invalid"
                ) from None
