"""Offline multilingual E5 embeddings powered by ONNX Runtime."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any

from .errors import KnowledgeBaseError


class LocalEmbeddingClient:
    """Generate normalized passage and query vectors without network access."""

    DIMENSION = 384
    MAX_LENGTH = 512

    def __init__(
        self,
        *,
        model_path: str | Path,
        tokenizer_path: str | Path,
        session: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self._lock = Lock()
        try:
            if tokenizer is None:
                from tokenizers import Tokenizer

                tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            if session is None:
                import onnxruntime as ort

                session = ort.InferenceSession(
                    str(self.model_path),
                    providers=["CPUExecutionProvider"],
                )
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            raise KnowledgeBaseError(
                "local_embedding_unavailable",
                "Local embedding model could not be loaded",
            ) from exc
        self._tokenizer = tokenizer
        self._session = session
        self._tokenizer.enable_truncation(max_length=self.MAX_LENGTH)
        self._tokenizer.enable_padding()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, prefix="passage: ")

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, prefix="query: ")

    def probe(self) -> int:
        vectors = self.embed_queries(["MPP embedding probe"])
        if not vectors or not vectors[0]:
            raise KnowledgeBaseError(
                "embedding_response_invalid", "Embedding response is invalid"
            )
        return len(vectors[0])

    def _embed(self, texts: Sequence[str], *, prefix: str) -> list[list[float]]:
        if isinstance(texts, (str, bytes)):
            raise KnowledgeBaseError(
                "embedding_input_invalid", "Embedding inputs are invalid"
            )
        values = list(texts)
        if not values:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in values):
            raise KnowledgeBaseError(
                "embedding_input_invalid", "Embedding inputs are invalid"
            )

        try:
            import numpy as np

            with self._lock:
                encodings = self._tokenizer.encode_batch(
                    [prefix + text.strip() for text in values]
                )
            input_values = {
                "input_ids": np.asarray([item.ids for item in encodings], dtype=np.int64),
                "attention_mask": np.asarray(
                    [item.attention_mask for item in encodings], dtype=np.int64
                ),
                "token_type_ids": np.asarray(
                    [item.type_ids for item in encodings], dtype=np.int64
                ),
            }
            expected_inputs = {item.name for item in self._session.get_inputs()}
            outputs = self._session.run(
                None,
                {key: value for key, value in input_values.items() if key in expected_inputs},
            )
            hidden = np.asarray(outputs[0], dtype=np.float32)
            if hidden.ndim == 3:
                mask = input_values["attention_mask"].astype(np.float32)[..., None]
                pooled = (hidden * mask).sum(axis=1) / np.clip(
                    mask.sum(axis=1), 1e-9, None
                )
            elif hidden.ndim == 2:
                pooled = hidden
            else:
                raise ValueError("unexpected ONNX output shape")
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            normalized = pooled / np.clip(norms, 1e-12, None)
        except KnowledgeBaseError:
            raise
        except Exception as exc:
            raise KnowledgeBaseError(
                "local_embedding_failed", "Local embedding inference failed"
            ) from exc

        vectors = normalized.tolist()
        if len(vectors) != len(values) or any(not vector for vector in vectors):
            raise KnowledgeBaseError(
                "embedding_response_invalid", "Embedding response is invalid"
            )
        return vectors
