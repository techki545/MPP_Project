from __future__ import annotations

import numpy as np
import pytest

from knowledge_base.errors import KnowledgeBaseError
from knowledge_base.local_embedding_client import LocalEmbeddingClient


class Encoding:
    def __init__(self, token: int):
        self.ids = [token, 0]
        self.attention_mask = [1, 0]
        self.type_ids = [0, 0]


class Tokenizer:
    def __init__(self):
        self.calls = []

    def enable_truncation(self, *, max_length):
        assert max_length == 512

    def enable_padding(self):
        return None

    def encode_batch(self, texts):
        self.calls.append(list(texts))
        return [Encoding(index + 1) for index, _ in enumerate(texts)]


class Input:
    def __init__(self, name):
        self.name = name


class Session:
    def get_inputs(self):
        return [Input("input_ids"), Input("attention_mask")]

    def run(self, output_names, values):
        assert output_names is None
        assert set(values) == {"input_ids", "attention_mask"}
        batch = values["input_ids"].shape[0]
        hidden = np.zeros((batch, 2, 3), dtype=np.float32)
        hidden[:, 0, :] = np.asarray([3.0, 4.0, 0.0])
        return [hidden]


def make_client():
    tokenizer = Tokenizer()
    client = LocalEmbeddingClient(
        model_path="unused.onnx",
        tokenizer_path="unused.json",
        session=Session(),
        tokenizer=tokenizer,
    )
    return client, tokenizer


def test_local_embedding_uses_distinct_e5_prefixes_and_normalizes_vectors():
    client, tokenizer = make_client()

    passage = client.embed(["medical evidence"])
    query = client.embed_queries(["clinical question"])

    assert tokenizer.calls == [
        ["passage: medical evidence"],
        ["query: clinical question"],
    ]
    assert passage[0] == pytest.approx([0.6, 0.8, 0.0])
    assert query[0] == pytest.approx([0.6, 0.8, 0.0])
    assert client.probe() == 3


def test_local_embedding_validates_input_without_running_inference():
    client, _ = make_client()

    assert client.embed([]) == []
    with pytest.raises(KnowledgeBaseError) as captured:
        client.embed([" "])
    assert captured.value.code == "embedding_input_invalid"
