"""Local Qdrant collections for metadata and full-text embeddings."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID, NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .errors import KnowledgeBaseError
from .models import RankedHit


class LocalVectorStore:
    """Small adapter that validates vectors before handing them to Qdrant."""

    METADATA_COLLECTION = "mpp_metadata"
    FULLTEXT_COLLECTION = "mpp_fulltext"
    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self._client = QdrantClient(path=str(path))
        self._dimension: int | None = None

    @property
    def client(self) -> QdrantClient:
        return self._client

    @staticmethod
    def point_id_for(record_id: object) -> UUID:
        return uuid5(NAMESPACE_URL, f"mpp-knowledge-base:{record_id}")

    def ensure_collections(self, *, dimension: int, model_name: str) -> None:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
            or not isinstance(model_name, str)
            or not model_name.strip()
        ):
            raise KnowledgeBaseError(
                "vector_collection_invalid", "Vector collection configuration is invalid"
            )
        manifest = {"embedding_model": model_name.strip(), "schema_version": self.SCHEMA_VERSION}
        for collection_name in (self.METADATA_COLLECTION, self.FULLTEXT_COLLECTION):
            if not self._client.collection_exists(collection_name):
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=dimension, distance=qmodels.Distance.COSINE
                    ),
                    metadata=manifest,
                )
            self._validate_collection(collection_name, dimension, manifest)
        self._dimension = dimension

    def load_existing(self, *, model_name: str) -> int:
        """Load and validate an existing index without calling the embedding API."""
        normalized_model = str(model_name).strip()
        if not normalized_model:
            raise KnowledgeBaseError(
                "vector_collection_invalid", "Vector collection configuration is invalid"
            )
        if not all(
            self._client.collection_exists(collection_name)
            for collection_name in (self.METADATA_COLLECTION, self.FULLTEXT_COLLECTION)
        ):
            raise KnowledgeBaseError(
                "vector_index_not_built", "Vector index has not been built"
            )
        metadata_info = self._client.get_collection(self.METADATA_COLLECTION)
        vectors = metadata_info.config.params.vectors
        if not isinstance(vectors, qmodels.VectorParams):
            raise KnowledgeBaseError(
                "vector_collection_mismatch", "Existing vector collection is incompatible"
            )
        dimension = int(vectors.size)
        manifest = {
            "embedding_model": normalized_model,
            "schema_version": self.SCHEMA_VERSION,
        }
        for collection_name in (self.METADATA_COLLECTION, self.FULLTEXT_COLLECTION):
            self._validate_collection(collection_name, dimension, manifest)
        self._dimension = dimension
        return dimension

    def upsert_metadata(
        self, items: Sequence[tuple[object, Sequence[float], Mapping[str, Any]]]
    ) -> None:
        self._upsert(self.METADATA_COLLECTION, items)

    def upsert_fulltext(
        self, items: Sequence[tuple[object, Sequence[float], Mapping[str, Any]]]
    ) -> None:
        self._upsert(self.FULLTEXT_COLLECTION, items)

    def prune_fulltext(self, record_ids: Sequence[object]) -> int:
        """Remove vectors whose source chunks no longer exist in SQLite."""

        expected = {str(record_id) for record_id in record_ids}
        stale_point_ids: list[Any] = []
        offset: Any | None = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self.FULLTEXT_COLLECTION,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                record_id = str((point.payload or {}).get("record_id", ""))
                if record_id not in expected:
                    stale_point_ids.append(point.id)
            if offset is None:
                break
        for start in range(0, len(stale_point_ids), 512):
            self._client.delete(
                collection_name=self.FULLTEXT_COLLECTION,
                points_selector=qmodels.PointIdsList(
                    points=stale_point_ids[start : start + 512]
                ),
                wait=True,
            )
        return len(stale_point_ids)

    def query_metadata(
        self,
        vector: Sequence[float],
        *,
        limit: int = 10,
        query_filter: qmodels.Filter | None = None,
    ) -> list[RankedHit]:
        return self._query(self.METADATA_COLLECTION, "metadata", vector, limit, query_filter)

    def query_fulltext(
        self,
        vector: Sequence[float],
        *,
        limit: int = 10,
        query_filter: qmodels.Filter | None = None,
    ) -> list[RankedHit]:
        return self._query(self.FULLTEXT_COLLECTION, "fulltext", vector, limit, query_filter)

    def close(self) -> None:
        self._client.close()

    def _validate_collection(
        self, collection_name: str, dimension: int, manifest: dict[str, Any]
    ) -> None:
        info = self._client.get_collection(collection_name)
        vectors = info.config.params.vectors
        if not isinstance(vectors, qmodels.VectorParams):
            raise KnowledgeBaseError(
                "vector_collection_mismatch", "Existing vector collection is incompatible"
            )
        if (
            vectors.size != dimension
            or vectors.distance != qmodels.Distance.COSINE
            or info.config.metadata != manifest
        ):
            raise KnowledgeBaseError(
                "vector_collection_mismatch", "Existing vector collection is incompatible"
            )

    def _upsert(
        self,
        collection_name: str,
        items: Sequence[tuple[object, Sequence[float], Mapping[str, Any]]],
    ) -> None:
        if not items:
            return
        dimension = self._require_dimension()
        points: list[qmodels.PointStruct] = []
        for record_id, vector, payload in items:
            identifier = str(record_id)
            if not identifier or not isinstance(payload, Mapping):
                raise KnowledgeBaseError("vector_invalid", "Vector item is invalid")
            values = self._validated_vector(vector, dimension)
            point_payload = dict(payload)
            point_payload["record_id"] = identifier
            points.append(
                qmodels.PointStruct(
                    id=str(self.point_id_for(identifier)),
                    vector=values,
                    payload=point_payload,
                )
            )
        self._client.upsert(collection_name=collection_name, points=points, wait=True)

    def _query(
        self,
        collection_name: str,
        source: str,
        vector: Sequence[float],
        limit: int,
        query_filter: qmodels.Filter | None,
    ) -> list[RankedHit]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise KnowledgeBaseError("vector_query_invalid", "Vector query is invalid")
        values = self._validated_vector(vector, self._require_dimension())
        response = self._client.query_points(
            collection_name=collection_name,
            query=values,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        hits: list[RankedHit] = []
        for rank, point in enumerate(response.points, start=1):
            payload = dict(point.payload or {})
            record_id = str(payload.get("record_id", point.id))
            document_id = str(payload.get("document_id", record_id))
            text = str(payload.get("text", payload.get("abstract", "")))
            hits.append(
                RankedHit(
                    record_id=record_id,
                    document_id=document_id,
                    source=source,
                    rank=rank,
                    score=float(point.score),
                    text=text,
                    payload=payload,
                )
            )
        return hits

    def _require_dimension(self) -> int:
        if self._dimension is None:
            raise KnowledgeBaseError(
                "vector_store_uninitialized", "Vector collections have not been initialized"
            )
        return self._dimension

    @staticmethod
    def _validated_vector(vector: Sequence[float], dimension: int) -> list[float]:
        if isinstance(vector, (str, bytes)) or len(vector) != dimension:
            raise KnowledgeBaseError("vector_invalid", "Vector values are invalid")
        values: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise KnowledgeBaseError("vector_invalid", "Vector values are invalid")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise KnowledgeBaseError("vector_invalid", "Vector values are invalid")
            values.append(numeric)
        return values
