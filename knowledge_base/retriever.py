"""Four-list hybrid retrieval with deterministic evidence-aware reranking."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
import unicodedata
from typing import Any, Iterable, Sequence

from qdrant_client.http import models as qmodels

from .errors import KnowledgeBaseError
from .models import RankedHit, SearchFilters


EVIDENCE_LEVEL_SCORE = {
    "guideline": 1.0,
    "systematic_review": 0.90,
    "randomized_controlled_trial": 0.85,
    "observational_study": 0.55,
    "narrative_review": 0.35,
    "case_report": 0.20,
    "unknown": 0.10,
}
QUALITY_SCORE = {
    "high": 1.0,
    "moderate": 0.7,
    "low": 0.4,
    "very_low": 0.2,
    "unknown": 0.3,
}

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*|[\u4e00-\u9fff]+")
_POPULATION_TERMS = (
    "children",
    "child",
    "pediatric",
    "paediatric",
    "infant",
    "adolescent",
    "儿童",
    "患儿",
    "婴幼儿",
    "青少年",
)
_INTERVENTION_TERMS = (
    "low dose",
    "steroid",
    "glucocorticoid",
    "methylprednisolone",
    "azithromycin",
    "低剂量",
    "糖皮质激素",
    "甲泼尼龙",
    "阿奇霉素",
)
_COMPARATOR_TERMS = (
    "pulse dose",
    "high dose",
    "versus",
    " vs ",
    "placebo",
    "冲击剂量",
    "冲击疗法",
    "高剂量",
    "安慰剂",
    "相比",
)
_OUTCOME_TERMS = (
    "fever",
    "mortality",
    "hospital stay",
    "adverse event",
    "hypertension",
    "hyperglycemia",
    "退热",
    "死亡",
    "住院",
    "不良反应",
    "高血压",
    "高血糖",
)
_SAFETY_TERMS = (
    "safety",
    "adverse",
    "harm",
    "hypertension",
    "hyperglycemia",
    "bleeding",
    "infection",
    "安全性",
    "不良反应",
    "风险",
    "高血压",
    "高血糖",
    "出血",
    "感染",
)
_LOWER_LEVEL_EVIDENCE = {
    "observational_study",
    "narrative_review",
    "case_report",
    "unknown",
}


@dataclass(frozen=True)
class QueryContext:
    raw_question: str
    normalized_fts_query: str
    population_terms: tuple[str, ...]
    intervention_terms: tuple[str, ...]
    comparator_terms: tuple[str, ...]
    outcome_terms: tuple[str, ...]
    safety_focused: bool
    year_from: int | None
    year_to: int | None
    evidence_types: tuple[str, ...]
    fulltext_only: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_question": self.raw_question,
            "normalized_fts_query": self.normalized_fts_query,
            "population_terms": list(self.population_terms),
            "intervention_terms": list(self.intervention_terms),
            "comparator_terms": list(self.comparator_terms),
            "outcome_terms": list(self.outcome_terms),
            "safety_focused": self.safety_focused,
            "year_from": self.year_from,
            "year_to": self.year_to,
            "evidence_types": list(self.evidence_types),
            "fulltext_only": self.fulltext_only,
        }


@dataclass(frozen=True)
class RetrievedDocument:
    document_id: str
    fused_score: float
    final_score: float
    supporting_hits: tuple[RankedHit, ...]
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "fused_score": self.fused_score,
            "final_score": self.final_score,
            "supporting_hits": [
                {
                    "record_id": item.record_id,
                    "document_id": item.document_id,
                    "source": item.source,
                    "rank": item.rank,
                    "score": item.score,
                    "text": item.text,
                    "payload": dict(item.payload),
                }
                for item in self.supporting_hits
            ],
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class RetrievalResult:
    mode: str
    degraded_reason: str
    documents: tuple[RetrievedDocument, ...]
    candidate_count: int
    query_context: QueryContext

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "degraded_reason": self.degraded_reason,
            "candidate_count": self.candidate_count,
            "query_context": self.query_context.as_dict(),
            "documents": [item.as_dict() for item in self.documents],
        }


@dataclass(frozen=True)
class _Candidate:
    document_id: str
    fused_score: float
    supporting_hits: tuple[RankedHit, ...]
    payload: dict[str, Any]


class HybridRetriever:
    METADATA_LIMIT = 40
    FULLTEXT_LIMIT = 80
    RRF_K = 60
    CANDIDATE_LIMIT = 200
    EVIDENCE_PACKAGE_LIMIT = 15
    POLICY_WINDOW = 30

    def __init__(self, embedding_client: Any, vector_store: Any, lexical_store: Any):
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.lexical_store = lexical_store

    def search(
        self, question: str, filters: SearchFilters | None = None
    ) -> RetrievalResult:
        active_filters = filters or SearchFilters()
        self._validate_filters(active_filters)
        context = build_query_context(question, active_filters)

        lists: list[list[RankedHit]] = []
        vector_available = False
        lexical_available = False
        vector_failure = False
        lexical_failure = False
        embedding_failure = False

        try:
            embedded = self.embedding_client.embed([context.raw_question])
            if len(embedded) != 1 or not embedded[0]:
                raise ValueError("query embedding response is invalid")
            query_filter = build_vector_filter(active_filters)
            vector_lists, vector_failure = self._vector_lists(
                embedded[0], query_filter
            )
            vector_available = bool(vector_lists)
            lists.extend(vector_lists)
        except Exception:
            embedding_failure = True

        lexical_lists, lexical_failure, lexical_available = self._lexical_lists(
            context.normalized_fts_query, active_filters
        )
        lists.extend(lexical_lists)

        if not vector_available and not lexical_available:
            raise KnowledgeBaseError(
                "retrieval_unavailable", "Literature retrieval is unavailable"
            )

        if embedding_failure:
            mode = "keyword"
            degraded_reason = "query_embedding_unavailable"
        elif not vector_available:
            mode = "keyword"
            degraded_reason = "vector_retrieval_unavailable"
        elif not lexical_available:
            mode = "vector"
            degraded_reason = "lexical_retrieval_unavailable"
        else:
            mode = "hybrid"
            if vector_failure:
                degraded_reason = "partial_vector_retrieval"
            elif lexical_failure:
                degraded_reason = "partial_lexical_retrieval"
            else:
                degraded_reason = ""

        candidates = self._fuse(lists, active_filters)
        ranked = self._rerank(candidates)
        selected = self._select_evidence_package(ranked, context)
        return RetrievalResult(
            mode=mode,
            degraded_reason=degraded_reason,
            documents=tuple(selected),
            candidate_count=len(candidates),
            query_context=context,
        )

    def _vector_lists(
        self, vector: Sequence[float], query_filter: qmodels.Filter | None
    ) -> tuple[list[list[RankedHit]], bool]:
        results: list[list[RankedHit]] = []
        failures = 0
        for source, method_name, limit in (
            ("vector_metadata", "query_metadata", self.METADATA_LIMIT),
            ("vector_fulltext", "query_fulltext", self.FULLTEXT_LIMIT),
        ):
            try:
                method = getattr(self.vector_store, method_name)
                hits = method(vector, limit=limit, query_filter=query_filter)
                results.append(_relabel_hits(hits, source, limit))
            except Exception:
                failures += 1
        return results, failures > 0

    def _lexical_lists(
        self, query: str, filters: SearchFilters
    ) -> tuple[list[list[RankedHit]], bool, bool]:
        results: list[list[RankedHit]] = []
        successes = 0
        failures = 0
        for source, method_name, limit in (
            ("fts_metadata", "search_metadata", self.METADATA_LIMIT),
            ("fts_fulltext", "search_fulltext", self.FULLTEXT_LIMIT),
        ):
            try:
                method = getattr(self.lexical_store, method_name)
                hits = method(query, limit, filters)
                results.append(_relabel_hits(hits, source, limit))
                successes += 1
            except Exception:
                failures += 1
        return results, failures > 0, successes > 0

    def _fuse(
        self, lists: Iterable[Iterable[RankedHit]], filters: SearchFilters
    ) -> list[_Candidate]:
        unique_hits: dict[tuple[str, str], RankedHit] = {}
        for result_list in lists:
            for item in result_list:
                key = (item.source, item.record_id)
                current = unique_hits.get(key)
                if current is None or item.rank < current.rank:
                    unique_hits[key] = item

        grouped: dict[str, list[RankedHit]] = {}
        for item in unique_hits.values():
            grouped.setdefault(item.document_id, []).append(item)

        candidates: list[_Candidate] = []
        for document_id, hits in grouped.items():
            selected_hits = _bounded_supporting_hits(hits)
            payload = _merge_payload(selected_hits)
            if not _matches_filters(payload, selected_hits, filters):
                continue
            fused_score = sum(1.0 / (self.RRF_K + item.rank) for item in selected_hits)
            candidates.append(
                _Candidate(
                    document_id=document_id,
                    fused_score=fused_score,
                    supporting_hits=selected_hits,
                    payload=payload,
                )
            )

        candidates.sort(key=lambda item: (-item.fused_score, item.document_id))
        return candidates[: self.CANDIDATE_LIMIT]

    def _rerank(self, candidates: list[_Candidate]) -> list[RetrievedDocument]:
        if not candidates:
            return []
        max_fused = max(item.fused_score for item in candidates)
        recency = _recency_scores(candidates)
        ranked: list[RetrievedDocument] = []
        for item in candidates:
            evidence_type = _evidence_type(item.payload)
            quality = _quality(item.payload, item.supporting_hits)
            relevance_score = item.fused_score / max_fused if max_fused else 0.0
            source_completeness = (
                1.0 if _has_fulltext_hit(item.supporting_hits) else 0.4
            )
            final_score = (
                0.65 * relevance_score
                + 0.15 * EVIDENCE_LEVEL_SCORE.get(evidence_type, 0.10)
                + 0.10 * QUALITY_SCORE.get(quality, 0.3)
                + 0.05 * recency[item.document_id]
                + 0.05 * source_completeness
            )
            payload = dict(item.payload)
            payload.update(
                {
                    "evidence_type": evidence_type,
                    "quality": quality,
                    "relevance_score": relevance_score,
                    "recency_score": recency[item.document_id],
                    "source_completeness_score": source_completeness,
                }
            )
            ranked.append(
                RetrievedDocument(
                    document_id=item.document_id,
                    fused_score=item.fused_score,
                    final_score=final_score,
                    supporting_hits=item.supporting_hits,
                    payload=payload,
                )
            )
        ranked.sort(
            key=lambda item: (-item.final_score, -item.fused_score, item.document_id)
        )
        return ranked

    def _select_evidence_package(
        self, ranked: list[RetrievedDocument], context: QueryContext
    ) -> list[RetrievedDocument]:
        if len(ranked) <= self.EVIDENCE_PACKAGE_LIMIT:
            return ranked

        policy_window = sorted(
            ranked,
            key=lambda item: (-item.fused_score, item.document_id),
        )[: self.POLICY_WINDOW]
        required: list[RetrievedDocument] = []
        for evidence_type in (
            "guideline",
            "systematic_review",
            "randomized_controlled_trial",
        ):
            match = next(
                (
                    item
                    for item in policy_window
                    if _evidence_type(item.payload) == evidence_type
                ),
                None,
            )
            if match is not None:
                required.append(match)

        safety = next(
            (
                item
                for item in policy_window
                if _evidence_type(item.payload) in _LOWER_LEVEL_EVIDENCE
                and _is_safety_focused(item)
            ),
            None,
        )
        if safety is not None:
            required.append(safety)

        selected = list(ranked[: self.EVIDENCE_PACKAGE_LIMIT])
        required_ids = {item.document_id for item in required}
        for item in required:
            if item.document_id in {current.document_id for current in selected}:
                continue
            replace_index = next(
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if selected[index].document_id not in required_ids
                ),
                None,
            )
            if replace_index is not None:
                selected[replace_index] = item
        selected.sort(
            key=lambda item: (-item.final_score, -item.fused_score, item.document_id)
        )
        return selected

    @staticmethod
    def _validate_filters(filters: SearchFilters) -> None:
        if (
            filters.year_from is not None
            and filters.year_to is not None
            and filters.year_from > filters.year_to
        ):
            raise KnowledgeBaseError("invalid_year_range", "Year range is invalid")


def build_query_context(
    question: str, filters: SearchFilters | None = None
) -> QueryContext:
    if not isinstance(question, str) or not question.strip():
        raise KnowledgeBaseError("invalid_query", "Clinical question is required")
    raw_question = " ".join(question.split())
    normalized = unicodedata.normalize("NFKC", raw_question).casefold()
    padded = f" {normalized} "
    tokens = _TOKEN_PATTERN.findall(normalized)
    active_filters = filters or SearchFilters()
    return QueryContext(
        raw_question=raw_question,
        normalized_fts_query=" ".join(tokens),
        population_terms=_detect_terms(padded, _POPULATION_TERMS),
        intervention_terms=_detect_terms(padded, _INTERVENTION_TERMS),
        comparator_terms=_detect_terms(padded, _COMPARATOR_TERMS),
        outcome_terms=_detect_terms(padded, _OUTCOME_TERMS),
        safety_focused=any(term in padded for term in _SAFETY_TERMS),
        year_from=active_filters.year_from,
        year_to=active_filters.year_to,
        evidence_types=tuple(sorted(active_filters.evidence_types)),
        fulltext_only=active_filters.fulltext_only,
    )


def build_vector_filter(filters: SearchFilters) -> qmodels.Filter | None:
    conditions: list[qmodels.Condition] = []
    if filters.evidence_types:
        conditions.append(
            qmodels.FieldCondition(
                key="evidence_type",
                match=qmodels.MatchAny(any=sorted(filters.evidence_types)),
            )
        )
    if filters.year_from is not None or filters.year_to is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="year",
                range=qmodels.Range(gte=filters.year_from, lte=filters.year_to),
            )
        )
    if filters.fulltext_only:
        conditions.append(
            qmodels.FieldCondition(
                key="has_fulltext", match=qmodels.MatchValue(value=True)
            )
        )
    return qmodels.Filter(must=conditions) if conditions else None


def _detect_terms(text: str, terms: Iterable[str]) -> tuple[str, ...]:
    return tuple(term.strip() for term in terms if term in text)


def _relabel_hits(
    hits: Iterable[RankedHit], source: str, limit: int
) -> list[RankedHit]:
    return [
        replace(item, source=source, rank=rank)
        for rank, item in enumerate(list(hits)[:limit], start=1)
    ]


def _bounded_supporting_hits(hits: Iterable[RankedHit]) -> tuple[RankedHit, ...]:
    ordered = sorted(hits, key=lambda item: (item.rank, item.source, item.record_id))
    selected: list[RankedHit] = []
    fulltext_record_ids: set[str] = set()
    for item in ordered:
        if "fulltext" in item.source:
            if (
                item.record_id not in fulltext_record_ids
                and len(fulltext_record_ids) >= 3
            ):
                continue
            fulltext_record_ids.add(item.record_id)
        selected.append(item)
    return tuple(selected)


def _merge_payload(hits: Iterable[RankedHit]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    evidence_type = "unknown"
    quality = "unknown"
    has_fulltext = False
    for item in hits:
        for key, value in item.payload.items():
            if key not in payload or payload[key] in (None, "", "unknown", False):
                payload[key] = value
        candidate_type = str(item.payload.get("evidence_type", "unknown"))
        if evidence_type == "unknown" and candidate_type in EVIDENCE_LEVEL_SCORE:
            evidence_type = candidate_type
        candidate_quality = str(item.payload.get("quality", "unknown"))
        if QUALITY_SCORE.get(candidate_quality, 0.0) > QUALITY_SCORE.get(quality, 0.0):
            quality = candidate_quality
        has_fulltext = has_fulltext or "fulltext" in item.source or bool(
            item.payload.get("has_fulltext")
        )
    payload["evidence_type"] = evidence_type
    payload["quality"] = quality
    payload["has_fulltext"] = has_fulltext
    return payload


def _matches_filters(
    payload: dict[str, Any], hits: Iterable[RankedHit], filters: SearchFilters
) -> bool:
    evidence_type = _evidence_type(payload)
    if filters.evidence_types and evidence_type not in filters.evidence_types:
        return False
    year = _year(payload)
    if filters.year_from is not None and (year is None or year < filters.year_from):
        return False
    if filters.year_to is not None and (year is None or year > filters.year_to):
        return False
    if filters.fulltext_only and not (
        bool(payload.get("has_fulltext")) or _has_fulltext_hit(hits)
    ):
        return False
    return True


def _recency_scores(candidates: Iterable[_Candidate]) -> dict[str, float]:
    items = list(candidates)
    known_years = [year for item in items if (year := _year(item.payload)) is not None]
    if not known_years or min(known_years) == max(known_years):
        return {item.document_id: 0.5 for item in items}
    minimum = min(known_years)
    span = max(known_years) - minimum
    return {
        item.document_id: (
            0.0
            if _year(item.payload) is None
            else (int(_year(item.payload)) - minimum) / span
        )
        for item in items
    }


def _year(payload: dict[str, Any]) -> int | None:
    value = payload.get("year")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _evidence_type(payload: dict[str, Any]) -> str:
    value = str(payload.get("evidence_type", "unknown"))
    return value if value in EVIDENCE_LEVEL_SCORE else "unknown"


def _quality(payload: dict[str, Any], hits: Iterable[RankedHit]) -> str:
    values = [str(payload.get("quality", "unknown"))]
    values.extend(str(item.payload.get("quality", "unknown")) for item in hits)
    return max(values, key=lambda value: QUALITY_SCORE.get(value, 0.0))


def _has_fulltext_hit(hits: Iterable[RankedHit]) -> bool:
    return any("fulltext" in item.source for item in hits)


def _is_safety_focused(item: RetrievedDocument) -> bool:
    text = " ".join(hit.text for hit in item.supporting_hits).casefold()
    return any(term in text for term in _SAFETY_TERMS)
