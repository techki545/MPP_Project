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
_CONDITION_TERMS = (
    "smpp",
    "rmpp",
    "mpp",
    "mycoplasma pneumoniae pneumonia",
    "mycoplasma pneumonia",
    "severe mycoplasma pneumonia",
    "重症支原体肺炎",
    "难治性支原体肺炎",
    "肺炎支原体肺炎",
    "支原体肺炎",
)
_ADULT_POPULATION_TERMS = (
    "adult",
    "adults",
    "elderly",
    "older patients",
    "成人",
    "成年人",
    "老年人",
    "老年患者",
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
    "大环内酯",
    "克拉霉素",
    "多西环素",
    "米诺环素",
    "左氧氟沙星",
    "免疫球蛋白",
    "丙种球蛋白",
    "ivig",
    "支气管镜",
    "肺泡灌洗",
    "抗凝",
    "溶栓",
    "运动",
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
    "疗程",
    "治疗周期",
    "住院时间",
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
_QUESTION_TYPE_TERMS = (
    ("diagnosis", ("诊断", "鉴别", "识别", "预测", "检查", "diagnos", "detect")),
    ("prognosis", ("预后", "随访", "长期结局", "复发", "prognos", "follow-up")),
    ("etiology", ("病因", "危险因素", "为什么", "etiolog", "risk factor")),
    ("prevention", ("预防", "疫苗", "prevent", "prophyl")),
    ("safety", ("安全吗", "不良反应", "副作用", "风险是什么", "safety", "adverse")),
    (
        "treatment",
        (
            "治疗",
            "用药",
            "吃什么药",
            "用什么药",
            "用哪些药",
            "哪些药",
            "是否应",
            "能否",
            "剂量",
            "疗程",
            "何时使用",
            "怎么处理",
            "调整治疗",
            "可考虑",
            "何时需要",
            "获益",
            "treat",
            "therapy",
            "dose",
        ),
    ),
)
_RETRIEVAL_FOCUS_RULES = (
    (
        (
            ("糖皮质激素", "甲泼尼龙", "steroid", "corticosteroid", "methylprednisolone"),
            ("何时", "时间", "开始", "时机", "早期", "early", "timing"),
        ),
        "早期 糖皮质激素 治疗时机 early corticosteroid therapy timing",
    ),
    (
        (
            ("大环内酯", "macrolide"),
            ("耐药", "resistan"),
        ),
        "大环内酯 耐药 macrolide resistant resistance",
    ),
    (
        (("四环素", "多西环素", "米诺环素", "tetracycline", "doxycycline", "minocycline"),),
        "多西环素 米诺环素 doxycycline minocycline tetracycline tosufloxacin",
    ),
    (
        (("血栓", "肺栓塞", "抗凝", "溶栓", "thrombo", "embol", "anticoag"),),
        "肺栓塞 血栓 溶栓 抗凝 pulmonary embolism thrombosis anticoagulation",
    ),
    (
        (
            ("难治性", "rmpp", "refractory"),
            ("诊断", "识别", "预测", "diagnos", "predict"),
        ),
        "难治性 早期诊断 预测 refractory early diagnosis prediction",
    ),
    (
        (
            ("重症", "smpp", "severe"),
            ("标准", "识别", "危险因素", "criteria", "diagnos", "risk factor"),
        ),
        "重症 诊断标准 危险因素 severe diagnostic criteria risk factors",
    ),
    (
        (
            ("闭塞性细支气管炎", "支气管扩张", "bronchiolitis obliterans", "bronchiectasis"),
            ("随访", "长期", "结局", "危险因素", "follow-up", "outcome", "risk factor"),
        ),
        "闭塞性细支气管炎 支气管扩张 危险因素 bronchiolitis obliterans bronchiectasis risk factors",
    ),
)
_LOWER_LEVEL_EVIDENCE = {
    "observational_study",
    "narrative_review",
    "case_report",
    "unknown",
}

_QUESTION_TOPIC_GROUPS = (
    ("steroid", ("糖皮质激素", "甲泼尼龙", "甲强龙", "地塞米松", "steroid", "corticosteroid", "methylprednisolone")),
    ("exercise", ("运动", "锻炼", "康复训练", "体力活动", "exercise", "physical activity")),
    ("macrolide", ("大环内酯", "阿奇霉素", "红霉素", "克拉霉素", "macrolide", "azithromycin")),
    ("tetracycline", ("四环素", "多西环素", "米诺环素", "tetracycline", "doxycycline", "minocycline")),
    ("bronchoscopy", ("支气管镜", "肺泡灌洗", "bronchoscopy", "bronchoalveolar lavage")),
    ("ivig", ("免疫球蛋白", "丙种球蛋白", "ivig", "immunoglobulin")),
    ("thrombosis", ("血栓", "肺栓塞", "抗凝", "溶栓", "thrombo", "embol", "anticoag")),
    ("timing", ("何时", "时机", "早期", "病程", "timing", "when", "early")),
    ("dose", ("剂量", "用量", "疗程", "mg/kg", "dose")),
    ("diagnosis", ("诊断", "识别", "预测", "标准", "危险因素", "diagnos", "predict", "risk factor")),
    ("long_term", ("长期", "随访", "闭塞性细支气管炎", "支气管扩张", "long-term", "follow-up", "bronchiolitis obliterans", "bronchiectasis")),
)

_SMPP_TERMS = (
    "smpp",
    "severe mycoplasma pneumoniae pneumonia",
    "severe mycoplasma pneumonia",
    "重症肺炎支原体肺炎",
    "重症支原体肺炎",
    "重症mpp",
)
_RMPP_TERMS = (
    "rmpp",
    "refractory mycoplasma pneumoniae pneumonia",
    "refractory mycoplasma pneumonia",
    "难治性肺炎支原体肺炎",
    "难治性支原体肺炎",
    "难治性mpp",
)
_MPP_TERMS = (
    "mpp",
    "mycoplasma pneumoniae pneumonia",
    "mycoplasma pneumonia",
    "肺炎支原体肺炎",
    "支原体肺炎",
)


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
    condition_terms: tuple[str, ...] = ()
    question_type: str = "general"
    retrieval_queries: tuple[str, ...] = ()

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
            "condition_terms": list(self.condition_terms),
            "question_type": self.question_type,
            "pico": {
                "population": list(dict.fromkeys((*self.population_terms, *self.condition_terms))),
                "intervention": list(self.intervention_terms),
                "comparator": list(self.comparator_terms),
                "outcome": list(self.outcome_terms),
            },
            "retrieval_queries": list(self.retrieval_queries),
            "subquestions": list(self.retrieval_queries[1:]),
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
        retrieval_queries = context.retrieval_queries or (context.raw_question,)

        lists: list[list[RankedHit]] = []
        vector_available = False
        lexical_available = False
        vector_failure = False
        lexical_failure = False
        embedding_failure = False

        try:
            embed_queries = getattr(self.embedding_client, "embed_queries", None)
            embedded = (
                embed_queries(list(retrieval_queries))
                if callable(embed_queries)
                else self.embedding_client.embed(list(retrieval_queries))
            )
            if len(embedded) != len(retrieval_queries) or not all(embedded):
                raise ValueError("query embedding response is invalid")
            query_filter = build_vector_filter(active_filters)
            vector_failures: list[bool] = []
            for query_index, vector in enumerate(embedded):
                vector_lists, failed = self._vector_lists(
                    vector,
                    query_filter,
                    query_index=query_index,
                )
                vector_failures.append(failed)
                lists.extend(vector_lists)
            vector_available = any("vector_" in item.source for group in lists for item in group)
            vector_failure = any(vector_failures)
        except Exception:
            embedding_failure = True

        lexical_failures: list[bool] = []
        lexical_successes: list[bool] = []
        for query_index, retrieval_query in enumerate(retrieval_queries):
            normalized_fts_query = " ".join(
                _TOKEN_PATTERN.findall(
                    unicodedata.normalize("NFKC", retrieval_query).casefold()
                )
            )
            lexical_lists, failed, available = self._lexical_lists(
                normalized_fts_query,
                active_filters,
                query_index=query_index,
            )
            lists.extend(lexical_lists)
            lexical_failures.append(failed)
            lexical_successes.append(available)
        lexical_failure = any(lexical_failures)
        lexical_available = any(lexical_successes)

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

        candidates = self._fuse(lists, active_filters, context)
        ranked = self._rerank(candidates, context)
        selected = self._select_evidence_package(ranked, context)
        return RetrievalResult(
            mode=mode,
            degraded_reason=degraded_reason,
            documents=tuple(selected),
            candidate_count=len(candidates),
            query_context=context,
        )

    def _vector_lists(
        self,
        vector: Sequence[float],
        query_filter: qmodels.Filter | None,
        *,
        query_index: int = 0,
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
                source_name = source if query_index == 0 else f"{source}_focus_{query_index}"
                results.append(_relabel_hits(hits, source_name, limit))
            except Exception:
                failures += 1
        return results, failures > 0

    def _lexical_lists(
        self,
        query: str,
        filters: SearchFilters,
        *,
        query_index: int = 0,
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
                source_name = source if query_index == 0 else f"{source}_focus_{query_index}"
                results.append(_relabel_hits(hits, source_name, limit))
                successes += 1
            except Exception:
                failures += 1
        return results, failures > 0, successes > 0

    def _fuse(
        self,
        lists: Iterable[Iterable[RankedHit]],
        filters: SearchFilters,
        context: QueryContext,
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
            if not _matches_population_context(payload, selected_hits, context):
                continue
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

    def _rerank(
        self, candidates: list[_Candidate], context: QueryContext
    ) -> list[RetrievedDocument]:
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
            applicability, applicability_score = _evidence_applicability(
                item.payload, item.supporting_hits, context
            )
            question_relevance = _question_relevance(
                item.payload, item.supporting_hits, context
            )
            semantic_similarity = _semantic_similarity(item.supporting_hits)
            final_score = (
                0.65 * relevance_score
                + 0.15 * EVIDENCE_LEVEL_SCORE.get(evidence_type, 0.10)
                + 0.10 * QUALITY_SCORE.get(quality, 0.3)
                + 0.05 * recency[item.document_id]
                + 0.05 * source_completeness
            )
            if _target_condition_scope(context) != "unspecified":
                final_score += 0.10 * (applicability_score - 0.5)
                final_score += 0.08 * (question_relevance - 0.5)
                final_score += 0.08 * (semantic_similarity - 0.5)
            payload = dict(item.payload)
            payload.update(
                {
                    "evidence_type": evidence_type,
                    "quality": quality,
                    "relevance_score": relevance_score,
                    "recency_score": recency[item.document_id],
                    "source_completeness_score": source_completeness,
                    "population_applicability": applicability,
                    "population_applicability_score": applicability_score,
                    "question_relevance_score": question_relevance,
                    "semantic_similarity_score": semantic_similarity,
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

        for query_index in range(1, len(context.retrieval_queries)):
            focus_suffix = f"_focus_{query_index}"
            for source_prefix, retain_count in (
                ("fts_metadata", 2),
                ("vector_metadata", 1),
            ):
                source_name = f"{source_prefix}{focus_suffix}"
                focus_candidates = [
                    item
                    for item in ranked
                    if any(hit.source == source_name for hit in item.supporting_hits)
                ]
                ordered_focus = sorted(
                    focus_candidates,
                    key=lambda item: (
                        min(
                            hit.rank
                            for hit in item.supporting_hits
                            if hit.source == source_name
                        ),
                        -item.final_score,
                        item.document_id,
                    ),
                )
                required.extend(ordered_focus[:retain_count])

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
    population_terms = _detect_terms(padded, _POPULATION_TERMS)
    condition_terms = _detect_terms(padded, _CONDITION_TERMS)
    intervention_terms = _detect_terms(padded, _INTERVENTION_TERMS)
    comparator_terms = _detect_terms(padded, _COMPARATOR_TERMS)
    outcome_terms = _detect_terms(padded, _OUTCOME_TERMS)
    question_type = _detect_question_type(padded)
    retrieval_queries = _build_retrieval_queries(
        raw_question,
        question_type=question_type,
        population_terms=population_terms,
        condition_terms=condition_terms,
        intervention_terms=intervention_terms,
    )
    return QueryContext(
        raw_question=raw_question,
        normalized_fts_query=" ".join(tokens),
        population_terms=population_terms,
        intervention_terms=intervention_terms,
        comparator_terms=comparator_terms,
        outcome_terms=outcome_terms,
        safety_focused=any(term in padded for term in _SAFETY_TERMS),
        year_from=active_filters.year_from,
        year_to=active_filters.year_to,
        evidence_types=tuple(sorted(active_filters.evidence_types)),
        fulltext_only=active_filters.fulltext_only,
        condition_terms=condition_terms,
        question_type=question_type,
        retrieval_queries=retrieval_queries,
    )


def _detect_question_type(text: str) -> str:
    matched = {
        question_type
        for question_type, terms in _QUESTION_TYPE_TERMS
        if any(term in text for term in terms)
    }
    if "safety" in matched and any(
        term in text for term in ("安全吗", "不良反应", "副作用", "safety", "adverse event")
    ):
        return "safety"
    if "treatment" in matched and any(
        term in text
        for term in (
            "治疗",
            "用药",
            "吃什么药",
            "用什么药",
            "用哪些药",
            "哪些药",
            "是否应",
            "能否",
            "剂量",
            "疗程",
            "何时使用",
            "怎么处理",
            "可考虑",
            "何时需要",
            "获益",
            "treat",
            "therapy",
            "dose",
        )
    ):
        return "treatment"
    for question_type, terms in _QUESTION_TYPE_TERMS:
        if question_type in matched:
            return question_type
    return "general"


def _build_retrieval_queries(
    raw_question: str,
    *,
    question_type: str,
    population_terms: tuple[str, ...],
    condition_terms: tuple[str, ...],
    intervention_terms: tuple[str, ...],
) -> tuple[str, ...]:
    queries = [raw_question]
    normalized = unicodedata.normalize("NFKC", raw_question).casefold()
    condition = condition_terms[0] if condition_terms else "肺炎支原体肺炎"
    population = population_terms[0] if population_terms else "儿童"
    for term_groups, focus_query in _RETRIEVAL_FOCUS_RULES:
        if all(any(term in normalized for term in group) for group in term_groups):
            queries.append(f"{condition} {population} {focus_query}")
    duration_focused = bool(
        re.search(
            r"疗程|治疗周期|治疗多久|多长时间|需要多久|持续多久|course of treatment|treatment duration",
            normalized,
            re.I,
        )
    )
    if (
        question_type == "treatment"
        and duration_focused
        and not intervention_terms
        and condition_terms
    ):
        queries.extend(
            (
                f"{condition} {population} 抗菌药物 阿奇霉素 疗程 治疗周期 指南",
                f"{condition} {population} 糖皮质激素 甲泼尼龙 疗程 减量 治疗周期",
            )
        )
    elif question_type == "treatment" and not intervention_terms and condition_terms:
        queries.extend(
            (
                f"{condition} {population} 抗菌药物 一线治疗 指南",
                f"{condition} {population} 糖皮质激素 免疫治疗",
            )
        )
    elif len(queries) == 1 and condition_terms:
        intervention = intervention_terms[0] if intervention_terms else ""
        focus_queries = {
            "diagnosis": f"{condition} {population} 诊断标准 临床指标 病原学检测 鉴别诊断",
            "prognosis": f"{condition} {population} 预后 危险因素 长期结局 随访",
            "etiology": f"{condition} {population} 重症 危险因素 相关因素 发病机制",
            "prevention": f"{condition} {population} 预防 早期干预 风险降低",
            "safety": (
                f"{condition} {population} {intervention} 不良反应 安全性 QT间期 禁忌"
            ),
        }
        focus_query = focus_queries.get(question_type)
        if focus_query:
            queries.append(focus_query)
    return tuple(dict.fromkeys(queries))[:3]


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


def _matches_population_context(
    payload: dict[str, Any],
    hits: Iterable[RankedHit],
    context: QueryContext,
) -> bool:
    if not context.population_terms:
        return True
    text_parts = [
        str(payload.get("title", "")),
        str(payload.get("abstract", "")),
        str(payload.get("population", "")),
    ]
    for hit in hits:
        text_parts.extend(
            (
                hit.text,
                str(hit.payload.get("title", "")),
                str(hit.payload.get("abstract", "")),
                str(hit.payload.get("population", "")),
            )
        )
    normalized = unicodedata.normalize("NFKC", " ".join(text_parts)).casefold()
    pediatric_signal = any(term in normalized for term in _POPULATION_TERMS)
    adult_signal = any(term in normalized for term in _ADULT_POPULATION_TERMS)
    return not (adult_signal and not pediatric_signal)


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


def _target_condition_scope(context: QueryContext) -> str:
    text = unicodedata.normalize("NFKC", context.raw_question).casefold()
    if any(term in text for term in _SMPP_TERMS):
        return "smpp"
    if any(term in text for term in _RMPP_TERMS):
        return "rmpp"
    if any(term in text for term in _MPP_TERMS):
        return "mpp"
    return "unspecified"


def _candidate_text(
    payload: dict[str, Any], hits: Iterable[RankedHit]
) -> str:
    values = [
        str(payload.get("title", "")),
        str(payload.get("abstract", "")),
        str(payload.get("population", "")),
    ]
    for hit in hits:
        values.extend(
            (
                hit.text,
                str(hit.payload.get("title", "")),
                str(hit.payload.get("abstract", "")),
                str(hit.payload.get("population", "")),
            )
        )
    return unicodedata.normalize("NFKC", " ".join(values)).casefold()


def _evidence_applicability(
    payload: dict[str, Any],
    hits: Iterable[RankedHit],
    context: QueryContext,
) -> tuple[str, float]:
    """Label whether a paper directly studies the condition asked about."""

    scope = _target_condition_scope(context)
    if scope == "unspecified":
        return "not_assessed", 0.5
    hit_list = tuple(hits)
    title = unicodedata.normalize(
        "NFKC",
        " ".join(
            value
            for value in (
                str(payload.get("title", "")),
                *(str(hit.payload.get("title", "")) for hit in hit_list),
            )
            if value
        ),
    ).casefold()
    metadata = unicodedata.normalize(
        "NFKC",
        " ".join(
            value
            for value in (
                str(payload.get("abstract", "")),
                str(payload.get("population", "")),
                *(str(hit.payload.get("abstract", "")) for hit in hit_list),
                *(str(hit.payload.get("population", "")) for hit in hit_list),
            )
            if value
        ),
    ).casefold()
    text = title or metadata or _candidate_text(payload, hit_list)

    def signals(value: str) -> tuple[bool, bool, bool]:
        return (
            any(term in value for term in _SMPP_TERMS),
            any(term in value for term in _RMPP_TERMS),
            any(term in value for term in _MPP_TERMS),
        )

    has_smpp, has_rmpp, has_mpp = signals(title)
    if not (has_smpp or has_rmpp or has_mpp):
        has_smpp, has_rmpp, has_mpp = signals(metadata)
    if not (has_smpp or has_rmpp or has_mpp):
        has_smpp, has_rmpp, has_mpp = signals(text)
    if scope == "smpp":
        if has_smpp:
            return "direct", 1.0
        if has_rmpp:
            return "indirect_rmpp", 0.65
        if has_mpp:
            return "general_mpp", 0.45
    elif scope == "rmpp":
        if has_rmpp:
            return "direct", 1.0
        if has_smpp:
            return "indirect_smpp", 0.65
        if has_mpp:
            return "general_mpp", 0.45
    elif scope == "mpp" and has_mpp:
        return "direct", 1.0
    return "unclear", 0.2


def _question_relevance(
    payload: dict[str, Any],
    hits: Iterable[RankedHit],
    context: QueryContext,
) -> float:
    """Return a small, explainable local reranking signal for clinical focus."""

    question = unicodedata.normalize("NFKC", context.raw_question).casefold()
    text = _candidate_text(payload, hits)
    requested_groups = [
        terms
        for _, terms in _QUESTION_TOPIC_GROUPS
        if any(term in question for term in terms)
    ]
    if not requested_groups:
        return 0.5
    matched = sum(
        1 for terms in requested_groups if any(term in text for term in terms)
    )
    return matched / len(requested_groups)


def _semantic_similarity(hits: Iterable[RankedHit]) -> float:
    """Reuse local E5 vector similarity as a post-fusion semantic reranker."""

    scores = [
        float(hit.score)
        for hit in hits
        if hit.source.startswith("vector_")
        and isinstance(hit.score, (int, float))
        and not isinstance(hit.score, bool)
    ]
    if not scores:
        return 0.5
    return max(0.0, min(1.0, max(scores)))
