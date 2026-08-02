"""Build a traceable local evidence graph from source-grounded claims."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from itertools import combinations
import re
import unicodedata
from typing import Any, Iterable


ALLOWED_RELATIONS = frozenset(
    {"supports", "updates", "supplements", "confirms", "conflicts", "cautions"}
)
CLINICAL_ASPECTS = frozenset(
    {
        "overall",
        "effectiveness",
        "dose",
        "timing",
        "safety",
        "diagnosis",
        "prognosis",
        "applicability",
        "other",
    }
)
EVIDENCE_ROLES = frozenset({"core", "supplement", "boundary"})
_STUDY_UPDATE_TYPES = {"randomized_controlled_trial", "systematic_review"}
_SAFETY_EVIDENCE_TYPES = {"case_report", "observational_study"}
_LOWER_LEVEL_EVIDENCE_TYPES = {
    "observational_study",
    "narrative_review",
    "case_report",
    "unknown",
}
_EVIDENCE_RANK = {
    "guideline": 0,
    "systematic_review": 1,
    "randomized_controlled_trial": 2,
    "observational_study": 3,
    "narrative_review": 4,
    "case_report": 5,
    "unknown": 6,
}
_DIRECTIONS = {"supports", "opposes", "uncertain"}
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "or",
    "the",
    "with",
    "in",
    "of",
    "for",
    "patients",
    "patient",
    "study",
}
_ALIASES = {
    "smpp": "mpp",
    "mpp": "mpp",
    "mycoplasma": "mpp",
    "pneumoniae": "mpp",
    "支原体肺炎": "mpp",
    "重症支原体肺炎": "mpp",
    "children": "children",
    "child": "children",
    "pediatric": "children",
    "paediatric": "children",
    "儿童": "children",
    "患儿": "children",
    "infant": "children",
    "infants": "children",
    "adolescent": "children",
    "adolescents": "children",
    "adult": "adults",
    "adults": "adults",
    "成人": "adults",
    "methylprednisolone": "steroid",
    "steroid": "steroid",
    "steroids": "steroid",
    "glucocorticoid": "steroid",
    "glucocorticoids": "steroid",
    "甲泼尼龙": "steroid",
    "糖皮质激素": "steroid",
    "defervescence": "fever",
    "fever": "fever",
    "退热": "fever",
}


@dataclass(frozen=True)
class EvidenceClaim:
    claim_id: str
    document_id: str
    evidence_type: str
    year: int | None
    population: str
    intervention: str
    comparator: str
    outcome: str
    direction: str
    statement: str
    source_chunk_ids: tuple[str, ...]
    evidence_cutoff_year: int | None = None
    safety_signal: bool = False
    design: str = ""
    dose: str = ""
    sample_size: str = ""
    effect_measures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    source_quote: str = ""
    follow_up: str = ""
    clinical_aspect: str = "other"
    evidence_role: str = "supplement"

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.document_id.strip():
            raise ValueError("Claim and document identifiers are required")
        if not self.source_chunk_ids or not all(
            isinstance(value, str) and value.strip() for value in self.source_chunk_ids
        ):
            raise ValueError("At least one source chunk is required")
        if self.direction not in _DIRECTIONS:
            raise ValueError("Claim direction is invalid")
        if self.clinical_aspect not in CLINICAL_ASPECTS:
            raise ValueError("Clinical aspect is invalid")
        if self.evidence_role not in EVIDENCE_ROLES:
            raise ValueError("Evidence role is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "document_id": self.document_id,
            "evidence_type": self.evidence_type,
            "year": self.year,
            "evidence_cutoff_year": self.evidence_cutoff_year,
            "population": self.population,
            "intervention": self.intervention,
            "comparator": self.comparator,
            "outcome": self.outcome,
            "direction": self.direction,
            "safety_signal": self.safety_signal,
            "design": self.design,
            "dose": self.dose,
            "sample_size": self.sample_size,
            "effect_measures": list(self.effect_measures),
            "limitations": list(self.limitations),
            "statement": self.statement,
            "source_quote": self.source_quote,
            "source_chunk_ids": list(self.source_chunk_ids),
            "follow_up": self.follow_up,
            "clinical_aspect": self.clinical_aspect,
            "evidence_role": self.evidence_role,
        }


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: str
    label: str
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    relation: str
    source_claim_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if self.relation not in ALLOWED_RELATIONS:
            raise ValueError("Evidence relation is invalid")
        if not self.source_claim_ids or not self.rationale.strip():
            raise ValueError("Evidence edge traceability is required")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "source_claim_ids": list(self.source_claim_ids),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class EvidenceGraph:
    question: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "nodes": [node.as_dict() for node in self.nodes],
            "edges": [edge.as_dict() for edge in self.edges],
        }


class LocalGraphBuilder:
    QUESTION_NODE_ID = "question"

    def build(
        self, question: str, claims: Iterable[EvidenceClaim]
    ) -> EvidenceGraph:
        normalized_question = " ".join(str(question).split())
        if not normalized_question:
            raise ValueError("Question is required")
        claim_list = list(claims)
        identifiers = [item.claim_id for item in claim_list]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Claim identifiers must be unique")

        nodes = self._nodes(normalized_question, claim_list)
        edges: list[GraphEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()

        for item in claim_list:
            if item.direction == "supports":
                self._append_edge(
                    edges,
                    seen_edges,
                    GraphEdge(
                        source=item.claim_id,
                        target=self.QUESTION_NODE_ID,
                        relation="supports",
                        source_claim_ids=(item.claim_id,),
                        rationale="The source claim directly supports the clinical question.",
                    ),
                )
                self._append_edge(
                    edges,
                    seen_edges,
                    GraphEdge(
                        source=item.claim_id,
                        target=_outcome_node_id(item.outcome),
                        relation="supports",
                        source_claim_ids=(item.claim_id,),
                        rationale="The source claim reports a supportive outcome.",
                    ),
                )
            if item.direction == "uncertain":
                self._append_edge(
                    edges,
                    seen_edges,
                    GraphEdge(
                        source=item.claim_id,
                        target=self.QUESTION_NODE_ID,
                        relation="supplements",
                        source_claim_ids=(item.claim_id,),
                        rationale=(
                            "The retrieved source contributes traceable evidence, but its "
                            "clinical direction was not inferred."
                        ),
                    ),
                )
            if item.safety_signal and item.evidence_type in _SAFETY_EVIDENCE_TYPES:
                self._append_edge(
                    edges,
                    seen_edges,
                    GraphEdge(
                        source=item.claim_id,
                        target=self.QUESTION_NODE_ID,
                        relation="cautions",
                        source_claim_ids=(item.claim_id,),
                        rationale="A lower-level source reports a relevant safety signal.",
                    ),
                )

        for first, second in combinations(claim_list, 2):
            relation = self._pair_relation(first, second)
            if relation is not None:
                self._append_edge(edges, seen_edges, relation)

        edges.sort(key=lambda edge: (edge.source, edge.target, edge.relation))
        return EvidenceGraph(
            question=normalized_question,
            nodes=tuple(nodes),
            edges=tuple(edges),
        )

    def _nodes(
        self, question: str, claims: list[EvidenceClaim]
    ) -> list[GraphNode]:
        nodes = [
            GraphNode(
                node_id=self.QUESTION_NODE_ID,
                node_type="question",
                label=question,
            )
        ]
        documents: dict[str, EvidenceClaim] = {}
        outcomes: dict[str, str] = {}
        for item in claims:
            documents.setdefault(item.document_id, item)
            outcome_id = _outcome_node_id(item.outcome)
            outcomes.setdefault(outcome_id, item.outcome)
            nodes.append(
                GraphNode(
                    node_id=item.claim_id,
                    node_type="claim",
                    label=item.statement or item.claim_id,
                    payload=item.as_dict(),
                )
            )
        for document_id, item in sorted(documents.items()):
            nodes.append(
                GraphNode(
                    node_id=f"document:{document_id}",
                    node_type="document",
                    label=document_id,
                    payload={
                        "document_id": document_id,
                        "evidence_type": item.evidence_type,
                        "year": item.year,
                    },
                )
            )
        for outcome_id, outcome in sorted(outcomes.items()):
            nodes.append(
                GraphNode(
                    node_id=outcome_id,
                    node_type="outcome",
                    label=outcome,
                )
            )
        nodes.sort(key=lambda node: (node.node_type, node.node_id))
        return nodes

    def _pair_relation(
        self, first: EvidenceClaim, second: EvidenceClaim
    ) -> GraphEdge | None:
        if _incompatible_context(first, second):
            return None
        source, target = _relation_roles(first, second)
        source_aspect = source.clinical_aspect
        target_aspect = target.clinical_aspect
        same_aspect = source_aspect == target_aspect and source_aspect != "other"
        different_aspect = (
            source_aspect != "other"
            and target_aspect != "other"
            and source_aspect != target_aspect
        )

        relation = ""
        rationale = ""
        if _is_update(source, target, same_aspect):
            relation = "updates"
            rationale = (
                "A newer trial or review addresses the same clinical aspect as the "
                "earlier guideline and may update its evidence base."
            )
        elif _is_caution(source):
            relation = "cautions"
            rationale = (
                "The source contributes a safety, applicability, or boundary signal "
                "that qualifies the target evidence."
            )
        elif same_aspect and {source.direction, target.direction} == {
            "supports",
            "opposes",
        }:
            relation = "conflicts"
            rationale = "Evidence on the same clinical aspect reports opposite directions."
        elif (
            same_aspect
            and source.direction == target.direction
            and source.direction != "uncertain"
            and target.evidence_type == "guideline"
        ):
            relation = "supports"
            rationale = "Concordant evidence supports the guideline on the same clinical aspect."
        elif (
            source.evidence_type in _LOWER_LEVEL_EVIDENCE_TYPES
            and (same_aspect or different_aspect)
        ):
            relation = "supplements"
            rationale = "Lower-level evidence supplements higher-level evidence or its applicability."
        elif _supplement_difference(source, target):
            relation = "supplements"
            rationale = "The source adds a different endpoint, dose, subgroup, or follow-up."
        elif (
            same_aspect
            and source.direction == target.direction
            and source.direction != "uncertain"
        ):
            relation = "confirms"
            rationale = "Independent evidence confirms the same directional finding."
        elif (
            different_aspect
            or (
                source.evidence_role == "supplement"
                and source.clinical_aspect != "other"
            )
        ):
            relation = "supplements"
            rationale = "The source adds a different clinical aspect, endpoint, subgroup, or follow-up."
        else:
            return None

        return GraphEdge(
            source=source.claim_id,
            target=target.claim_id,
            relation=relation,
            source_claim_ids=(source.claim_id, target.claim_id),
            rationale=rationale,
        )

    @staticmethod
    def _append_edge(
        edges: list[GraphEdge],
        seen: set[tuple[str, str, str]],
        edge: GraphEdge,
    ) -> None:
        key = (edge.source, edge.target, edge.relation)
        if key not in seen:
            edges.append(edge)
            seen.add(key)


def _relation_roles(
    first: EvidenceClaim, second: EvidenceClaim
) -> tuple[EvidenceClaim, EvidenceClaim]:
    first_rank = _EVIDENCE_RANK.get(first.evidence_type, len(_EVIDENCE_RANK))
    second_rank = _EVIDENCE_RANK.get(second.evidence_type, len(_EVIDENCE_RANK))
    if first_rank != second_rank:
        return (first, second) if first_rank > second_rank else (second, first)
    return _newer_first(first, second)


def _is_update(
    source: EvidenceClaim, target: EvidenceClaim, same_aspect: bool
) -> bool:
    if (
        not same_aspect
        or source.evidence_type not in _STUDY_UPDATE_TYPES
        or target.evidence_type != "guideline"
        or source.year is None
        or source.direction == "uncertain"
        or target.direction == "uncertain"
        or source.direction != target.direction
    ):
        return False
    target_cutoff = (
        target.evidence_cutoff_year
        if target.evidence_cutoff_year is not None
        else target.year
    )
    return target_cutoff is not None and source.year > target_cutoff


def _is_caution(source: EvidenceClaim) -> bool:
    return (
        source.safety_signal
        or source.evidence_role == "boundary"
        or source.clinical_aspect == "safety"
    )


def _incompatible_context(first: EvidenceClaim, second: EvidenceClaim) -> bool:
    if (
        first.population.strip()
        and second.population.strip()
        and not _population_overlap(first.population, second.population)
    ):
        return True
    if (
        first.intervention.strip()
        and second.intervention.strip()
        and not _terms_overlap(first.intervention, second.intervention)
    ):
        return True
    return False


def _same_pio(first: EvidenceClaim, second: EvidenceClaim) -> bool:
    return _core_overlap(first, second) and _terms_overlap(
        first.outcome, second.outcome
    )


def _core_overlap(first: EvidenceClaim, second: EvidenceClaim) -> bool:
    return _population_overlap(first.population, second.population) and _terms_overlap(
        first.intervention, second.intervention
    )


def _comparable(first: EvidenceClaim, second: EvidenceClaim) -> bool:
    if not _same_pio(first, second):
        return False
    return all(
        _compatible_or_empty(left, right)
        for left, right in (
            (first.dose, second.dose),
            (first.follow_up, second.follow_up),
        )
    ) and _canonical_terms(first.population) == _canonical_terms(second.population)


def _supplement_difference(first: EvidenceClaim, second: EvidenceClaim) -> bool:
    if not _core_overlap(first, second):
        return False
    return any(
        (
            _different_nonempty(first.dose, second.dose),
            _different_nonempty(first.follow_up, second.follow_up),
            _canonical_terms(first.population) != _canonical_terms(second.population),
            not _terms_overlap(first.outcome, second.outcome),
        )
    )


def _newer_first(
    first: EvidenceClaim, second: EvidenceClaim
) -> tuple[EvidenceClaim, EvidenceClaim]:
    first_key = (first.year if first.year is not None else -1, first.claim_id)
    second_key = (second.year if second.year is not None else -1, second.claim_id)
    return (first, second) if first_key >= second_key else (second, first)


def _compatible_or_empty(first: str, second: str) -> bool:
    return not first.strip() or not second.strip() or _normalize(first) == _normalize(second)


def _different_nonempty(first: str, second: str) -> bool:
    return bool(first.strip() and second.strip() and _normalize(first) != _normalize(second))


def _terms_overlap(first: str, second: str) -> bool:
    first_terms = _canonical_terms(first)
    second_terms = _canonical_terms(second)
    return bool(first_terms and second_terms and first_terms.intersection(second_terms))


def _population_overlap(first: str, second: str) -> bool:
    first_terms = _canonical_terms(first)
    second_terms = _canonical_terms(second)
    population_groups = {"children", "adults"}
    first_groups = first_terms.intersection(population_groups)
    second_groups = second_terms.intersection(population_groups)
    if first_groups and second_groups and first_groups.isdisjoint(second_groups):
        return False
    return bool(first_terms and second_terms and first_terms.intersection(second_terms))


def _canonical_terms(value: str) -> frozenset[str]:
    normalized = _normalize(value)
    terms: set[str] = set()
    for token in _TOKEN_PATTERN.findall(normalized):
        if token in _STOPWORDS:
            continue
        terms.add(_ALIASES.get(token, token))
        for alias, canonical in _ALIASES.items():
            if any("\u4e00" <= char <= "\u9fff" for char in alias) and alias in token:
                terms.add(canonical)
    return frozenset(terms)


def _outcome_node_id(outcome: str) -> str:
    normalized = _normalize(outcome) or "unknown"
    digest = sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"outcome:{digest}"


def _normalize(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())
