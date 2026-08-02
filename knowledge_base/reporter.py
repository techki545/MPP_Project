"""Source-grounded claim extraction and two-part evidence report generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any

from .errors import KnowledgeBaseError
from .evidence_classifier import EvidenceAssessment
from .graph_builder import EvidenceClaim


_EVIDENCE_TYPES = {
    "guideline",
    "systematic_review",
    "randomized_controlled_trial",
    "observational_study",
    "narrative_review",
    "case_report",
    "unknown",
}
_DIRECTIONS = {"supports", "opposes", "uncertain"}
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_NUMBER_PATTERN = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])")
_NUMBER_UNIT_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|mg|g|kg|ml|mmhg|day|days|week|weeks|month|months|year|years|h|hours|例|天|周|月|年)",
    flags=re.IGNORECASE,
)
_MODEL_REPORT_PROMPT = """
You are a clinical evidence synthesis engine. Return one JSON object with exactly
analysis_steps and final_answer_markdown. Explain retrieval inventory, evidence
hierarchy, chronology, agreement, lower-level supplementation, limitations, and
the final answer. Cite only the numbered sources supplied as [N]. Never invent a
study, number, dose, effect size, confidence interval, source, or citation. Any
quantitative value must appear in a cited source snippet. This is evidence support,
not a substitute for an individual clinician's judgment.
""".strip()
_CLAIM_EXTRACTION_PROMPT = """
Extract only source-grounded clinical claims from the numbered evidence snippets.
Return JSON with a claims array. Every claim must include source_number,
source_chunk_ids, population, intervention, comparator, design, sample_size, dose,
outcome, direction (supports, opposes, or uncertain), effect_measures,
safety_signal, limitations, statement, and source_quote. Copy source_chunk_ids
exactly. source_quote must be present verbatim in the supplied snippets. Do not
infer missing numbers or details.
""".strip()
_REFINEMENT_PROMPT = """
Classify one title/abstract into exactly one allowed evidence type: guideline,
systematic_review, randomized_controlled_trial, observational_study,
narrative_review, case_report, or unknown. Return evidence_type and a short basis
using words visible in the supplied title or abstract. Do not infer study design.
""".strip()


@dataclass(frozen=True)
class EvidenceSource:
    source_number: int
    document_id: str
    title: str
    evidence_type: str
    year: int | None
    chunk_ids: tuple[str, ...]
    snippets: tuple[str, ...]
    page_ranges: tuple[str, ...]
    fulltext: bool
    abstract: str = ""
    classification_confidence: float = 1.0
    classification_basis: str = ""
    quality: str = "unknown"
    journal: str = ""
    doi: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_number, bool)
            or not isinstance(self.source_number, int)
            or self.source_number < 1
            or not self.document_id.strip()
            or not self.title.strip()
        ):
            raise ValueError("Evidence source is invalid")
        if not all(value.strip() for value in self.chunk_ids + self.snippets):
            raise ValueError("Evidence source text identifiers are invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_number": self.source_number,
            "document_id": self.document_id,
            "title": self.title,
            "evidence_type": self.evidence_type,
            "year": self.year,
            "chunk_ids": list(self.chunk_ids),
            "snippets": list(self.snippets),
            "page_ranges": list(self.page_ranges),
            "fulltext": self.fulltext,
            "abstract": self.abstract,
            "classification_confidence": self.classification_confidence,
            "classification_basis": self.classification_basis,
            "quality": self.quality,
            "journal": self.journal,
            "doi": self.doi,
        }


@dataclass(frozen=True)
class EvidenceBundle:
    sources: tuple[EvidenceSource, ...]
    graph: dict[str, Any]

    def __post_init__(self) -> None:
        numbers = [item.source_number for item in self.sources]
        if len(set(numbers)) != len(numbers):
            raise ValueError("Evidence source numbers must be unique")

    def as_dict(self) -> dict[str, Any]:
        return {
            "sources": [item.as_dict() for item in self.sources],
            "graph": dict(self.graph),
        }


@dataclass(frozen=True)
class ValidatedClaims:
    claims: tuple[EvidenceClaim, ...]
    audit: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "claims": [item.as_dict() for item in self.claims],
            "audit": [dict(item) for item in self.audit],
        }


@dataclass(frozen=True)
class GeneratedReport:
    analysis_steps: tuple[dict[str, Any], ...]
    final_answer_markdown: str
    model_used: bool
    model_name: str
    model_error: str | None
    evidence_inventory: tuple[dict[str, Any], ...]
    graph: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysis_steps": [dict(item) for item in self.analysis_steps],
            "final_answer_markdown": self.final_answer_markdown,
            "model_used": self.model_used,
            "model_name": self.model_name,
            "model_error": self.model_error,
            "evidence_inventory": [dict(item) for item in self.evidence_inventory],
            "graph": dict(self.graph),
        }


class ChatEvidenceRefiner:
    def __init__(self, chat_client: Any) -> None:
        self.chat_client = chat_client

    def refine(
        self,
        title: str,
        abstract: str,
        rule_assessment: EvidenceAssessment,
    ) -> EvidenceAssessment:
        if not rule_assessment.needs_llm_refinement:
            return rule_assessment
        try:
            response = self.chat_client.complete_json(
                _REFINEMENT_PROMPT,
                {"title": title, "abstract": abstract},
            )
            evidence_type = response.get("evidence_type")
            basis = response.get("basis")
            if (
                evidence_type not in _EVIDENCE_TYPES
                or not isinstance(basis, str)
                or not _basis_is_grounded(basis, title, abstract)
            ):
                return rule_assessment
            return EvidenceAssessment(
                evidence_type=evidence_type,
                confidence=0.6,
                quality=rule_assessment.quality,
                basis=f"model:{' '.join(basis.split())}",
                quality_signals=rule_assessment.quality_signals,
            )
        except Exception:
            return rule_assessment


class GroundedClaimExtractor:
    def __init__(self, chat_client: Any) -> None:
        self.chat_client = chat_client

    def extract(self, question: str, bundle: EvidenceBundle) -> ValidatedClaims:
        response = self.chat_client.complete_json(
            _CLAIM_EXTRACTION_PROMPT,
            {
                "question": question,
                "sources": [source.as_dict() for source in bundle.sources],
            },
        )
        raw_claims = response.get("claims") if isinstance(response, Mapping) else None
        if not isinstance(raw_claims, list):
            raise KnowledgeBaseError(
                "claim_response_invalid", "Claim extraction response is invalid"
            )

        source_lookup = {item.source_number: item for item in bundle.sources}
        claims: list[EvidenceClaim] = []
        audit: list[dict[str, Any]] = []
        for index, raw_claim in enumerate(raw_claims, start=1):
            claim, reason = self._validate_claim(index, raw_claim, source_lookup)
            if claim is None:
                source_number = (
                    raw_claim.get("source_number")
                    if isinstance(raw_claim, Mapping)
                    else None
                )
                source = source_lookup.get(source_number)
                audit.append(
                    {
                        "claim_index": index,
                        "reason": reason,
                        "raw_claim": dict(raw_claim)
                        if isinstance(raw_claim, Mapping)
                        else raw_claim,
                        "source_text": list(source.snippets) if source else [],
                    }
                )
            else:
                claims.append(claim)
        return ValidatedClaims(tuple(claims), tuple(audit))

    @staticmethod
    def _validate_claim(
        index: int,
        raw_claim: object,
        source_lookup: dict[int, EvidenceSource],
    ) -> tuple[EvidenceClaim | None, str]:
        if not isinstance(raw_claim, Mapping):
            return None, "invalid_claim_shape"
        source_number = raw_claim.get("source_number")
        if isinstance(source_number, bool) or not isinstance(source_number, int):
            return None, "unknown_source_number"
        source = source_lookup.get(source_number)
        if source is None:
            return None, "unknown_source_number"
        chunk_ids = _string_tuple(raw_claim.get("source_chunk_ids"))
        if not chunk_ids or not set(chunk_ids).issubset(set(source.chunk_ids)):
            return None, "unknown_source_chunk"
        direction = raw_claim.get("direction")
        if direction not in _DIRECTIONS:
            return None, "invalid_direction"
        required_text = (
            "population",
            "intervention",
            "design",
            "outcome",
            "statement",
            "source_quote",
        )
        if any(
            not isinstance(raw_claim.get(field), str)
            or not str(raw_claim.get(field)).strip()
            for field in required_text
        ):
            return None, "missing_required_text"
        source_quote = " ".join(str(raw_claim["source_quote"]).split())
        source_text = " ".join(" ".join(source.snippets).split())
        if source_quote.casefold() not in source_text.casefold():
            return None, "source_quote_not_found"
        effect_measures = _string_tuple(raw_claim.get("effect_measures"), allow_empty=True)
        limitations = _string_tuple(raw_claim.get("limitations"), allow_empty=True)
        if effect_measures is None or limitations is None:
            return None, "invalid_list_field"
        safety_signal = raw_claim.get("safety_signal")
        if not isinstance(safety_signal, bool):
            return None, "invalid_safety_signal"

        cutoff = raw_claim.get("evidence_cutoff_year")
        if isinstance(cutoff, bool) or not isinstance(cutoff, (int, type(None))):
            cutoff = None
        return (
            EvidenceClaim(
                claim_id=f"claim-{source_number}-{index}",
                document_id=source.document_id,
                evidence_type=source.evidence_type,
                year=source.year,
                evidence_cutoff_year=cutoff,
                population=str(raw_claim["population"]).strip(),
                intervention=str(raw_claim["intervention"]).strip(),
                comparator=_optional_text(raw_claim.get("comparator")),
                outcome=str(raw_claim["outcome"]).strip(),
                direction=str(direction),
                safety_signal=safety_signal,
                design=str(raw_claim["design"]).strip(),
                dose=_optional_text(raw_claim.get("dose")),
                sample_size=_optional_scalar(raw_claim.get("sample_size")),
                effect_measures=effect_measures,
                limitations=limitations,
                statement=str(raw_claim["statement"]).strip(),
                source_quote=source_quote,
                source_chunk_ids=chunk_ids,
                follow_up=_optional_text(raw_claim.get("follow_up")),
            ),
            "",
        )


class GroundedReporter:
    def __init__(self, chat_client: Any) -> None:
        self.chat_client = chat_client

    def generate(self, question: str, bundle: EvidenceBundle) -> GeneratedReport:
        response = self.chat_client.complete_json(
            _MODEL_REPORT_PROMPT,
            {
                "question": question,
                "evidence_bundle": bundle.as_dict(),
                "required_output": {
                    "analysis_steps": [
                        {"title": "string", "body": "string", "source_ids": [1]}
                    ],
                    "final_answer_markdown": "string with [N] citations",
                },
            },
        )
        analysis_steps, final_answer = self._validate_response(response, bundle)
        return GeneratedReport(
            analysis_steps=analysis_steps,
            final_answer_markdown=final_answer,
            model_used=True,
            model_name=str(getattr(self.chat_client, "model", "configured-model")),
            model_error=None,
            evidence_inventory=tuple(source.as_dict() for source in bundle.sources),
            graph=dict(bundle.graph),
        )

    def generate_with_fallback(
        self, question: str, bundle: EvidenceBundle
    ) -> GeneratedReport:
        try:
            return self.generate(question, bundle)
        except KnowledgeBaseError as error:
            code = error.code
        except Exception:
            code = "model_generation_failed"
        return self._deterministic_fallback(bundle, code)

    @staticmethod
    def _validate_response(
        response: object, bundle: EvidenceBundle
    ) -> tuple[tuple[dict[str, Any], ...], str]:
        if not isinstance(response, Mapping):
            raise _ungrounded()
        raw_steps = response.get("analysis_steps")
        final_answer = response.get("final_answer_markdown")
        if not isinstance(raw_steps, list) or not isinstance(final_answer, str):
            raise _ungrounded()
        final_answer = final_answer.strip()
        if not final_answer:
            raise _ungrounded()

        sources = {item.source_number: item for item in bundle.sources}
        known_numbers = set(sources)
        validated_steps: list[dict[str, Any]] = []
        for raw_step in raw_steps:
            if not isinstance(raw_step, Mapping):
                raise _ungrounded()
            title = raw_step.get("title")
            body = raw_step.get("body")
            source_ids = raw_step.get("source_ids")
            if (
                not isinstance(title, str)
                or not title.strip()
                or not isinstance(body, str)
                or not body.strip()
                or not isinstance(source_ids, list)
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in source_ids
                )
            ):
                raise _ungrounded()
            cited = _citation_ids(f"{title} {body}")
            declared = set(source_ids)
            if not cited.issubset(known_numbers) or not declared.issubset(known_numbers):
                raise _ungrounded()
            if cited and not cited.issubset(declared):
                raise _ungrounded()
            grounding_ids = cited or declared
            _validate_statistics(f"{title} {body}", grounding_ids, sources)
            validated_steps.append(
                {
                    "title": title.strip(),
                    "body": body.strip(),
                    "source_ids": sorted(declared),
                }
            )

        final_citations = _citation_ids(final_answer)
        if not final_citations.issubset(known_numbers):
            raise _ungrounded()
        if known_numbers and not final_citations:
            raise _ungrounded()
        _validate_statistics(final_answer, final_citations, sources)
        return tuple(validated_steps), final_answer

    def _deterministic_fallback(
        self, bundle: EvidenceBundle, error_code: str
    ) -> GeneratedReport:
        inventory_lines = [
            f"- [{source.source_number}] {source.title} "
            f"({source.evidence_type}, {source.year or 'year unknown'})"
            for source in bundle.sources
        ]
        body = "\n".join(inventory_lines) if inventory_lines else "未检索到可用证据。"
        return GeneratedReport(
            analysis_steps=(
                {
                    "title": "证据清单（确定性回退）",
                    "body": body,
                    "source_ids": [source.source_number for source in bundle.sources],
                },
                {
                    "title": "证据关系图",
                    "body": "已保留本次检索生成的证据节点与关系，供人工核验。",
                    "source_ids": [],
                },
            ),
            final_answer_markdown=(
                "模型生成结果未通过可追溯性校验。请依据上方证据清单和关系图进行人工判断。"
            ),
            model_used=False,
            model_name=str(getattr(self.chat_client, "model", "configured-model")),
            model_error=error_code,
            evidence_inventory=tuple(source.as_dict() for source in bundle.sources),
            graph=dict(bundle.graph),
        )


def _basis_is_grounded(basis: str, title: str, abstract: str) -> bool:
    normalized_source = " ".join(f"{title} {abstract}".casefold().split())
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", basis.casefold())
    meaningful = [token for token in tokens if len(token) >= 3]
    return bool(meaningful) and any(token in normalized_source for token in meaningful)


def _string_tuple(value: object, *, allow_empty: bool = False) -> tuple[str, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return () if allow_empty and value in (None, ()) else None
    if any(not isinstance(item, str) or not item.strip() for item in value):
        return None
    result = tuple(item.strip() for item in value)
    if not result and not allow_empty:
        return None
    return result


def _optional_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _optional_scalar(value: object) -> str:
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, (str, int, float)):
        return str(value).strip()
    return ""


def _citation_ids(text: str) -> set[int]:
    return {int(value) for value in _CITATION_PATTERN.findall(text)}


def _validate_statistics(
    text: str,
    cited_ids: set[int],
    sources: dict[int, EvidenceSource],
) -> None:
    without_citations = _CITATION_PATTERN.sub("", text)
    values = _statistical_values(without_citations)
    if not values:
        return
    cited_text = " ".join(
        snippet
        for source_number in cited_ids
        for snippet in sources[source_number].snippets
    )
    grounded_values = {match.group(1) for match in _NUMBER_PATTERN.finditer(cited_text)}
    if not values.issubset(grounded_values):
        raise _ungrounded()


def _statistical_values(text: str) -> set[str]:
    unit_spans = [match.span() for match in _NUMBER_UNIT_PATTERN.finditer(text)]
    values: set[str] = set()
    for match in _NUMBER_PATTERN.finditer(text):
        raw = match.group(1)
        numeric = float(raw)
        if raw.isdigit() and 1900 <= int(raw) <= 2100:
            continue
        in_unit = any(start <= match.start() < end for start, end in unit_spans)
        nearby = text[max(0, match.start() - 8) : min(len(text), match.end() + 4)]
        effect_context = bool(
            re.search(r"(?:rr|or|hr|ci|p\s*[<=>]|%|率|例)", nearby, re.IGNORECASE)
        )
        if in_unit or effect_context or "." in raw or numeric >= 20:
            values.add(raw)
    return values


def _ungrounded() -> KnowledgeBaseError:
    return KnowledgeBaseError(
        "ungrounded_model_response", "Model response is not grounded in retrieved evidence"
    )
