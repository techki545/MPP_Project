"""Source-grounded claim extraction and two-part evidence report generation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any

from .errors import KnowledgeBaseError
from .evidence_classifier import EvidenceAssessment
from .graph_builder import CLINICAL_ASPECTS, EVIDENCE_ROLES, EvidenceClaim
from .report_structure import (
    EXPECTED_STAGE_KEYS,
    FIRST_DEMO_STAGES,
    compose_deterministic_report,
)


_EVIDENCE_TYPES = {
    "guideline",
    "systematic_review",
    "randomized_controlled_trial",
    "observational_study",
    "narrative_review",
    "case_report",
    "unknown",
}
_EVIDENCE_ORDER = (
    "guideline",
    "systematic_review",
    "randomized_controlled_trial",
    "observational_study",
    "narrative_review",
    "case_report",
    "unknown",
)
_EVIDENCE_LABELS = {
    "guideline": "指南",
    "systematic_review": "系统综述/Meta 分析",
    "randomized_controlled_trial": "随机对照试验（RCT）",
    "observational_study": "观察性研究",
    "narrative_review": "叙述性综述",
    "case_report": "病例报告",
    "unknown": "未分类证据",
}
_DIRECTIONS = {"supports", "opposes", "uncertain"}
_MODEL_SOURCE_LIMIT = 8
_MODEL_SNIPPET_CHAR_LIMIT = 1600
_MODEL_ABSTRACT_CHAR_LIMIT = 1200
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_CITATION_GROUP_PATTERN = re.compile(r"(?:\s*\[\d+\])+")
_NUMBER_PATTERN = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])")
_QUANTITY_PATTERN = re.compile(
    r"(?<![\d.])(?P<number>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>mg\s*/\s*kg\s*/\s*(?:d|day|days)|"
    r"mg\s*/\s*kg|mg|g|kg|ml|mmhg|%|"
    r"children|child|patients?|participants?|cases?|"
    r"days?|weeks?|months?|years?|hours?|h|"
    r"名(?:儿童|患儿)?|例|天|日|周|月|年)",
    flags=re.IGNORECASE,
)
_GROUNDING_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*|[\u4e00-\u9fff]{2,}")
_GROUNDING_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "was",
    "were",
    "are",
    "is",
    "of",
    "in",
    "to",
    "a",
    "an",
}
_REPORT_HEADING_LABELS = frozenset(
    {
        "evidence answer",
        "final answer",
        "recommendation",
        "conclusion",
        "循证回答",
        "综合回答",
        "推荐意见",
        "结论",
        "证据链",
        "时间更新",
        "安全性与适用边界",
        "证据缺口",
    }
)
_MODEL_REPORT_PROMPT = """
You are a clinical evidence synthesis engine. Return one JSON object with exactly
analysis_steps and final_answer_markdown. analysis_steps must contain exactly six
items in this stage_key order: inventory, guidelines, systematic_reviews,
randomized_trials, lower_level_evidence, synthesis. Use the supplied Chinese title
for each stage. Explicitly state when an evidence level is absent. Follow the
evidence pyramid, then use chronology to decide whether newer lower-level evidence
updates, confirms, supplements, conflicts with, or cautions higher-level evidence.

The final answer must start with "## 综合回答", immediately give a conclusion, and
then contain these headings in order: "### 证据链", "### 时间更新",
"### 安全性与适用边界", and "### 证据缺口". Cite only the numbered sources
supplied as [N]. Never invent a study, number, dose, effect size, confidence
interval, source, or citation. Any quantitative value must appear in a cited
source snippet. Put citations immediately after every substantive sentence and
reuse a validated graph claim statement verbatim for each clinical assertion.
Do not translate or paraphrase those claim statements. This is evidence support,
not a substitute for an individual clinician's judgment.
""".strip()
_CLAIM_EXTRACTION_PROMPT = """
Extract only source-grounded clinical claims from the numbered evidence snippets.
Return JSON with a claims array containing at most one claim per supplied source.
Every claim must include source_number, source_chunk_ids, source_quote_id,
clinical_aspect, direction, evidence_role, population, intervention, comparator,
design, sample_size, dose, outcome, effect_measures, safety_signal, limitations,
and statement. clinical_aspect must be one of overall, effectiveness, dose, timing,
safety, diagnosis, prognosis, applicability, or other. direction must be supports,
opposes, or uncertain relative to the user's proposition. evidence_role must be
core, supplement, or boundary. Select exactly one supplied quote_options
item and copy its quote_id into source_quote_id and its source_chunk_id into
source_chunk_ids. Do not return or rewrite the quote text; the server resolves it by
ID. effect_measures and limitations must always be JSON arrays of strings; use []
when absent. For population, intervention, comparator, design, sample_size, dose,
outcome, and follow_up, copy an exact phrase from the selected quote or use an empty
string when it is not reported. Do not infer missing numbers or details.
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
    claims: tuple[EvidenceClaim, ...] = ()

    def __post_init__(self) -> None:
        numbers = [item.source_number for item in self.sources]
        if len(set(numbers)) != len(numbers):
            raise ValueError("Evidence source numbers must be unique")

    def as_dict(self) -> dict[str, Any]:
        return {
            "sources": [item.as_dict() for item in self.sources],
            "graph": dict(self.graph),
            "claims": [item.as_dict() for item in self.claims],
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


def extractive_fallback_claims(
    sources: Sequence[EvidenceSource], *, limit: int = 8
) -> ValidatedClaims:
    """Create traceable, direction-neutral claims without semantic inference."""
    claims: list[EvidenceClaim] = []
    audit: list[dict[str, Any]] = []
    for source in sources[: max(0, limit)]:
        quote = next((text.strip() for text in source.snippets if text.strip()), "")
        if not quote or not source.chunk_ids:
            audit.append(
                {
                    "source_number": source.source_number,
                    "reason": "extractive_fallback_missing_grounding",
                }
            )
            continue
        claims.append(
            EvidenceClaim(
                claim_id=f"fallback-{source.source_number}",
                document_id=source.document_id,
                evidence_type=source.evidence_type,
                year=source.year,
                population="",
                intervention="",
                comparator="",
                outcome="相关证据发现",
                direction="uncertain",
                statement=_grounded_excerpt(quote),
                source_chunk_ids=source.chunk_ids,
                design=source.evidence_type,
                limitations=("模型不可用时的提取式回退，未判断效应方向。",),
                source_quote=quote,
            )
        )
    return ValidatedClaims(tuple(claims), tuple(audit))


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
                "sources": _sources_for_model(bundle.sources),
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
        quote_id = _optional_text(raw_claim.get("source_quote_id"))
        quote_option = {
            option["quote_id"]: option for option in _quote_options(source)
        }.get(quote_id)
        if quote_id:
            if quote_option is None:
                return None, "unknown_source_quote"
            chunk_ids = (quote_option["source_chunk_id"],)
            source_quote = quote_option["text"]
            statement = source_quote
        else:
            chunk_ids = _string_tuple(raw_claim.get("source_chunk_ids"))
            if not chunk_ids or not set(chunk_ids).issubset(set(source.chunk_ids)):
                return None, "unknown_source_chunk"
            source_quote = " ".join(str(raw_claim.get("source_quote", "")).split())
            statement = _optional_text(raw_claim.get("statement"))
        if not statement or not source_quote:
            return None, "missing_required_text"
        source_text = ". ".join(
            " ".join(value.split())
            for value in (source.title, source.abstract, *source.snippets)
            if value.strip()
        )
        if source_quote.casefold() not in source_text.casefold():
            return None, "source_quote_not_found"
        if quote_id:
            cutoff = raw_claim.get("evidence_cutoff_year")
            if isinstance(cutoff, bool) or not isinstance(cutoff, (int, type(None))):
                cutoff = None
            direction = _closed_label(raw_claim.get("direction"), _DIRECTIONS, "uncertain")
            clinical_aspect = _closed_label(
                raw_claim.get("clinical_aspect"), CLINICAL_ASPECTS, "other"
            )
            evidence_role = _closed_label(
                raw_claim.get("evidence_role"), EVIDENCE_ROLES, "supplement"
            )
            grounded_fields = {
                field: _grounded_optional_text(raw_claim.get(field), source_quote)
                for field in (
                    "population",
                    "intervention",
                    "comparator",
                    "design",
                    "sample_size",
                    "dose",
                    "outcome",
                    "follow_up",
                )
            }
            effect_measures = _grounded_string_tuple(
                raw_claim.get("effect_measures"), source_quote
            )
            limitations = _grounded_string_tuple(
                raw_claim.get("limitations"), source_quote
            )
            return (
                EvidenceClaim(
                    claim_id=f"claim-{source_number}-{index}",
                    document_id=source.document_id,
                    evidence_type=source.evidence_type,
                    year=source.year,
                    evidence_cutoff_year=cutoff,
                    population=grounded_fields["population"],
                    intervention=grounded_fields["intervention"],
                    comparator=grounded_fields["comparator"],
                    outcome=grounded_fields["outcome"],
                    direction=direction,
                    safety_signal=_quoted_safety_signal(source_quote) is True,
                    design=grounded_fields["design"],
                    dose=grounded_fields["dose"],
                    sample_size=grounded_fields["sample_size"],
                    effect_measures=effect_measures,
                    limitations=limitations,
                    statement=source_quote,
                    source_quote=source_quote,
                    source_chunk_ids=chunk_ids,
                    follow_up=grounded_fields["follow_up"],
                    clinical_aspect=clinical_aspect,
                    evidence_role=evidence_role,
                ),
                "",
            )

        direction = raw_claim.get("direction")
        if direction not in _DIRECTIONS:
            return None, "invalid_direction"
        effect_measures = _string_tuple(raw_claim.get("effect_measures"), allow_empty=True)
        limitations = _string_tuple(raw_claim.get("limitations"), allow_empty=True)
        if effect_measures is None or limitations is None:
            return None, "invalid_list_field"
        quoted_safety_signal = _quoted_safety_signal(source_quote)
        safety_signal = raw_claim.get("safety_signal")
        if not isinstance(safety_signal, bool):
            return None, "invalid_safety_signal"
        grounded_fields: list[object] = [
            raw_claim.get("population"),
            raw_claim.get("intervention"),
            raw_claim.get("comparator"),
            raw_claim.get("design"),
            raw_claim.get("sample_size"),
            raw_claim.get("dose"),
            raw_claim.get("outcome"),
            raw_claim.get("follow_up"),
            *effect_measures,
            *limitations,
        ]
        if any(
            not _claim_field_is_grounded(value, source_text)
            for value in grounded_fields
            if _optional_scalar(value)
        ):
            return None, "claim_not_grounded"
        if not _claim_statement_is_grounded(statement, source_quote):
            return None, "claim_not_grounded"
        if safety_signal != (quoted_safety_signal is True):
            return None, "claim_not_grounded"

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
                population=_optional_text(raw_claim.get("population")),
                intervention=_optional_text(raw_claim.get("intervention")),
                comparator=_optional_text(raw_claim.get("comparator")),
                outcome=_optional_text(raw_claim.get("outcome")),
                direction=str(direction),
                safety_signal=safety_signal,
                design=_optional_text(raw_claim.get("design")),
                dose=_optional_text(raw_claim.get("dose")),
                sample_size=_optional_scalar(raw_claim.get("sample_size")),
                effect_measures=effect_measures,
                limitations=limitations,
                statement=statement,
                source_quote=source_quote,
                source_chunk_ids=chunk_ids,
                follow_up=_optional_text(raw_claim.get("follow_up")),
                clinical_aspect=_closed_label(
                    raw_claim.get("clinical_aspect"), CLINICAL_ASPECTS, "other"
                ),
                evidence_role=_closed_label(
                    raw_claim.get("evidence_role"), EVIDENCE_ROLES, "supplement"
                ),
            ),
            "",
        )


class GroundedReporter:
    def __init__(self, chat_client: Any) -> None:
        self.chat_client = chat_client

    def generate(self, question: str, bundle: EvidenceBundle) -> GeneratedReport:
        response = self._request_report(question, bundle)
        analysis_steps, final_answer = self._validate_response(response, bundle)
        return self._build_report(analysis_steps, final_answer, bundle)

    def generate_grounded_claim_report(
        self,
        question: str,
        bundle: EvidenceBundle,
        *,
        model_used: bool,
        model_error: str | None,
    ) -> GeneratedReport:
        """Render validated model claims without a second free-text model call."""
        structured = compose_deterministic_report(question, bundle)
        return GeneratedReport(
            analysis_steps=structured.analysis_steps,
            final_answer_markdown=structured.final_answer_markdown,
            model_used=model_used,
            model_name=str(
                getattr(self.chat_client, "model", "configured-model")
            ),
            model_error=model_error,
            evidence_inventory=tuple(
                source.as_dict() for source in bundle.sources
            ),
            graph=dict(bundle.graph),
        )

    def _request_report(self, question: str, bundle: EvidenceBundle) -> object:
        return self.chat_client.complete_json(
            _MODEL_REPORT_PROMPT,
            {
                "question": question,
                "evidence_bundle": _bundle_for_model(bundle),
                "required_output": {
                    "analysis_steps": [
                        {
                            "stage_key": stage_key,
                            "title": title,
                            "body": "source-grounded string",
                            "source_ids": [1],
                        }
                        for stage_key, title in FIRST_DEMO_STAGES
                    ],
                    "final_answer_markdown": (
                        "## 综合回答\n\n**结论：...**[N]\n\n"
                        "### 证据链\n...\n\n### 时间更新\n...\n\n"
                        "### 安全性与适用边界\n...\n\n### 证据缺口\n..."
                    ),
                },
            },
        )

    def _build_report(
        self,
        analysis_steps: tuple[dict[str, Any], ...],
        final_answer: str,
        bundle: EvidenceBundle,
    ) -> GeneratedReport:
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
            response = self._request_report(question, bundle)
            try:
                analysis_steps, final_answer = self._validate_response(response, bundle)
            except KnowledgeBaseError as error:
                if error.code != "ungrounded_model_response":
                    raise
                sources = {item.source_number: item for item in bundle.sources}
                try:
                    analysis_steps, raw_final = self._validate_response_envelope(
                        response, bundle
                    )
                except KnowledgeBaseError:
                    if not isinstance(response, Mapping):
                        raise
                    raw_final = response.get("final_answer_markdown")
                    if not isinstance(raw_final, str) or not raw_final.strip():
                        raise
                    analysis_steps = self._deterministic_fallback(
                        question, bundle, "ungrounded_model_response"
                    ).analysis_steps
                final_answer = _canonicalize_report_claims(
                    raw_final, bundle, sources
                )
            return self._build_report(analysis_steps, final_answer, bundle)
        except KnowledgeBaseError as error:
            code = error.code
        except Exception:
            code = "model_generation_failed"
        return self._deterministic_fallback(question, bundle, code)

    @staticmethod
    def _validate_response(
        response: object, bundle: EvidenceBundle
    ) -> tuple[tuple[dict[str, Any], ...], str]:
        validated_steps, final_answer = GroundedReporter._validate_response_envelope(
            response, bundle
        )
        sources = {item.source_number: item for item in bundle.sources}
        _validate_report_claims(final_answer, bundle, sources)
        return validated_steps, final_answer

    @staticmethod
    def _validate_response_envelope(
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
        required_headings = (
            "## 综合回答",
            "### 证据链",
            "### 时间更新",
            "### 安全性与适用边界",
            "### 证据缺口",
        )
        heading_positions = [final_answer.find(heading) for heading in required_headings]
        if (
            not final_answer.startswith(required_headings[0])
            or any(position < 0 for position in heading_positions)
            or heading_positions != sorted(heading_positions)
        ):
            raise _ungrounded()

        sources = {item.source_number: item for item in bundle.sources}
        known_numbers = set(sources)
        validated_steps: list[dict[str, Any]] = []
        if len(raw_steps) != len(FIRST_DEMO_STAGES):
            raise _ungrounded()
        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, Mapping):
                raise _ungrounded()
            stage_key = raw_step.get("stage_key")
            title = raw_step.get("title")
            body = raw_step.get("body")
            source_ids = raw_step.get("source_ids")
            if (
                stage_key != EXPECTED_STAGE_KEYS[index]
                or not isinstance(title, str)
                or not title.strip()
                or title.strip() != FIRST_DEMO_STAGES[index][1]
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
                    "stage_key": str(stage_key),
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
        self, question: str, bundle: EvidenceBundle, error_code: str
    ) -> GeneratedReport:
        structured = compose_deterministic_report(question, bundle)
        return GeneratedReport(
            analysis_steps=structured.analysis_steps,
            final_answer_markdown=structured.final_answer_markdown,
            model_used=False,
            model_name=str(getattr(self.chat_client, "model", "configured-model")),
            model_error=error_code,
            evidence_inventory=tuple(source.as_dict() for source in bundle.sources),
            graph=dict(bundle.graph),
        )


def _grounded_excerpt(text: str, *, limit: int = 320) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    for index, character in enumerate(normalized[:limit], start=1):
        if index >= 80 and character in "。！？.!?":
            return normalized[:index]
    return normalized[:limit].rstrip()


def _sources_for_model(sources: Sequence[EvidenceSource]) -> list[dict[str, Any]]:
    return [_source_for_model(source) for source in sources[:_MODEL_SOURCE_LIMIT]]


def _source_for_model(source: EvidenceSource) -> dict[str, Any]:
    payload = source.as_dict()
    payload["quote_options"] = _quote_options(source)
    payload["snippets"] = []
    payload["abstract"] = ""
    return payload


def _bundle_for_model(bundle: EvidenceBundle) -> dict[str, Any]:
    model_sources = tuple(bundle.sources[:_MODEL_SOURCE_LIMIT])
    allowed_documents = {source.document_id for source in model_sources}
    return {
        "sources": _sources_for_model(model_sources),
        "graph": dict(bundle.graph),
        "claims": [
            claim.as_dict()
            for claim in bundle.claims
            if claim.document_id in allowed_documents
        ],
    }


def _quote_options(source: EvidenceSource) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for chunk_id, snippet in zip(source.chunk_ids, source.snippets):
        for clause in _grounding_clauses(snippet):
            text = " ".join(clause.split())
            if len(text) < 16:
                continue
            options.append(
                {
                    "quote_id": f"quote-{source.source_number}-{len(options) + 1}",
                    "source_chunk_id": chunk_id,
                    "text": text[:360].rstrip(),
                }
            )
            if len(options) == 4:
                return options
    return options


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


def _closed_label(value: object, allowed: frozenset[str] | set[str], default: str) -> str:
    normalized = value.strip().casefold() if isinstance(value, str) else ""
    return normalized if normalized in allowed else default


def _optional_scalar(value: object) -> str:
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, (str, int, float)):
        return str(value).strip()
    return ""


def _citation_ids(text: str) -> set[int]:
    return {int(value) for value in _CITATION_PATTERN.findall(text)}


def _validate_report_claims(
    text: str,
    bundle: EvidenceBundle,
    sources: dict[int, EvidenceSource],
) -> None:
    claims_by_document = _graph_claims_by_document(bundle)

    saw_claim = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or _is_allowed_report_heading(line):
            continue
        cursor = 0
        for match in _CITATION_GROUP_PATTERN.finditer(line):
            segment = _clean_report_segment(line[cursor : match.start()])
            cited_ids = _citation_ids(match.group(0))
            if not segment or not cited_ids:
                raise _ungrounded()
            candidate_claims = [
                claim
                for source_number in cited_ids
                for claim in claims_by_document.get(
                    sources[source_number].document_id, []
                )
            ]
            if not any(
                _report_segment_is_grounded(segment, claim)
                for claim in candidate_claims
            ):
                raise _ungrounded()
            saw_claim = True
            cursor = match.end()
        if _clean_report_segment(line[cursor:]):
            raise _ungrounded()
    if not saw_claim:
        raise _ungrounded()


def _canonicalize_report_claims(
    text: str,
    bundle: EvidenceBundle,
    sources: dict[int, EvidenceSource],
) -> str:
    """Replace unsupported paraphrases with their cited, validated graph claims."""
    claims_by_document = _graph_claims_by_document(bundle)
    cited_ids = _citation_ids(text)
    if not cited_ids or not cited_ids.issubset(sources):
        raise _ungrounded()
    output_lines: list[str] = []
    saw_claim = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or _is_allowed_report_heading(line):
            output_lines.append(raw_line)
            continue
        cursor = 0
        line_parts: list[str] = []
        matches = list(_CITATION_GROUP_PATTERN.finditer(line))
        if not matches:
            raise _ungrounded()
        for match in matches:
            raw_segment = line[cursor : match.start()]
            segment = _clean_report_segment(raw_segment)
            segment_ids = _citation_ids(match.group(0))
            if not segment or not segment_ids or not segment_ids.issubset(sources):
                raise _ungrounded()
            candidate_claims = [
                (source_number, claim)
                for source_number in sorted(segment_ids)
                for claim in claims_by_document.get(
                    sources[source_number].document_id, []
                )
            ]
            if any(
                _report_segment_is_grounded(segment, claim)
                for _, claim in candidate_claims
            ):
                line_parts.append(raw_segment + match.group(0))
            else:
                replacements = list(
                    dict.fromkeys(
                        f"{claim['statement']}[{source_number}]"
                        for source_number, claim in candidate_claims
                    )
                )
                if not replacements:
                    raise _ungrounded()
                line_parts.append(" ".join(replacements))
            saw_claim = True
            cursor = match.end()
        if _clean_report_segment(line[cursor:]):
            raise _ungrounded()
        output_lines.append("".join(line_parts))
    if not saw_claim:
        raise _ungrounded()
    return "\n".join(output_lines)


def _graph_claims_by_document(
    bundle: EvidenceBundle,
) -> dict[str, list[dict[str, str]]]:
    claims_by_document: dict[str, list[dict[str, str]]] = {}
    for claim in bundle.claims:
        if claim.statement and claim.source_quote:
            claims_by_document.setdefault(claim.document_id, []).append(
                {
                    "statement": claim.statement,
                    "source_quote": claim.source_quote,
                }
            )
    if claims_by_document:
        return claims_by_document
    for node in bundle.graph.get("nodes", []):
        if not isinstance(node, Mapping) or node.get("node_type") != "claim":
            continue
        payload = node.get("payload")
        if not isinstance(payload, Mapping):
            continue
        document_id = _optional_text(payload.get("document_id"))
        statement = _optional_text(payload.get("statement"))
        source_quote = _optional_text(payload.get("source_quote"))
        if document_id and statement and source_quote:
            claims_by_document.setdefault(document_id, []).append(
                {"statement": statement, "source_quote": source_quote}
            )
    return claims_by_document


def _clean_report_segment(text: str) -> str:
    cleaned = re.sub(r"^[\s>*#`_\-+\d.)]+", "", text)
    cleaned = re.sub(r"[*_`]+", "", cleaned)
    return cleaned.strip(" \t\r\n:;,.")


def _is_allowed_report_heading(line: str) -> bool:
    if _citation_ids(line):
        return False
    cleaned = re.sub(r"^#+\s*", "", line.strip())
    cleaned = cleaned.strip("*_` \t\r\n:：").casefold()
    return cleaned in _REPORT_HEADING_LABELS


def _report_segment_is_grounded(
    segment: str, claim: Mapping[str, Any]
) -> bool:
    statement = _optional_text(claim.get("statement"))
    source_quote = _optional_text(claim.get("source_quote"))
    if _unicase(segment) == _unicase(statement).strip(" .;:"):
        return True
    return _claim_statement_is_grounded(segment, source_quote)


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
    quantities = _quantity_values(without_citations)
    grounded_quantities = _quantity_values(cited_text)
    if not quantities.issubset(grounded_quantities):
        raise _ungrounded()
    grounded_values = {match.group(1) for match in _NUMBER_PATTERN.finditer(cited_text)}
    if not values.issubset(grounded_values):
        raise _ungrounded()


def _statistical_values(text: str) -> set[str]:
    unit_spans = [match.span() for match in _QUANTITY_PATTERN.finditer(text)]
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


def _claim_text_is_grounded(value: object, source_text: str) -> bool:
    text = _optional_scalar(value)
    if not text:
        return True
    if not _quantity_values(text).issubset(_quantity_values(source_text)):
        return False
    values = _statistical_values(text)
    grounded_values = {
        match.group(1) for match in _NUMBER_PATTERN.finditer(source_text)
    }
    if not values.issubset(grounded_values):
        return False
    source_normalized = _unicase(source_text)
    tokens = {
        token
        for token in _GROUNDING_TOKEN_PATTERN.findall(_unicase(text))
        if token not in _GROUNDING_STOPWORDS and not token.isdigit()
    }
    if not tokens:
        return bool(values or _quantity_values(text))
    return all(token in source_normalized for token in tokens)


def _grounded_optional_text(value: object, source_quote: str) -> str:
    text = _optional_text(value)
    if text and _claim_field_is_grounded(text, source_quote):
        return text
    return ""


def _grounded_string_tuple(value: object, source_quote: str) -> tuple[str, ...]:
    items = _string_tuple(value, allow_empty=True)
    if items is None:
        return ()
    return tuple(
        item for item in items if _claim_field_is_grounded(item, source_quote)
    )


def _claim_field_is_grounded(value: object, source_text: str) -> bool:
    text = _optional_scalar(value)
    if not text:
        return True
    text_has_negation = _has_negation(text)
    return any(
        _claim_text_is_grounded(text, sentence)
        and text_has_negation == _has_negation(sentence)
        for sentence in _grounding_clauses(source_text)
        if sentence.strip()
    )


def _claim_statement_is_grounded(statement: str, source_quote: str) -> bool:
    statement_has_negation = _has_negation(statement)
    for sentence in _grounding_clauses(source_quote):
        if not sentence.strip() or not _claim_text_is_grounded(statement, sentence):
            continue
        if statement_has_negation == _has_negation(sentence):
            return True
    return False


def _grounding_clauses(text: str) -> tuple[str, ...]:
    return tuple(
        clause.strip()
        for clause in re.split(
            r"[.!?;。！？；]+|\b(?:but|however|whereas|although|while)\b|(?:但是|但|然而|而|却|相比之下)",
            text,
            flags=re.IGNORECASE,
        )
        if clause.strip()
    )


def _has_negation(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:no|not|without|neither|nor|non-significant)\b|无|未|不|没有|未见",
            text,
            flags=re.IGNORECASE,
        )
    )


def _quoted_safety_signal(text: str) -> bool | None:
    signals = [
        not _has_negation(clause)
        for clause in _grounding_clauses(text)
        if re.search(
            r"\b(?:safety|adverse|harm|risk|bleeding|infection)\b|安全|不良|风险|出血|感染",
            clause,
            flags=re.IGNORECASE,
        )
    ]
    return any(signals) if signals else None


def _quantity_values(text: str) -> set[tuple[str, str]]:
    return {
        (_normalize_number(match.group("number")), _normalize_unit(match.group("unit")))
        for match in _QUANTITY_PATTERN.finditer(text)
    }


def _normalize_number(value: str) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else format(numeric, "g")


def _normalize_unit(value: str) -> str:
    unit = re.sub(r"\s+", "", value.casefold())
    if unit in {"child", "children", "patient", "patients", "participant", "participants", "case", "cases", "例", "名", "名儿童", "名患儿"}:
        return "person"
    if unit in {"d", "day", "days", "天", "日"}:
        return "day"
    if unit in {"week", "weeks", "周"}:
        return "week"
    if unit in {"month", "months", "月"}:
        return "month"
    if unit in {"year", "years", "年"}:
        return "year"
    if unit in {"h", "hour", "hours"}:
        return "hour"
    if unit in {"mg/kg/d", "mg/kg/day", "mg/kg/days"}:
        return "mg/kg/day"
    return unit


def _unicase(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _ungrounded() -> KnowledgeBaseError:
    return KnowledgeBaseError(
        "ungrounded_model_response", "Model response is not grounded in retrieved evidence"
    )
