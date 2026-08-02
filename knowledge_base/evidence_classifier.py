"""Rule-first evidence classification with auditable quality signals."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Iterable


@dataclass(frozen=True)
class EvidenceAssessment:
    evidence_type: str
    confidence: float
    quality: str
    basis: str
    quality_signals: tuple[str, ...]

    @property
    def needs_llm_refinement(self) -> bool:
        return self.evidence_type == "unknown" or self.confidence < 0.7

    def as_dict(self) -> dict[str, object]:
        return {
            "evidence_type": self.evidence_type,
            "confidence": self.confidence,
            "quality": self.quality,
            "basis": self.basis,
            "quality_signals": list(self.quality_signals),
            "needs_llm_refinement": self.needs_llm_refinement,
        }


_TYPE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "guideline",
        (
            r"\bclinical practice guideline\b",
            r"\bguideline(?:s)?\b",
            r"\bexpert consensus\b",
            r"\bconsensus statement\b",
            r"指南",
            r"专家共识",
            r"诊疗共识",
        ),
    ),
    (
        "systematic_review",
        (
            r"\bsystematic review\b",
            r"\bmeta[ -]?analysis\b",
            r"系统综述",
            r"系统评价",
            r"荟萃分析",
            r"meta分析",
        ),
    ),
    (
        "randomized_controlled_trial",
        (
            r"\brandomi[sz]ed controlled trial\b",
            r"\brandomi[sz]ed trial\b",
            r"\brandomised clinical trial\b",
            r"随机对照试验",
            r"随机临床试验",
        ),
    ),
    (
        "observational_study",
        (
            r"\bcohort stud(?:y|ies)\b",
            r"\bcase[ -]?control stud(?:y|ies)\b",
            r"\bcross[ -]?sectional stud(?:y|ies)\b",
            r"\bobservational stud(?:y|ies)\b",
            r"\bretrospective (?:study|analysis|cohort)\b",
            r"\bprospective cohort\b",
            r"队列研究",
            r"病例对照研究",
            r"横断面研究",
            r"观察性研究",
            r"回顾性研究",
            r"回顾性分析",
        ),
    ),
    (
        "case_report",
        (
            r"\bcase report\b",
            r"\bcase series\b",
            r"病例报告",
            r"个案报告",
            r"病例系列",
        ),
    ),
    (
        "narrative_review",
        (
            r"\bnarrative review\b",
            r"\bliterature review\b",
            r"\breview of (?:current|available|the)\b",
            r"叙述性综述",
            r"文献综述",
        ),
    ),
)

_ABSTRACT_RCT_DESIGN_PATTERNS = (
    r"\bwe (?:conducted|performed|undertook) (?:an? )?(?:multicenter |multicentre )?randomi[sz]ed",
    r"\bparticipants? (?:were )?random(?:ly|ized|ised) assign",
    r"\bpatients? (?:were )?random(?:ly|ized|ised) assign",
    r"\bmulticent(?:er|re)[ -]randomi[sz]ed controlled trial\b",
    r"采用随机对照(?:试验|研究)",
    r"随机分配(?:至|到|为)",
    r"随机(?:数字表|硬币投掷|抽签)(?:法)?.{0,12}分(?:为|组)",
)

_SECONDARY_REVIEW_TITLE_PATTERNS = (
    r"研究进展",
    r"进展综述",
    r"指南.{0,20}(?:解读|述评)",
    r"(?:重点|专家)解读",
)


def classify_evidence(
    title: str,
    abstract: str,
    publication_types: Iterable[str] | str = (),
) -> EvidenceAssessment:
    normalized_title = _normalize(title)
    normalized_abstract = _normalize(abstract)
    if isinstance(publication_types, str):
        publication_types = (publication_types,)
    normalized_publication_types = tuple(
        _normalize(value) for value in publication_types if str(value).strip()
    )

    evidence_type = "unknown"
    confidence = 0.0
    basis = "no_explicit_evidence_type_signal"

    publication_text = " | ".join(normalized_publication_types)
    publication_match = _match_type(publication_text)
    if publication_match is not None:
        evidence_type, matched_pattern = publication_match
        confidence = 1.0
        basis = f"publication_type:{matched_pattern}"
    else:
        secondary_review_match = _match_secondary_review_title(normalized_title)
        title_match = secondary_review_match or _match_type(normalized_title)
        if title_match is not None:
            evidence_type, matched_pattern = title_match
            confidence = 0.9
            basis = f"title:{matched_pattern}"
        else:
            abstract_match = _match_abstract_type(normalized_abstract)
            if abstract_match is not None:
                evidence_type, matched_pattern = abstract_match
                confidence = 0.7
                basis = f"abstract:{matched_pattern}"

    quality_signals = _quality_signals(
        " ".join(
            value
            for value in (normalized_title, normalized_abstract, publication_text)
            if value
        )
    )
    quality = _quality_from_signals(quality_signals)
    return EvidenceAssessment(
        evidence_type=evidence_type,
        confidence=confidence,
        quality=quality,
        basis=basis,
        quality_signals=quality_signals,
    )


def _match_type(text: str) -> tuple[str, str] | None:
    if not text:
        return None
    for evidence_type, patterns in _TYPE_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return evidence_type, pattern
    return None


def _match_secondary_review_title(text: str) -> tuple[str, str] | None:
    for pattern in _SECONDARY_REVIEW_TITLE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return "narrative_review", pattern
    return None


def _match_abstract_type(text: str) -> tuple[str, str] | None:
    if not text:
        return None
    for evidence_type, patterns in _TYPE_PATTERNS:
        if evidence_type == "randomized_controlled_trial":
            for pattern in _ABSTRACT_RCT_DESIGN_PATTERNS:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    return evidence_type, pattern
            continue
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return evidence_type, pattern
    return None


def _quality_signals(text: str) -> tuple[str, ...]:
    signals: list[str] = []
    randomized = bool(re.search(r"randomi[sz]|随机", text, flags=re.IGNORECASE))
    multicenter = bool(
        re.search(r"multicent(?:er|re)|多中心", text, flags=re.IGNORECASE)
    )
    if randomized and multicenter:
        signals.append("multicenter_randomization")
    if re.search(
        r"\b(?:prisma|consort|strobe|grade)\b|报告规范|证据分级",
        text,
        flags=re.IGNORECASE,
    ):
        signals.append("reporting_standard")
    if re.search(
        r"\b(?:n\s*=\s*)?\d{2,}\s+(?:patients?|participants?|children|cases)\b"
        r"|\b(?:included|enrolled|recruited)\s+\d{2,}\b"
        r"|纳入\s*\d{2,}\s*例",
        text,
        flags=re.IGNORECASE,
    ):
        signals.append("sample_size")
    if re.search(
        r"\badjusted (?:analysis|model|odds|hazard|risk)\b|多因素(?:分析|回归)|校正后",
        text,
        flags=re.IGNORECASE,
    ):
        signals.append("adjusted_analysis")
    if re.search(
        r"evidence (?:search|cutoff|through|up to).{0,20}\b(?:19|20)\d{2}\b"
        r"|证据检索.{0,12}(?:截至|截止).{0,8}(?:19|20)\d{2}",
        text,
        flags=re.IGNORECASE,
    ):
        signals.append("guideline_evidence_cutoff")
    return tuple(signals)


def _quality_from_signals(signals: tuple[str, ...]) -> str:
    if "multicenter_randomization" in signals:
        return "high"
    if signals:
        return "moderate"
    return "unknown"


def _normalize(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())
