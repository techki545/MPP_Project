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
from .text_cleaning import clean_evidence_text, reference_like_text


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
        "综合文献结论",
        "推荐意见",
        "结论",
        "证据链",
        "时间更新",
        "安全性与适用边界",
        "证据缺口",
    }
)
_MODEL_REPORT_PROMPT = """
You are a clinical literature synthesis assistant. Answer the user's actual
question by integrating the conclusions of multiple relevant sources instead of
mechanically restating retrieved snippets. First classify the question's actual
intent and scope (for example: yes/no recommendation, treatment options, dose,
duration, treatment timing, diagnosis, examination, prognosis, safety, cause,
mechanism, or prevention). Then identify consensus, conflicts,
chronological updates, and evidence gaps across studies. Then use clinical
reasoning to explain what that combined literature means for the question. The
literature supplies the factual findings; your role is comparison, synthesis,
and clearly labelled interpretation. Answer the identified intent directly. Do
not substitute an adjacent question merely because retrieved evidence is richer
for it. If a question asks for overall treatment duration while sources report
only component-specific regimens, state that no single overall duration can be
inferred, separate durations by intervention, and identify missing dimensions.
Never equate one drug's course with the entire treatment course.

Return one JSON object with exactly analysis_steps and final_answer_markdown.
analysis_steps must contain exactly six items in this stage_key order: inventory,
guidelines, systematic_reviews, randomized_trials, lower_level_evidence,
synthesis. Use the supplied Chinese title for each stage. These steps are an
auditable evidence workflow, not private chain-of-thought. Explicitly state when
an evidence level is absent. Follow the evidence pyramid, then use chronology to
decide whether newer lower-level evidence updates, confirms, supplements,
conflicts with, or cautions higher-level evidence.

The final answer must start with "## 综合回答", immediately give a useful,
question-specific paragraph beginning with "**综合文献结论：", and then contain
these headings in order:
"### 证据链", "### 时间更新", "### 安全性与适用边界", and "### 证据缺口".
You may paraphrase, synthesize across sources, and add clearly framed clinical
interpretation. Cite numbered sources as [N] when stating what a retrieved source
found or recommended. General clinical reasoning and transparent inferences may
be uncited, but must not be presented as a result of a specific study. Never
invent a study, number, dose, effect size, confidence interval, source, or
citation. Every quantitative value must appear in a cited source snippet. Make
clear where evidence ends and synthesis begins. This is decision support, not a
substitute for an individual clinician's judgment.

Each source includes population_applicability. Treat direct as evidence for the
question's target condition, indirect_rmpp or indirect_smpp as extrapolated
evidence, general_mpp as general background, and unclear as uncertain. Never hide
an extrapolation: state it near the conclusion and reduce certainty when the main
evidence is indirect.

Write one cross-source synthesis, not a list of isolated paper summaries or an
extractive collage. State the dominant conclusion, then name meaningful
exceptions or conflicts and explain whether newer evidence changes older
guidance. Never paste bibliography entries,
journal reference strings, garbled column text, or long verbatim passages. The
only valid citation syntax is [N], where N is a supplied source_number; bracketed
numbers that may have appeared inside an article are not system citations. Keep
the conclusion focused and group the evidence chain by evidence level.

When the evidence bundle contains enough relevant claims, make the report
substantive rather than terse: aim for about 1200-2000 Chinese characters in
final_answer_markdown. This is a depth target, not permission to invent or pad.
Use two or three paragraphs for the conclusion and explain the recommendation,
its rationale, certainty, and decision boundary. In the evidence chain, compare
the strongest two to four sources across levels instead of merely naming them.
Explain whether their populations, interventions, outcomes, and directions are
consistent. Give chronology its own interpretation, not only a year range.
Describe concrete safety concerns, applicable patients, exceptions, and at least
three evidence gaps when supported by the bundle. Each analysis_steps body should
normally contain 2-4 complete, source-grounded sentences. Avoid repetition, but
do not compress a multi-source clinical question into a few bullet fragments.
""".strip()
_CLAIM_EXTRACTION_PROMPT = """
Extract only source-grounded clinical claims from the numbered evidence snippets.
Return JSON with a claims array containing at most one claim per supplied source.
Omit a source when its quote options discuss the same disease but do not directly
address the intervention, comparison, outcome, or clinical focus in the user's
question. Prefer a result, conclusion, or recommendation sentence over background
mechanism, introduction, bibliography, or methods text.
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
string when it is not reported. Do not infer missing numbers or details. Classify
treatment duration, course length, treatment cycles, and stopping or reassessment
time points as timing, not dose.
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
    population_applicability: str = "not_assessed"
    population_applicability_score: float = 0.5
    question_relevance_score: float = 0.5

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
            "population_applicability": self.population_applicability,
            "population_applicability_score": self.population_applicability_score,
            "question_relevance_score": self.question_relevance_score,
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


_QUESTION_FOCUS_CONCEPTS = (
    ("exercise", (r"运动", r"锻炼", r"康复训练", r"体力活动", r"exercise", r"physical\s+activity")),
    ("steroid", (r"糖皮质激素", r"甲泼尼龙", r"甲强龙", r"地塞米松", r"glucocorticoid", r"corticosteroid", r"methylprednisolone")),
    ("azithromycin", (r"阿奇霉素", r"azithromycin")),
    ("macrolide", (r"大环内酯", r"macrolide")),
    ("tetracycline", (r"四环素", r"多西环素", r"米诺环素", r"tetracycline", r"doxycycline", r"minocycline")),
    ("fluoroquinolone", (r"氟喹诺酮", r"左氧氟沙星", r"莫西沙星", r"fluoroquinolone", r"levofloxacin", r"moxifloxacin")),
    ("bronchoscopy", (r"支气管镜", r"肺泡灌洗", r"bronchoscop", r"bronchoalveolar\s+lavage")),
    ("ivig", (r"静脉注射免疫球蛋白", r"丙种球蛋白", r"\bivig\b", r"immunoglobulin")),
    ("thrombosis", (r"血栓", r"肺栓塞", r"thrombo", r"pulmonary\s+embol")),
)


def filter_claims_for_question(
    question: str, validated: ValidatedClaims
) -> ValidatedClaims:
    """Discard claims that do not mention every intervention focus in the question."""

    focus_groups = [
        (name, patterns)
        for name, patterns in _QUESTION_FOCUS_CONCEPTS
        if any(re.search(pattern, question, re.I) for pattern in patterns)
    ]
    if not focus_groups:
        return validated

    kept: list[EvidenceClaim] = []
    audit = list(validated.audit)
    for claim in validated.claims:
        claim_text = " ".join(
            value
            for value in (
                claim.population,
                claim.intervention,
                claim.comparator,
                claim.outcome,
                claim.statement,
                claim.source_quote,
                claim.dose,
            )
            if value
        )
        missing = [
            name
            for name, patterns in focus_groups
            if not any(re.search(pattern, claim_text, re.I) for pattern in patterns)
        ]
        if missing:
            audit.append(
                {
                    "claim_id": claim.claim_id,
                    "document_id": claim.document_id,
                    "reason": "claim_question_focus_mismatch",
                    "missing_focus": missing,
                }
            )
            continue
        kept.append(claim)
    return ValidatedClaims(tuple(kept), tuple(audit))


def extractive_fallback_claims(
    sources: Sequence[EvidenceSource], *, question: str = "", limit: int = 15
) -> ValidatedClaims:
    """Create concise, traceable claims when model extraction is unavailable."""
    claims: list[EvidenceClaim] = []
    audit: list[dict[str, Any]] = []
    for source in sources[: max(0, limit)]:
        quote = _best_fallback_excerpt(
            (*source.snippets, source.abstract),
            question=question,
            title=source.title,
        )
        grounding_ids = ()
        if quote:
            grounding_ids = (
                (source.document_id,)
                if _normalized_text_key(quote) in _normalized_text_key(source.abstract)
                else source.chunk_ids
            )
        if not quote or not grounding_ids:
            audit.append(
                {
                    "source_number": source.source_number,
                    "reason": "extractive_fallback_missing_grounding",
                }
            )
            continue
        # The intervention is often stated in the paper title while the selected
        # result sentence only says "different doses" or "the intervention".
        intervention = _fallback_intervention(f"{source.title} {quote}")
        dose = _fallback_dose(quote)
        clinical_aspect = _fallback_clinical_aspect(quote)
        direction = _fallback_direction(quote)
        safety_signal = bool(
            re.search(r"不良反应|安全性|感染风险|副作用|高血压|高血糖", quote, re.I)
        )
        evidence_role = (
            "boundary"
            if safety_signal
            else "core"
            if source.evidence_type
            in {"guideline", "systematic_review", "randomized_controlled_trial"}
            else "supplement"
        )
        claims.append(
            EvidenceClaim(
                claim_id=f"fallback-{source.source_number}",
                document_id=source.document_id,
                evidence_type=source.evidence_type,
                year=source.year,
                population="",
                intervention=intervention,
                comparator="",
                outcome=_fallback_outcome(quote),
                direction=direction,
                statement=_fallback_statement(quote, intervention, dose),
                source_chunk_ids=grounding_ids,
                design=source.evidence_type,
                dose=dose,
                limitations=("模型不可用时由原文完整句提取，需结合全文复核。",),
                source_quote=quote,
                clinical_aspect=clinical_aspect,
                evidence_role=evidence_role,
                safety_signal=safety_signal,
            )
        )
    return ValidatedClaims(tuple(claims), tuple(audit))


_FALLBACK_MEDICATIONS = (
    ("甲泼尼龙", r"甲泼尼龙|甲强龙|methylprednisolone"),
    ("地塞米松", r"地塞米松|dexamethasone"),
    ("阿奇霉素", r"阿奇霉素|azithromycin"),
    ("红霉素", r"红霉素|erythromycin"),
    ("多西环素", r"多西环素|doxycycline"),
    ("米诺环素", r"米诺环素|minocycline"),
    ("静脉注射免疫球蛋白（IVIG）", r"丙种球蛋白|免疫球蛋白|ivig|gammaglobulin"),
    ("糖皮质激素", r"糖皮质激素|皮质类固醇|glucocorticoid|corticosteroid"),
)
_FALLBACK_DOSE_PATTERN = re.compile(
    r"\d+(?:\.\d+)?(?:\s*[~～\-–至]\s*\d+(?:\.\d+)?)?\s*"
    r"(?:mg|g)\s*/?\s*(?:\(\s*kg\s*[·./]\s*d\s*\)|kg\s*(?:[·./]\s*(?:d|day))?)",
    re.I,
)
_FALLBACK_SENTENCE_SPLIT = re.compile(
    r"(?<=[。！？!?])\s*|(?<=\.)\s+|"
    r"(?<=\.)(?=(?:目的|方法|结果|结论)\s*[:：]?)"
)


def _normalized_text_key(text: str) -> str:
    return re.sub(r"\W+", "", str(text or "").casefold())


def _best_fallback_excerpt(
    snippets: Sequence[str], *, question: str = "", title: str = ""
) -> str:
    candidates: list[tuple[int, int, str]] = []
    descriptive_candidates: list[tuple[int, int, str]] = []
    question_terms = {
        token
        for token in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z]{3,}", question.casefold())
        if token not in {"儿童", "肺炎", "什么", "怎么", "应该"}
    }
    focus_groups = [
        patterns
        for _, patterns in _QUESTION_FOCUS_CONCEPTS
        if any(re.search(pattern, question, re.I) for pattern in patterns)
    ]
    duration_question = bool(
        re.search(
            r"疗程|治疗周期|治疗多久|多长时间|需要多久|持续多久|"
            r"course of treatment|treatment duration|how long",
            question,
            re.I,
        )
    )
    duration_pattern = re.compile(
        r"疗程|周期|连续|连用|持续|间隔|停药|减量|"
        r"(?:\d+|[一二三四五六七八九十]+)\s*(?:天|周|月|d)|"
        r"48\s*[-~～]\s*72\s*h",
        re.I,
    )
    intent_pattern: re.Pattern[str] | None = None
    if re.search(r"诊断|确诊|鉴别|如何识别|检查|检测|diagnos", question, re.I):
        intent_pattern = re.compile(
            r"诊断标准|符合.{0,12}标准|确诊|鉴别|病原学|核酸检测|抗体|"
            r"影像学|敏感度|特异度|临界值|cutoff|diagnos",
            re.I,
        )
    elif re.search(r"不良反应|副作用|安全性|风险是什么|adverse|safety|harm", question, re.I):
        intent_pattern = re.compile(
            r"不良反应|副作用|安全性|风险|禁忌|慎用|过敏|心律失常|"
            r"QT|肝功能|胃肠道|恶心|呕吐|腹泻|adverse|safety",
            re.I,
        )
    elif re.search(r"病因|原因|为什么|如何发展|机制|etiolog|cause|mechanism", question, re.I):
        intent_pattern = re.compile(
            r"危险因素|风险因素|独立因素|相关因素|发病机制|机制|由于|导致|"
            r"与.{0,18}相关|预测|OR\s*=|risk factor|mechanism",
            re.I,
        )
    elif re.search(r"预后|复发|长期结局|危险因素|风险因素|prognos|follow-up", question, re.I):
        intent_pattern = re.compile(
            r"预后|复发|长期结局|危险因素|风险因素|独立因素|相关因素|"
            r"随访|后遗症|支气管扩张|闭塞性细支气管炎|prognos|follow-up",
            re.I,
        )
    for snippet in snippets:
        raw_text = str(snippet or "").replace("\r\n", "\n").replace("\r", "\n")
        # PDF extraction commonly inserts blank lines at every visual line. Join
        # those layout breaks before sentence splitting so evidence stays whole.
        raw_text = re.sub(r"\s*\n+\s*", " ", raw_text)
        for raw_clause in _FALLBACK_SENTENCE_SPLIT.split(raw_text):
            author_list = re.search(
                r"[\u4e00-\u9fff]{2,4}[,，][\u4e00-\u9fff]{2,4}[,，]"
                r"[\u4e00-\u9fff]{2,4}(?:[,，]等|等)",
                raw_clause,
            )
            if author_list and author_list.start() >= 18:
                raw_clause = raw_clause[: author_list.start()]
            clause = clean_evidence_text(raw_clause, limit=220)
            if len(clause) < 18 or reference_like_text(raw_clause):
                continue
            if focus_groups and any(
                not any(
                    re.search(pattern, f"{title} {clause}", re.I)
                    for pattern in patterns
                )
                for patterns in focus_groups
            ):
                continue
            duration_markers = duration_pattern.findall(clause)
            if duration_question and not duration_markers:
                continue
            intent_markers = intent_pattern.findall(clause) if intent_pattern else []
            if intent_pattern is not None and not intent_markers:
                continue
            if re.search(
                r"^(?:尼龙|霉素)\s*治疗|[、,，]\s*[\u4e00-\u9fff]{2,4}\.?$",
                clause,
            ):
                continue
            if re.search(
                r"参考文献|\bet\s*al\b|doi\s*:|CHINA\s+MODERN|\bVol\.|"
                r"第\s*\d+\s*卷\s*第\s*\d+\s*期|"
                r"有效性\s+(?:IL-?\d+|CRP|TNF)|总结(?:研究)?结果|结果.*报道如下",
                clause,
                re.I,
            ):
                continue
            digit_ratio = sum(character.isdigit() for character in clause) / len(clause)
            if digit_ratio > 0.2:
                continue
            direction_markers = re.findall(
                r"推荐|建议|可考虑|有效|显著|良好|获益|改善|缩短|"
                r"降低|高于|低于|短于|优于|提高|未增加|差异有统计学意义|"
                r"差异无统计学意义|首选|不推荐|无效|风险因素|危险因素|预测|"
                r"相当|更高|更低|增加|减少|更多|较少",
                clause,
                re.I,
            )
            result_markers = re.findall(r"结论|结果|综上|提示|显示", clause, re.I)
            method_markers = re.findall(
                r"目的|纳入标准|入选标准|排除标准|一般资料|研究对象|随机数字表|"
                r"分为(?:对照|观察|治疗)组|方法选取|观察指标|疗效判定|"
                r"判定(?:为|治疗效果)|知情同意|伦理委员会",
                clause,
                re.I,
            )
            if re.search(
                r"排除标准|纳入标准|入选标准|随机数字表|观察指标|疗效判定|"
                r"判定治疗效果|知情同意|伦理委员会",
                clause,
                re.I,
            ):
                continue
            if re.search(r"^(?:目的|研究目的)\s*[:：]?", clause):
                continue
            if method_markers and not direction_markers:
                continue
            if (
                not direction_markers
                and not _FALLBACK_DOSE_PATTERN.search(clause)
                and not duration_markers
            ):
                descriptive_score = sum(
                    3 for term in question_terms if term in clause.casefold()
                )
                descriptive_score += 12 * len(intent_markers)
                descriptive_candidates.append(
                    (descriptive_score, -len(clause), clause)
                )
                continue
            if re.search(r"(?:与|和|及|为|在|对|使用|治疗|联合|比较)$", clause):
                continue
            score = 0
            score += sum(3 for term in question_terms if term in clause.casefold())
            score += 7 * len(direction_markers)
            score += 3 * len(result_markers)
            score += 12 * len(duration_markers) if duration_question else 0
            score += 12 * len(intent_markers)
            score += 3 * len(
                re.findall(
                    r"联合|治疗|不良反应|安全|剂量|疗程|mg\s*/?\s*kg",
                    clause,
                    re.I,
                )
            )
            score -= 12 * len(method_markers)
            score -= 5 * len(re.findall(r"表\s*\d|图\s*\d|t\s*值|P\s*值", clause, re.I))
            if 35 <= len(clause) <= 170:
                score += 3
            candidates.append((score, -len(clause), clause))
    if not candidates:
        if not descriptive_candidates:
            return ""
        descriptive_candidates.sort(reverse=True)
        return descriptive_candidates[0][2]
    candidates.sort(reverse=True)
    return candidates[0][2]


def _fallback_intervention(text: str) -> str:
    names = [
        label
        for label, pattern in _FALLBACK_MEDICATIONS
        if re.search(pattern, text, re.I)
    ]
    if "甲泼尼龙" in names or "地塞米松" in names:
        names = [name for name in names if name != "糖皮质激素"]
    return "、".join(dict.fromkeys(names))


def _fallback_dose(text: str) -> str:
    match = _FALLBACK_DOSE_PATTERN.search(text)
    return " ".join(match.group(0).split()) if match else ""


def _fallback_clinical_aspect(text: str) -> str:
    if re.search(r"不良反应|安全性|感染风险|副作用|高血压|高血糖", text, re.I):
        return "safety"
    if re.search(
        r"疗程|治疗周期|持续.{0,8}(?:天|周|月|d)|连用.{0,8}(?:天|周|d)|"
        r"病程.{0,8}(?:天|周|d)|间隔.{0,8}(?:天|周|d)|总疗程|时机|早期|晚期",
        text,
        re.I,
    ):
        return "timing"
    if _FALLBACK_DOSE_PATTERN.search(text) or re.search(r"剂量|小剂量|大剂量", text):
        return "dose"
    if re.search(
        r"诊断标准|确诊|鉴别诊断|病原学|核酸检测|抗体检测|敏感度|特异度|临界值",
        text,
        re.I,
    ):
        return "diagnosis"
    if re.search(
        r"预后|复发|长期结局|危险因素|风险因素|独立因素|随访|后遗症|"
        r"支气管扩张|闭塞性细支气管炎",
        text,
        re.I,
    ):
        return "prognosis"
    if re.search(r"病因|发病机制|机制|由于|导致|相关因素", text, re.I):
        return "other"
    if re.search(r"适用|指征|危重|重症|难治性", text):
        return "applicability"
    return "effectiveness"


def _fallback_direction(text: str) -> str:
    ineffective = re.search(
        r"(\d+(?:\.\d+)?)\s*%[^，,。；;%]{0,45}(?:治疗)?无效", text
    )
    effective = re.search(
        r"(\d+(?:\.\d+)?)\s*%[^，,。；;%]{0,45}(?:治疗)?(?<!无)有效", text
    )
    if ineffective and effective:
        ineffective_rate = float(ineffective.group(1))
        effective_rate = float(effective.group(1))
        if effective_rate > ineffective_rate:
            return "supports"
        if ineffective_rate > effective_rate:
            return "opposes"
    if re.search(r"不推荐|不支持|(?:治疗|方案)无效|未见改善", text):
        return "opposes"
    if re.search(r"尚无|无定论|不明确|差异无统计学意义|未能检索", text):
        return "uncertain"
    if re.search(r"推荐|建议|有效|改善|缩短|降低|优于|提高|可考虑", text):
        return "supports"
    return "uncertain"


def _fallback_outcome(text: str) -> str:
    outcomes = []
    patterns = (
        ("退热时间", r"退热|热程|发热"),
        ("咳嗽等症状", r"咳嗽|临床症状"),
        ("住院时间", r"住院"),
        ("肺部炎症或影像学改善", r"肺部|炎症|影像|CRP"),
        ("不良反应", r"不良反应|安全"),
    )
    for label, pattern in patterns:
        if re.search(pattern, text, re.I):
            outcomes.append(label)
    return "、".join(outcomes[:3]) or "临床疗效"


def _fallback_statement(text: str, intervention: str, dose: str) -> str:
    """Prefer a compact structured sentence over a damaged PDF line."""
    if intervention and dose:
        route = next(
            (
                label
                for label, pattern in (
                    ("静脉滴注", r"静脉滴注|静滴|intravenous"),
                    ("口服", r"口服|oral"),
                    ("雾化吸入", r"雾化|吸入|inhal"),
                )
                if re.search(pattern, text, re.I)
            ),
            "",
        )
        route_text = f"，给药途径为{route}" if route else ""
        return f"该来源报告{intervention}剂量为{dose}{route_text}。"
    excerpt = _grounded_excerpt(text, limit=170)
    return re.sub(r"^(?:结论|结果)\s*[:：]?\s*", "", excerpt).strip()


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
        if not quote_id and source_quote.casefold() not in source_text.casefold():
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
                        "## 综合回答\n\n**综合文献结论：...**[N]\n\n"
                        "### 证据链\n...\n\n### 时间更新\n...\n\n"
                        "### 安全性与适用边界\n...\n\n### 证据缺口\n..."
                    ),
                    "detail_requirements": {
                        "target_length": "1200-2000 Chinese characters when supported",
                        "analysis_step_depth": "2-4 complete sentences per step",
                        "conclusion": "direct answer, rationale, certainty, decision boundary",
                        "evidence_chain": "compare 2-4 strongest sources across levels",
                        "safety_and_gaps": "concrete boundaries and at least 3 supported gaps",
                    },
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
                envelope_valid = True
                try:
                    analysis_steps, raw_final = self._validate_response_envelope(
                        response, bundle
                    )
                except KnowledgeBaseError:
                    envelope_valid = False
                    if not isinstance(response, Mapping):
                        raise
                    raw_final = response.get("final_answer_markdown")
                    if not isinstance(raw_final, str) or not raw_final.strip():
                        raise
                    analysis_steps = self._deterministic_fallback(
                        question, bundle, "ungrounded_model_response"
                    ).analysis_steps
                if envelope_valid:
                    try:
                        final_answer = _validate_flexible_synthesis(
                            raw_final, sources, require_readability=True
                        )
                    except KnowledgeBaseError as flexible_error:
                        if flexible_error.code != "ungrounded_model_response":
                            raise
                        final_answer = _canonicalize_report_claims(
                            raw_final, bundle, sources
                        )
                        _validate_report_readability(final_answer)
                else:
                    try:
                        final_answer = _canonicalize_report_claims(
                            raw_final, bundle, sources
                        )
                    except KnowledgeBaseError as canonical_error:
                        if canonical_error.code != "ungrounded_model_response":
                            raise
                        final_answer = _validate_flexible_synthesis(
                            raw_final, sources
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
        "sources": [_report_source_metadata(source) for source in model_sources],
        "graph": dict(bundle.graph),
        "claims": [
            _clean_claim_for_model(claim.as_dict())
            for claim in bundle.claims
            if claim.document_id in allowed_documents
        ],
    }


def _clean_claim_for_model(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(payload)
    for key in (
        "statement",
        "source_quote",
        "population",
        "intervention",
        "comparator",
        "outcome",
        "dose",
        "design",
        "follow_up",
    ):
        if key in cleaned:
            cleaned[key] = clean_evidence_text(cleaned[key], limit=420)
    for key in ("effect_measures", "limitations"):
        value = cleaned.get(key)
        if isinstance(value, list):
            cleaned[key] = [
                item
                for item in (clean_evidence_text(text, limit=200) for text in value)
                if item
            ]
    return cleaned


def _report_source_metadata(source: EvidenceSource) -> dict[str, Any]:
    """Keep the synthesis request to bibliographic metadata and validated claims."""
    return {
        "source_number": source.source_number,
        "document_id": source.document_id,
        "title": source.title,
        "evidence_type": source.evidence_type,
        "year": source.year,
        "quality": source.quality,
        "journal": source.journal,
        "doi": source.doi,
        "population_applicability": source.population_applicability,
        "question_relevance_score": source.question_relevance_score,
    }


def _quote_options(source: EvidenceSource) -> list[dict[str, str]]:
    candidates: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for chunk_id, snippet in zip(source.chunk_ids, source.snippets):
        for clause in _grounding_clauses(snippet):
            text = clean_evidence_text(clause, limit=360)
            key = re.sub(r"\W+", "", text.casefold())
            if (
                len(text) < 16
                or reference_like_text(clause)
                or not key
                or key in seen
            ):
                continue
            seen.add(key)
            candidates.append((_quote_quality_score(text), chunk_id, text))
    candidates.sort(key=lambda item: (item[0], -len(item[2])), reverse=True)
    return [
        {
            "quote_id": f"quote-{source.source_number}-{index}",
            "source_chunk_id": chunk_id,
            "text": text,
        }
        for index, (_, chunk_id, text) in enumerate(candidates[:4], start=1)
    ]


def _quote_quality_score(text: str) -> int:
    score = 0
    score += 5 * len(
        re.findall(
            r"结论|推荐|建议|结果|可考虑|联合|小剂量|常规剂量|疗程|剂量|"
            r"有效|改善|缩短|降低|安全性|不良反应|mg\s*/\s*kg",
            text,
            re.I,
        )
    )
    score += 3 if re.search(r"甲泼尼龙|糖皮质激素|阿奇霉素", text, re.I) else 0
    score += 2 if text.rstrip().endswith(("。", ".", "！", "?", "？")) else 0
    score -= 8 * len(
        re.findall(
            r"t\s*值|P\s*值|表\s*\d|图\s*\d|病迁延|观目前|察其|对研泼|"
            r"\b(?:vol|doi)\b",
            text,
            re.I,
        )
    )
    digit_ratio = sum(character.isdigit() for character in text) / max(len(text), 1)
    if digit_ratio > 0.2:
        score -= 10
    if len(text) > 260:
        score -= 3
    return score


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
    _validate_statistics(text, cited_ids, sources)
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
            output_lines.append(raw_line)
            continue
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
        trailing = line[cursor:]
        output_lines.append("".join(line_parts) + trailing)
    if not saw_claim:
        raise _ungrounded()
    return "\n".join(output_lines)


def _validate_flexible_synthesis(
    text: str,
    sources: dict[int, EvidenceSource],
    *,
    require_readability: bool = False,
) -> str:
    """Accept model synthesis while enforcing real citations and source-backed numbers."""

    final_answer = text.strip()
    required_headings = (
        "## 综合回答",
        "### 证据链",
        "### 时间更新",
        "### 安全性与适用边界",
        "### 证据缺口",
    )
    positions = [final_answer.find(heading) for heading in required_headings]
    if (
        not final_answer.startswith(required_headings[0])
        or any(position < 0 for position in positions)
        or positions != sorted(positions)
    ):
        raise _ungrounded()
    cited_ids = _citation_ids(final_answer)
    if not cited_ids or not cited_ids.issubset(sources):
        raise _ungrounded()
    _validate_statistics(final_answer, cited_ids, sources)
    if require_readability:
        _validate_report_readability(final_answer)
    return final_answer


def _validate_report_readability(text: str) -> None:
    """Reject extractive-looking model reports before they reach the UI."""

    if re.search(
        r"参考文献|(?:硕士|博士)(?:研究生)?学位论文|当代医药论丛|"
        r"[\[［【]\s*[JM]\s*[\]］】]",
        text,
        re.I,
    ):
        raise _ungrounded()
    content_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if any(len(line) > 420 for line in content_lines):
        raise _ungrounded()
    conclusion_end = text.find("### 证据链")
    introduction = text[:conclusion_end] if conclusion_end >= 0 else text[:500]
    if not re.search(r"结论|推荐|建议|不推荐|证据不足", introduction[:260]):
        raise _ungrounded()


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
    if _claim_statement_is_grounded(segment, source_quote):
        return True
    if _has_negation(segment) != _has_negation(source_quote):
        return False
    segment_tokens = _semantic_tokens(segment)
    source_tokens = _semantic_tokens(f"{statement} {source_quote}")
    overlap = segment_tokens.intersection(source_tokens)
    return len(overlap) >= 2 and len(overlap) / max(1, len(segment_tokens)) >= 0.45


def _semantic_tokens(text: str) -> set[str]:
    normalized = _unicase(text)
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized)
        if len(token) >= 2 and token not in _GROUNDING_STOPWORDS
    }
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    for run in chinese_runs:
        tokens.discard(run)
        tokens.update(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


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
