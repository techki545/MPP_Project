"""Deterministic six-step evidence reasoning and conclusion-first reporting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph_builder import EvidenceClaim
    from .reporter import EvidenceBundle, EvidenceSource


FIRST_DEMO_STAGES = (
    ("inventory", "第一步：检索并盘点证据库存"),
    ("guidelines", "第二步：优先查看指南"),
    ("systematic_reviews", "第三步：查阅系统综述，检验指南结论"),
    ("randomized_trials", "第四步：聚焦关键随机对照试验"),
    ("lower_level_evidence", "第五步：用下级证据补充安全性与边界"),
    ("synthesis", "第六步：检查一致性并形成综合判断"),
)
EXPECTED_STAGE_KEYS = tuple(stage[0] for stage in FIRST_DEMO_STAGES)

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
_LOWER_LEVEL_TYPES = (
    "observational_study",
    "narrative_review",
    "case_report",
    "unknown",
)
_RELATION_LABELS = {
    "supports": "支持",
    "updates": "更新",
    "supplements": "补充",
    "confirms": "确认",
    "conflicts": "冲突",
    "cautions": "警示",
}
_MEDICATION_PATTERNS = (
    ("甲泼尼龙", r"甲泼尼龙|甲强龙|methylprednisolone"),
    ("地塞米松", r"地塞米松|dexamethasone"),
    ("阿奇霉素", r"阿奇霉素|azithromycin"),
    ("红霉素", r"红霉素|erythromycin"),
    ("克拉霉素", r"克拉霉素|clarithromycin"),
    ("多西环素", r"多西环素|doxycycline"),
    ("米诺环素", r"米诺环素|minocycline"),
    ("左氧氟沙星", r"左氧氟沙星|levofloxacin"),
    ("莫西沙星", r"莫西沙星|moxifloxacin"),
    ("静脉注射免疫球蛋白（IVIG）", r"丙种球蛋白|免疫球蛋白|ivig|gammaglobulin"),
    ("糖皮质激素", r"糖皮质激素|皮质类固醇|皮质激素|glucocorticoid|corticosteroid"),
    ("大环内酯类抗菌药物", r"大环内酯|macrolide"),
    ("四环素类抗菌药物", r"四环素|tetracycline"),
)


@dataclass(frozen=True)
class StructuredReport:
    analysis_steps: tuple[dict[str, Any], ...]
    final_answer_markdown: str


def compose_deterministic_report(
    question: str, bundle: EvidenceBundle
) -> StructuredReport:
    grouped = {
        evidence_type: tuple(
            source
            for source in bundle.sources
            if source.evidence_type == evidence_type
        )
        for evidence_type in _EVIDENCE_ORDER
    }
    relation_counts = Counter(
        str(edge.get("relation", ""))
        for edge in bundle.graph.get("edges", [])
        if isinstance(edge, dict)
    )
    claims_by_document: dict[str, list[EvidenceClaim]] = {}
    for claim in bundle.claims:
        claims_by_document.setdefault(claim.document_id, []).append(claim)
    steps = _compose_first_demo_steps(
        question,
        bundle,
        grouped,
        relation_counts,
        claims_by_document,
    )
    answer = _compose_conclusion_first_answer(
        question,
        bundle,
        grouped,
        relation_counts,
        claims_by_document,
    )
    return StructuredReport(tuple(steps), answer)


def _compose_first_demo_steps(
    question: str,
    bundle: EvidenceBundle,
    grouped: dict[str, tuple[EvidenceSource, ...]],
    relation_counts: Counter[str],
    claims_by_document: dict[str, list[EvidenceClaim]],
) -> list[dict[str, Any]]:
    counts = Counter(source.evidence_type for source in bundle.sources)
    inventory = "、".join(
        f"{_EVIDENCE_LABELS[evidence_type]} {counts[evidence_type]} 项"
        for evidence_type in _EVIDENCE_ORDER
        if counts[evidence_type]
    ) or "没有可用证据"
    inventory_body = (
        f"围绕“{question}”，本次纳入 {len(bundle.sources)} 项来源：{inventory}。"
        "后续按证据金字塔顺序审阅，同时检查较新研究是否补充或更新上级证据。"
    )

    guideline_body = _evidence_level_body(
        grouped["guideline"],
        claims_by_document,
        missing="未检索到指南；本轮不能把指南作为直接依据，将继续查看系统综述和原始研究。",
        lead="优先查看指南及其证据时间范围：",
    )
    review_body = _evidence_level_body(
        grouped["systematic_review"],
        claims_by_document,
        missing="未检索到系统综述或 Meta 分析，无法用汇总证据直接检验指南结论。",
        lead="系统综述用于检验上级建议与汇总研究方向是否一致：",
    )
    trial_body = _evidence_level_body(
        grouped["randomized_controlled_trial"],
        claims_by_document,
        missing="未检索到随机对照试验，干预效果或比较结论主要依赖其他证据层级。",
        lead="优先核对关键随机对照试验、发表时间和临床维度：",
    )
    lower_sources = tuple(
        source
        for evidence_type in _LOWER_LEVEL_TYPES
        for source in grouped[evidence_type]
    )
    lower_body = _evidence_level_body(
        lower_sources,
        claims_by_document,
        missing="未检索到观察性研究、叙述性综述或病例报告，安全性和适用边界信息有限。",
        lead="下级证据不替代上级结论，主要用于补充安全性、适用性和特殊情境：",
    )
    synthesis_body = _synthesis_body(bundle, relation_counts)

    bodies = (
        inventory_body,
        guideline_body,
        review_body,
        trial_body,
        lower_body,
        synthesis_body,
    )
    step_sources = (
        bundle.sources,
        grouped["guideline"],
        grouped["systematic_review"],
        grouped["randomized_controlled_trial"],
        lower_sources,
        bundle.sources,
    )
    return [
        {
            "stage_key": stage_key,
            "title": title,
            "body": body,
            "source_ids": [source.source_number for source in sources],
        }
        for (stage_key, title), body, sources in zip(
            FIRST_DEMO_STAGES, bodies, step_sources
        )
    ]


def _compose_conclusion_first_answer(
    question: str,
    bundle: EvidenceBundle,
    grouped: dict[str, tuple[EvidenceSource, ...]],
    relation_counts: Counter[str],
    claims_by_document: dict[str, list[EvidenceClaim]],
) -> str:
    intent = _question_intent(question)
    relevant_aspects = _question_aspects(question)
    decision_claims = tuple(
        claim
        for claim in bundle.claims
        if claim.evidence_role == "core"
        and claim.clinical_aspect in relevant_aspects
    )
    if intent != "yes_no" and not decision_claims:
        decision_claims = tuple(
            claim
            for claim in bundle.claims
            if claim.evidence_role != "boundary"
            and claim.clinical_aspect in relevant_aspects
        )
    directions = Counter(claim.direction for claim in decision_claims)
    clean_question = " ".join(question.split()).rstrip("?？")
    yes_no_question = intent == "yes_no"
    directional_sources = [
        source.source_number
        for source in bundle.sources
        if any(
            claim.direction != "uncertain"
            for claim in decision_claims
            if claim.document_id == source.document_id
        )
    ]
    citation_suffix = "".join(f"[{number}]" for number in directional_sources)
    if not yes_no_question:
        conclusion = _open_question_conclusion(
            intent,
            clean_question,
            decision_claims,
            bundle.sources,
        )
    elif not directions or directions["supports"] + directions["opposes"] == 0:
        if yes_no_question:
            conclusion = f"**结论：对于“{clean_question}”，当前证据不足，不能回答“是”或“否”。**"
        else:
            conclusion = "**结论：当前可验证声明没有提供足够的方向信息，不能形成肯定或否定建议。**"
    elif directions["supports"] > directions["opposes"] and not relation_counts[
        "conflicts"
    ]:
        if yes_no_question:
            conclusion = (
                f"**结论：对于“{clean_question}”，当前证据倾向于“是”，即总体支持题述做法；"
                f"但仍需结合安全性、适用人群和证据缺口审慎决策。**{citation_suffix}"
            )
        else:
            conclusion = (
                "**结论：现有检索证据总体支持问题所聚焦的临床判断，但仍需结合"
                f"安全性、适用人群和证据缺口审慎解释。**{citation_suffix}"
            )
    elif directions["opposes"] > directions["supports"] and not relation_counts[
        "conflicts"
    ]:
        if yes_no_question:
            conclusion = (
                f"**结论：对于“{clean_question}”，当前证据倾向于“否”，即总体不支持题述做法；"
                f"仍需结合具体人群和证据局限审慎决策。**{citation_suffix}"
            )
        else:
            conclusion = (
                "**结论：现有检索证据总体不支持问题所述临床判断，且仍需结合"
                f"具体人群和证据局限审慎决策。**{citation_suffix}"
            )
    else:
        if yes_no_question:
            conclusion = (
                f"**结论：对于“{clean_question}”，各来源方向不一致，当前不能简单回答“是”或“否”。**"
                f"{citation_suffix}"
            )
        else:
            conclusion = (
                "**结论：各来源的方向并不一致，当前证据不足以形成单一肯定或"
                f"否定结论。**{citation_suffix}"
            )

    decision_document_ids = {claim.document_id for claim in decision_claims}
    evidence_document_ids = decision_document_ids or {
        claim.document_id for claim in bundle.claims
    }
    evidence_lines = [
        _claim_line(source, claims_by_document)
        for evidence_type in _EVIDENCE_ORDER
        for source in grouped[evidence_type]
        if source.document_id in evidence_document_ids
    ]
    evidence_chain = "\n".join(line for line in evidence_lines if line)
    if not evidence_chain:
        evidence_chain = "当前没有可用于综合的来源声明。"

    chronology = _chronology_text(bundle, relation_counts)
    boundary_lines = [
        _claim_line(source, claims_by_document)
        for source in bundle.sources
        if any(
            claim.safety_signal
            or claim.clinical_aspect in {"safety", "applicability"}
            or claim.evidence_role == "boundary"
            for claim in claims_by_document.get(source.document_id, [])
        )
    ]
    boundaries = "\n".join(line for line in boundary_lines if line) or (
        "当前检索声明未提供明确的安全性或适用边界信息，不能据此推断其不存在。"
    )
    gaps = _gap_text(grouped, directions, relation_counts)

    return "\n\n".join(
        (
            "## 综合回答",
            conclusion,
            "### 证据链\n" + evidence_chain,
            "### 时间更新\n" + chronology,
            "### 安全性与适用边界\n" + boundaries,
            "### 证据缺口\n" + gaps,
        )
    )


def _question_intent(question: str) -> str:
    normalized = " ".join(question.casefold().split())
    patterns = (
        ("yes_no", r"是否|应否|能否|可否|要不要|该不该|\b(?:should|whether|can)\b"),
        ("dose", r"剂量|用量|怎么用|dose|mg/kg"),
        ("timing", r"时机|何时|什么时候|早期|晚期|timing|when|early"),
        ("diagnosis", r"诊断|鉴别|确诊|diagnos"),
        ("prognosis", r"预后|预测|危险因素|风险因素|predict|prognos|risk factor"),
        ("safety", r"安全|不良反应|副作用|毒性|safety|adverse|harm"),
        (
            "medication",
            r"吃什么药|用什么药|哪些药|药物|用药|怎么治疗|如何治疗|治疗方案|medicine|medication|which drug|treatment",
        ),
        ("applicability", r"适用|哪些人|什么人|人群|年龄|applicab|population"),
    )
    return next(
        (
            intent
            for intent, pattern in patterns
            if re.search(pattern, normalized, re.I)
        ),
        "general",
    )


def _open_question_conclusion(
    intent: str,
    question: str,
    claims: tuple[EvidenceClaim, ...],
    sources: tuple[EvidenceSource, ...],
) -> str:
    labels = {
        "medication": "用药信息",
        "dose": "剂量信息",
        "timing": "治疗时机信息",
        "diagnosis": "诊断信息",
        "prognosis": "预后信息",
        "safety": "安全性信息",
        "applicability": "适用人群信息",
        "general": "回答信息",
    }
    source_numbers = {
        source.document_id: source.source_number for source in sources
    }
    lines: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        source_number = source_numbers.get(claim.document_id)
        if source_number is None:
            continue
        if intent == "medication":
            answer = _medication_answer(claim)
        elif intent == "dose" and (claim.intervention or claim.dose):
            answer = "：".join(
                value for value in (claim.intervention, claim.dose) if value
            )
        else:
            answer = _clean_claim_statement(claim.statement)
        key = answer.casefold()
        if not answer or key in seen:
            continue
        seen.add(key)
        lines.append(f"- {answer}[{source_number}]")
        if len(lines) == 4:
            break
    label = labels.get(intent, labels["general"])
    if not lines:
        return f"**结论：针对“{question}”，当前验证后的证据不足以形成直接回答。**"
    safety_note = (
        "这些内容是检索证据中的治疗选项，不等同于针对个体患儿的处方。"
        if intent in {"medication", "dose", "timing"}
        else "以下内容均来自本次检索后通过验证的原文声明。"
    )
    return (
        f"**结论：针对“{question}”，当前证据可直接提取的核心{label}如下。**"
        f"{safety_note}\n" + "\n".join(lines)
    )


def _medication_answer(claim: EvidenceClaim) -> str:
    source_text = " ".join(
        value for value in (claim.intervention, claim.statement) if value
    )
    compact = re.sub(r"\s+", "", source_text).casefold()
    names = [
        label
        for label, pattern in _MEDICATION_PATTERNS
        if re.search(pattern, compact, re.I)
    ]
    if "甲泼尼龙" in names or "地塞米松" in names:
        names = [name for name in names if name != "糖皮质激素"]
    if "阿奇霉素" in names:
        names = [name for name in names if name != "大环内酯类抗菌药物"]
    dose = claim.dose or _dose_from_text(source_text)
    if names:
        answer = "、".join(dict.fromkeys(names))
        return f"{answer}（{dose}）" if dose and len(names) == 1 else answer
    intervention = _clean_claim_statement(claim.intervention)
    if intervention:
        return f"{intervention}（{dose}）" if dose else intervention
    return _clean_claim_statement(claim.statement)


def _dose_from_text(text: str) -> str:
    match = re.search(
        r"\b\d+(?:\.\d+)?\s*(?:mg|g)/kg(?:/(?:d|day))?\b",
        text,
        re.I,
    )
    return " ".join(match.group(0).split()) if match else ""


def _question_aspects(question: str) -> set[str]:
    normalized = " ".join(question.casefold().split())
    specific_patterns = (
        ("diagnosis", r"诊断|鉴别|确诊|diagnos"),
        ("prognosis", r"预后|预测|危险因素|风险因素|predict|prognos|risk factor"),
        ("safety", r"安全|不良反应|副作用|毒性|safety|adverse|harm"),
        ("dose", r"剂量|用量|dose|mg/kg"),
        ("timing", r"时机|何时|早期|晚期|timing|when|early"),
        ("applicability", r"适用|人群|年龄|applicab|population"),
    )
    matched = {
        aspect
        for aspect, pattern in specific_patterns
        if re.search(pattern, normalized, re.I)
    }
    if matched:
        return {"overall", *matched}
    return {"overall", "effectiveness", "dose", "timing"}


def _evidence_level_body(
    sources: tuple[EvidenceSource, ...],
    claims_by_document: dict[str, list[EvidenceClaim]],
    *,
    missing: str,
    lead: str,
) -> str:
    if not sources:
        return missing
    lines = [
        _claim_line(source, claims_by_document)
        for source in sorted(
            sources,
            key=lambda item: (
                -(item.year or 0),
                item.source_number,
            ),
        )
    ]
    return lead + "\n" + "\n".join(line for line in lines if line)


def _claim_line(
    source: EvidenceSource,
    claims_by_document: dict[str, list[EvidenceClaim]],
) -> str:
    claims = claims_by_document.get(source.document_id, [])
    if claims:
        statement = _clean_claim_statement(claims[0].statement)
        return f"- {statement}[{source.source_number}]"
    year = str(source.year) if source.year is not None else "年份未知"
    return f"- 《{source.title}》（{year}）[{source.source_number}]"


def _clean_claim_statement(statement: str) -> str:
    without_internal_citations = re.sub(
        r"[\[［]\s*\d+\s*[\]］]", "", statement
    )
    return " ".join(without_internal_citations.split())


def _synthesis_body(
    bundle: EvidenceBundle, relation_counts: Counter[str]
) -> str:
    directions = Counter(claim.direction for claim in bundle.claims)
    relation_text = "、".join(
        f"{_RELATION_LABELS[relation]} {relation_counts[relation]} 条"
        for relation in _RELATION_LABELS
        if relation_counts[relation]
    ) or "未形成可验证的文献间关系"
    if relation_counts["conflicts"] or (
        directions["supports"] and directions["opposes"]
    ):
        consistency = "各级证据存在方向不一致，需要保留冲突并降低结论确定性。"
    elif directions["supports"] or directions["opposes"]:
        consistency = "当前可验证声明的主要方向一致，但一致性不等同于证据充分。"
    else:
        consistency = "当前声明方向均不确定，不能形成肯定或否定判断。"
    return f"关系盘点：{relation_text}。{consistency}"


def _chronology_text(
    bundle: EvidenceBundle, relation_counts: Counter[str]
) -> str:
    dated = [source for source in bundle.sources if source.year is not None]
    if not dated:
        return "当前来源缺少可用发表年份，无法判断时间先后。"
    oldest = min(dated, key=lambda source: (source.year or 0, source.source_number))
    newest = max(dated, key=lambda source: (source.year or 0, -source.source_number))
    update_text = (
        f"关系图识别到 {relation_counts['updates']} 条更新关系。"
        if relation_counts["updates"]
        else "未识别到满足临床维度一致条件的更新关系。"
    )
    return (
        f"证据年份覆盖 {oldest.year} 至 {newest.year}；最新来源为"
        f"《{newest.title}》[{newest.source_number}]。{update_text}"
    )


def _gap_text(
    grouped: dict[str, tuple[EvidenceSource, ...]],
    directions: Counter[str],
    relation_counts: Counter[str],
) -> str:
    missing = [
        _EVIDENCE_LABELS[evidence_type]
        for evidence_type in (
            "guideline",
            "systematic_review",
            "randomized_controlled_trial",
        )
        if not grouped[evidence_type]
    ]
    gaps = []
    if missing:
        gaps.append("未检索到" + "、".join(missing))
    if directions["uncertain"]:
        gaps.append(f"{directions['uncertain']} 项声明的效应方向不确定")
    if relation_counts["conflicts"]:
        gaps.append(f"存在 {relation_counts['conflicts']} 条冲突关系")
    if not gaps:
        gaps.append("仍需核对全文质量、偏倚风险、随访长度和目标人群适用性")
    return "；".join(gaps) + "。"
