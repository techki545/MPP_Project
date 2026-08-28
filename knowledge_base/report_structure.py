"""Deterministic six-step evidence reasoning and conclusion-first reporting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Any

from .text_cleaning import clean_evidence_text

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
_APPLICABILITY_LABELS = {
    "direct": "直接证据",
    "indirect_rmpp": "RMPP间接外推",
    "indirect_smpp": "SMPP间接外推",
    "general_mpp": "一般MPP证据",
    "unclear": "人群不明确",
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
        + _inventory_applicability_text(bundle.sources)
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
    if intent != "yes_no":
        decision_claims = tuple(
            claim
            for claim in bundle.claims
            if claim.clinical_aspect in relevant_aspects
        )
    elif not any(
        claim.direction in {"supports", "opposes"} for claim in decision_claims
    ):
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
            conclusion = f"**综合文献结论：对于“{clean_question}”，当前证据不足，不能回答“是”或“否”。**"
        else:
            conclusion = "**综合文献结论：当前可验证声明没有提供足够的方向信息，不能形成肯定或否定建议。**"
    elif directions["supports"] > directions["opposes"] and not relation_counts[
        "conflicts"
    ]:
        if yes_no_question:
            conclusion = (
                f"**综合文献结论：对于“{clean_question}”，当前证据倾向于“是”，即总体支持题述做法；"
                f"但仍需结合安全性、适用人群和证据缺口审慎决策。**{citation_suffix}"
            )
        else:
            conclusion = (
                "**综合文献结论：现有检索证据总体支持问题所聚焦的临床判断，但仍需结合"
                f"安全性、适用人群和证据缺口审慎解释。**{citation_suffix}"
            )
    elif directions["opposes"] > directions["supports"] and not relation_counts[
        "conflicts"
    ]:
        if yes_no_question:
            conclusion = (
                f"**综合文献结论：对于“{clean_question}”，当前证据倾向于“否”，即总体不支持题述做法；"
                f"仍需结合具体人群和证据局限审慎决策。**{citation_suffix}"
            )
        else:
            conclusion = (
                "**综合文献结论：现有检索证据总体不支持问题所述临床判断，且仍需结合"
                f"具体人群和证据局限审慎决策。**{citation_suffix}"
            )
    else:
        if yes_no_question:
            conclusion = (
                f"**综合文献结论：对于“{clean_question}”，各来源方向不一致，当前不能简单回答“是”或“否”。**"
                f"{citation_suffix}"
            )
        else:
            conclusion = (
                "**综合文献结论：各来源的方向并不一致，当前证据不足以形成单一肯定或"
                f"否定结论。**{citation_suffix}"
            )

    decision_document_ids = {claim.document_id for claim in decision_claims}
    evidence_document_ids = decision_document_ids or {
        claim.document_id for claim in bundle.claims
    }
    for evidence_type in (
        "guideline",
        "systematic_review",
        "randomized_controlled_trial",
    ):
        grounded_sources = [
            source
            for source in grouped[evidence_type]
            if any(
                claim.clinical_aspect in relevant_aspects
                for claim in claims_by_document.get(source.document_id, [])
            )
        ]
        grounded = (
            max(
                grounded_sources,
                key=lambda source: _source_display_score(
                    source, claims_by_document
                ),
            )
            if grounded_sources
            else None
        )
        if grounded is not None:
            evidence_document_ids.add(grounded.document_id)
    evidence_chain = _grouped_evidence_chain(
        grouped,
        claims_by_document,
        evidence_document_ids,
    )
    if not evidence_chain:
        evidence_chain = "当前没有可用于综合的来源声明。"

    chronology = _chronology_text(bundle, relation_counts)
    boundary_sources = sorted(
        bundle.sources,
        key=lambda source: _source_display_score(source, claims_by_document),
        reverse=True,
    )
    boundary_lines = [
        _claim_line(source, claims_by_document)
        for source in boundary_sources
        if any(
            claim.safety_signal
            or claim.clinical_aspect in {"safety", "applicability"}
            or claim.evidence_role == "boundary"
            for claim in claims_by_document.get(source.document_id, [])
        )
    ]
    boundaries = "\n".join(line for line in boundary_lines[:2] if line) or (
        "当前检索声明未提供明确的安全性或适用边界信息，不能据此推断其不存在。"
    )
    gaps = _gap_text(grouped, directions, relation_counts)
    applicability_note = _applicability_note(bundle.sources)

    return "\n\n".join(
        (
            "## 综合回答",
            conclusion,
            applicability_note,
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
        (
            "duration",
            r"疗程|治疗周期|治疗多久|多长时间|需要多久|持续多久|course of treatment|treatment duration|how long",
        ),
        ("dose", r"剂量|用量|怎么用|dose|mg/kg"),
        ("timing", r"时机|何时|什么时候|早期|晚期|timing|when|early"),
        ("diagnosis", r"诊断|鉴别|确诊|diagnos"),
        ("prognosis", r"预后|预测|危险因素|风险因素|predict|prognos|risk factor"),
        ("safety", r"安全|不良反应|副作用|毒性|safety|adverse|harm"),
        ("examination", r"检查|化验|检测|影像|检验|exam|test|imaging"),
        ("etiology", r"病因|原因|为什么|危险因素|etiolog|cause|risk factor"),
        ("mechanism", r"机制|原理|如何起效|mechanism|pathway"),
        ("prevention", r"预防|避免|疫苗|prevent|prophyl"),
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
        "duration": "疗程信息",
        "timing": "治疗时机信息",
        "diagnosis": "诊断信息",
        "prognosis": "预后信息",
        "safety": "安全性信息",
        "applicability": "适用人群信息",
        "examination": "检查信息",
        "etiology": "病因信息",
        "mechanism": "机制信息",
        "prevention": "预防信息",
        "general": "回答信息",
    }
    if intent == "duration" and _is_broad_duration_question(question):
        return _broad_duration_conclusion(question, claims, sources)

    source_numbers = {
        source.document_id: source.source_number for source in sources
    }
    lines: list[str] = []
    seen: set[str] = set()
    ranked_claims = sorted(
        claims,
        key=lambda claim: _answer_claim_score(claim, intent, question),
        reverse=True,
    )
    for claim in ranked_claims:
        source_number = source_numbers.get(claim.document_id)
        if source_number is None:
            continue
        if not _claim_answers_question(claim, intent, question):
            continue
        if intent == "medication":
            answer = _medication_answer(claim)
        elif intent == "dose":
            if claim.dose:
                answer = "：".join(
                    value for value in (claim.intervention, claim.dose) if value
                )
            elif re.search(r"剂量|疗程|小剂量|常规剂量|大剂量", claim.statement):
                answer = clean_evidence_text(claim.statement, limit=150)
            else:
                continue
        elif intent == "duration":
            answer = _duration_answer(claim)
            if not answer:
                continue
        else:
            answer = _clean_claim_statement(claim.statement)
        answer = clean_evidence_text(answer, limit=160)
        key = re.sub(r"\W+", "", answer.casefold())
        if not answer or key in seen:
            continue
        seen.add(key)
        lines.append(f"- {answer}[{source_number}]")
        if len(lines) == 3:
            break
    label = labels.get(intent, labels["general"])
    if not lines:
        return f"**综合文献结论：针对“{question}”，当前验证后的证据不足以形成直接回答。**"
    safety_note = (
        "这些内容是检索证据中的治疗选项，不等同于针对个体患儿的处方。"
        if intent in {"medication", "dose", "duration", "timing"}
        else "以下内容均来自本次检索后通过验证的原文声明。"
    )
    lead = (
        f"**综合文献结论：针对“{question}”，检索证据给出的核心{label}如下。**"
        "应先确认患儿符合重症或难治性肺炎支原体肺炎的临床指征，再由儿科医生结合病程、炎症指标和并发症决定方案。"
        if intent in {"medication", "dose", "duration", "timing"}
        else f"**综合文献结论：针对“{question}”，检索证据给出的核心{label}如下。**"
    )
    return lead + "\n\n" + safety_note + "\n" + "\n".join(lines)


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


def _is_broad_duration_question(question: str) -> bool:
    if not re.search(
        r"疗程|治疗周期|治疗多久|多长时间|需要多久|持续多久|treatment duration|how long",
        question,
        re.I,
    ):
        return False
    intervention_pattern = "|".join(pattern for _, pattern in _MEDICATION_PATTERNS)
    return not re.search(intervention_pattern, question, re.I)


def _duration_answer(claim: EvidenceClaim) -> str:
    statement = _clean_claim_statement(claim.statement)
    if not re.search(
        r"疗程|周期|连续|连用|持续|间隔|停药|减量|"
        r"(?:\d+|[一二三四五六七八九十]+)\s*(?:天|周|月|d)\b|"
        r"48\s*[-~～]\s*72\s*h",
        statement,
        re.I,
    ):
        return ""
    if re.search(r"国药准字|有限公司|^\s*\d+(?:\.\d+)+\s*方法", statement):
        fragments: list[str] = []
        fragment_pattern = re.compile(
            r"连续(?:使用|用药|治疗)?\s*\d+(?:\s*[~～至-]\s*\d+)?\s*(?:d|天|周|月)(?:\s*后)?|"
            r"停药.{0,8}?间隔\s*\d+(?:\s*[~～至-]\s*\d+)?\s*(?:d|天|周|月)(?:\s*后(?:再次给药)?)?|"
            r"间隔\s*\d+(?:\s*[~～至-]\s*\d+)?\s*(?:d|天|周|月)(?:\s*后(?:再次给药)?)?|"
            r"(?:总)?疗程.{0,12}?\d+(?:\s*[~～至-]\s*\d+)?\s*(?:d|天|周|月)|"
            r"建议不超过\s*\d+\s*(?:d|天|周|月)",
            re.I,
        )
        for match in fragment_pattern.finditer(statement):
            fragment = " ".join(match.group(0).split()).strip(" ,，;；")
            key = re.sub(r"\W+", "", fragment.casefold())
            if fragment and key not in {re.sub(r"\W+", "", item.casefold()) for item in fragments}:
                fragments.append(fragment)
        if fragments:
            medication_matches = [
                (match.start(), label)
                for label, pattern in _MEDICATION_PATTERNS
                if (match := re.search(pattern, statement, re.I)) is not None
            ]
            intervention = (
                min(medication_matches)[1]
                if medication_matches
                else claim.intervention or "该治疗"
            )
            return f"{intervention}疗程：" + "，".join(fragments[:4]) + "。"
    return statement


def _duration_group(claim: EvidenceClaim) -> str:
    statement = claim.statement
    if re.search(
        r"甲泼尼龙|地塞米松|糖皮质激素|免疫球蛋白|丙种球蛋白|ivig|免疫治疗",
        statement,
        re.I,
    ):
        return "抗炎或免疫治疗"
    if re.search(
        r"阿奇霉素|红霉素|克拉霉素|大环内酯|多西环素|米诺环素|"
        r"左氧氟沙星|抗菌|抗生素",
        statement,
        re.I,
    ):
        return "抗菌治疗"
    text = " ".join((claim.intervention, statement, claim.outcome))
    if re.search(r"免疫球蛋白|丙种球蛋白|ivig", text, re.I):
        return "抗炎或免疫治疗"
    if re.search(
        r"阿奇霉素|红霉素|克拉霉素|大环内酯|多西环素|米诺环素|"
        r"左氧氟沙星|抗菌|抗生素",
        text,
        re.I,
    ):
        return "抗菌治疗"
    if re.search(r"甲泼尼龙|地塞米松|糖皮质激素|免疫治疗", text, re.I):
        return "抗炎或免疫治疗"
    if re.search(r"住院|随访|复查|评估|退热|病程|48\s*[-~～]\s*72", text, re.I):
        return "疗效评估与随访"
    return "其他治疗环节"


def _broad_duration_conclusion(
    question: str,
    claims: tuple[EvidenceClaim, ...],
    sources: tuple[EvidenceSource, ...],
) -> str:
    source_numbers = {source.document_id: source.source_number for source in sources}
    grouped: dict[str, list[str]] = {}
    seen: set[str] = set()
    ranked = sorted(
        claims,
        key=lambda claim: _answer_claim_score(claim, "duration", question),
        reverse=True,
    )
    for claim in ranked:
        source_number = source_numbers.get(claim.document_id)
        answer = _duration_answer(claim)
        if source_number is None or not answer:
            continue
        key = re.sub(r"\W+", "", answer.casefold())
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(_duration_group(claim), []).append(
            f"{clean_evidence_text(answer, limit=180)}[{source_number}]"
        )

    if not grouped:
        return (
            f"**综合文献结论：针对“{question}”，当前验证后的证据未给出可直接采用的疗程信息，"
            "因此不能推定一个固定治疗周期。**"
        )

    sections = []
    for label in ("抗菌治疗", "抗炎或免疫治疗", "疗效评估与随访", "其他治疗环节"):
        values = grouped.get(label, [])[:2]
        if values:
            sections.append(f"**{label}：**" + "；".join(values))
    missing = [
        label
        for label in ("抗菌治疗", "抗炎或免疫治疗", "疗效评估与随访")
        if label not in grouped
    ]
    gap_note = (
        "本次证据尚未直接覆盖" + "、".join(missing) + "的疗程，不能用已检索到的单项方案替代。"
        if missing
        else "各部分疗程仍需根据病原耐药、临床反应、并发症和复查结果动态调整。"
    )
    return (
        f"**综合文献结论：针对“{question}”，SMPP 没有一个可由当前文献统一概括的固定“总治疗周期”。**"
        "现有证据给出的是不同治疗环节各自的疗程和复评节点，不能把阿奇霉素或任何一种药物的疗程等同于全部治疗周期。\n\n"
        + "\n\n".join(sections)
        + "\n\n"
        + gap_note
        + "以上为文献综合，不是针对个体患儿的处方。"
    )


def _question_aspects(question: str) -> set[str]:
    normalized = " ".join(question.casefold().split())
    intent = _question_intent(question)
    intent_aspects = {
        "duration": {"overall", "timing"},
        "dose": {"overall", "dose"},
        "timing": {"overall", "timing"},
        "diagnosis": {"overall", "diagnosis"},
        "examination": {"overall", "diagnosis"},
        "prognosis": {"overall", "prognosis"},
        "safety": {"overall", "safety"},
        "applicability": {"overall", "applicability"},
        "etiology": {"overall", "prognosis", "other"},
        "mechanism": {"overall", "other"},
        "prevention": {"overall", "effectiveness", "other"},
        "medication": {
            "overall",
            "effectiveness",
            "dose",
            "timing",
            "safety",
            "applicability",
        },
    }
    if intent in intent_aspects:
        return intent_aspects[intent]
    specific_patterns = (
        ("diagnosis", r"诊断|鉴别|确诊|diagnos"),
        ("prognosis", r"预后|预测|危险因素|风险因素|predict|prognos|risk factor"),
        ("safety", r"安全|不良反应|副作用|毒性|safety|adverse|harm"),
        ("dose", r"剂量|用量|怎么用|如何用|dose|mg/kg"),
        (
            "timing",
            r"疗程|治疗周期|治疗多久|多长时间|持续多久|时机|何时|早期|晚期|"
            r"timing|when|early|duration|how long",
        ),
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
    grounded_sources = tuple(
        source for source in sources if claims_by_document.get(source.document_id)
    )
    if not grounded_sources:
        return (
            f"检索到 {len(sources)} 项该层级来源，但没有可验证声明直接匹配问题中的临床焦点，"
            "因此不将其作为结论依据。"
        )
    lines = [
        _claim_line(source, claims_by_document)
        for source in sorted(
            grounded_sources,
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
    applicability = _APPLICABILITY_LABELS.get(source.population_applicability)
    prefix = f"[{applicability}] " if applicability else ""
    if claims:
        statement = clean_evidence_text(claims[0].statement, limit=170)
        return f"- {prefix}{statement}[{source.source_number}]"
    year = str(source.year) if source.year is not None else "年份未知"
    return f"- {prefix}《{source.title}》（{year}）[{source.source_number}]"


def _clean_claim_statement(statement: str) -> str:
    return clean_evidence_text(statement, limit=320)


def _grouped_evidence_chain(
    grouped: dict[str, tuple[EvidenceSource, ...]],
    claims_by_document: dict[str, list[EvidenceClaim]],
    included_document_ids: set[str],
) -> str:
    sections: list[str] = []
    seen_statements: set[str] = set()
    for evidence_type in _EVIDENCE_ORDER:
        lines: list[str] = []
        ranked_sources = sorted(
            grouped[evidence_type],
            key=lambda source: _source_display_score(source, claims_by_document),
            reverse=True,
        )
        for source in ranked_sources:
            if source.document_id not in included_document_ids:
                continue
            line = _claim_line(source, claims_by_document)
            key = re.sub(r"\[\d+\]", "", line)
            key = re.sub(r"\W+", "", key.casefold())
            if not line or key in seen_statements:
                continue
            seen_statements.add(key)
            lines.append(line)
            if len(lines) == 3:
                break
        if lines:
            sections.append(f"**{_EVIDENCE_LABELS[evidence_type]}**\n" + "\n".join(lines))
    return "\n\n".join(sections) or "当前没有可用于综合的来源声明。"


def _answer_claim_score(claim: EvidenceClaim, intent: str, question: str) -> int:
    score = 0
    if intent == "dose":
        score += 10 if claim.dose else 0
        score += 6 if re.search(r"小剂量.{0,30}(?:常规剂量|安全性|相当)", claim.statement) else 0
        score += 3 if re.search(r"尚无统一|无定论|不明确", claim.statement) else 0
    if intent == "duration":
        score += 10 if _duration_answer(claim) else 0
        score += 4 if re.search(r"疗程|总疗程|周期|连用|间隔|减量", claim.statement) else 0
        score += 3 if re.search(r"尚无统一|无定论|不明确|依据病情", claim.statement) else 0
    if claim.intervention:
        score += 2
    if claim.direction != "uncertain":
        score += 1
    if not _claim_answers_question(claim, intent, question):
        score -= 20
    return score


def _claim_answers_question(
    claim: EvidenceClaim, intent: str, question: str
) -> bool:
    statement = claim.statement
    if len(statement) < 18:
        return False
    if intent == "dose" and re.search(r"糖皮质激素|甲泼尼龙|甲强龙", question):
        grounded_treatment = " ".join((claim.intervention, statement))
        if not re.search(r"糖皮质激素|甲泼尼龙|甲强龙|methylprednisolone", grounded_treatment, re.I):
            return False
    if intent == "duration" and not _duration_answer(claim):
        return False
    if statement.rstrip().endswith(("甲泼", "联合", "治疗基础上")):
        return False
    return True


def _source_display_score(
    source: EvidenceSource,
    claims_by_document: dict[str, list[EvidenceClaim]],
) -> int:
    claims = claims_by_document.get(source.document_id, [])
    if not claims:
        return 0
    claim = claims[0]
    score = _answer_claim_score(claim, claim.clinical_aspect, claim.statement)
    score += {
        "guideline": 18,
        "systematic_review": 15,
        "randomized_controlled_trial": 12,
        "observational_study": 6,
        "narrative_review": 4,
        "case_report": 2,
        "unknown": 0,
    }.get(source.evidence_type, 0)
    if claim.dose:
        score += 8
    if re.search(r"结论|推荐|建议|相当|尚无统一", claim.statement):
        score += 3
    if re.search(r"t\s*值|P\s*值|病迁延|观目前|察其|对研泼", claim.statement, re.I):
        score -= 8
    score += {
        "direct": 6,
        "indirect_rmpp": 1,
        "indirect_smpp": 1,
        "general_mpp": -1,
        "unclear": -4,
    }.get(source.population_applicability, 0)
    return score


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
    if not bundle.claims:
        return (
            "检索结果中没有直接匹配问题焦点的可验证声明，"
            "因此不进行证据时间更新判断。"
        )
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


def _inventory_applicability_text(sources: tuple[EvidenceSource, ...]) -> str:
    assessed = [
        source
        for source in sources
        if source.population_applicability != "not_assessed"
    ]
    if not assessed:
        return ""
    counts = Counter(source.population_applicability for source in assessed)
    parts = [
        f"{_APPLICABILITY_LABELS[key]} {counts[key]} 项"
        for key in _APPLICABILITY_LABELS
        if counts[key]
    ]
    return " 人群适用性盘点：" + "、".join(parts) + "。"


def _applicability_note(sources: tuple[EvidenceSource, ...]) -> str:
    assessed = [
        source
        for source in sources
        if source.population_applicability != "not_assessed"
    ]
    if not assessed:
        return ""
    direct = [source for source in assessed if source.population_applicability == "direct"]
    indirect = [
        source
        for source in assessed
        if source.population_applicability in {"indirect_rmpp", "indirect_smpp"}
    ]
    if not direct and indirect:
        citations = "".join(f"[{source.source_number}]" for source in indirect[:3])
        return (
            f"**适用性说明：**本轮核心证据主要来自相邻临床人群，属于间接外推{citations}。"
            "可用于辅助判断，但不能等同于目标人群的直接研究，结论确定性应相应下调。"
        )
    if direct and indirect:
        return (
            f"**适用性说明：**检索结果同时包含 {len(direct)} 项目标人群直接证据和 "
            f"{len(indirect)} 项相邻人群间接证据；综合时以前者为主，后者仅作补充。"
        )
    if direct:
        return f"**适用性说明：**本轮包含 {len(direct)} 项目标人群直接证据。"
    return "**适用性说明：**现有来源主要是一般MPP背景证据，目标人群适用性仍需核对。"


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
    all_sources = tuple(
        source for sources in grouped.values() for source in sources
    )
    assessed = [
        source
        for source in all_sources
        if source.population_applicability != "not_assessed"
    ]
    if assessed and not any(
        source.population_applicability == "direct" for source in assessed
    ):
        gaps.append("缺少目标人群直接证据，当前结论主要依赖间接外推")
    if not directions:
        gaps.append("未找到直接匹配问题焦点的可验证声明")
    if missing:
        gaps.append("未检索到" + "、".join(missing))
    if directions["uncertain"]:
        gaps.append(f"{directions['uncertain']} 项声明的效应方向不确定")
    if relation_counts["conflicts"]:
        gaps.append(f"存在 {relation_counts['conflicts']} 条冲突关系")
    if not gaps:
        gaps.append("仍需核对全文质量、偏倚风险、随访长度和目标人群适用性")
    return "；".join(gaps) + "。"
