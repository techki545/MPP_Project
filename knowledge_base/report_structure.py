"""Deterministic six-step evidence reasoning and conclusion-first reporting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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
    bundle: EvidenceBundle,
    grouped: dict[str, tuple[EvidenceSource, ...]],
    relation_counts: Counter[str],
    claims_by_document: dict[str, list[EvidenceClaim]],
) -> str:
    directions = Counter(claim.direction for claim in bundle.claims)
    directional_sources = [
        source.source_number
        for source in bundle.sources
        if any(
            claim.direction != "uncertain"
            for claim in claims_by_document.get(source.document_id, [])
        )
    ]
    citation_suffix = "".join(f"[{number}]" for number in directional_sources)
    if not directions or directions["supports"] + directions["opposes"] == 0:
        conclusion = (
            "**结论：当前可验证声明没有提供足够的方向信息，不能形成肯定或否定建议。**"
        )
    elif directions["supports"] > directions["opposes"] and not relation_counts[
        "conflicts"
    ]:
        conclusion = (
            "**结论：现有检索证据总体支持问题所聚焦的临床判断，但仍需结合"
            f"安全性、适用人群和证据缺口审慎解释。**{citation_suffix}"
        )
    elif directions["opposes"] > directions["supports"] and not relation_counts[
        "conflicts"
    ]:
        conclusion = (
            "**结论：现有检索证据总体不支持问题所述临床判断，且仍需结合"
            f"具体人群和证据局限审慎决策。**{citation_suffix}"
        )
    else:
        conclusion = (
            "**结论：各来源的方向并不一致，当前证据不足以形成单一肯定或"
            f"否定结论。**{citation_suffix}"
        )

    evidence_lines = [
        _claim_line(source, claims_by_document)
        for evidence_type in _EVIDENCE_ORDER
        for source in grouped[evidence_type]
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
        return f"- {claims[0].statement}[{source.source_number}]"
    year = str(source.year) if source.year is not None else "年份未知"
    return f"- 《{source.title}》（{year}）[{source.source_number}]"


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
