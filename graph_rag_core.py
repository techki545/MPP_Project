# -*- coding: utf-8 -*-
"""Minimal evidence-graph RAG core for the pediatric MPP MVP.

The module is intentionally deterministic. It demonstrates the framework
logic first: retrieve evidence nodes, rank them by evidence pyramid, inspect
time-sensitive update edges, then render a transparent clinical answer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
import re
from typing import Dict, Iterable, List, Sequence


EVIDENCE_ORDER = {
    "guideline": 0,
    "systematic_review": 1,
    "randomized_controlled_trial": 2,
    "observational_study": 3,
    "narrative_review": 4,
    "case_report": 5,
}

TYPE_LABELS = {
    "guideline": "指南",
    "systematic_review": "系统综述/Meta 分析",
    "randomized_controlled_trial": "随机对照试验（RCT）",
    "observational_study": "观察性研究",
    "narrative_review": "叙述性综述",
    "case_report": "病例报告",
}

QUALITY_ORDER = {
    "high": 0,
    "moderate": 1,
    "low": 2,
    "very_low": 3,
}


class EvidenceGraph:
    def __init__(self, nodes: Sequence[Dict], edges: Sequence[Dict]):
        self.nodes = list(nodes)
        self.edges = list(edges)

    @classmethod
    def from_files(cls, nodes_path: str, edges_path: str) -> "EvidenceGraph":
        with open(nodes_path, "r", encoding="utf-8") as file:
            nodes = json.load(file)
        with open(edges_path, "r", encoding="utf-8") as file:
            edges = json.load(file)
        return cls(nodes=nodes, edges=edges)

    def validate(self) -> None:
        node_ids = {node.get("id") for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("Evidence node IDs must be unique.")

        required = {"id", "title", "evidence_type", "year", "main_findings"}
        for node in self.nodes:
            missing = sorted(required.difference(node))
            if missing:
                raise ValueError(f"{node.get('id', '<unknown>')} missing fields: {missing}")
            if node["evidence_type"] not in EVIDENCE_ORDER:
                raise ValueError(f"Unsupported evidence type: {node['evidence_type']}")

        for edge in self.edges:
            if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
                raise ValueError(f"Invalid edge endpoint: {edge}")

    def retrieve(self, query: str) -> List[Dict]:
        tokens = _tokenize(query)
        if not tokens:
            return list(self.nodes)

        scored = []
        for node in self.nodes:
            haystack = json.dumps(node, ensure_ascii=False).lower()
            score = sum(1 for token in tokens if token in haystack)
            if score:
                scored.append((score, node))

        if not scored:
            return list(self.nodes)

        scored.sort(
            key=lambda item: (
                -item[0],
                EVIDENCE_ORDER[item[1]["evidence_type"]],
                -int(item[1].get("year", 0)),
                item[1]["id"],
            )
        )
        return [node for _, node in scored]

    def rank(self, nodes: Iterable[Dict]) -> List[Dict]:
        return sorted(
            nodes,
            key=lambda node: (
                EVIDENCE_ORDER[node["evidence_type"]],
                QUALITY_ORDER.get(node.get("quality", "low"), 9),
                -int(node.get("year", 0)),
                -int(node.get("month", 0)),
                node["id"],
            ),
        )

    def edges_by_relation(self, node_ids: Iterable[str]) -> Dict[str, List[Dict]]:
        selected = set(node_ids)
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for edge in self.edges:
            if edge["source"] in selected and edge["target"] in selected:
                grouped[edge["relation"]].append(edge)
        return dict(grouped)


def load_default_graph(base_dir: str | None = None) -> EvidenceGraph:
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    graph = EvidenceGraph.from_files(
        os.path.join(base_dir, "evidence_nodes.json"),
        os.path.join(base_dir, "evidence_edges.json"),
    )
    graph.validate()
    return graph


def build_evidence_payload(question: str, graph: EvidenceGraph) -> Dict:
    retrieved = graph.retrieve(question)
    ranked = graph.rank(retrieved)
    node_ids = [node["id"] for node in ranked]
    return {
        "question": question,
        "ranked_evidence": ranked,
        "evidence_edges": [
            edge
            for edge in graph.edges
            if edge["source"] in node_ids and edge["target"] in node_ids
        ],
        "pyramid_order": [
            "guideline",
            "systematic_review",
            "randomized_controlled_trial",
            "observational_study",
            "narrative_review",
            "case_report",
        ],
    }


def render_evidence_answer(question: str, graph: EvidenceGraph, llm_final_answer: Dict | None = None) -> str:
    retrieved = graph.retrieve(question)
    ranked = graph.rank(retrieved)
    node_ids = [node["id"] for node in ranked]
    relation_groups = graph.edges_by_relation(node_ids)
    by_type = _group_by_type(ranked)

    guideline_nodes = by_type.get("guideline", [])
    review_nodes = by_type.get("systematic_review", [])
    rct_nodes = by_type.get("randomized_controlled_trial", [])
    lower_nodes = (
        by_type.get("observational_study", [])
        + by_type.get("narrative_review", [])
        + by_type.get("case_report", [])
    )

    lines: List[str] = []
    lines.append("## 临床问题")
    lines.append("")
    lines.append(f"**{question}**")
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>第一部分：循证推理过程</summary>")
    lines.append("")
    lines.extend(_render_inventory_step(ranked))
    lines.extend(_render_guideline_step(guideline_nodes, relation_groups))
    lines.extend(_render_review_step(review_nodes))
    lines.extend(_render_rct_step(rct_nodes, relation_groups))
    lines.extend(_render_lower_evidence_step(lower_nodes))
    lines.extend(_render_synthesis_step(ranked, relation_groups))
    lines.append("")
    lines.append("</details>")
    lines.append("")
    lines.extend(_render_final_answer(llm_final_answer))
    lines.append("")
    return "\n".join(lines)


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", text)]


def _group_by_type(nodes: Sequence[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for node in nodes:
        grouped[node["evidence_type"]].append(node)
    return dict(grouped)


def _render_inventory_step(nodes: Sequence[Dict]) -> List[str]:
    counts = Counter(node["evidence_type"] for node in nodes)
    lines = [
        "### 第一步：检索并盘点证据库存",
        "",
        "系统先把问题拆成 PICO 要素：P 为儿童重症肺炎支原体肺炎（SMPP），I 为糖皮质激素/甲泼尼龙，C 为常规抗菌治疗或不同剂量方案，O 为退热、住院时间、长期肺部损害和安全性。",
        "",
        "本次 MVP 证据图中命中的证据如下：",
        "",
    ]
    for evidence_type in EVIDENCE_ORDER:
        count = counts.get(evidence_type, 0)
        if count:
            lines.append(f"- **{TYPE_LABELS[evidence_type]}** {count} 条")
    lines.extend(
        [
            "",
            "排序规则不是简单按年份，而是先按**证据金字塔**确定基础权重，再检查是否存在更新的下级证据能够修正上级证据中的空白或过时部分。",
            "",
        ]
    )
    return lines


def _render_guideline_step(nodes: Sequence[Dict], relation_groups: Dict[str, List[Dict]]) -> List[str]:
    lines = [
        "### 第二步：优先查看指南",
        "",
        "证据金字塔的顶层优先级给到指南和专家共识，因为它们通常已经完成了系统检索、证据评价和推荐形成。",
        "",
    ]
    for node in nodes:
        lines.append(
            f"- **{node['citation']}**：{node['main_findings']}"
        )
    lines.extend(
        [
            "",
            "这里的关键细节是：指南支持“在 SMPP 中使用糖皮质激素”，但对“用多大剂量”没有形成明确共识。因此，指南可以确定大方向，但不能直接解决剂量问题。",
        ]
    )
    update_edges = relation_groups.get("updates", [])
    if update_edges:
        lines.append(
            f"另外，证据图发现 {len(update_edges)} 条“更新”关系：有更新的 RCT 晚于指南制定时间，需要回到原始研究层面对剂量建议进行修正。"
        )
    lines.append("")
    return lines


def _render_review_step(nodes: Sequence[Dict]) -> List[str]:
    lines = [
        "### 第三步：查阅系统综述和 Meta 分析",
        "",
        "第二层证据用于检验指南方向是否稳定，同时补充指南没有展开的细节。",
        "",
    ]
    for node in nodes:
        lines.append(f"- **{node['citation']}**：{node['main_findings']}")
    lines.extend(
        [
            "",
            "两类 Meta 分析给出的方向基本一致：联合使用糖皮质激素有临床获益，但高剂量并没有稳定显示更好疗效，安全性反而更差。这个结论为后续查看 RCT 提供了重点：不是只问“用不用”，而是进一步问“低剂量还是高剂量”。",
            "",
        ]
    )
    return lines


def _render_rct_step(nodes: Sequence[Dict], relation_groups: Dict[str, List[Dict]]) -> List[str]:
    lines = [
        "### 第四步：聚焦关键 RCT",
        "",
        "RCT 位于指南和系统综述之下，但如果 RCT 发布时间更新、设计质量更高，并且正好回答指南留下的问题，就可以对局部结论形成重要更新。",
        "",
    ]
    for node in nodes:
        lines.append(f"- **{node['citation']}**：{node['main_findings']}")
    lines.extend(
        [
            "",
            "这条证据对指南的修正意义很明确：它没有否定糖皮质激素本身，而是把剂量选择从“可能需要高剂量”拉回到“低剂量优先”。高剂量在长期肺部结局上没有优势，却带来更高的高血压风险。",
        ]
    )
    confirm_edges = relation_groups.get("confirms", [])
    if confirm_edges:
        lines.append(
            f"证据图中还存在 {len(confirm_edges)} 条“一致/确认”关系，说明新 RCT 与既往剂量 Meta 分析方向一致。"
        )
    lines.append("")
    return lines


def _render_lower_evidence_step(nodes: Sequence[Dict]) -> List[str]:
    lines = [
        "### 第五步：查看观察性研究和病例报告",
        "",
        "下级证据不用于推翻高质量证据，但可以补充真实世界使用场景、安全性信号和边界条件。",
        "",
    ]
    for node in nodes:
        lines.append(f"- **{node['citation']}**：{node['main_findings']}")
    lines.extend(
        [
            "",
            "因此，下级证据在这里主要承担两个作用：一是补充联合治疗在真实临床中的短期表现；二是提醒极端并发症病例不能机械套用常规路径，需要重新评估并发症和个体化干预。",
            "",
        ]
    )
    return lines


def _render_synthesis_step(nodes: Sequence[Dict], relation_groups: Dict[str, List[Dict]]) -> List[str]:
    lines = [
        "### 第六步：合成判断",
        "",
        "| 证据层次 | 数量 | 结论方向 | 在本问题中的作用 |",
        "|---|---:|---|---|",
    ]
    counts = Counter(node["evidence_type"] for node in nodes)
    role_map = {
        "guideline": ("支持使用", "确定临床推荐的大方向"),
        "systematic_review": ("支持使用，倾向低剂量", "验证指南方向，并补充剂量问题"),
        "randomized_controlled_trial": ("支持低剂量，不支持常规高剂量", "用更新证据修正剂量选择"),
        "observational_study": ("方向一致", "补充真实世界短期结局"),
        "narrative_review": ("补充安全性", "提示治疗时机和长期安全性缺口"),
        "case_report": ("提示边界条件", "提醒复杂并发症需个体化"),
    }
    for evidence_type in EVIDENCE_ORDER:
        count = counts.get(evidence_type, 0)
        if not count:
            continue
        direction, role = role_map[evidence_type]
        lines.append(f"| {TYPE_LABELS[evidence_type]} | {count} | {direction} | {role} |")

    relation_total = sum(len(items) for items in relation_groups.values())
    lines.extend(
        [
            "",
            f"证据图共识别到 {relation_total} 条证据关系，包括 supports、supplements、updates、confirms 和 cautions。整体判断是：各级证据在“是否使用糖皮质激素”上基本一致；在“剂量选择”上，更新的 RCT 可以补充并修正早期指南的空白。",
            "",
        ]
    )
    return lines


def _render_final_answer(llm_final_answer: Dict | None = None) -> List[str]:
    if llm_final_answer and llm_final_answer.get("final_answer_markdown"):
        lines = [
            "## 第二部分：综合回答",
            "",
            str(llm_final_answer["final_answer_markdown"]).strip(),
            "",
        ]
        model = llm_final_answer.get("model")
        if model:
            lines.append(
                f"_大模型辅助生成：{model}；证据检索、证据金字塔排序、时间更新判断和安全性约束由程序规则固定。_"
            )
        return lines

    return [
        "## 第二部分：综合回答",
        "",
        "**结论：对于符合 SMPP 诊断标准的儿童，可以在抗菌治疗基础上联合使用糖皮质激素；剂量上推荐低剂量甲泼尼龙方案，不建议把高剂量或冲击治疗作为常规选择。**",
        "",
        "这个结论来自两条证据链的合并：第一，指南和系统综述均支持在重症或难治性肺炎支原体肺炎中使用糖皮质激素，说明“使用”这个方向比较稳定；第二，2025 年多中心 RCT 对“低剂量 vs 高剂量”给出了更直接、更新的证据，显示低剂量在长期肺部结局上不劣于高剂量，而高剂量带来的高血压等安全性问题更突出。",
        "",
        "因此，MVP 推荐表达为：在明确 SMPP、已有抗菌治疗基础、并完成并发症评估后，可考虑甲泼尼龙 **2 mg/kg/d** 静脉使用，短程后逐步减量；不建议常规使用 **10 mg/kg/d** 这类高剂量方案。若患儿合并肺栓塞、严重免疫异常、消化道出血风险或继发感染风险，则需要重新进入个体化决策流程。",
        "",
        "需要说明的是，这个 demo 的定位是 Graph RAG 风格的循证推荐框架 MVP：当前证据来自内置示例知识库，用于展示检索、证据排序、时间更新和答案组织方式；正式使用前还需要接入标准数据库、做文献去重、质量评价和小范围病例验证。",
    ]
