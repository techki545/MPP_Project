# -*- coding: utf-8 -*-
"""Application service for the interactive evidence Graph RAG web demo."""

from __future__ import annotations

from collections import Counter
import json
from typing import Callable, Dict, List, Sequence
import urllib.error
import urllib.parse
import urllib.request

from graph_rag_core import EVIDENCE_ORDER, TYPE_LABELS, EvidenceGraph
from MycoplasmaPneumonia.llm_client import extract_json_object


ModelGateway = Callable[[Dict, str, Dict], Dict]
DEFAULT_CLINICAL_QUESTION = "重症支原体肺炎（SMPP）儿童是否应常规使用糖皮质激素？"


class APIError(ValueError):
    """Stable, user-facing API error without sensitive request data."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def as_dict(self) -> Dict:
        return {"error": {"code": self.code, "message": self.message}}


class GraphRAGWebService:
    """Build structured, deterministic analysis from an evidence graph."""

    def __init__(
        self,
        graph: EvidenceGraph,
        model_gateway: ModelGateway | None = None,
    ):
        self.graph = graph
        self.model_gateway = model_gateway or call_openai_compatible

    def health(self) -> Dict:
        self.graph.validate()
        return {
            "status": "ok",
            "graph_valid": True,
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
        }

    def evidence_catalog(self) -> Dict:
        ranked = self.graph.rank(self.graph.nodes)
        counts = Counter(node["evidence_type"] for node in ranked)
        return {
            "default_question": DEFAULT_CLINICAL_QUESTION,
            "nodes": [dict(node) for node in ranked],
            "edges": [dict(edge) for edge in self.graph.edges],
            "evidence_types": [
                {
                    "key": evidence_type,
                    "label": TYPE_LABELS[evidence_type],
                    "count": counts[evidence_type],
                }
                for evidence_type in EVIDENCE_ORDER
            ],
        }

    def analyze_local(
        self,
        question: str,
        evidence_types: Sequence[str] | None,
    ) -> Dict:
        clean_question = str(question or "").strip()
        if not clean_question:
            raise APIError("empty_question", "请输入临床问题。")

        selected_types = self._validate_evidence_types(evidence_types)
        retrieved = self.graph.retrieve(clean_question)
        if selected_types is not None:
            retrieved = [
                node for node in retrieved if node["evidence_type"] in selected_types
            ]
        ranked = self.graph.rank(retrieved)
        visible_ids = {node["id"] for node in ranked}
        edges = [
            dict(edge)
            for edge in self.graph.edges
            if edge["source"] in visible_ids and edge["target"] in visible_ids
        ]
        return {
            "question": clean_question,
            "evidence": [dict(node) for node in ranked],
            "edges": edges,
            "reasoning_steps": self._build_reasoning_steps(ranked, edges),
            "summary": self._build_summary(ranked, edges),
        }

    def analyze(
        self,
        question: str,
        evidence_types: Sequence[str] | None,
        model_config: Dict,
    ) -> Dict:
        local_result = self.analyze_local(question, evidence_types)
        config = self._validate_model_config(model_config)
        payload = {
            "question": local_result["question"],
            "ranked_evidence": local_result["evidence"],
            "evidence_edges": local_result["edges"],
            "pyramid_order": list(EVIDENCE_ORDER),
            "reasoning_steps": local_result["reasoning_steps"],
            "output_constraints": [
                "仅使用提供的证据，不得虚构文献、数据或引用。",
                "明确回答临床问题，并说明证据一致性、剂量与安全性。",
                "说明该输出是研究演示，最终决策需由临床专业人员完成。",
                "只返回 JSON 对象，键名为 final_answer_markdown。",
            ],
        }
        system_prompt = (
            "你是儿童肺炎支原体肺炎循证决策 Graph RAG 的回答生成器。"
            "程序已经完成证据检索、金字塔排序、时间更新和关系检查。"
            "你只能根据提供的证据生成综合回答，不得补充未提供的研究或引用。"
        )
        try:
            generated = self._invoke_model(config, system_prompt, payload)
            answer_value = generated.get("final_answer_markdown")
            if not isinstance(answer_value, str) or not answer_value.strip():
                raise APIError(
                    "invalid_model_response",
                    "大模型未返回有效的结构化综合回答。",
                    status=502,
                )
            answer = answer_value.strip()
        except APIError as exc:
            return {
                **local_result,
                "answer_markdown": "",
                "model_name": config["model_name"],
                "model_used": False,
                "model_error": {
                    "code": exc.code,
                    "message": exc.message,
                },
            }
        return {
            **local_result,
            "answer_markdown": answer,
            "model_name": config["model_name"],
            "model_used": True,
        }

    def test_model(self, model_config: Dict) -> Dict:
        config = self._validate_model_config(model_config)
        generated = self._invoke_model(
            config,
            "你正在执行连接测试。只返回 JSON 对象，键名为 final_answer_markdown。",
            {
                "task": "connection_test",
                "instruction": "将 final_answer_markdown 设置为连接正常。",
            },
        )
        if not str(generated.get("final_answer_markdown", "")).strip():
            raise APIError(
                "invalid_model_response",
                "模型已响应，但没有返回预期的 JSON 内容。",
                status=502,
            )
        return {
            "connected": True,
            "model_name": config["model_name"],
        }

    def _invoke_model(self, config: Dict, system_prompt: str, payload: Dict) -> Dict:
        try:
            generated = self.model_gateway(config, system_prompt, payload)
        except APIError:
            raise
        except Exception as exc:
            raise APIError(
                "model_request_failed",
                "大模型请求失败，请检查 API Key、Base URL、模型名称和网络连接。",
                status=502,
            ) from exc
        if not isinstance(generated, dict):
            raise APIError(
                "invalid_model_response",
                "大模型返回格式不是 JSON 对象。",
                status=502,
            )
        return generated

    @staticmethod
    def _validate_model_config(model_config: Dict) -> Dict:
        if not isinstance(model_config, dict):
            raise APIError("invalid_model_config", "请填写完整的大模型配置。")
        config = {
            "api_key": str(model_config.get("api_key", "")).strip(),
            "base_url": str(model_config.get("base_url", "")).strip().rstrip("/"),
            "model_name": str(model_config.get("model_name", "")).strip(),
        }
        missing = [name for name, value in config.items() if not value]
        if missing:
            raise APIError(
                "invalid_model_config",
                "请填写完整的大模型配置：" + "、".join(missing) + "。",
            )
        parsed = urllib.parse.urlparse(config["base_url"])
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise APIError(
                "invalid_model_config",
                "Base URL 必须是有效的 HTTP 或 HTTPS 地址。",
            )
        return config

    @staticmethod
    def _validate_evidence_types(
        evidence_types: Sequence[str] | None,
    ) -> set[str] | None:
        if evidence_types is None:
            return None
        if isinstance(evidence_types, (str, bytes)):
            raise APIError(
                "invalid_evidence_type",
                "证据类型必须使用数组格式。",
            )
        selected = {str(item) for item in evidence_types}
        invalid = sorted(selected.difference(EVIDENCE_ORDER))
        if invalid:
            raise APIError(
                "invalid_evidence_type",
                "Unsupported evidence type: " + ", ".join(invalid),
            )
        return selected

    @staticmethod
    def _build_reasoning_steps(nodes: Sequence[Dict], edges: Sequence[Dict]) -> List[Dict]:
        by_type = {
            evidence_type: [
                node for node in nodes if node["evidence_type"] == evidence_type
            ]
            for evidence_type in EVIDENCE_ORDER
        }
        counts = Counter(node["evidence_type"] for node in nodes)
        relation_counts = Counter(edge["relation"] for edge in edges)

        inventory = "；".join(
            f"{TYPE_LABELS[evidence_type]} {counts[evidence_type]} 条"
            for evidence_type in EVIDENCE_ORDER
            if counts[evidence_type]
        ) or "当前筛选条件下没有命中证据"

        guideline_nodes = by_type["guideline"]
        review_nodes = by_type["systematic_review"]
        rct_nodes = by_type["randomized_controlled_trial"]
        lower_nodes = (
            by_type["observational_study"]
            + by_type["narrative_review"]
            + by_type["case_report"]
        )

        return [
            {
                "key": "inventory",
                "title": "第一步：检索并盘点证据库存",
                "body": (
                    f"命中证据：{inventory}。系统先按证据金字塔确定基础权重，"
                    "再检查较新的下级证据能否补充上级证据的空白。"
                ),
                "node_ids": [node["id"] for node in nodes],
            },
            {
                "key": "guidelines",
                "title": "第二步：优先查看指南",
                "body": _summarize_nodes(
                    guideline_nodes,
                    "指南确定推荐大方向，并检查制定时间与剂量共识是否存在空白。",
                    "当前筛选条件下未保留指南证据。",
                ),
                "node_ids": [node["id"] for node in guideline_nodes],
            },
            {
                "key": "reviews",
                "title": "第三步：查阅系统综述和 Meta 分析",
                "body": _summarize_nodes(
                    review_nodes,
                    "系统综述用于检验指南方向是否稳定，并补充剂量和安全性细节。",
                    "当前筛选条件下未保留系统综述证据。",
                ),
                "node_ids": [node["id"] for node in review_nodes],
            },
            {
                "key": "rct",
                "title": "第四步：聚焦关键 RCT",
                "body": _summarize_nodes(
                    rct_nodes,
                    (
                        "较新的高质量 RCT 可针对指南尚未解决的局部问题形成更新。"
                        f"当前识别到 {relation_counts.get('updates', 0)} 条更新关系。"
                    ),
                    "当前筛选条件下未保留 RCT 证据。",
                ),
                "node_ids": [node["id"] for node in rct_nodes],
            },
            {
                "key": "lower_evidence",
                "title": "第五步：查看观察性研究和病例报告",
                "body": _summarize_nodes(
                    lower_nodes,
                    "下级证据用于补充真实世界结局、安全性信号和边界条件。",
                    "当前筛选条件下未保留下级证据。",
                ),
                "node_ids": [node["id"] for node in lower_nodes],
            },
            {
                "key": "synthesis",
                "title": "第六步：合成判断",
                "body": (
                    f"共纳入 {len(nodes)} 条证据、{len(edges)} 条关系。"
                    f"支持关系 {relation_counts.get('supports', 0)} 条，"
                    f"补充关系 {relation_counts.get('supplements', 0)} 条，"
                    f"更新关系 {relation_counts.get('updates', 0)} 条，"
                    f"确认关系 {relation_counts.get('confirms', 0)} 条，"
                    f"警示关系 {relation_counts.get('cautions', 0)} 条。"
                ),
                "node_ids": [node["id"] for node in nodes],
            },
        ]

    @staticmethod
    def _build_summary(nodes: Sequence[Dict], edges: Sequence[Dict]) -> Dict:
        relations = Counter(edge["relation"] for edge in edges)
        high_level_types = {
            "guideline",
            "systematic_review",
            "randomized_controlled_trial",
        }
        supporting_nodes = [
            node
            for node in nodes
            if node["evidence_type"] in high_level_types
            and str(node.get("recommendation_direction", "")).startswith("support")
        ]
        low_dose_supported = any(
            node.get("dose_signal")
            in {
                "low_dose_preferred",
                "high_dose_not_preferred",
                "avoid_unnecessary_high_dose",
            }
            for node in supporting_nodes
        )
        if not supporting_nodes:
            consistency = "证据不足或方向不确定"
            conclusion = (
                "当前筛选证据不足以形成常规治疗推荐，需恢复高层级证据或扩大证据范围后再判断。"
            )
        elif low_dose_supported:
            consistency = "证据方向总体一致"
            conclusion = (
                "支持在抗菌治疗基础上使用糖皮质激素，低剂量优先；"
                "当前证据不支持将高剂量作为常规方案。"
            )
        else:
            consistency = "高层级证据支持使用，剂量仍需核对"
            conclusion = (
                "当前高层级证据支持在适用人群中使用糖皮质激素，"
                "但所选证据不足以确定最优剂量。"
            )
        return {
            "evidence_count": len(nodes),
            "relation_count": len(edges),
            "update_count": relations.get("updates", 0),
            "consistency": consistency,
            "conclusion": conclusion,
        }


def _summarize_nodes(
    nodes: Sequence[Dict],
    introduction: str,
    empty_message: str,
) -> str:
    if not nodes:
        return empty_message
    findings = "；".join(
        f"{node['citation']}：{node['main_findings']}" for node in nodes
    )
    return f"{introduction} {findings}"


def call_openai_compatible(config: Dict, system_prompt: str, payload: Dict) -> Dict:
    """Call an OpenAI-compatible chat-completions endpoint."""

    request_payload = {
        "model": config["model_name"],
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ],
    }
    raw = ""
    for include_response_format in (True, False):
        attempt_payload = dict(request_payload)
        if include_response_format:
            attempt_payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            url=_chat_completions_url(config["base_url"]),
            data=json.dumps(attempt_payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + config["api_key"],
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            upstream_detail = _read_upstream_error_detail(exc)
            if (
                include_response_format
                and exc.code == 400
                and _is_response_format_unsupported(upstream_detail)
            ):
                continue
            if exc.code in {401, 403}:
                raise APIError(
                    "model_auth_failed",
                    f"模型服务拒绝了 API Key（HTTP {exc.code}），请检查令牌是否有效、是否已过期以及是否允许访问当前模型。",
                    status=502,
                ) from exc
            if exc.code == 400:
                raise APIError(
                    "model_request_rejected",
                    "模型服务拒绝了请求（HTTP 400）。请核对模型名称；若名称正确，则该网关可能不兼容当前 Chat Completions 请求格式。",
                    status=502,
                ) from exc
            raise RuntimeError(f"model endpoint returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("model endpoint is unreachable") from exc
        except TimeoutError as exc:
            raise RuntimeError("model request timed out") from exc

    try:
        response_data = json.loads(raw)
        content = response_data["choices"][0]["message"]["content"]
        return extract_json_object(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("model response could not be parsed") from exc


def _read_upstream_error_detail(error: urllib.error.HTTPError) -> str:
    try:
        return error.read(4096).decode("utf-8", errors="replace").lower()
    except (AttributeError, OSError, UnicodeError):
        return ""


def _is_response_format_unsupported(detail: str) -> bool:
    return "response_format" in detail or "json_object" in detail


def _chat_completions_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean
    return clean + "/chat/completions"
