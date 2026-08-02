# -*- coding: utf-8 -*-
"""Compatibility facade for the demo graph and the real literature service."""

from __future__ import annotations

import ipaddress
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from graph_rag_core import EVIDENCE_ORDER, TYPE_LABELS, EvidenceGraph
from knowledge_base.errors import KnowledgeBaseError
from knowledge_base.models import SearchFilters


DEFAULT_CLINICAL_QUESTION = "重症支原体肺炎（SMPP）儿童是否应常规使用糖皮质激素？"
MODEL_CONFIG_KEYS = frozenset({"base_url", "api_key", "model_name"})
MODEL_CONFIG_MAX_LENGTHS = {
    "base_url": 2048,
    "api_key": 4096,
    "model_name": 256,
}
HOST_LABEL_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
)


class APIError(ValueError):
    """Stable user-facing API error without provider or filesystem details."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def as_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message}}


class GraphRAGWebService:
    def __init__(
        self,
        graph: EvidenceGraph,
        *,
        knowledge_service: Any | None = None,
        job_manager: Any | None = None,
    ) -> None:
        self.graph = graph
        self.knowledge_service = knowledge_service
        self.job_manager = job_manager

    def health(self) -> dict[str, Any]:
        self.graph.validate()
        knowledge_base = self.knowledge_base_status()
        model = self.model_status()
        return {
            "status": "ok" if knowledge_base.get("status") == "ready" else "degraded",
            "demo": {
                "graph_valid": True,
                "nodes": len(self.graph.nodes),
                "edges": len(self.graph.edges),
            },
            "knowledge_base": knowledge_base,
            "model": model,
        }

    def evidence_catalog(self) -> dict[str, Any]:
        ranked = self.graph.rank(self.graph.nodes)
        counts = Counter(node["evidence_type"] for node in ranked)
        return {
            "default_question": DEFAULT_CLINICAL_QUESTION,
            "data_source": "demo",
            "nodes": [dict(node) for node in ranked],
            "edges": [_normalized_demo_edge(edge) for edge in self.graph.edges],
            "evidence_types": [
                {
                    "key": evidence_type,
                    "label": TYPE_LABELS[evidence_type],
                    "count": counts[evidence_type],
                }
                for evidence_type in EVIDENCE_ORDER
            ],
        }

    def knowledge_base_status(self) -> dict[str, Any]:
        if self.knowledge_service is None:
            return {"status": "not_built", "metadata_records": 0}
        operation = getattr(self.knowledge_service, "knowledge_base_status", None)
        if not callable(operation):
            operation = self.knowledge_service.health
        return self._call_knowledge(operation)

    def model_status(self) -> dict[str, Any]:
        if self.knowledge_service is None:
            return {"configured": False, "chat_model": ""}
        return self._call_knowledge(self.knowledge_service.model_status)

    def query(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise APIError("invalid_json", "请求体必须是 JSON 对象。")
        if any(
            key in payload for key in ("model", "api_key", "base_url", "model_name")
        ):
            raise APIError(
                "client_model_config_forbidden",
                "请使用 model_config 提供单次查询的模型配置。",
            )
        model_config = (
            _parse_model_config(payload["model_config"])
            if "model_config" in payload
            else None
        )
        question = " ".join(str(payload.get("question", "")).split())
        if not question:
            raise APIError("empty_question", "请输入临床问题。")
        filters = self._search_filters(payload)

        status = self.knowledge_base_status()
        if self.knowledge_service is not None and status.get("status") == "ready":
            if model_config is None:
                return self._call_knowledge(
                    self.knowledge_service.query, question, filters
                )
            return self._call_knowledge(
                self.knowledge_service.query,
                question,
                filters,
                model_config=model_config,
            )
        return self._demo_fallback(question, filters)

    def analyze(
        self,
        question: str,
        evidence_types: Sequence[str] | None = None,
        model_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": question,
            "evidence_types": list(evidence_types or []),
        }
        if model_config is not None:
            payload["model_config"] = model_config
        return self.query(payload)

    def document_detail(self, document_id: str) -> dict[str, Any]:
        if self.knowledge_service is None:
            raise APIError("document_not_found", "文献不存在。", status=404)
        return self._call_knowledge(
            self.knowledge_service.document_detail, document_id
        )

    def resolve_pdf(self, document_id: str) -> Path:
        if self.knowledge_service is None:
            raise APIError("pdf_unavailable", "文献 PDF 不可用。", status=404)
        return self._call_knowledge(self.knowledge_service.resolve_pdf, document_id)

    def job_status(self, job_id: str) -> dict[str, Any]:
        if self.job_manager is None:
            raise APIError("job_not_found", "构建任务不存在。", status=404)
        return self._call_knowledge(self.job_manager.get, job_id).as_dict()

    def start_build(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.job_manager is None:
            raise APIError("build_unavailable", "知识库构建服务不可用。", status=503)
        if any(
            key in payload
            for key in ("model", "api_key", "base_url", "model_name", "model_config")
        ):
            raise APIError(
                "client_model_config_forbidden",
                "模型配置只能由服务器环境变量提供。",
            )
        confirmation = payload.get("confirm_embedding_cost", False)
        if not isinstance(confirmation, bool):
            raise APIError("invalid_embedding_confirmation", "嵌入费用确认字段无效。")
        full_confirmation = payload.get("confirm_full_embedding_cost", False)
        if not isinstance(full_confirmation, bool):
            raise APIError("invalid_embedding_confirmation", "完整嵌入确认字段无效。")
        embedding_limit = payload.get("embedding_limit")
        if confirmation and not full_confirmation and (
            isinstance(embedding_limit, bool)
            or not isinstance(embedding_limit, int)
            or embedding_limit < 1
            or embedding_limit > 32
        ):
            raise APIError(
                "embedding_probe_limit_required",
                "首次向量化必须限制为最多 32 条文本。",
            )
        if full_confirmation and not confirmation:
            raise APIError(
                "invalid_embedding_confirmation",
                "完整向量化需要同时确认嵌入费用。",
            )
        options = {
            key: payload[key]
            for key in (
                "document_limit",
                "embedding_limit",
                "confirm_full_embedding_cost",
                "stage",
            )
            if key in payload
        }
        record = self._call_knowledge(
            self.job_manager.start_build,
            confirm_embedding_cost=confirmation,
            options=options,
        )
        return record.as_dict()

    def retry_build(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if "stage" not in payload:
            raise APIError("retry_stage_required", "请选择需要重试的阶段。")
        return self.start_build(payload)

    def pause_build(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.job_manager is None:
            raise APIError("build_unavailable", "知识库构建服务不可用。", status=503)
        job_id = str(payload.get("job_id", "")).strip()
        if not job_id:
            latest = self.job_manager.latest()
            if latest is None:
                raise APIError("job_not_found", "构建任务不存在。", status=404)
            job_id = latest.job_id
        return self._call_knowledge(self.job_manager.pause, job_id).as_dict()

    def _demo_fallback(
        self, question: str, filters: SearchFilters
    ) -> dict[str, Any]:
        retrieved = self.graph.retrieve(question)
        if filters.evidence_types:
            retrieved = [
                node
                for node in retrieved
                if node["evidence_type"] in filters.evidence_types
            ]
        if filters.year_from is not None:
            retrieved = [
                node for node in retrieved if int(node.get("year", 0)) >= filters.year_from
            ]
        if filters.year_to is not None:
            retrieved = [
                node for node in retrieved if int(node.get("year", 9999)) <= filters.year_to
            ]
        ranked = self.graph.rank(retrieved)
        visible_ids = {node["id"] for node in ranked}
        edges = [
            _normalized_demo_edge(edge)
            for edge in self.graph.edges
            if edge["source"] in visible_ids and edge["target"] in visible_ids
        ]
        sources = [
            {
                "source_number": index,
                "document_id": str(node["id"]),
                "title": str(node.get("title", node["id"])),
                "evidence_type": node["evidence_type"],
                "year": node.get("year"),
                "data_source": "demo",
            }
            for index, node in enumerate(ranked, start=1)
        ]
        inventory = "；".join(
            f"{TYPE_LABELS[key]} {value} 项"
            for key, value in Counter(
                node["evidence_type"] for node in ranked
            ).items()
        ) or "没有命中示例证据"
        citation = "[1]" if sources else ""
        return {
            "question": question,
            "mode": "demo",
            "degraded_reason": "knowledge_base_not_built",
            "filters": {
                "evidence_types": sorted(filters.evidence_types),
                "year_from": filters.year_from,
                "year_to": filters.year_to,
                "fulltext_only": filters.fulltext_only,
            },
            "reasoning_steps": [
                {
                    "title": "第一步：示例证据盘点",
                    "body": f"当前展示静态示例：{inventory}。{citation}",
                    "source_ids": [1] if sources else [],
                }
            ],
            "answer_markdown": "当前真实知识库尚未构建，不能据此形成真实语料结论。",
            "model_used": False,
            "model_name": "",
            "model_error": "knowledge_base_not_built",
            "summary": {"evidence_count": len(sources), "relation_count": len(edges)},
            "sources": sources,
            "graph": {"nodes": [dict(node) for node in ranked], "edges": edges},
            "retrieval_stats": {"candidate_count": len(ranked)},
            "data_source": "demo",
            "warning": "当前结果来自内置示例图，不是对本地文献库的检索结果。",
        }

    @staticmethod
    def _search_filters(payload: Mapping[str, Any]) -> SearchFilters:
        raw_types = payload.get("evidence_types", [])
        if not isinstance(raw_types, list) or any(
            not isinstance(value, str) for value in raw_types
        ):
            raise APIError("invalid_evidence_type", "证据类型必须使用字符串数组。")
        evidence_types = frozenset(raw_types)
        invalid = evidence_types.difference(EVIDENCE_ORDER)
        if invalid:
            raise APIError(
                "invalid_evidence_type",
                "不支持的证据类型：" + "，".join(sorted(invalid)),
            )
        year_from = _optional_year(payload.get("year_from"), "year_from")
        year_to = _optional_year(payload.get("year_to"), "year_to")
        if year_from is not None and year_to is not None and year_from > year_to:
            raise APIError("invalid_year_range", "起始年份不能晚于结束年份。")
        fulltext_only = payload.get("fulltext_only", False)
        if not isinstance(fulltext_only, bool):
            raise APIError("invalid_fulltext_filter", "全文筛选字段无效。")
        return SearchFilters(evidence_types, year_from, year_to, fulltext_only)

    @staticmethod
    def _call_knowledge(operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except APIError:
            raise
        except KnowledgeBaseError as error:
            raise _api_error_from_knowledge(error) from None


def _optional_year(value: object, name: str) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise APIError("invalid_year", f"{name} 必须是整数年份。")
    if value < 1900 or value > 2100:
        raise APIError("invalid_year", f"{name} 超出允许范围。")
    return value


def _parse_model_config(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != MODEL_CONFIG_KEYS:
        raise _invalid_model_config()

    normalized: dict[str, str] = {}
    for key, max_length in MODEL_CONFIG_MAX_LENGTHS.items():
        raw_value = value[key]
        if not isinstance(raw_value, str):
            raise _invalid_model_config()
        clean_value = raw_value.strip()
        if not clean_value or len(clean_value) > max_length:
            raise _invalid_model_config()
        normalized[key] = clean_value

    normalized["base_url"] = _normalize_model_base_url(normalized["base_url"])
    return normalized


def _normalize_model_base_url(base_url: str) -> str:
    if (
        "?" in base_url
        or "#" in base_url
        or "\\" in base_url
        or any(
            character.isspace() or unicodedata.category(character) == "Cc"
            for character in base_url
        )
    ):
        raise _invalid_model_config()
    try:
        parsed = urlsplit(base_url)
        parsed.port
    except ValueError:
        raise _invalid_model_config() from None

    hostname = parsed.hostname
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise _invalid_model_config()
    if not _is_valid_model_hostname(hostname):
        raise _invalid_model_config()
    if parsed.scheme.lower() == "http" and not _is_loopback_host(hostname):
        raise _invalid_model_config()
    return base_url.rstrip("/")


def _is_valid_model_hostname(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass

    try:
        ascii_hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    if len(ascii_hostname) > 253:
        return False

    labels = ascii_hostname.split(".")
    return all(
        label
        and len(label) <= 63
        and not label.startswith("-")
        and not label.endswith("-")
        and all(character in HOST_LABEL_CHARACTERS for character in label)
        for label in labels
    )


def _is_loopback_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _invalid_model_config() -> APIError:
    return APIError("invalid_model_config", "模型配置无效。")


def _api_error_from_knowledge(error: KnowledgeBaseError) -> APIError:
    if error.code.endswith("_not_found"):
        status = 404
    elif error.code in {"build_already_running", "job_not_running"}:
        status = 409
    elif error.code in {
        "retrieval_unavailable",
        "chat_unavailable",
        "build_unavailable",
    }:
        status = 503
    elif error.code in {"chat_auth_failed", "model_auth_failed"}:
        status = 502
    elif error.code in {"pdf_unavailable"}:
        status = 404
    else:
        status = 400
    return APIError(error.code, error.message, status=status)


def _normalized_demo_edge(edge: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(edge)
    if normalized.get("relation") == "confirms":
        normalized["relation"] = "supports"
    return normalized
