"""Project the internal claim graph into a compact document evidence graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .graph_builder import ALLOWED_RELATIONS, EvidenceClaim
from .reporter import EvidenceSource


_EVIDENCE_ORDER = (
    "guideline",
    "systematic_review",
    "randomized_controlled_trial",
    "observational_study",
    "narrative_review",
    "case_report",
    "unknown",
)
_EVIDENCE_RANK = {
    evidence_type: rank for rank, evidence_type in enumerate(_EVIDENCE_ORDER)
}
_RELATION_PRIORITY = {
    "conflicts": 0,
    "updates": 1,
    "cautions": 2,
    "supports": 3,
    "confirms": 4,
    "supplements": 5,
}


def build_document_graph(
    sources: Sequence[EvidenceSource],
    claims: Sequence[EvidenceClaim],
    claim_graph: Mapping[str, Any],
    *,
    limit: int = 10,
) -> dict[str, Any]:
    """Return first-demo-compatible document nodes and traceable relations."""
    selected_sources = _select_sources(sources, limit=max(0, limit))
    selected_ids = {source.document_id for source in selected_sources}
    claim_documents = {
        claim.claim_id: claim.document_id
        for claim in claims
        if claim.document_id in selected_ids
    }

    nodes = [
        {
            "node_id": source.document_id,
            "node_type": "document",
            "label": source.title,
            "payload": {
                "document_id": source.document_id,
                "source_number": source.source_number,
                "title": source.title,
                "evidence_type": source.evidence_type,
                "year": source.year,
                "quality": source.quality,
            },
        }
        for source in selected_sources
    ]
    nodes.sort(
        key=lambda node: (
            _EVIDENCE_RANK.get(
                str(node["payload"]["evidence_type"]), len(_EVIDENCE_RANK)
            ),
            int(node["payload"]["source_number"]),
            str(node["node_id"]),
        )
    )

    collapsed: dict[tuple[str, str, str], dict[str, Any]] = {}
    raw_edges = claim_graph.get("edges", [])
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
        raw_edges = []
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, Mapping):
            continue
        source_claim = str(raw_edge.get("source", ""))
        target_claim = str(raw_edge.get("target", ""))
        source_document = claim_documents.get(source_claim)
        target_document = claim_documents.get(target_claim)
        relation = str(raw_edge.get("relation", ""))
        rationale = str(raw_edge.get("rationale", "")).strip()
        if (
            not source_document
            or not target_document
            or source_document == target_document
            or relation not in ALLOWED_RELATIONS
            or not rationale
        ):
            continue
        key = (source_document, target_document, relation)
        entry = collapsed.setdefault(
            key,
            {
                "source": source_document,
                "target": target_document,
                "relation": relation,
                "source_claim_ids": set(),
                "rationale": rationale,
            },
        )
        claim_ids = raw_edge.get("source_claim_ids", ())
        if isinstance(claim_ids, Sequence) and not isinstance(
            claim_ids, (str, bytes)
        ):
            entry["source_claim_ids"].update(
                str(claim_id) for claim_id in claim_ids if str(claim_id).strip()
            )

    edges = []
    for key in sorted(collapsed):
        entry = collapsed[key]
        edges.append(
            {
                "source": entry["source"],
                "target": entry["target"],
                "relation": entry["relation"],
                "source_claim_ids": sorted(entry["source_claim_ids"]),
                "rationale": entry["rationale"],
            }
        )
    edges = _sparse_relation_network(edges, tuple(selected_ids))

    return {
        "question": str(claim_graph.get("question", "")).strip(),
        "nodes": nodes,
        "edges": edges,
    }


def _sparse_relation_network(
    edges: list[dict[str, Any]], node_ids: tuple[str, ...]
) -> list[dict[str, Any]]:
    max_edges = min(len(edges), len(node_ids) + 2)
    if len(edges) <= max_edges:
        return edges
    ordered = sorted(
        edges,
        key=lambda edge: (
            _RELATION_PRIORITY.get(str(edge["relation"]), 99),
            str(edge["source"]),
            str(edge["target"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, str]] = set()

    def add(edge: dict[str, Any]) -> None:
        key = (str(edge["source"]), str(edge["target"]), str(edge["relation"]))
        if key not in selected_keys and len(selected) < max_edges:
            selected.append(edge)
            selected_keys.add(key)

    for relation in _RELATION_PRIORITY:
        representative = next(
            (edge for edge in ordered if edge["relation"] == relation),
            None,
        )
        if representative is not None:
            add(representative)

    parent = {node_id: node_id for node_id in node_ids}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(left: str, right: str) -> bool:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return False
        parent[right_root] = left_root
        return True

    for edge in selected:
        union(str(edge["source"]), str(edge["target"]))
    for edge in ordered:
        if len(selected) >= max_edges:
            break
        if union(str(edge["source"]), str(edge["target"])):
            add(edge)
    for edge in ordered:
        if len(selected) >= max_edges:
            break
        add(edge)
    return sorted(
        selected,
        key=lambda edge: (
            str(edge["source"]),
            str(edge["target"]),
            str(edge["relation"]),
        ),
    )


def _select_sources(
    sources: Sequence[EvidenceSource], *, limit: int
) -> tuple[EvidenceSource, ...]:
    if limit == 0:
        return ()
    ranked = sorted(sources, key=lambda item: (item.source_number, item.document_id))
    by_document: dict[str, EvidenceSource] = {}
    for source in ranked:
        by_document.setdefault(source.document_id, source)
    unique_sources = tuple(by_document.values())

    selected: list[EvidenceSource] = []
    selected_ids: set[str] = set()
    for evidence_type in _EVIDENCE_ORDER:
        representative = next(
            (
                source
                for source in unique_sources
                if source.evidence_type == evidence_type
            ),
            None,
        )
        if representative is not None and len(selected) < limit:
            selected.append(representative)
            selected_ids.add(representative.document_id)
    for source in unique_sources:
        if len(selected) >= limit:
            break
        if source.document_id not in selected_ids:
            selected.append(source)
            selected_ids.add(source.document_id)
    return tuple(selected)
