from __future__ import annotations

from knowledge_base.document_graph import build_document_graph
from knowledge_base.graph_builder import EvidenceClaim, LocalGraphBuilder
from knowledge_base.reporter import EvidenceBundle, EvidenceSource


def source(
    number: int,
    evidence_type: str,
    *,
    title: str | None = None,
    year: int | None = None,
) -> EvidenceSource:
    return EvidenceSource(
        source_number=number,
        document_id=f"doc-{number}",
        title=title or f"Evidence source {number}",
        evidence_type=evidence_type,
        year=year,
        chunk_ids=(f"chunk-{number}",),
        snippets=(f"Grounded excerpt {number}",),
        page_ranges=(str(number),),
        fulltext=True,
        quality="moderate",
    )


def claim(
    number: int,
    evidence_type: str,
    year: int,
    *,
    aspect: str = "effectiveness",
    direction: str = "supports",
    role: str = "core",
) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=f"claim-{number}",
        document_id=f"doc-{number}",
        evidence_type=evidence_type,
        year=year,
        population="children",
        intervention="treatment",
        comparator="usual care",
        outcome=aspect,
        direction=direction,
        statement=f"Grounded claim {number}",
        source_chunk_ids=(f"chunk-{number}",),
        source_quote=f"Grounded excerpt {number}",
        clinical_aspect=aspect,
        evidence_role=role,
    )


def test_document_graph_projects_claim_edges_to_sources() -> None:
    sources = (
        source(1, "guideline", title="Clinical guideline", year=2023),
        source(2, "randomized_controlled_trial", title="New trial", year=2025),
    )
    claims = (
        claim(1, "guideline", 2023, aspect="dose"),
        claim(2, "randomized_controlled_trial", 2025, aspect="dose"),
    )
    claim_graph = LocalGraphBuilder().build("question", claims).as_dict()

    graph = build_document_graph(sources, claims, claim_graph, limit=10)

    assert {node["node_type"] for node in graph["nodes"]} == {"document"}
    assert graph["nodes"][0] == {
        "node_id": "doc-1",
        "node_type": "document",
        "label": "Clinical guideline",
        "payload": {
            "document_id": "doc-1",
            "source_number": 1,
            "title": "Clinical guideline",
            "evidence_type": "guideline",
            "year": 2023,
            "quality": "moderate",
        },
    }
    assert graph["edges"] == [
        {
            "source": "doc-2",
            "target": "doc-1",
            "relation": "updates",
            "source_claim_ids": ["claim-1", "claim-2"],
            "rationale": (
                "A newer trial or review addresses the same clinical aspect as the "
                "earlier guideline and may update its evidence base."
            ),
        }
    ]


def test_document_graph_limits_nodes_and_retains_evidence_type_representatives() -> None:
    evidence_types = (
        "guideline",
        "systematic_review",
        "randomized_controlled_trial",
        "observational_study",
        "narrative_review",
        "case_report",
    )
    sources = tuple(
        source(
            number,
            evidence_types[(number - 1) % len(evidence_types)],
            year=2010 + number,
        )
        for number in range(1, 13)
    )
    claims = tuple(
        claim(
            item.source_number,
            item.evidence_type,
            item.year or 2020,
            role="boundary" if item.evidence_type == "case_report" else "core",
        )
        for item in sources
    )
    claim_graph = LocalGraphBuilder().build("question", claims).as_dict()

    graph = build_document_graph(sources, claims, claim_graph, limit=10)

    node_ids = {node["node_id"] for node in graph["nodes"]}
    represented_types = {
        node["payload"]["evidence_type"] for node in graph["nodes"]
    }
    assert len(graph["nodes"]) == 10
    assert len(graph["edges"]) <= 12
    assert represented_types == set(evidence_types)
    assert all(
        edge["source"] in node_ids and edge["target"] in node_ids
        for edge in graph["edges"]
    )
    assert all(edge["source_claim_ids"] for edge in graph["edges"])


def test_evidence_bundle_preserves_internal_claims_with_document_graph() -> None:
    sources = (source(1, "guideline", year=2023),)
    claims = (claim(1, "guideline", 2023),)
    graph = {"question": "question", "nodes": [], "edges": []}

    bundle = EvidenceBundle(sources=sources, graph=graph, claims=claims)

    assert bundle.claims == claims
    assert bundle.as_dict()["claims"][0]["claim_id"] == "claim-1"
