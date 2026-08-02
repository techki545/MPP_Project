from __future__ import annotations

import pytest

from knowledge_base.graph_builder import EvidenceClaim, LocalGraphBuilder


def claim(
    identifier: str,
    evidence_type: str,
    year: int,
    direction: str,
    outcome: str,
    *,
    safety: bool = False,
    cutoff: int | None = None,
    dose: str = "",
    population: str = "SMPP children",
    intervention: str = "methylprednisolone",
    clinical_aspect: str = "effectiveness",
    evidence_role: str = "core",
) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=identifier,
        document_id=f"doc-{identifier}",
        evidence_type=evidence_type,
        year=year,
        evidence_cutoff_year=cutoff,
        population=population,
        intervention=intervention,
        comparator="antibiotics or another dose",
        outcome=outcome,
        direction=direction,
        safety_signal=safety,
        dose=dose,
        statement="source-grounded statement",
        source_chunk_ids=(f"chunk-{identifier}",),
        source_quote="verbatim source excerpt",
        clinical_aspect=clinical_aspect,
        evidence_role=evidence_role,
    )


def test_new_rct_updates_old_guideline_and_case_adds_caution() -> None:
    guideline = claim("g", "guideline", 2023, "supports", "lung injury", cutoff=2022)
    rct = claim("r", "randomized_controlled_trial", 2025, "supports", "lung injury")
    case = claim(
        "c", "case_report", 2025, "uncertain", "pulmonary embolism", safety=True
    )

    graph = LocalGraphBuilder().build(
        "Should steroids be used?", [guideline, rct, case]
    )

    relations = {(edge.source, edge.target, edge.relation) for edge in graph.edges}
    assert ("r", "g", "updates") in relations
    assert any(edge.relation == "cautions" and edge.source == "c" for edge in graph.edges)


def test_comparable_opposite_directions_create_conflict() -> None:
    first = claim("a", "randomized_controlled_trial", 2024, "supports", "fever duration")
    second = claim("b", "randomized_controlled_trial", 2025, "opposes", "fever duration")

    graph = LocalGraphBuilder().build("question", [first, second])

    assert any(
        edge.source == "b" and edge.target == "a" and edge.relation == "conflicts"
        for edge in graph.edges
    )


def test_different_dose_creates_supplement_instead_of_support() -> None:
    low = claim(
        "low",
        "randomized_controlled_trial",
        2025,
        "supports",
        "fever duration",
        dose="2 mg/kg/day",
    )
    high = claim(
        "high",
        "randomized_controlled_trial",
        2024,
        "supports",
        "fever duration",
        dose="10 mg/kg/day",
    )

    graph = LocalGraphBuilder().build("question", [low, high])
    pair_edges = [
        edge for edge in graph.edges if {edge.source, edge.target} == {"low", "high"}
    ]

    assert [edge.relation for edge in pair_edges] == ["supplements"]


def test_insufficient_population_overlap_creates_no_pair_relation() -> None:
    children = claim("child", "guideline", 2024, "supports", "fever duration")
    adults = claim(
        "adult",
        "randomized_controlled_trial",
        2025,
        "supports",
        "fever duration",
        population="adult MPP patients",
    )

    graph = LocalGraphBuilder().build("question", [children, adults])

    assert not any(
        {edge.source, edge.target} == {"child", "adult"} for edge in graph.edges
    )


def test_claim_without_source_chunks_is_rejected() -> None:
    with pytest.raises(ValueError, match="source chunk"):
        EvidenceClaim(
            claim_id="bad",
            document_id="doc-bad",
            evidence_type="unknown",
            year=None,
            population="children",
            intervention="steroid",
            comparator="",
            outcome="fever",
            direction="uncertain",
            statement="unsupported",
            source_chunk_ids=(),
        )


def test_graph_json_uses_only_allowed_relations_and_has_traceability() -> None:
    first = claim("a", "systematic_review", 2024, "supports", "fever duration")
    second = claim("b", "randomized_controlled_trial", 2025, "supports", "fever duration")

    payload = LocalGraphBuilder().build("question", [first, second]).as_dict()

    assert {node["node_type"] for node in payload["nodes"]} >= {
        "question",
        "document",
        "claim",
        "outcome",
    }
    assert all(
        edge["relation"]
        in {
            "supports",
            "updates",
            "supplements",
            "confirms",
            "conflicts",
            "cautions",
        }
        for edge in payload["edges"]
    )
    assert all(edge["source_claim_ids"] and edge["rationale"] for edge in payload["edges"])
    assert "confirms" in str(payload)


@pytest.mark.parametrize(
    (
        "source_type",
        "target_type",
        "source_year",
        "target_year",
        "source_direction",
        "target_direction",
        "aspect",
        "expected",
    ),
    [
        (
            "systematic_review",
            "guideline",
            2022,
            2023,
            "supports",
            "supports",
            "overall",
            "supports",
        ),
        (
            "randomized_controlled_trial",
            "systematic_review",
            2025,
            2020,
            "supports",
            "supports",
            "dose",
            "confirms",
        ),
        (
            "randomized_controlled_trial",
            "guideline",
            2025,
            2023,
            "supports",
            "supports",
            "dose",
            "updates",
        ),
        (
            "observational_study",
            "systematic_review",
            2024,
            2022,
            "supports",
            "supports",
            "applicability",
            "supplements",
        ),
        (
            "randomized_controlled_trial",
            "systematic_review",
            2025,
            2024,
            "opposes",
            "supports",
            "effectiveness",
            "conflicts",
        ),
        (
            "case_report",
            "guideline",
            2025,
            2023,
            "uncertain",
            "supports",
            "safety",
            "cautions",
        ),
    ],
)
def test_first_demo_relation_vocabulary(
    source_type: str,
    target_type: str,
    source_year: int,
    target_year: int,
    source_direction: str,
    target_direction: str,
    aspect: str,
    expected: str,
) -> None:
    source = claim(
        "source",
        source_type,
        source_year,
        source_direction,
        aspect,
        safety=expected == "cautions",
        clinical_aspect=aspect,
        evidence_role="boundary" if expected == "cautions" else "core",
    )
    target = claim(
        "target",
        target_type,
        target_year,
        target_direction,
        aspect,
        clinical_aspect=aspect,
        evidence_role="core",
    )

    graph = LocalGraphBuilder().build("question", [source, target])

    assert any(
        edge.source == "source"
        and edge.target == "target"
        and edge.relation == expected
        for edge in graph.edges
    )


def test_newer_study_with_different_clinical_aspect_cannot_update_guideline() -> None:
    guideline = claim(
        "guide",
        "guideline",
        2023,
        "supports",
        "overall treatment",
        clinical_aspect="overall",
    )
    rct = claim(
        "trial",
        "randomized_controlled_trial",
        2025,
        "supports",
        "dose response",
        clinical_aspect="dose",
    )

    graph = LocalGraphBuilder().build("question", [guideline, rct])

    assert not any(edge.relation == "updates" for edge in graph.edges)
