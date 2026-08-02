from __future__ import annotations

from knowledge_base.document_graph import build_document_graph
from knowledge_base.graph_builder import EvidenceClaim, LocalGraphBuilder
from knowledge_base.report_structure import (
    EXPECTED_STAGE_KEYS,
    compose_deterministic_report,
)
from knowledge_base.reporter import EvidenceBundle, EvidenceSource


def evidence(
    number: int,
    evidence_type: str,
    year: int,
    *,
    direction: str = "supports",
    aspect: str = "effectiveness",
    role: str = "core",
) -> tuple[EvidenceSource, EvidenceClaim]:
    title = f"{evidence_type} source {number}"
    statement = f"Grounded {aspect} finding from source {number}."
    source = EvidenceSource(
        source_number=number,
        document_id=f"doc-{number}",
        title=title,
        evidence_type=evidence_type,
        year=year,
        chunk_ids=(f"chunk-{number}",),
        snippets=(statement,),
        page_ranges=(str(number),),
        fulltext=True,
        quality="moderate",
    )
    claim = EvidenceClaim(
        claim_id=f"claim-{number}",
        document_id=source.document_id,
        evidence_type=evidence_type,
        year=year,
        population="children",
        intervention="clinical intervention",
        comparator="usual care",
        outcome=aspect,
        direction=direction,
        statement=statement,
        source_chunk_ids=source.chunk_ids,
        source_quote=statement,
        clinical_aspect=aspect,
        evidence_role=role,
        safety_signal=aspect == "safety",
    )
    return source, claim


def bundle_from(
    records: tuple[tuple[EvidenceSource, EvidenceClaim], ...]
) -> EvidenceBundle:
    sources = tuple(record[0] for record in records)
    claims = tuple(record[1] for record in records)
    claim_graph = LocalGraphBuilder().build("Clinical question", claims).as_dict()
    document_graph = build_document_graph(sources, claims, claim_graph)
    return EvidenceBundle(sources=sources, graph=document_graph, claims=claims)


def test_deterministic_report_always_has_first_demo_six_steps() -> None:
    bundle = bundle_from(
        (evidence(1, "randomized_controlled_trial", 2025),)
    )

    report = compose_deterministic_report("Clinical question", bundle)

    assert tuple(step["stage_key"] for step in report.analysis_steps) == (
        EXPECTED_STAGE_KEYS
    )
    assert report.analysis_steps[1]["title"] == "第二步：优先查看指南"
    assert "未检索到指南" in report.analysis_steps[1]["body"]
    assert report.final_answer_markdown.startswith("## 综合回答")


def test_deterministic_answer_is_conclusion_first_and_source_grounded() -> None:
    bundle = bundle_from(
        (
            evidence(1, "guideline", 2023, aspect="overall"),
            evidence(2, "systematic_review", 2024, aspect="overall"),
            evidence(3, "randomized_controlled_trial", 2025, aspect="dose"),
            evidence(
                4,
                "case_report",
                2025,
                direction="uncertain",
                aspect="safety",
                role="boundary",
            ),
        )
    )

    report = compose_deterministic_report("Clinical question", bundle)

    first_paragraph = report.final_answer_markdown.split("\n\n", 2)[1]
    assert first_paragraph.startswith("**结论：")
    assert report.final_answer_markdown.index("### 证据链") < (
        report.final_answer_markdown.index("### 时间更新")
    )
    assert report.final_answer_markdown.index("### 时间更新") < (
        report.final_answer_markdown.index("### 安全性与适用边界")
    )
    assert report.final_answer_markdown.index("### 安全性与适用边界") < (
        report.final_answer_markdown.index("### 证据缺口")
    )
    assert "Grounded overall finding from source 1.[1]" in (
        report.final_answer_markdown
    )
    assert "Grounded safety finding from source 4.[4]" in (
        report.final_answer_markdown
    )


def test_uncertain_claims_do_not_create_positive_or_negative_recommendation() -> None:
    bundle = bundle_from(
        (
            evidence(
                1,
                "observational_study",
                2024,
                direction="uncertain",
                aspect="prognosis",
                role="supplement",
            ),
        )
    )

    report = compose_deterministic_report("Which findings predict risk?", bundle)

    assert "不能形成肯定或否定建议" in report.final_answer_markdown
