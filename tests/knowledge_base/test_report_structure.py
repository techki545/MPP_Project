from __future__ import annotations

from dataclasses import replace

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


def test_uncertain_open_claims_report_findings_without_a_recommendation() -> None:
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

    assert "核心预后信息" in report.final_answer_markdown
    assert "倾向于“是”" not in report.final_answer_markdown
    assert "倾向于“否”" not in report.final_answer_markdown
    assert "1 项声明的效应方向不确定" in report.final_answer_markdown


def test_yes_no_question_receives_an_explicit_answer_in_the_conclusion() -> None:
    bundle = bundle_from(
        (evidence(1, "randomized_controlled_trial", 2025),)
    )

    report = compose_deterministic_report(
        "重症支原体肺炎儿童是否应使用糖皮质激素？",
        bundle,
    )

    first_paragraph = report.final_answer_markdown.split("\n\n", 2)[1]
    assert "倾向于“是”" in first_paragraph
    assert "重症支原体肺炎儿童是否应使用糖皮质激素" in first_paragraph


def test_evidence_chain_excludes_unvalidated_sources_and_internal_citations() -> None:
    source, claim = evidence(1, "randomized_controlled_trial", 2025)
    unvalidated_source, _ = evidence(2, "observational_study", 2024)
    claim = EvidenceClaim(
        **{
            **claim.__dict__,
            "statement": "Earlier evidence [44] supports treatment.",
            "source_quote": "Earlier evidence [44] supports treatment.",
        }
    )
    claim_graph = LocalGraphBuilder().build("Clinical question", (claim,)).as_dict()
    document_graph = build_document_graph(
        (source, unvalidated_source),
        (claim,),
        claim_graph,
    )
    bundle = EvidenceBundle(
        sources=(source, unvalidated_source),
        graph=document_graph,
        claims=(claim,),
    )

    report = compose_deterministic_report("Should treatment be used?", bundle)
    evidence_chain = report.final_answer_markdown.split("### 证据链\n", 1)[1].split(
        "\n\n### 时间更新", 1
    )[0]

    assert "[44]" not in evidence_chain
    assert "[1]" in evidence_chain
    assert unvalidated_source.title not in evidence_chain


def test_evidence_chain_keeps_boundary_claims_in_the_boundary_section_only() -> None:
    bundle = bundle_from(
        (
            evidence(1, "randomized_controlled_trial", 2025),
            evidence(
                2,
                "observational_study",
                2024,
                direction="uncertain",
                aspect="safety",
                role="boundary",
            ),
        )
    )

    report = compose_deterministic_report("Should treatment be used?", bundle)
    evidence_chain = report.final_answer_markdown.split("### 证据链\n", 1)[1].split(
        "\n\n### 时间更新", 1
    )[0]
    boundary_section = report.final_answer_markdown.split(
        "### 安全性与适用边界\n", 1
    )[1].split("\n\n### 证据缺口", 1)[0]

    assert "Grounded effectiveness finding from source 1" in evidence_chain
    assert "Grounded safety finding from source 2" not in evidence_chain
    assert "Grounded safety finding from source 2" in boundary_section


def test_open_medication_question_returns_grounded_drug_options() -> None:
    source, claim = evidence(
        1,
        "randomized_controlled_trial",
        2025,
        direction="uncertain",
        aspect="effectiveness",
        role="core",
    )
    claim = replace(
        claim,
        intervention="阿奇霉素",
        dose="10 mg/kg/day",
        statement="阿奇霉素可作为重症支原体肺炎的抗菌治疗药物。",
        source_quote="阿奇霉素可作为重症支原体肺炎的抗菌治疗药物，剂量为10 mg/kg/day。",
    )
    claim_graph = LocalGraphBuilder().build("用药问题", (claim,)).as_dict()
    bundle = EvidenceBundle(
        sources=(source,),
        graph=build_document_graph((source,), (claim,), claim_graph),
        claims=(claim,),
    )

    report = compose_deterministic_report(
        "重症支原体肺炎儿童要吃什么药？",
        bundle,
    )

    assert "核心用药信息" in report.final_answer_markdown
    assert "阿奇霉素（10 mg/kg/day）[1]" in report.final_answer_markdown
    assert "不能形成肯定或否定建议" not in report.final_answer_markdown


def test_medication_answer_normalizes_drug_names_from_grounded_ocr_text() -> None:
    source, claim = evidence(
        1,
        "observational_study",
        2024,
        direction="uncertain",
        aspect="effectiveness",
        role="core",
    )
    claim = replace(
        claim,
        intervention="",
        dose="",
        statement="糖皮质 激素可以降低免疫介导的肺损伤。",
        source_quote="糖皮质 激素可以降低免疫介导的肺损伤。",
    )
    claim_graph = LocalGraphBuilder().build("用药问题", (claim,)).as_dict()
    bundle = EvidenceBundle(
        sources=(source,),
        graph=build_document_graph((source,), (claim,), claim_graph),
        claims=(claim,),
    )

    report = compose_deterministic_report("儿童要吃什么药？", bundle)

    direct_answer = report.final_answer_markdown.split("### 证据链", 1)[0]
    assert "- 糖皮质激素[1]" in direct_answer
    assert "可以降低免疫介导的肺损伤" not in direct_answer


def test_open_prognosis_question_returns_grounded_findings_without_voting() -> None:
    source, claim = evidence(
        1,
        "observational_study",
        2025,
        direction="uncertain",
        aspect="prognosis",
        role="core",
    )
    bundle = bundle_from(((source, claim),))

    report = compose_deterministic_report(
        "哪些指标可以预测难治性支原体肺炎？",
        bundle,
    )

    assert "核心预后信息" in report.final_answer_markdown
    assert claim.statement in report.final_answer_markdown
    assert "不能形成肯定或否定建议" not in report.final_answer_markdown
