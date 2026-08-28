from __future__ import annotations

import pytest

from knowledge_base.errors import KnowledgeBaseError
from knowledge_base.evidence_classifier import EvidenceAssessment
from knowledge_base.graph_builder import EvidenceClaim
from knowledge_base.reporter import (
    ChatEvidenceRefiner,
    EvidenceBundle,
    EvidenceSource,
    GroundedClaimExtractor,
    GroundedReporter,
    ValidatedClaims,
    extractive_fallback_claims,
    filter_claims_for_question,
)


class FakeChatClient:
    model = "test-model"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, system_prompt, payload):
        self.calls.append((system_prompt, payload))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def evidence_bundle() -> EvidenceBundle:
    source = EvidenceSource(
                source_number=1,
                document_id="doc-1",
                title="Randomized trial",
                evidence_type="randomized_controlled_trial",
                year=2025,
                chunk_ids=("chunk-1",),
                snippets=(
                    "In a randomized controlled trial of 424 children, low-dose "
                    "methylprednisolone 2 mg/kg/day was supported for fever duration.",
                ),
                page_ranges=("3",),
                fulltext=True,
                classification_confidence=0.9,
            )
    claim = EvidenceClaim(
        claim_id="claim-1-0",
        document_id="doc-1",
        evidence_type="randomized_controlled_trial",
        year=2025,
        population="children",
        intervention="methylprednisolone",
        comparator="",
        outcome="fever duration",
        direction="supports",
        statement="Low-dose methylprednisolone was supported for fever duration.",
        source_chunk_ids=("chunk-1",),
        source_quote=(
            "low-dose methylprednisolone 2 mg/kg/day was supported for fever duration"
        ),
        clinical_aspect="effectiveness",
        evidence_role="core",
    )
    return EvidenceBundle(
        sources=(source,),
        graph={
            "nodes": [
                {"node_id": "question", "node_type": "question", "payload": {}},
                {
                    "node_id": "claim-1-0",
                    "node_type": "claim",
                    "payload": {
                        "claim_id": "claim-1-0",
                        "document_id": "doc-1",
                        "statement": "Low-dose methylprednisolone was supported for fever duration.",
                        "source_quote": (
                            "low-dose methylprednisolone 2 mg/kg/day was supported "
                            "for fever duration"
                        ),
                    },
                },
            ],
            "edges": [],
        },
        claims=(claim,),
    )


def test_extractive_fallback_prefers_result_sentence_over_methods_and_pdf_header() -> None:
    source = EvidenceSource(
        source_number=1,
        document_id="doc-result",
        title="甲泼尼龙联合阿奇霉素治疗儿童SMPP",
        evidence_type="randomized_controlled_trial",
        year=2024,
        chunk_ids=("chunk-result",),
        snippets=(
            "1.2 方法\n\n排除标准：认知障碍或交流困难者；采用随机数字表法分为对照组和治疗组。\n\n"
            "结果：治疗组退热时间和住院时间均短于对照组，差异有统计学意义。\n\n"
            "CHINA MODERN DOCTOR Vol. 62 第5卷第3期",
        ),
        page_ranges=("3",),
        fulltext=True,
    )

    validated = extractive_fallback_claims(
        (source,), question="SMPP儿童使用甲泼尼龙是否有效？"
    )

    assert len(validated.claims) == 1
    assert "退热时间和住院时间均短于对照组" in validated.claims[0].source_quote
    assert "排除标准" not in validated.claims[0].source_quote


def test_extractive_fallback_prefers_course_sentence_for_duration_question() -> None:
    source = EvidenceSource(
        source_number=1,
        document_id="doc-duration",
        title="阿奇霉素治疗儿童SMPP",
        evidence_type="guideline",
        year=2024,
        chunk_ids=("chunk-duration",),
        snippets=(
            "阿奇霉素治疗总有效率高于对照组，差异有统计学意义。",
            "阿奇霉素连续使用3 d后停药4 d，再根据临床反应决定是否重复该疗程。",
        ),
        page_ranges=("3",),
        fulltext=True,
    )

    validated = extractive_fallback_claims(
        (source,), question="SMPP儿童治疗周期是多长时间？"
    )

    assert len(validated.claims) == 1
    assert "连续使用3 d后停药4 d" in validated.claims[0].source_quote
    assert validated.claims[0].clinical_aspect == "timing"


@pytest.mark.parametrize(
    ("question", "focused_sentence", "expected_aspect"),
    [
        (
            "如何诊断儿童重症支原体肺炎？",
            "诊断标准包括持续高热、影像学快速进展和炎症指标明显升高。",
            "diagnosis",
        ),
        (
            "阿奇霉素治疗儿童SMPP有哪些不良反应？",
            "阿奇霉素可能引起胃肠道不良反应，并需警惕QT间期延长风险。",
            "safety",
        ),
        (
            "儿童支原体肺炎为什么会发展成重症？",
            "多因素分析显示高CRP和肺实变范围是发展为重症的独立危险因素。",
            "prognosis",
        ),
    ],
)
def test_extractive_fallback_obeys_non_treatment_question_intent(
    question: str, focused_sentence: str, expected_aspect: str
) -> None:
    source = EvidenceSource(
        source_number=1,
        document_id=f"doc-{expected_aspect}",
        title="儿童SMPP临床研究",
        evidence_type="observational_study",
        year=2024,
        chunk_ids=(f"chunk-{expected_aspect}",),
        snippets=(
            "联合治疗组总有效率高于对照组，退热时间也更短。",
            focused_sentence,
        ),
        page_ranges=("2",),
        fulltext=True,
    )

    validated = extractive_fallback_claims((source,), question=question)

    assert len(validated.claims) == 1
    assert validated.claims[0].source_quote.replace(",", "，") == focused_sentence
    assert validated.claims[0].clinical_aspect == expected_aspect


def test_extractive_fallback_rejects_dangling_treatment_fragment() -> None:
    source = EvidenceSource(
        source_number=1,
        document_id="doc-fragment",
        title="糖皮质激素治疗儿童SMPP",
        evidence_type="observational_study",
        year=2023,
        chunk_ids=("chunk-fragment",),
        snippets=(
            "总之，小剂量甲泼尼龙联合阿奇霉素与常规剂量使用\n\n"
            "尼龙治疗后，治疗效果显著高于对照组；观察组发热、咳嗽、 杨花珍.\n\n"
            "研究结果显示，小剂量方案可缩短退热时间，且未增加不良反应。",
        ),
        page_ranges=("2",),
        fulltext=True,
    )

    validated = extractive_fallback_claims(
        (source,), question="SMPP儿童是否使用糖皮质激素？"
    )

    assert "研究结果显示" in validated.claims[0].source_quote
    assert "杨花珍" not in validated.claims[0].source_quote
    assert not validated.claims[0].source_quote.endswith("使用")


def test_extractive_fallback_splits_compact_structured_abstract() -> None:
    source = EvidenceSource(
        source_number=1,
        document_id="doc-abstract",
        title="抗生素联合糖皮质激素治疗儿童SMPP",
        evidence_type="randomized_controlled_trial",
        year=2024,
        chunk_ids=("chunk-abstract",),
        snippets=("PDF全文片段",),
        page_ranges=("1",),
        fulltext=True,
        abstract=(
            "目的 探讨联合治疗效果.方法 采用随机数字表法分组."
            "结果 联合治疗组退热时间短于对照组,差异有统计学意义."
            "结论 联合治疗可改善临床症状且未增加不良反应."
        ),
    )

    validated = extractive_fallback_claims(
        (source,), question="儿童SMPP是否联合使用糖皮质激素？"
    )

    quote = validated.claims[0].source_quote
    assert quote.startswith(("结果", "结论"))
    assert "随机数字表" not in quote
    assert validated.claims[0].source_chunk_ids == ("doc-abstract",)


def test_extractive_fallback_uses_relevant_guideline_fulltext_over_generic_abstract() -> None:
    source = EvidenceSource(
        source_number=1,
        document_id="doc-guideline",
        title="儿童重症肺炎支原体肺炎诊疗建议",
        evidence_type="guideline",
        year=2024,
        chunk_ids=("chunk-guideline",),
        snippets=(
            "对于常规剂量甲泼尼龙1~2 mg/(kg·d)治疗不敏感的SMPP患儿，可增加剂量至4~6 mg/(kg·d)。",
            "D-二聚体水平明显升高者可考虑给予预防性抗凝治疗，并每日评估效果。",
        ),
        page_ranges=("4",),
        fulltext=True,
        abstract="本建议提出集束化治疗策略，为儿童重症肺炎支原体肺炎的诊疗提供参考。",
    )

    validated = extractive_fallback_claims(
        (source,), question="SMPP儿童是否应使用糖皮质激素？"
    )

    assert len(validated.claims) == 1
    assert "甲泼尼龙" in validated.claims[0].source_quote
    assert validated.claims[0].source_chunk_ids == ("chunk-guideline",)


def test_extractive_fallback_does_not_treat_ineffective_case_objective_as_opposition() -> None:
    source = EvidenceSource(
        source_number=1,
        document_id="doc-objective",
        title="常规剂量甲泼尼龙治疗无效的RMPP病例特征",
        evidence_type="observational_study",
        year=2023,
        chunk_ids=("chunk-objective",),
        snippets=("目的：研究常规剂量甲泼尼龙治疗无效的RMPP病例并建立预测模型。",),
        page_ranges=("1",),
        fulltext=True,
    )

    validated = extractive_fallback_claims(
        (source,), question="SMPP儿童是否应使用糖皮质激素？"
    )

    assert validated.claims == ()


def test_extractive_fallback_uses_comparative_response_rates_for_direction() -> None:
    source = EvidenceSource(
        source_number=1,
        document_id="doc-response-rates",
        title="甲泼尼龙治疗RMPP的临床特征",
        evidence_type="observational_study",
        year=2023,
        chunk_ids=("chunk-response-rates",),
        snippets=(
            "在纳入并完成随访的儿童病例中，5例（19.2%）为甲泼尼龙治疗无效病例，"
            "21例（80.8%）为甲泼尼龙治疗有效病例，提示多数患儿获得治疗反应。",
        ),
        page_ranges=("3",),
        fulltext=True,
    )

    validated = extractive_fallback_claims(
        (source,), question="SMPP儿童是否应使用糖皮质激素？"
    )

    assert validated.claims[0].direction == "supports"


def valid_report_response() -> dict:
    return {
        "analysis_steps": [
            {
                "stage_key": "inventory",
                "title": "第一步：检索并盘点证据库存",
                "body": "纳入一项随机对照试验。[1]",
                "source_ids": [1],
            },
            {
                "stage_key": "guidelines",
                "title": "第二步：优先查看指南",
                "body": "未检索到指南。",
                "source_ids": [],
            },
            {
                "stage_key": "systematic_reviews",
                "title": "第三步：查阅系统综述，检验指南结论",
                "body": "未检索到系统综述。",
                "source_ids": [],
            },
            {
                "stage_key": "randomized_trials",
                "title": "第四步：聚焦关键随机对照试验",
                "body": "核对随机试验原文。[1]",
                "source_ids": [1],
            },
            {
                "stage_key": "lower_level_evidence",
                "title": "第五步：用下级证据补充安全性与边界",
                "body": "未检索到下级补充证据。",
                "source_ids": [],
            },
            {
                "stage_key": "synthesis",
                "title": "第六步：检查一致性并形成综合判断",
                "body": "综合一项可追溯声明。[1]",
                "source_ids": [1],
            },
        ],
        "final_answer_markdown": (
            "## 综合回答\n\n"
            "Low-dose methylprednisolone was supported for fever duration.[1]\n\n"
            "### 证据链\n"
            "Low-dose methylprednisolone was supported for fever duration.[1]\n\n"
            "### 时间更新\n"
            "Low-dose methylprednisolone was supported for fever duration.[1]\n\n"
            "### 安全性与适用边界\n"
            "Low-dose methylprednisolone was supported for fever duration.[1]\n\n"
            "### 证据缺口\n"
            "Low-dose methylprednisolone was supported for fever duration.[1]"
        ),
    }


def test_reporter_accepts_only_known_source_numbers() -> None:
    report = GroundedReporter(FakeChatClient(valid_report_response())).generate(
        "question", evidence_bundle()
    )

    assert report.final_answer_markdown.endswith("[1]")
    assert report.model_used is True
    assert report.model_error is None


def test_report_request_requires_first_demo_stage_and_heading_contract() -> None:
    chat = FakeChatClient(valid_report_response())

    report = GroundedReporter(chat).generate("question", evidence_bundle())

    required = chat.calls[0][1]["required_output"]
    assert [step["stage_key"] for step in required["analysis_steps"]] == [
        "inventory",
        "guidelines",
        "systematic_reviews",
        "randomized_trials",
        "lower_level_evidence",
        "synthesis",
    ]
    assert "### 证据链" in required["final_answer_markdown"]
    assert required["detail_requirements"]["target_length"].startswith("1200-2000")
    assert "1200-2000 Chinese characters" in chat.calls[0][0]
    assert "Never equate one drug's course with the entire treatment course" in chat.calls[0][0]
    synthesis_bundle = chat.calls[0][1]["evidence_bundle"]
    assert "snippets" not in synthesis_bundle["sources"][0]
    assert "quote_options" not in synthesis_bundle["sources"][0]
    assert synthesis_bundle["claims"][0]["source_quote"]
    assert len(report.analysis_steps) == 6


def test_reporter_rejects_clinical_assertion_disguised_as_heading() -> None:
    response = valid_report_response()
    response["final_answer_markdown"] = (
        "## High dose increases mortality\n\n"
        "Low-dose methylprednisolone was supported for fever duration.[1]"
    )

    with pytest.raises(KnowledgeBaseError) as captured:
        GroundedReporter(FakeChatClient(response)).generate(
            "question", evidence_bundle()
        )

    assert captured.value.code == "ungrounded_model_response"


def test_reporter_rejects_unknown_citation() -> None:
    response = valid_report_response()
    response["final_answer_markdown"] = "结论。[99]"

    with pytest.raises(KnowledgeBaseError) as captured:
        GroundedReporter(FakeChatClient(response)).generate("question", evidence_bundle())

    assert captured.value.code == "ungrounded_model_response"


def test_reporter_rejects_statistic_absent_from_cited_snippet() -> None:
    response = valid_report_response()
    response["final_answer_markdown"] = "共纳入999例患儿。[1]"

    with pytest.raises(KnowledgeBaseError) as captured:
        GroundedReporter(FakeChatClient(response)).generate("question", evidence_bundle())

    assert captured.value.code == "ungrounded_model_response"


def test_reporter_rejects_number_with_wrong_clinical_unit() -> None:
    source = EvidenceSource(
        source_number=1,
        document_id="doc-1",
        title="Small case series",
        evidence_type="case_report",
        year=2025,
        chunk_ids=("chunk-1",),
        snippets=("The report described 10 children with pneumonia.",),
        page_ranges=("1",),
        fulltext=True,
    )
    response = valid_report_response()
    response["final_answer_markdown"] = "推荐甲泼尼龙 10 mg/kg/day。[1]"

    with pytest.raises(KnowledgeBaseError) as captured:
        GroundedReporter(FakeChatClient(response)).generate(
            "question", EvidenceBundle((source,), {"nodes": [], "edges": []})
        )

    assert captured.value.code == "ungrounded_model_response"


def test_reporter_rejects_qualitative_claim_absent_from_validated_graph() -> None:
    response = valid_report_response()
    response["final_answer_markdown"] = response["final_answer_markdown"].replace(
        "Low-dose methylprednisolone was supported for fever duration.[1]",
        "High dose increases mortality.[1]",
        1,
    )

    with pytest.raises(KnowledgeBaseError) as captured:
        GroundedReporter(FakeChatClient(response)).generate(
            "question", evidence_bundle()
        )

    assert captured.value.code == "ungrounded_model_response"


def test_reporter_preserves_supported_cited_model_paraphrase() -> None:
    response = valid_report_response()
    response["final_answer_markdown"] = response["final_answer_markdown"].replace(
        "Low-dose methylprednisolone was supported for fever duration.[1]",
        "The evidence favors low-dose methylprednisolone for shortening fever.[1]",
        1,
    )

    report = GroundedReporter(FakeChatClient(response)).generate_with_fallback(
        "question", evidence_bundle()
    )

    assert report.model_used is True
    assert report.model_error is None
    assert report.analysis_steps[0]["source_ids"] == [1]
    assert "## 综合回答" in report.final_answer_markdown
    assert "### 证据链" in report.final_answer_markdown
    assert "The evidence favors low-dose methylprednisolone" in (
        report.final_answer_markdown
    )


def test_reporter_canonicalizes_when_reasoning_metadata_is_invalid() -> None:
    response = valid_report_response()
    response["analysis_steps"][0]["source_ids"] = [99]
    response["final_answer_markdown"] = response["final_answer_markdown"].replace(
        "Low-dose methylprednisolone was supported for fever duration.[1]",
        "A translated evidence summary.[1]",
        1,
    )

    report = GroundedReporter(FakeChatClient(response)).generate_with_fallback(
        "question", evidence_bundle()
    )

    assert report.model_used is True
    assert report.model_error is None
    assert len(report.analysis_steps) == 6
    assert "Low-dose methylprednisolone was supported for fever duration.[1]" in (
        report.final_answer_markdown
    )


def test_reporter_canonicalization_preserves_uncited_clinical_interpretation() -> None:
    response = valid_report_response()
    response["analysis_steps"][0]["source_ids"] = [99]
    interpretation = (
        "Clinical interpretation: treatment choice should consider severity and "
        "contraindications."
    )
    response["final_answer_markdown"] = response["final_answer_markdown"].replace(
        "## 综合回答",
        f"## 综合回答\n\n{interpretation}",
        1,
    )

    report = GroundedReporter(FakeChatClient(response)).generate_with_fallback(
        "question", evidence_bundle()
    )

    assert report.model_used is True
    assert report.model_error is None
    assert interpretation in report.final_answer_markdown


def test_reporter_flexible_synthesis_can_reference_source_without_graph_claim() -> None:
    response = valid_report_response()
    source_without_claim = EvidenceSource(
        source_number=2,
        document_id="doc-2",
        title="Guideline context",
        evidence_type="guideline",
        year=2025,
        chunk_ids=("chunk-2",),
        snippets=("Treatment selection should consider disease severity.",),
        page_ranges=("2",),
        fulltext=True,
    )
    bundle = evidence_bundle()
    bundle = EvidenceBundle(
        sources=(*bundle.sources, source_without_claim),
        graph=bundle.graph,
        claims=bundle.claims,
    )
    response["analysis_steps"][0]["source_ids"] = [99]
    response["final_answer_markdown"] = response["final_answer_markdown"].replace(
        "Low-dose methylprednisolone was supported for fever duration.[1]",
        "Treatment selection should consider disease severity.[2]",
        1,
    )

    report = GroundedReporter(FakeChatClient(response)).generate_with_fallback(
        "question", bundle
    )

    assert report.model_used is True
    assert report.model_error is None
    assert "disease severity.[2]" in report.final_answer_markdown


def test_reporter_allows_uncited_clinical_interpretation_alongside_evidence() -> None:
    response = valid_report_response()
    interpretation = (
        "Clinical interpretation: treatment choice should also consider severity "
        "and contraindications."
    )
    response["final_answer_markdown"] = response["final_answer_markdown"].replace(
        "## 综合回答",
        f"## 综合回答\n\n{interpretation}",
        1,
    )

    report = GroundedReporter(FakeChatClient(response)).generate(
        "question", evidence_bundle()
    )

    assert interpretation in report.final_answer_markdown
    assert report.model_used is True


def test_reporter_fallback_preserves_inventory_and_graph() -> None:
    response = valid_report_response()
    response["final_answer_markdown"] = "结论。[99]"
    bundle = evidence_bundle()

    report = GroundedReporter(FakeChatClient(response)).generate_with_fallback(
        "question", bundle
    )

    assert report.model_used is False
    assert report.model_name == "test-model"
    assert report.model_error == "ungrounded_model_response"
    assert report.evidence_inventory[0]["document_id"] == "doc-1"
    assert report.graph == bundle.graph
    assert len(report.analysis_steps) == 6
    assert report.final_answer_markdown.startswith("## 综合回答")
    assert "### 证据链" in report.final_answer_markdown
    assert "### 时间更新" in report.final_answer_markdown
    assert "### 安全性与适用边界" in report.final_answer_markdown
    assert "### 证据缺口" in report.final_answer_markdown
    assert "Low-dose methylprednisolone was supported for fever duration.[1]" in (
        report.final_answer_markdown
    )


def test_grounded_claim_report_is_deterministic_without_second_model_call() -> None:
    chat = FakeChatClient(valid_report_response())
    bundle = evidence_bundle()

    report = GroundedReporter(chat).generate_grounded_claim_report(
        "question",
        bundle,
        model_used=True,
        model_error=None,
    )

    assert chat.calls == []
    assert report.model_used is True
    assert report.model_error is None
    assert len(report.analysis_steps) == 6
    assert report.final_answer_markdown.startswith("## 综合回答")
    assert "### 证据链" in report.final_answer_markdown


def test_claim_extractor_discards_unknown_chunks_and_keeps_audit_record() -> None:
    response = {
        "claims": [
            {
                "source_number": 1,
                "source_chunk_ids": ["chunk-1"],
                "population": "children",
                "intervention": "methylprednisolone",
                "comparator": "",
                "design": "randomized controlled trial",
                "sample_size": "424",
                "dose": "2 mg/kg/day",
                "outcome": "fever duration",
                "direction": "supports",
                "effect_measures": [],
                "safety_signal": False,
                "limitations": [],
                "statement": "Low dose was supported.",
                "source_quote": "low-dose methylprednisolone 2 mg/kg/day was supported for fever duration",
            },
            {
                "source_number": 1,
                "source_chunk_ids": ["invented-chunk"],
                "population": "children",
                "intervention": "steroid",
                "comparator": "",
                "design": "trial",
                "sample_size": "",
                "dose": "",
                "outcome": "fever",
                "direction": "supports",
                "effect_measures": [],
                "safety_signal": False,
                "limitations": [],
                "statement": "Invented claim",
                "source_quote": "Invented quote",
            },
        ]
    }

    result = GroundedClaimExtractor(FakeChatClient(response)).extract(
        "question", evidence_bundle()
    )

    assert len(result.claims) == 1
    assert result.claims[0].source_chunk_ids == ("chunk-1",)
    assert result.claims[0].design == "randomized controlled trial"
    assert result.claims[0].sample_size == "424"
    assert len(result.audit) == 1
    assert result.audit[0]["reason"] == "unknown_source_chunk"


def test_quote_id_claim_preserves_only_grounded_structured_fields() -> None:
    response = {
        "claims": [
            {
                "source_number": 1,
                "source_quote_id": "quote-1-1",
                "clinical_aspect": "effectiveness",
                "direction": "uncertain",
                "evidence_role": "core",
                "population": "424 children",
                "intervention": "methylprednisolone",
                "comparator": "invented placebo",
                "design": "randomized controlled trial",
                "sample_size": "424",
                "dose": "2 mg/kg/day",
                "outcome": "fever duration",
                "follow_up": "invented follow-up",
                "effect_measures": [],
                "limitations": [],
                "safety_signal": False,
                "statement": "ignored when a quote ID is used",
            }
        ]
    }

    result = GroundedClaimExtractor(FakeChatClient(response)).extract(
        "What medicine should be used?", evidence_bundle()
    )

    claim = result.claims[0]
    assert claim.intervention == "methylprednisolone"
    assert claim.dose == "2 mg/kg/day"
    assert claim.population == "424 children"
    assert claim.design == "randomized controlled trial"
    assert claim.comparator == ""
    assert claim.follow_up == ""


def test_claim_extractor_discards_semantics_not_supported_by_source_text() -> None:
    response = {
        "claims": [
            {
                "source_number": 1,
                "source_chunk_ids": ["chunk-1"],
                "population": "children",
                "intervention": "methylprednisolone",
                "comparator": "",
                "design": "randomized trial",
                "sample_size": "424",
                "dose": "100 mg/kg/day",
                "outcome": "fever duration",
                "direction": "supports",
                "effect_measures": [],
                "safety_signal": False,
                "limitations": [],
                "statement": "A 100 mg/kg/day dose was supported.",
                "source_quote": "In a randomized controlled trial of 424 children, low-dose methylprednisolone 2 mg/kg/day was supported for fever duration.",
            }
        ]
    }

    result = GroundedClaimExtractor(FakeChatClient(response)).extract(
        "question", evidence_bundle()
    )

    assert result.claims == ()
    assert result.audit[0]["reason"] == "claim_not_grounded"


@pytest.mark.parametrize(
    ("snippet", "statement", "safety_signal"),
    [
        (
            "High dose was studied. Treatment was supported.",
            "High dose was supported.",
            False,
        ),
        (
            "High dose was not supported.",
            "High dose was supported.",
            False,
        ),
        (
            "Serious adverse events occurred after treatment.",
            "Serious adverse events occurred after treatment.",
            False,
        ),
        (
            "Treatment was supported.",
            "Treatment was supported.",
            True,
        ),
        (
            "No bleeding occurred, but infection risk increased.",
            "Infection risk increased.",
            False,
        ),
    ],
)
def test_claim_extractor_rejects_scattered_negated_or_false_safety_claims(
    snippet: str, statement: str, safety_signal: bool
) -> None:
    source = EvidenceSource(
        source_number=1,
        document_id="doc-1",
        title="Treatment study",
        evidence_type="observational_study",
        year=2025,
        chunk_ids=("chunk-1",),
        snippets=(snippet,),
        page_ranges=("1",),
        fulltext=True,
    )
    response = {
        "claims": [
            {
                "source_number": 1,
                "source_chunk_ids": ["chunk-1"],
                "population": "treatment",
                "intervention": "treatment",
                "comparator": "",
                "design": "study",
                "sample_size": "",
                "dose": "",
                "outcome": "treatment",
                "direction": "supports",
                "effect_measures": [],
                "safety_signal": safety_signal,
                "limitations": [],
                "statement": statement,
                "source_quote": snippet,
            }
        ]
    }

    result = GroundedClaimExtractor(FakeChatClient(response)).extract(
        "question", EvidenceBundle((source,), {"nodes": [], "edges": []})
    )

    assert result.claims == ()
    assert result.audit[0]["reason"] == "claim_not_grounded"


def test_claim_extractor_rejects_effect_terms_scattered_across_sentences() -> None:
    snippet = (
        "Hospital stay was measured. Duration was reduced. Treatment was supported."
    )
    source = EvidenceSource(
        source_number=1,
        document_id="doc-1",
        title="Treatment study",
        evidence_type="observational_study",
        year=2025,
        chunk_ids=("chunk-1",),
        snippets=(snippet,),
        page_ranges=("1",),
        fulltext=True,
    )
    response = {
        "claims": [
            {
                "source_number": 1,
                "source_chunk_ids": ["chunk-1"],
                "population": "treatment",
                "intervention": "treatment",
                "comparator": "",
                "design": "study",
                "sample_size": "",
                "dose": "",
                "outcome": "hospital stay",
                "direction": "supports",
                "effect_measures": ["hospital stay reduced"],
                "safety_signal": False,
                "limitations": [],
                "statement": "Treatment was supported.",
                "source_quote": snippet,
            }
        ]
    }

    result = GroundedClaimExtractor(FakeChatClient(response)).extract(
        "question", EvidenceBundle((source,), {"nodes": [], "edges": []})
    )

    assert result.claims == ()
    assert result.audit[0]["reason"] == "claim_not_grounded"


@pytest.mark.parametrize(
    ("snippet", "statement"),
    [
        (
            "High dose was supported but low dose was not supported.",
            "High dose was not supported.",
        ),
        (
            "High dose was supported. Low dose was not supported.",
            "High dose was not supported.",
        ),
    ],
)
def test_claim_extractor_binds_negation_to_the_same_clause(
    snippet: str, statement: str
) -> None:
    source = EvidenceSource(
        source_number=1,
        document_id="doc-1",
        title="Dose study",
        evidence_type="observational_study",
        year=2025,
        chunk_ids=("chunk-1",),
        snippets=(snippet,),
        page_ranges=("1",),
        fulltext=True,
    )
    response = {
        "claims": [
            {
                "source_number": 1,
                "source_chunk_ids": ["chunk-1"],
                "population": "dose",
                "intervention": "high dose",
                "comparator": "low dose",
                "design": "study",
                "sample_size": "",
                "dose": "",
                "outcome": "dose",
                "direction": "opposes",
                "effect_measures": [],
                "safety_signal": False,
                "limitations": [],
                "statement": statement,
                "source_quote": snippet,
            }
        ]
    }

    result = GroundedClaimExtractor(FakeChatClient(response)).extract(
        "question", EvidenceBundle((source,), {"nodes": [], "edges": []})
    )

    assert result.claims == ()
    assert result.audit[0]["reason"] == "claim_not_grounded"


def test_claim_extractor_does_not_join_title_and_abstract_into_one_claim() -> None:
    source = EvidenceSource(
        source_number=1,
        document_id="doc-1",
        title="High dose",
        abstract="was supported",
        evidence_type="observational_study",
        year=2025,
        chunk_ids=("chunk-1",),
        snippets=("Treatment was measured.",),
        page_ranges=("1",),
        fulltext=True,
    )
    response = {
        "claims": [
            {
                "source_number": 1,
                "source_chunk_ids": ["chunk-1"],
                "population": "treatment",
                "intervention": "high dose supported",
                "comparator": "",
                "design": "dose",
                "sample_size": "",
                "dose": "",
                "outcome": "treatment",
                "direction": "supports",
                "effect_measures": [],
                "safety_signal": False,
                "limitations": [],
                "statement": "Treatment was measured.",
                "source_quote": "Treatment was measured.",
            }
        ]
    }

    result = GroundedClaimExtractor(FakeChatClient(response)).extract(
        "question", EvidenceBundle((source,), {"nodes": [], "edges": []})
    )

    assert result.claims == ()
    assert result.audit[0]["reason"] == "claim_not_grounded"


def test_evidence_refiner_only_calls_model_for_low_confidence_results() -> None:
    low = EvidenceAssessment("unknown", 0.0, "unknown", "no signal", ())
    high = EvidenceAssessment(
        "guideline", 0.9, "moderate", "title:guideline", ("reporting_standard",)
    )
    chat = FakeChatClient(
        {
            "evidence_type": "systematic_review",
            "basis": "systematic review",
        }
    )
    refiner = ChatEvidenceRefiner(chat)

    refined = refiner.refine(
        "A systematic review of treatment", "Evidence was synthesized.", low
    )
    unchanged = refiner.refine("Clinical guideline", "", high)

    assert refined.evidence_type == "systematic_review"
    assert refined.basis.startswith("model:")
    assert unchanged is high
    assert len(chat.calls) == 1


def test_invalid_evidence_refinement_falls_back_to_rule_result() -> None:
    original = EvidenceAssessment("unknown", 0.0, "unknown", "no signal", ())
    refiner = ChatEvidenceRefiner(
        FakeChatClient({"evidence_type": "expert_opinion", "basis": "invented"})
    )

    assert refiner.refine("Treatment study", "Outcomes", original) is original


def test_question_focus_filter_rejects_steroid_claim_for_exercise_question() -> None:
    bundle = evidence_bundle()

    filtered = filter_claims_for_question(
        "重症支原体肺炎（SMPP）儿童是否应常规运动？",
        ValidatedClaims(bundle.claims, ()),
    )

    assert filtered.claims == ()
    assert filtered.audit[0]["reason"] == "claim_question_focus_mismatch"
    assert filtered.audit[0]["missing_focus"] == ["exercise"]


def test_question_focus_filter_keeps_matching_steroid_claim() -> None:
    bundle = evidence_bundle()

    filtered = filter_claims_for_question(
        "重症支原体肺炎儿童是否应使用甲泼尼龙？",
        ValidatedClaims(bundle.claims, ()),
    )

    assert filtered.claims == bundle.claims
    assert filtered.audit == ()
