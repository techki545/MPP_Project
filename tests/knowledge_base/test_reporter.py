from __future__ import annotations

import pytest

from knowledge_base.errors import KnowledgeBaseError
from knowledge_base.evidence_classifier import EvidenceAssessment
from knowledge_base.reporter import (
    ChatEvidenceRefiner,
    EvidenceBundle,
    EvidenceSource,
    GroundedClaimExtractor,
    GroundedReporter,
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
    return EvidenceBundle(
        sources=(
            EvidenceSource(
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
            ),
        ),
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
    )


def valid_report_response() -> dict:
    return {
        "analysis_steps": [
            {
                "title": "第一步：盘点证据",
                "body": "纳入一项随机对照试验。[1]",
                "source_ids": [1],
            }
        ],
        "final_answer_markdown": (
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


def test_reporter_allows_uncited_markdown_heading_before_grounded_claim() -> None:
    response = valid_report_response()
    response["final_answer_markdown"] = (
        "**Evidence answer**\n\n"
        "Low-dose methylprednisolone was supported for fever duration.[1]"
    )

    report = GroundedReporter(FakeChatClient(response)).generate(
        "question", evidence_bundle()
    )

    assert report.model_used is True


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
    response["final_answer_markdown"] = "High dose increases mortality.[1]"

    with pytest.raises(KnowledgeBaseError) as captured:
        GroundedReporter(FakeChatClient(response)).generate(
            "question", evidence_bundle()
        )

    assert captured.value.code == "ungrounded_model_response"


def test_reporter_rejects_uncited_qualitative_claim() -> None:
    response = valid_report_response()
    response["final_answer_markdown"] = (
        "High dose increases mortality. "
        "Low-dose methylprednisolone was supported for fever duration.[1]"
    )

    with pytest.raises(KnowledgeBaseError) as captured:
        GroundedReporter(FakeChatClient(response)).generate(
            "question", evidence_bundle()
        )

    assert captured.value.code == "ungrounded_model_response"


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
    assert report.analysis_steps
    assert "Low-dose methylprednisolone was supported for fever duration.[1]" in (
        report.final_answer_markdown
    )


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
