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
                    "In 424 children, low-dose methylprednisolone 2 mg/kg/day was supported.",
                ),
                page_ranges=("3",),
                fulltext=True,
                classification_confidence=0.9,
            ),
        ),
        graph={"nodes": [{"node_id": "question"}], "edges": []},
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
        "final_answer_markdown": "现有证据支持低剂量方案。[1]",
    }


def test_reporter_accepts_only_known_source_numbers() -> None:
    report = GroundedReporter(FakeChatClient(valid_report_response())).generate(
        "question", evidence_bundle()
    )

    assert report.final_answer_markdown.endswith("[1]")
    assert report.model_used is True
    assert report.model_error is None


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


def test_claim_extractor_discards_unknown_chunks_and_keeps_audit_record() -> None:
    response = {
        "claims": [
            {
                "source_number": 1,
                "source_chunk_ids": ["chunk-1"],
                "population": "SMPP children",
                "intervention": "methylprednisolone",
                "comparator": "high dose",
                "design": "multicenter randomized controlled trial",
                "sample_size": "424",
                "dose": "2 mg/kg/day",
                "outcome": "lung injury",
                "direction": "supports",
                "effect_measures": ["no significant difference"],
                "safety_signal": False,
                "limitations": ["single country"],
                "statement": "Low dose was supported.",
                "source_quote": "low-dose methylprednisolone 2 mg/kg/day was supported",
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
    assert result.claims[0].design == "multicenter randomized controlled trial"
    assert result.claims[0].sample_size == "424"
    assert len(result.audit) == 1
    assert result.audit[0]["reason"] == "unknown_source_chunk"


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
