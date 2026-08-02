import pytest

from knowledge_base.evidence_classifier import classify_evidence


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Clinical practice guideline for MPP", "guideline"),
        ("Systematic review and meta-analysis of corticosteroids", "systematic_review"),
        ("A multicenter randomized controlled trial", "randomized_controlled_trial"),
        ("A retrospective cohort study", "observational_study"),
        ("Narrative review of current treatment", "narrative_review"),
        ("Case report of pulmonary embolism", "case_report"),
        ("儿童支原体肺炎诊疗专家共识", "guideline"),
        ("糖皮质激素治疗的随机对照试验", "randomized_controlled_trial"),
    ],
)
def test_classifies_major_evidence_types_from_title(
    title: str, expected: str
) -> None:
    assessment = classify_evidence(title, "")

    assert assessment.evidence_type == expected
    assert assessment.confidence == 0.9
    assert assessment.basis.startswith("title:")


def test_publication_type_has_priority_and_full_confidence() -> None:
    assessment = classify_evidence(
        "Steroid treatment in children",
        "This article discusses earlier trials.",
        publication_types=("Randomized Controlled Trial",),
    )

    assert assessment.evidence_type == "randomized_controlled_trial"
    assert assessment.confidence == 1.0
    assert assessment.basis.startswith("publication_type:")


def test_abstract_only_design_has_lower_confidence() -> None:
    assessment = classify_evidence(
        "Steroid treatment in children",
        "We conducted a multicenter randomized controlled trial in five hospitals.",
    )

    assert assessment.evidence_type == "randomized_controlled_trial"
    assert assessment.confidence == 0.7
    assert assessment.quality == "high"
    assert "multicenter_randomization" in assessment.quality_signals


def test_reference_to_previous_rcts_is_not_classified_as_an_rct() -> None:
    assessment = classify_evidence(
        "Current treatment options",
        "We reviewed previous randomized controlled trials of corticosteroids.",
    )

    assert assessment.evidence_type != "randomized_controlled_trial"


def test_unknown_classification_does_not_invent_quality() -> None:
    assessment = classify_evidence(
        "Steroid therapy in children", "Clinical outcomes were assessed."
    )

    assert assessment.evidence_type == "unknown"
    assert assessment.quality == "unknown"
    assert assessment.confidence == 0.0
    assert assessment.basis
    assert assessment.needs_llm_refinement is True


def test_explicit_reporting_and_analysis_signals_set_moderate_quality() -> None:
    assessment = classify_evidence(
        "A retrospective cohort study",
        "The adjusted analysis included 165 patients and reported a sample size calculation.",
    )

    assert assessment.quality == "moderate"
    assert set(assessment.quality_signals) >= {"sample_size", "adjusted_analysis"}
    assert assessment.needs_llm_refinement is False
