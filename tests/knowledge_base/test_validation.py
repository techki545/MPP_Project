from validation.run_validation import evaluate_case


def test_validation_checks_expected_source_and_citation_support() -> None:
    case = {
        "id": "steroid-dose",
        "question": "SMPP儿童糖皮质激素应选择低剂量还是高剂量？",
        "expected_title_terms": ["methylprednisolone", "randomized"],
    }
    result = {
        "sources": [{"title": "Randomized methylprednisolone dose trial"}],
        "answer_markdown": "低剂量与高剂量疗效相近。[1]",
        "source_support": {"1": True},
    }

    assessment = evaluate_case(case, result)

    assert assessment.expected_source_hit is True
    assert assessment.all_citations_supported is True


def test_validation_reports_duplicates_unsupported_citations_and_type_coverage() -> None:
    case = {
        "id": "macrolide-resistance",
        "question": "大环内酯耐药MPP如何治疗？",
        "expected_title_terms": ["macrolide", "resistance"],
        "expected_evidence_types": ["guideline", "systematic_review"],
    }
    result = {
        "mode": "keyword",
        "sources": [
            {
                "document_id": "doc-1",
                "title": "Macrolide resistance guideline",
                "evidence_type": "guideline",
            },
            {
                "document_id": "doc-1",
                "title": "Macrolide resistance guideline",
                "evidence_type": "guideline",
            },
        ],
        "answer_markdown": "应根据耐药风险调整治疗。[1][2]",
        "source_support": {"1": True, "2": False},
    }

    assessment = evaluate_case(case, result)

    assert assessment.duplicate_document_count == 1
    assert assessment.unsupported_citation_count == 1
    assert assessment.evidence_type_coverage == 0.5
    assert assessment.retrieval_mode == "keyword"


def test_validation_does_not_assume_production_citations_are_supported() -> None:
    case = {
        "id": "production-shape",
        "question": "问题",
        "expected_title_terms": ["trial"],
    }
    result = {
        "mode": "keyword",
        "sources": [
            {
                "source_number": 1,
                "document_id": "doc-1",
                "title": "Trial",
                "snippets": [],
                "abstract": "",
            }
        ],
        "answer_markdown": "结论。[1]",
    }

    assessment = evaluate_case(case, result)

    assert assessment.unsupported_citation_count == 1
    assert assessment.all_citations_supported is False


def test_validation_marks_an_uncited_answer_as_unsupported() -> None:
    assessment = evaluate_case(
        {"id": "uncited", "question": "问题", "expected_title_terms": ["trial"]},
        {
            "mode": "keyword",
            "sources": [
                {
                    "source_number": 1,
                    "document_id": "doc-1",
                    "title": "Trial",
                    "snippets": ["Grounded evidence."],
                }
            ],
            "answer_markdown": "没有引用的结论。",
        },
    )

    assert assessment.citation_count == 0
    assert assessment.unsupported_citation_count == 1
    assert assessment.all_citations_supported is False


def test_validation_scores_question_understanding_and_page_locators() -> None:
    assessment = evaluate_case(
        {
            "id": "typed-treatment",
            "question": "重症支原体肺炎儿童要吃什么药？",
            "expected_question_type": "treatment",
            "expected_title_terms": ["guideline"],
        },
        {
            "mode": "hybrid",
            "query_context": {
                "question_type": "treatment",
                "pico": {"population": ["儿童"], "intervention": [], "comparator": [], "outcome": []},
            },
            "sources": [
                {
                    "source_number": 1,
                    "document_id": "doc-1",
                    "title": "Treatment guideline",
                    "snippets": ["Grounded evidence."],
                    "page_ranges": ["3-4"],
                }
            ],
            "answer_markdown": "结论。[1]",
        },
    )

    assert assessment.query_understanding_present is True
    assert assessment.question_type_match is True
    assert assessment.citation_locator_coverage == 1.0
