from knowledge_base.text_cleaning import clean_evidence_text, reference_like_text


def test_clean_evidence_text_removes_internal_citations_and_cjk_column_gaps():
    raw = (
        "病程5~10 天,最好在病程6~7 天开始足量甲泼尼龙治疗[98-100] "
        "可有效预防坏死性肺 炎。"
    )

    cleaned = clean_evidence_text(raw)

    assert cleaned == (
        "病程5~10 天,最好在病程6~7 天开始足量甲泼尼龙治疗可有效预防坏死性肺炎。"
    )


def test_clean_evidence_text_cuts_bibliography_spill_after_clinical_sentence():
    raw = (
        "相关指南建议疗程3~5 d,不超过10 d。 "
        "[8]胡群.小剂量糖皮质激素治疗[J].当代医药论丛,2019,17(1):132-133."
    )

    assert clean_evidence_text(raw) == "相关指南建议疗程3~5 d,不超过10 d。"


def test_reference_like_text_rejects_journal_entries_but_keeps_clinical_prose():
    assert reference_like_text("临床研究,2019,27(4):71-73.") is True
    assert reference_like_text("甲泼尼龙治疗后退热时间缩短。") is False
