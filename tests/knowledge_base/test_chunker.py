from __future__ import annotations

import pytest

from knowledge_base.chunker import approximate_token_count, chunk_pages
from knowledge_base.pdf_parser import ParsedPage


def test_approximate_token_count_counts_cjk_and_alphanumeric_words() -> None:
    assert approximate_token_count("儿童 MPP trial 2025") == 5


def test_chunks_preserve_section_page_range_and_deterministic_id() -> None:
    pages = (
        ParsedPage(1, "摘要\n研究目的和背景。", False, "extracted"),
        ParsedPage(2, "结果\n低剂量与高剂量疗效相近。", False, "extracted"),
    )

    first = chunk_pages("doc-1", "file-1", pages, target_tokens=12, overlap_tokens=3)
    second = chunk_pages("doc-1", "file-1", pages, target_tokens=12, overlap_tokens=3)

    assert first == second
    assert first[0].document_id == "doc-1"
    assert first[0].page_start == 1
    assert first[-1].page_end == 2
    assert any(chunk.section == "结果" for chunk in first)
    assert all(len(chunk.chunk_id) == 64 for chunk in first)


def test_chunk_uses_ocr_and_conservative_quality_when_any_page_requires_it() -> None:
    pages = (
        ParsedPage(1, "Methods\nThis section has enough text for a chunk.", False, "extracted"),
        ParsedPage(2, "More material from OCR is present here.", True, "ocr"),
    )

    chunks = chunk_pages("doc-1", "file-1", pages, target_tokens=100, overlap_tokens=10)

    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert chunks[0].is_ocr is True
    assert chunks[0].quality == "ocr"


@pytest.mark.parametrize(
    ("page_quality", "used_ocr", "expected_quality"),
    [
        ("extracted", False, "extracted"),
        ("ocr", True, "ocr"),
        ("ocr_reduced", True, "ocr_reduced"),
        ("low_quality", False, "low_quality"),
        ("low_quality_reduced", False, "low_quality_reduced"),
        ("ocr_error", False, "degraded"),
        ("ocr_too_large", False, "degraded"),
        ("page_error", False, "degraded"),
    ],
)
def test_chunker_preserves_conservative_single_page_quality(
    page_quality: str, used_ocr: bool, expected_quality: str
) -> None:
    pages = (ParsedPage(1, "Retained page text for quality propagation.", used_ocr, page_quality),)

    chunks = chunk_pages("doc", "file", pages, target_tokens=100, overlap_tokens=10)

    assert chunks[0].quality == expected_quality


@pytest.mark.parametrize(
    ("qualities", "ocr_flags", "expected_quality"),
    [
        (("ocr", "ocr_reduced"), (True, True), "ocr_reduced"),
        (("ocr_reduced", "low_quality_reduced"), (True, False), "low_quality_reduced"),
        (("ocr", "ocr_too_large"), (True, False), "degraded"),
        (("extracted", "page_error"), (False, False), "degraded"),
    ],
)
def test_chunker_uses_deterministic_worst_quality_for_mixed_pages(
    qualities: tuple[str, ...], ocr_flags: tuple[bool, ...], expected_quality: str
) -> None:
    pages = tuple(
        ParsedPage(index + 1, f"Retained page {index} text for mixed quality.", ocr_flags[index], quality)
        for index, quality in enumerate(qualities)
    )

    chunks = chunk_pages("doc", "file", pages, target_tokens=100, overlap_tokens=10)

    assert chunks[0].quality == expected_quality


def test_chunker_detects_english_sections_case_insensitively() -> None:
    pages = (ParsedPage(1, "ABSTRACT\nBackground text.\n\nDISCUSSION\nInterpretation text.", False, "extracted"),)

    chunks = chunk_pages("doc", "file", pages, target_tokens=3, overlap_tokens=0)

    assert {chunk.section for chunk in chunks} >= {"ABSTRACT", "DISCUSSION"}


def test_overlap_and_hard_limit_split_a_long_paragraph_without_looping() -> None:
    text = " ".join("word%s" % index for index in range(120))
    pages = (ParsedPage(1, text, False, "extracted"),)

    chunks = chunk_pages("doc", "file", pages, target_tokens=25, overlap_tokens=5, max_chunk_tokens=30)

    assert len(chunks) > 1
    assert all(0 < chunk.token_count <= 30 for chunk in chunks)
    assert all(chunk.token_count >= 25 for chunk in chunks)
    assert "word24" in chunks[1].text
    assert chunks[-1].page_end == 1


def test_long_paragraph_windows_preserve_exact_medical_notation_and_spacing() -> None:
    phrase = "剂量为 2 mg/kg/d，P < 0.05；β-lactam 方案有效。"
    pages = (ParsedPage(1, " ".join([phrase] * 8), False, "extracted"),)

    chunks = chunk_pages("doc", "file", pages, target_tokens=12, overlap_tokens=4, max_chunk_tokens=16)

    assert len(chunks) > 1
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "2 mg/kg/d" in combined
    assert "P < 0.05" in combined
    assert "β-lactam" in combined
    assert all(approximate_token_count(chunk.text) == chunk.token_count for chunk in chunks)
    assert all(chunk.text in pages[0].text for chunk in chunks)


@pytest.mark.parametrize(
    ("target_tokens", "overlap_tokens", "max_chunk_tokens"),
    [(0, 0, 900), (-1, 0, 900), (10, -1, 900), (10, 10, 900), (10, 0, 0), (50, 0, 40)],
)
def test_chunker_rejects_invalid_window_parameters(
    target_tokens: int, overlap_tokens: int, max_chunk_tokens: int
) -> None:
    with pytest.raises(ValueError):
        chunk_pages("doc", "file", (), target_tokens, overlap_tokens, max_chunk_tokens)


def test_empty_pages_produce_no_chunks() -> None:
    pages = (ParsedPage(1, "", False, "low_quality"), ParsedPage(2, "  ", False, "low_quality"))

    assert chunk_pages("doc", "file", pages) == ()
