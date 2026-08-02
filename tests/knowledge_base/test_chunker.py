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
