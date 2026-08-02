from pathlib import Path

import pytest

from knowledge_base.deduplicator import DocumentLookup, hash_file, match_pdf
from knowledge_base.models import DocumentRecord
from knowledge_base.normalization import normalize_title


def _document(
    document_id: str,
    title: str,
    year: int | None = 2025,
    *,
    source_id: str = "",
    doi: str = "",
    author: str = "Zhang",
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        source_row=2,
        title=title,
        normalized_title=normalize_title(title),
        authors=(author,),
        year=year,
        journal="Journal",
        doi=doi,
        normalized_doi=doi.lower(),
        abstract="",
        language="en",
    )


def test_hash_file_detects_exact_duplicate_using_large_streamed_content(tmp_path: Path) -> None:
    content = b"pdf-content" * 150_000
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.pdf"
    first.write_bytes(content)
    second.write_bytes(content)

    assert len(content) > 1024 * 1024
    assert hash_file(first) == hash_file(second)


@pytest.mark.parametrize("block_size", [0, -1])
def test_hash_file_rejects_non_positive_block_sizes(tmp_path: Path, block_size: int) -> None:
    path = tmp_path / "article.pdf"
    path.write_bytes(b"pdf")

    with pytest.raises(ValueError, match="positive"):
        hash_file(path, block_size=block_size)


def test_match_pdf_prefers_doi_before_source_id_and_title() -> None:
    doi_document = _document("doc-doi", "Different Trial", doi="10.1000/abc")
    source_document = _document("doc-source", "MPP Steroid Trial", source_id="12")
    lookup = DocumentLookup.from_documents(
        (("12", source_document), ("99", doi_document))
    )

    result = match_pdf(Path("12-10.1000%2FABC MPP Steroid Trial.pdf"), lookup)

    assert result.document_id == "doc-doi"
    assert result.method == "doi"
    assert result.confidence == 1.0


def test_match_pdf_does_not_truncate_a_doi_to_match_a_prefix() -> None:
    document = _document("doc-doi", "Different Trial", doi="10.1000/abc")
    lookup = DocumentLookup.from_documents((("99", document),))

    result = match_pdf(Path("10.1000%2Fabc2.pdf"), lookup)

    assert result.document_id is None
    assert result.method == "unmatched"


def test_match_pdf_does_not_treat_an_unmatched_doi_prefix_as_a_source_id() -> None:
    source_document = _document("doc-source", "Source Ten")
    lookup = DocumentLookup.from_documents((("10", source_document),))

    result = match_pdf(Path("10.1000%2Fnot-in-index.pdf"), lookup)

    assert result.document_id is None
    assert result.method == "unmatched"


@pytest.mark.parametrize(
    ("doi", "filename"),
    [
        ("10.1000/abc_def", "10.1000%2Fabc_def.pdf"),
        ("10.1000/abc(test)", "https%3A%2F%2Fdoi.org%2F10.1000%2Fabc%28test%29.pdf"),
    ],
)
def test_match_pdf_accepts_complete_legal_doi_filename_characters(
    doi: str, filename: str
) -> None:
    document = _document("doc-doi", "Trial", doi=doi)
    lookup = DocumentLookup.from_documents((("1", document),))

    result = match_pdf(Path(filename), lookup)

    assert result.document_id == "doc-doi"
    assert result.method == "doi"


def test_match_pdf_prefers_the_longest_complete_doi_in_a_filename() -> None:
    shorter = _document("doc-short", "Short Trial", doi="10.1000/a")
    longer = _document("doc-long", "Long Trial", doi="10.1000/abcdef")
    lookup = DocumentLookup.from_documents((("1", shorter), ("2", longer)))

    result = match_pdf(Path("10.1000%2Fa 10.1000%2Fabcdef.pdf"), lookup)

    assert result.document_id == "doc-long"
    assert result.method == "doi"


def test_match_pdf_uses_normalized_numeric_source_id_prefix() -> None:
    document = _document("doc-12", "MPP Steroid Trial", source_id="12")

    result = match_pdf(
        Path("0012-MPP Steroid Trial.pdf"),
        DocumentLookup.from_documents(((" 12.0 ", document),)),
    )

    assert result.document_id == "doc-12"
    assert result.method == "source_id"
    assert result.confidence == 1.0


@pytest.mark.parametrize("filename", ["15572同5320.pdf", "18662张春风.pdf"])
def test_match_pdf_uses_numeric_source_prefix_before_cjk(filename: str) -> None:
    document = _document("doc-source", "MPP Trial")
    lookup = DocumentLookup.from_documents((("15572" if filename.startswith("15572") else "18662", document),))

    result = match_pdf(Path(filename), lookup)

    assert result.document_id == "doc-source"
    assert result.method == "source_id"
def test_match_pdf_accepts_source_id_mapping_for_pipeline_callers() -> None:
    document = _document("doc-12", "MPP Steroid Trial")

    result = match_pdf(
        Path("12-MPP Steroid Trial.pdf"), documents_by_source_id={"12": document}
    )

    assert result.document_id == "doc-12"
    assert result.method == "source_id"


def test_match_pdf_uses_exact_normalized_title_after_identifier_rules() -> None:
    document = _document("doc-title", "MPP Steroid Trial")

    result = match_pdf(Path("MPP Steroid Trial.pdf"), DocumentLookup.from_documents((("7", document),)))

    assert result.document_id == "doc-title"
    assert result.method == "title"
    assert result.confidence == 1.0


def test_match_pdf_uses_blocked_fuzzy_title_with_year_or_first_author() -> None:
    target = _document("doc-target", "Randomized MPP Steroid Treatment Trial", 2024)
    distractors = tuple(
        (
            str(index),
            _document(f"doc-{index}", f"Unrelated study {index}", 2024, author="Other"),
        )
        for index in range(100)
    )
    lookup = DocumentLookup.from_documents(distractors + (("200", target),))

    result = match_pdf(
        Path("Randomized MPP Steroid Treatment Tria1 2024 Zhang.pdf"), lookup
    )

    assert result.document_id == "doc-target"
    assert result.method == "fuzzy_title"
    assert result.confidence >= 0.92
    assert lookup.candidate_document_ids_for_filename(
        "Randomized MPP Steroid Treatment Tria1 2024 Zhang"
    ) == ("doc-target",)


def test_match_pdf_never_force_matches_ambiguous_candidates() -> None:
    first = _document("doc-a", "MPP Steroid Trial")
    second = _document("doc-b", "MPP Steroid Trial")
    lookup = DocumentLookup.from_documents((("1", first), ("2", second)))

    result = match_pdf(Path("MPP Steroid Trial.pdf"), lookup)

    assert result.document_id is None
    assert result.method == "ambiguous"
    assert result.ambiguous_ids == ("doc-a", "doc-b")


def test_match_pdf_returns_all_fuzzy_candidates_even_when_scores_differ() -> None:
    first = _document("doc-a", "Randomized MPP Steroid Treatment Trial", 2024)
    second = _document("doc-b", "Randomized MPP Steroid Treatment Trials", 2024)
    lookup = DocumentLookup.from_documents((("1", first), ("2", second)))

    result = match_pdf(
        Path("Randomized MPP Steroid Treatment Tria1 2024 Zhang.pdf"), lookup
    )

    assert result.document_id is None
    assert result.method == "ambiguous"
    assert result.ambiguous_ids == ("doc-a", "doc-b")


def test_author_year_blocking_uses_bounded_latin_filename_tokens() -> None:
    target = _document(
        "doc-target", "Different Trial", 2024, source_id="1", author="Zhang Wei"
    )
    distractors = tuple(
        (str(index), _document(f"doc-{index}", f"Other study {index}", 2024, author="Other"))
        for index in range(100)
    )
    lookup = DocumentLookup.from_documents(distractors + (("1", target),))

    assert lookup.candidate_document_ids_for_filename("Supplement Zhang Wei 2024") == (
        "doc-target",
    )


def test_author_year_blocking_uses_bounded_cjk_filename_ngrams() -> None:
    target = _document("doc-target", "Different Trial", 2024, author="张三")
    distractor = _document("doc-other", "Other Trial", 2024, author="李四")
    lookup = DocumentLookup.from_documents((("1", target), ("2", distractor)))

    assert lookup.candidate_document_ids_for_filename("补充材料 张三 2024") == (
        "doc-target",
    )
