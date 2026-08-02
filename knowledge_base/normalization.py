from __future__ import annotations

import hashlib
import re
import unicodedata

import jieba


_DOI_PREFIX_PATTERN = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
_INTEGER_FLOAT_PATTERN = re.compile(r"^(\d+)\.0+$")


def clean_text(value: object) -> str:
    """Return a whitespace-normalized text value without a byte-order mark."""
    if value is None:
        return ""
    return " ".join(str(value).replace("\ufeff", "").split())


def normalize_doi(value: object) -> str:
    """Normalize common DOI prefixes while preserving the DOI suffix."""
    doi = clean_text(value).lower()
    while True:
        normalized = _DOI_PREFIX_PATTERN.sub("", doi, count=1)
        if normalized == doi:
            break
        doi = normalized
    return doi.rstrip(" .")


def normalize_title(value: object) -> str:
    """Produce a comparable NFKC title key for Latin and CJK text."""
    title = unicodedata.normalize("NFKC", clean_text(value)).lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", title)


def normalize_source_id(value: object) -> str:
    """Normalize numeric spreadsheet IDs without changing nonnumeric identifiers."""
    source_id = clean_text(value)
    integer_float = _INTEGER_FLOAT_PATTERN.fullmatch(source_id)
    if integer_float is not None:
        source_id = integer_float.group(1)
    if source_id.isdigit():
        return str(int(source_id))
    return source_id


def stable_document_id(
    doi: str, title: str, first_author: str, year: int | None
) -> str:
    """Build a deterministic metadata identifier, preferring an available DOI."""
    normalized_doi = normalize_doi(doi)
    if normalized_doi:
        return "doi-" + hashlib.sha256(normalized_doi.encode("utf-8")).hexdigest()[:24]
    basis = "|".join(
        (
            normalize_title(title),
            normalize_title(first_author),
            str(year or ""),
        )
    )
    return "meta-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def tokenize_for_fts(text: str) -> str:
    """Tokenize mixed Chinese and Latin text for SQLite FTS5 indexing."""
    normalized = unicodedata.normalize("NFKC", clean_text(text)).lower()
    terms = (term.strip() for term in jieba.cut_for_search(normalized))
    return " ".join(term for term in terms if term)
