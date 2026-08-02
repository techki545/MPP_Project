from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from collections.abc import Iterable, Mapping
from pathlib import Path
import re
from types import MappingProxyType
from typing import TypeAlias
import unicodedata
from urllib.parse import unquote

from .models import DocumentRecord
from .normalization import clean_text, normalize_doi, normalize_source_id, normalize_title


HASH_BLOCK_SIZE = 1024 * 1024
_DOI_IN_FILENAME = re.compile(
    r"(?<![0-9a-z])(10\.\d{4,9}/[-._;()/:a-z0-9]+)", re.IGNORECASE
)
_DOI_FILENAME_PREFIX = re.compile(r"^\s*10\.\d", re.IGNORECASE)
_SOURCE_ID_PREFIX = re.compile(r"^\s*(\d+)(?=$|[\s_.-]|[\u4e00-\u9fff])")
_FILENAME_SOURCE_PREFIX = re.compile(r"^\s*\d+(?:[\s_.-]+)")
_YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")
_PREFIX_LENGTHS = (8, 12, 16, 24)

DocumentItems: TypeAlias = Iterable[tuple[str, DocumentRecord]] | Mapping[str, DocumentRecord]


@dataclass(frozen=True)
class MatchResult:
    document_id: str | None
    method: str
    confidence: float
    ambiguous_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentLookup:
    """Reusable, bounded lookup structures for PDF-to-metadata matching."""

    documents_by_id: Mapping[str, DocumentRecord]
    source_ids: Mapping[str, tuple[str, ...]]
    dois: Mapping[str, tuple[str, ...]]
    titles: Mapping[str, tuple[str, ...]]
    title_prefixes: Mapping[str, tuple[str, ...]]
    year_author: Mapping[tuple[int, str], tuple[str, ...]]
    authors: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_documents(cls, documents: DocumentItems) -> "DocumentLookup":
        items = documents.items() if isinstance(documents, Mapping) else documents
        documents_by_id: dict[str, DocumentRecord] = {}
        source_ids: dict[str, list[str]] = {}
        dois: dict[str, list[str]] = {}
        titles: dict[str, list[str]] = {}
        title_prefixes: dict[str, list[str]] = {}
        year_author: dict[tuple[int, str], list[str]] = {}
        authors: dict[str, list[str]] = {}

        for raw_source_id, document in items:
            document_id = document.document_id
            documents_by_id[document_id] = document
            cls._append(source_ids, normalize_source_id(raw_source_id), document_id)
            cls._append(dois, normalize_doi(document.normalized_doi or document.doi), document_id)
            title = document.normalized_title or normalize_title(document.title)
            cls._append(titles, title, document_id)
            for prefix_length in _PREFIX_LENGTHS:
                if title:
                    cls._append(title_prefixes, title[:prefix_length], document_id)
            first_author = cls._first_author(document)
            if first_author:
                cls._append(authors, first_author, document_id)
                if document.year is not None:
                    cls._append(year_author, (document.year, first_author), document_id)

        return cls(
            documents_by_id=MappingProxyType(documents_by_id),
            source_ids=cls._freeze(source_ids),
            dois=cls._freeze(dois),
            titles=cls._freeze(titles),
            title_prefixes=cls._freeze(title_prefixes),
            year_author=cls._freeze(year_author),
            authors=cls._freeze(authors),
        )

    @staticmethod
    def _append(index: dict[object, list[str]], key: object, document_id: str) -> None:
        if key:
            index.setdefault(key, []).append(document_id)

    @staticmethod
    def _freeze(index: dict[object, list[str]]) -> Mapping[object, tuple[str, ...]]:
        return MappingProxyType(
            {key: tuple(sorted(set(document_ids))) for key, document_ids in index.items()}
        )

    @staticmethod
    def _first_author(document: DocumentRecord) -> str:
        return normalize_title(document.authors[0]) if document.authors else ""

    def candidate_document_ids_for_filename(self, filename: str) -> tuple[str, ...]:
        """Return a small candidate block without scanning the document corpus."""
        filename_title = _filename_title(filename)
        candidate_ids: set[str] = set()
        for prefix_length in _PREFIX_LENGTHS:
            if filename_title:
                candidate_ids.update(
                    self.title_prefixes.get(filename_title[:prefix_length], ())
                )

        years = {int(value) for value in _YEAR_PATTERN.findall(filename)}
        for author in _filename_author_keys(filename):
            if years:
                for year in years:
                    candidate_ids.update(self.year_author.get((year, author), ()))
            else:
                candidate_ids.update(self.authors.get(author, ()))
        return tuple(sorted(candidate_ids))


def hash_file(path: Path, block_size: int = HASH_BLOCK_SIZE) -> str:
    """Return a SHA256 digest while reading at most one MiB per default block."""
    import hashlib

    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def match_pdf(
    path: Path,
    lookup: DocumentLookup | None = None,
    *,
    documents_by_source_id: Mapping[str, DocumentRecord] | None = None,
) -> MatchResult:
    """Match one PDF through deterministic identifiers and a blocked fuzzy fallback."""
    if lookup is None:
        if documents_by_source_id is None:
            raise ValueError("A document lookup or documents_by_source_id is required")
        lookup = DocumentLookup.from_documents(documents_by_source_id)
    elif documents_by_source_id is not None:
        raise ValueError("Pass either lookup or documents_by_source_id, not both")
    filename = unquote(Path(path).stem)

    extracted_dois = sorted(
        {normalize_doi(doi) for doi in _DOI_IN_FILENAME.findall(filename)},
        key=len,
        reverse=True,
    )
    for doi in extracted_dois:
        result = _result_for_candidates(
            lookup.dois.get(doi, ()), "doi", 1.0
        )
        if result is not None:
            return result

    source_id_match = _SOURCE_ID_PREFIX.match(filename)
    if source_id_match is not None and _DOI_FILENAME_PREFIX.match(filename) is None:
        result = _result_for_candidates(
            lookup.source_ids.get(normalize_source_id(source_id_match.group(1)), ()),
            "source_id",
            1.0,
        )
        if result is not None:
            return result

    filename_title = _filename_title(filename)
    result = _result_for_candidates(lookup.titles.get(filename_title, ()), "title", 1.0)
    if result is not None:
        return result

    fuzzy_matches: list[tuple[str, float]] = []
    for document_id in lookup.candidate_document_ids_for_filename(filename):
        document = lookup.documents_by_id[document_id]
        normalized_title = document.normalized_title or normalize_title(document.title)
        similarity = SequenceMatcher(
            None, normalized_title, filename_title[: len(normalized_title)]
        ).ratio()
        if similarity < 0.92 or not _has_matching_year_or_author(document, filename, filename_title):
            continue
        fuzzy_matches.append((document_id, similarity))
    if fuzzy_matches:
        best_score = max(score for _, score in fuzzy_matches)
        qualifying_ids = tuple(document_id for document_id, _ in fuzzy_matches)
        return _result_for_candidates(qualifying_ids, "fuzzy_title", best_score) or MatchResult(
            None, "unmatched", 0.0
        )

    return MatchResult(None, "unmatched", 0.0)


def _result_for_candidates(
    candidates: Iterable[str], method: str, confidence: float
) -> MatchResult | None:
    candidate_ids = tuple(sorted(set(candidates)))
    if not candidate_ids:
        return None
    if len(candidate_ids) > 1:
        return MatchResult(None, "ambiguous", confidence, candidate_ids)
    return MatchResult(candidate_ids[0], method, confidence)


def _filename_title(filename: str) -> str:
    return normalize_title(_FILENAME_SOURCE_PREFIX.sub("", filename))


def _filename_author_keys(filename: str) -> tuple[str, ...]:
    """Build bounded author-key candidates from filename tokens, not corpus rows."""
    normalized = unicodedata.normalize("NFKC", clean_text(filename)).lower()
    tokens = re.findall(r"[a-z]+|[\u4e00-\u9fff]+", normalized)
    candidates: set[str] = set()
    for index, token in enumerate(tokens):
        if token.isascii():
            for end in range(index + 1, min(index + 3, len(tokens)) + 1):
                parts = tokens[index:end]
                if not all(part.isascii() for part in parts):
                    break
                candidates.add("".join(parts))
            continue
        maximum = min(len(token), 8)
        for size in range(2, maximum + 1):
            for start in range(0, len(token) - size + 1):
                candidates.add(token[start : start + size])
    return tuple(sorted(candidates))


def _has_matching_year_or_author(
    document: DocumentRecord, filename: str, filename_title: str
) -> bool:
    if document.year is not None and str(document.year) in filename:
        return True
    first_author = DocumentLookup._first_author(document)
    return bool(first_author and first_author in filename_title)
