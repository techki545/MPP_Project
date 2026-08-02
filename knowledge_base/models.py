from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    source_row: int
    title: str
    normalized_title: str
    authors: tuple[str, ...]
    year: int | None
    journal: str
    doi: str
    normalized_doi: str
    abstract: str
    language: str
    evidence_type: str = "unknown"
    classification_confidence: float = 0.0
    classification_basis: str = ""
    has_fulltext: bool = False
    fulltext_status: str = "missing"


@dataclass(frozen=True)
class FileRecord:
    file_id: str
    path: str
    sha256: str
    size_bytes: int
    document_id: str | None
    match_method: str
    match_confidence: float
    status: str
    error_code: str = ""
    error_message: str = ""


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    file_id: str
    section: str
    page_start: int
    page_end: int
    text: str
    token_count: int
    is_ocr: bool
    quality: str
    content_hash: str


@dataclass(frozen=True)
class RankedHit:
    record_id: str
    document_id: str
    source: str
    rank: int
    score: float
    text: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchFilters:
    evidence_types: frozenset[str] = frozenset()
    year_from: int | None = None
    year_to: int | None = None
    fulltext_only: bool = False


@dataclass(frozen=True)
class EmbeddingItem:
    record_id: str
    document_id: str
    kind: str
    text: str
    content_hash: str
