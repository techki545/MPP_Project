# MPP Hybrid Graph RAG Knowledge Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable local literature knowledge base that indexes 35,408 metadata records and the available PDF full text, performs Qdrant plus SQLite FTS5 hybrid retrieval, constructs a question-specific evidence graph, and produces an auditable two-part evidence report in the existing web app.

**Architecture:** Add a focused `knowledge_base` package behind the existing HTTP service. Offline indexing stores manifests, FTS5 terms, build state, and embedding cache in SQLite while Qdrant local mode stores metadata and full-text vectors; online querying fuses four ranked lists, applies bounded evidence/time reranking, builds a local evidence graph, and passes only cited evidence to the chat model. The static JSON graph remains available as demo fallback until the real index is ready.

**Tech Stack:** Python 3.10+, SQLite/FTS5, qdrant-client local mode, PyMuPDF, RapidOCR with ONNX Runtime, jieba, OpenAI-compatible HTTP APIs, vanilla HTML/CSS/JavaScript, pytest, Playwright.

---

## Scope And File Map

Create these focused production files:

```text
knowledge_base/
├── __init__.py              # Public package exports
├── errors.py                # Stable domain error codes
├── config.py                # Environment and path configuration
├── models.py                # Shared dataclasses and enums
├── normalization.py         # DOI/title/text normalization and tokenization
├── sqlite_store.py          # Manifest, FTS5, jobs, errors, embedding cache
├── metadata_loader.py       # CSV/XLSX ingestion
├── deduplicator.py          # File hashing, document aliases, PDF matching
├── pdf_parser.py            # PyMuPDF extraction and RapidOCR fallback
├── chunker.py               # Section-aware, page-aware chunks
├── embedding_client.py      # OpenAI-compatible embeddings with retry
├── vector_store.py          # Qdrant local collections
├── indexer.py               # Resumable pipeline orchestration
├── retriever.py             # Four-list retrieval, RRF and diversity
├── evidence_classifier.py   # Evidence type, quality and temporal signals
├── graph_builder.py         # Claim nodes and evidence relations
├── chat_client.py           # Server-side OpenAI-compatible chat client
├── reporter.py              # Grounded two-part report generation
├── service.py               # Query and document application service
├── jobs.py                  # Background build/pause/retry lifecycle
└── cli.py                   # inspect/build/status/retry/query commands
```

Create tests under `tests/knowledge_base/`, add shared HTTP test support in `tests/web_test_support.py`, add API tests in `tests/test_web_api.py` and `tests/test_web_server.py`, and add a Playwright check in `tests/test_web_ui.py`.

Modify only these existing production files:

- `.gitignore`: ignore local corpus/index/cache artifacts.
- `README.md`: document the real knowledge-base mode and local configuration.
- `web_api.py`: delegate real queries and model status to `KnowledgeBaseService` while retaining demo fallback.
- `web_server.py`: expose knowledge-base, job, query, document, and safe PDF routes.
- `web/index.html`: replace browser API-key settings with knowledge-base status and filters.
- `web/app.js`: use the new query/status/job/document response contracts.
- `web/styles.css`: support progress, source details, status tables, and responsive layout.

Do not modify disease/case behavior in this implementation. They continue to run against their existing interfaces.

### Task 1: Establish Dependencies, Configuration, And Secret Boundaries

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `.env.example`
- Create: `knowledge_base/__init__.py`
- Create: `knowledge_base/errors.py`
- Create: `knowledge_base/config.py`
- Create: `tests/knowledge_base/test_config.py`
- Modify: `.gitignore`
- Modify: `README.md`

- [ ] **Step 1: Add the failing configuration tests**

```python
# tests/knowledge_base/test_config.py
from pathlib import Path

import pytest

from knowledge_base.config import Settings
from knowledge_base.errors import ConfigurationError


def test_settings_keep_corpus_and_index_local(tmp_path: Path) -> None:
    env = {
        "MPP_KB_SOURCE": str(tmp_path / "corpus"),
        "MPP_KB_DATA": str(tmp_path / "index"),
        "MPP_API_BASE": "https://provider.example/v1",
        "MPP_API_KEY": "secret-for-test",
        "MPP_EMBEDDING_MODEL": "text-embedding-3-large",
        "MPP_CHAT_MODEL": "chat-model",
    }
    settings = Settings.from_mapping(env, project_root=tmp_path)
    assert settings.source_dir == tmp_path / "corpus"
    assert settings.data_dir == tmp_path / "index"
    assert settings.qdrant_dir == tmp_path / "index" / "qdrant"
    assert settings.sqlite_path == tmp_path / "index" / "manifest.sqlite3"


def test_chat_settings_report_exact_missing_fields(tmp_path: Path) -> None:
    settings = Settings.from_mapping(
        {"MPP_KB_SOURCE": str(tmp_path / "corpus")},
        project_root=tmp_path,
    )
    with pytest.raises(ConfigurationError) as error:
        settings.require_chat_access()
    assert error.value.code == "model_config_missing"
    assert error.value.details == {
        "missing": ["MPP_API_BASE", "MPP_API_KEY", "MPP_CHAT_MODEL"]
    }


def test_embedding_can_run_without_chat_model(tmp_path: Path) -> None:
    settings = Settings.from_mapping(
        {
            "MPP_KB_SOURCE": str(tmp_path / "corpus"),
            "MPP_API_BASE": "https://provider.example/v1",
            "MPP_API_KEY": "secret-for-test",
        },
        project_root=tmp_path,
    )
    settings.require_embedding_access()
```

- [ ] **Step 2: Run the tests and verify the package does not exist yet**

Run: `python -m pytest tests/knowledge_base/test_config.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'knowledge_base'`.

- [ ] **Step 3: Add pinned-compatible dependencies and local-only examples**

```text
# requirements.txt
PyMuPDF>=1.24,<2
filelock>=3.16,<4
jieba==0.42.1
onnxruntime>=1.20,<2
openpyxl>=3.1,<4
qdrant-client>=1.18,<2
rapidocr>=3.9,<4
```

```text
# requirements-dev.txt
-r requirements.txt
playwright>=1.50,<2
pytest>=8,<10
pytest-cov>=5,<7
```

```text
# .env.example
MPP_KB_SOURCE=C:\Users\LTC\Desktop\MPP
MPP_KB_DATA=.local\knowledge_base
MPP_API_BASE=https://provider.example/v1
MPP_API_KEY=
MPP_EMBEDDING_MODEL=text-embedding-3-large
MPP_CHAT_MODEL=
```

Append these exact ignore entries to `.gitignore`:

```gitignore
.local/
*.sqlite3
*.sqlite3-shm
*.sqlite3-wal
qdrant/
ocr-cache/
embedding-cache/
```

The existing `.env.*` rule also matches `.env.example`; add `!.env.example` immediately after it so the empty example is tracked while every real environment file remains ignored.

- [ ] **Step 4: Implement stable configuration and domain errors**

```python
# knowledge_base/errors.py
from __future__ import annotations

from typing import Any


class KnowledgeBaseError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class ConfigurationError(KnowledgeBaseError):
    pass
```

```python
# knowledge_base/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import ConfigurationError


@dataclass(frozen=True)
class Settings:
    project_root: Path
    source_dir: Path
    data_dir: Path
    api_base: str
    api_key: str
    embedding_model: str
    chat_model: str
    embedding_batch_size: int = 32

    @property
    def qdrant_dir(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "manifest.sqlite3"

    @classmethod
    def from_mapping(cls, env: Mapping[str, str], *, project_root: Path) -> "Settings":
        source_value = env.get("MPP_KB_SOURCE", r"C:\Users\LTC\Desktop\MPP")
        data_value = env.get("MPP_KB_DATA", ".local/knowledge_base")
        source_dir = Path(source_value).expanduser()
        data_dir = Path(data_value).expanduser()
        if not data_dir.is_absolute():
            data_dir = project_root / data_dir
        return cls(
            project_root=project_root,
            source_dir=source_dir,
            data_dir=data_dir,
            api_base=env.get("MPP_API_BASE", "").strip().rstrip("/"),
            api_key=env.get("MPP_API_KEY", "").strip(),
            embedding_model=env.get("MPP_EMBEDDING_MODEL", "text-embedding-3-large").strip(),
            chat_model=env.get("MPP_CHAT_MODEL", "").strip(),
            embedding_batch_size=int(env.get("MPP_EMBEDDING_BATCH_SIZE", "32")),
        )

    def require_embedding_access(self) -> None:
        required = {
            "MPP_API_BASE": self.api_base,
            "MPP_API_KEY": self.api_key,
            "MPP_EMBEDDING_MODEL": self.embedding_model,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ConfigurationError(
                "embedding_config_missing",
                "向量模型环境变量不完整。",
                details={"missing": missing},
            )

    def require_chat_access(self) -> None:
        required = {
            "MPP_API_BASE": self.api_base,
            "MPP_API_KEY": self.api_key,
            "MPP_CHAT_MODEL": self.chat_model,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ConfigurationError(
                "model_config_missing",
                "模型环境变量不完整。",
                details={"missing": missing},
            )
```

Export `Settings`, `KnowledgeBaseError`, and `ConfigurationError` from `knowledge_base/__init__.py`.

- [ ] **Step 5: Run the focused tests**

Run: `python -m pytest tests/knowledge_base/test_config.py -v`

Expected: `3 passed`.

- [ ] **Step 6: Update the runtime baseline in README**

Change the documented baseline from Python 3.9 to Python 3.10+, add `python -m pip install -r requirements.txt`, and state that API keys are backend environment variables only. Do not document a browser key input.

- [ ] **Step 7: Commit the foundation**

```powershell
git add .gitignore .env.example README.md requirements.txt requirements-dev.txt knowledge_base tests/knowledge_base/test_config.py
git commit -m "build: add knowledge base configuration"
```

### Task 2: Add Shared Models And The SQLite Manifest/FTS Store

**Files:**
- Create: `knowledge_base/models.py`
- Create: `knowledge_base/sqlite_store.py`
- Create: `tests/knowledge_base/test_sqlite_store.py`

- [ ] **Step 1: Write failing persistence and FTS tests**

```python
# tests/knowledge_base/test_sqlite_store.py
from pathlib import Path

from knowledge_base.models import ChunkRecord, DocumentRecord, FileRecord
from knowledge_base.sqlite_store import SQLiteStore


def test_document_and_chunk_round_trip_with_fts(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "manifest.sqlite3")
    store.initialize()
    document = DocumentRecord(
        document_id="doc-1",
        source_row=2,
        title="儿童重症支原体肺炎糖皮质激素治疗",
        normalized_title="儿童重症支原体肺炎糖皮质激素治疗",
        authors=("张三",),
        year=2025,
        journal="测试期刊",
        doi="10.1000/test",
        normalized_doi="10.1000/test",
        abstract="比较低剂量和高剂量甲泼尼龙。",
        language="zh",
    )
    file_record = FileRecord(
        file_id="file-1",
        path="C:/corpus/article.pdf",
        sha256="pdf-hash",
        size_bytes=100,
        document_id="doc-1",
        match_method="source_id",
        match_confidence=1.0,
        status="matched",
    )
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        file_id="file-1",
        section="结果",
        page_start=3,
        page_end=3,
        text="低剂量组与高剂量组疗效相近。",
        token_count=15,
        is_ocr=False,
        quality="extracted",
        content_hash="hash-1",
    )
    store.upsert_document(document)
    store.upsert_file(file_record)
    store.upsert_chunk(chunk)
    assert store.get_document("doc-1") == document
    hits = store.search_fulltext("低剂量 高剂量", limit=10)
    assert [hit.record_id for hit in hits] == ["chunk-1"]
    assert hits[0].document_id == "doc-1"


def test_initialize_enables_wal_and_schema_version(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "manifest.sqlite3")
    store.initialize()
    assert store.schema_version() == 1
    assert store.journal_mode().lower() == "wal"
```

- [ ] **Step 2: Run the tests and verify missing models/store failures**

Run: `python -m pytest tests/knowledge_base/test_sqlite_store.py -v`

Expected: collection fails because `knowledge_base.models` and `knowledge_base.sqlite_store` do not exist.

- [ ] **Step 3: Define stable shared records**

```python
# knowledge_base/models.py
from __future__ import annotations

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
```

- [ ] **Step 4: Implement SQLite schema and explicit store methods**

`SQLiteStore.initialize()` must create `documents`, `files`, `document_aliases`, `chunks`, `embedding_cache`, `build_jobs`, and `ingest_errors` tables plus `metadata_fts` and `fulltext_fts` virtual tables. Use `PRAGMA foreign_keys=ON`, `PRAGMA journal_mode=WAL`, a `schema_info` row with version `1`, parameterized SQL only, and one transaction per batch.

Use these exact FTS definitions:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS metadata_fts USING fts5(
    document_id UNINDEXED,
    title_tokens,
    abstract_tokens
);
CREATE VIRTUAL TABLE IF NOT EXISTS fulltext_fts USING fts5(
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    section_tokens,
    body_tokens
);
```

Implement `upsert_document`, `get_document`, `upsert_file`, `upsert_chunk`, `search_metadata`, `search_fulltext`, `list_pending_embedding_items`, `mark_embedding_indexed`, `set_job_state`, `record_error`, `get_cached_embedding`, and `put_cached_embedding`. Store vectors as zlib-compressed `array('f').tobytes()` blobs keyed by `(content_hash, model_name)`.

The FTS queries must use `ORDER BY rank LIMIT ?`; never interpolate a raw user query into SQL. Escape FTS operators by quoting each jieba-tokenized term and joining the quoted terms with `OR` for recall.

- [ ] **Step 5: Run SQLite tests**

Run: `python -m pytest tests/knowledge_base/test_sqlite_store.py -v`

Expected: `2 passed` and no SQLite files outside pytest temporary directories.

- [ ] **Step 6: Commit the manifest store**

```powershell
git add knowledge_base/models.py knowledge_base/sqlite_store.py tests/knowledge_base/test_sqlite_store.py
git commit -m "feat: add literature manifest store"
```

### Task 3: Import Metadata, Normalize IDs, And Match/Deduplicate PDFs

**Files:**
- Create: `knowledge_base/normalization.py`
- Create: `knowledge_base/metadata_loader.py`
- Create: `knowledge_base/deduplicator.py`
- Create: `tests/knowledge_base/test_metadata_loader.py`
- Create: `tests/knowledge_base/test_deduplicator.py`

- [ ] **Step 1: Write failing normalization and metadata tests**

```python
# tests/knowledge_base/test_metadata_loader.py
from pathlib import Path

from knowledge_base.metadata_loader import load_csv_documents


def test_load_csv_ignores_blank_header_and_builds_stable_id(tmp_path: Path) -> None:
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text(
        "ID,Item Type,Author,,Publication Year,Title,Publication Title,DOI,Url,Abstract Note,Pages,Language\n"
        "12,journalArticle,Zhang,,2025, MPP Steroid Trial ,Journal,https://doi.org/10.1/ABC,,Results,1-8,en\n",
        encoding="utf-8",
    )
    documents = list(load_csv_documents(csv_path))
    assert len(documents) == 1
    assert documents[0].normalized_doi == "10.1/abc"
    assert documents[0].document_id.startswith("doi-")
    assert documents[0].title == "MPP Steroid Trial"
```

```python
# tests/knowledge_base/test_deduplicator.py
from pathlib import Path

from knowledge_base.deduplicator import hash_file, match_pdf
from knowledge_base.models import DocumentRecord


def _document(document_id: str, title: str, year: int) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        source_row=2,
        title=title,
        normalized_title=title.lower(),
        authors=("Zhang",),
        year=year,
        journal="Journal",
        doi="",
        normalized_doi="",
        abstract="",
        language="en",
    )


def test_hash_file_detects_exact_duplicate(tmp_path: Path) -> None:
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.pdf"
    first.write_bytes(b"same-pdf")
    second.write_bytes(b"same-pdf")
    assert hash_file(first) == hash_file(second)


def test_match_pdf_prefers_numeric_dataset_id() -> None:
    documents = {"12": _document("doc-12", "MPP Steroid Trial", 2025)}
    result = match_pdf(Path("12-MPP Steroid Trial.pdf"), documents_by_source_id=documents)
    assert result.document_id == "doc-12"
    assert result.method == "source_id"
    assert result.confidence == 1.0
```

- [ ] **Step 2: Run tests and verify import failures**

Run: `python -m pytest tests/knowledge_base/test_metadata_loader.py tests/knowledge_base/test_deduplicator.py -v`

Expected: collection fails because the new modules do not exist.

- [ ] **Step 3: Implement canonical normalization and stable IDs**

```python
# knowledge_base/normalization.py
from __future__ import annotations

import hashlib
import re
import unicodedata

import jieba


def clean_text(value: object) -> str:
    return " ".join(str(value or "").replace("\ufeff", "").split())


def normalize_doi(value: object) -> str:
    doi = clean_text(value).lower()
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", doi)
    return doi.rstrip(" .")


def normalize_title(value: object) -> str:
    title = unicodedata.normalize("NFKC", clean_text(value)).lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", title)


def stable_document_id(doi: str, title: str, first_author: str, year: int | None) -> str:
    if doi:
        return "doi-" + hashlib.sha256(doi.encode("utf-8")).hexdigest()[:24]
    basis = "|".join((title, first_author.lower(), str(year or "")))
    return "meta-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def tokenize_for_fts(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", clean_text(text)).lower()
    terms = [term.strip() for term in jieba.cut_for_search(normalized) if term.strip()]
    return " ".join(terms)
```

- [ ] **Step 4: Implement streaming CSV/XLSX import and PDF matching**

`load_csv_documents(path)` must open with `utf-8-sig`, fall back to `gb18030` only on decode failure, ignore the blank fourth header, and stream one `DocumentRecord` per data row. Map the exact source columns `ID`, `Author`, `Publication Year`, `Title`, `Publication Title`, `DOI`, `Url`, `Abstract Note`, and `Language`. Rows without both title and abstract remain stored with a low-information status but are not embedded.

Add `load_download_flags(paths)` using `openpyxl.load_workbook(read_only=True, data_only=True)` and return `{source_id: tuple[flag, ...]}` without loading all cell objects into memory.

`deduplicator.py` must provide:

```python
@dataclass(frozen=True)
class MatchResult:
    document_id: str | None
    method: str
    confidence: float
    ambiguous_ids: tuple[str, ...] = ()
```

Implement matching in this order: normalized DOI found in filename, numeric filename prefix/source ID, exact normalized title, then title similarity at least `0.92` plus matching year or first author. If more than one candidate satisfies the same strongest rule, return `document_id=None`, `method="ambiguous"`, and all candidate IDs. Use streaming SHA256 reads in 1 MiB blocks.

Build lookup maps for source ID, DOI, exact normalized title, normalized-title prefix, and `(year, first_author)`. Run expensive title similarity only inside the prefix/author/year candidate block; never compare every PDF against all35,408 metadata rows.

- [ ] **Step 5: Run metadata and deduplication tests**

Run: `python -m pytest tests/knowledge_base/test_metadata_loader.py tests/knowledge_base/test_deduplicator.py -v`

Expected: all tests pass.

- [ ] **Step 6: Add a 35,408-row count regression using the real CSV without writing to the corpus**

Run:

```powershell
python -c "from pathlib import Path; from knowledge_base.metadata_loader import load_csv_documents; p=Path(r'C:\Users\LTC\Desktop\MPP\MPP-fenlei-35408-初筛完.csv'); print(sum(1 for _ in load_csv_documents(p)))"
```

Expected: `35408`.

- [ ] **Step 7: Commit metadata ingestion**

```powershell
git add knowledge_base/normalization.py knowledge_base/metadata_loader.py knowledge_base/deduplicator.py tests/knowledge_base/test_metadata_loader.py tests/knowledge_base/test_deduplicator.py
git commit -m "feat: import and deduplicate MPP literature"
```

### Task 4: Parse PDFs, Apply Page-Level OCR, And Produce Traceable Chunks

**Files:**
- Create: `knowledge_base/pdf_parser.py`
- Create: `knowledge_base/chunker.py`
- Create: `tests/knowledge_base/test_pdf_parser.py`
- Create: `tests/knowledge_base/test_chunker.py`

- [ ] **Step 1: Write failing parser and chunker tests**

```python
# tests/knowledge_base/test_pdf_parser.py
from pathlib import Path

import pymupdf

from knowledge_base.pdf_parser import PDFParser


def test_parser_keeps_page_numbers_for_text_pdf(tmp_path: Path) -> None:
    path = tmp_path / "article.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Methods randomized trial")
    document.save(path)
    document.close()
    result = PDFParser(ocr=lambda image: "").parse(path)
    assert result.status == "parsed"
    assert result.pages[0].page_number == 1
    assert "randomized trial" in result.pages[0].text
    assert result.pages[0].used_ocr is False


def test_parser_uses_injected_ocr_for_empty_page(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(path)
    document.close()
    result = PDFParser(ocr=lambda image: "OCR识别结果").parse(path)
    assert result.pages[0].text == "OCR识别结果"
    assert result.pages[0].used_ocr is True
```

```python
# tests/knowledge_base/test_chunker.py
from knowledge_base.chunker import chunk_pages
from knowledge_base.pdf_parser import ParsedPage


def test_chunks_preserve_section_and_page_range() -> None:
    pages = (
        ParsedPage(1, "摘要\n研究目的。", False, "extracted"),
        ParsedPage(2, "结果\n低剂量与高剂量疗效相近。", False, "extracted"),
    )
    chunks = chunk_pages("doc-1", "file-1", pages, target_tokens=12, overlap_tokens=3)
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert any(chunk.section == "结果" for chunk in chunks)
```

- [ ] **Step 2: Run tests and verify missing parser failures**

Run: `python -m pytest tests/knowledge_base/test_pdf_parser.py tests/knowledge_base/test_chunker.py -v`

Expected: collection fails because parser and chunker modules do not exist.

- [ ] **Step 3: Implement page extraction with an injectable OCR boundary**

Define immutable `ParsedPage(page_number, text, used_ocr, quality)` and `ParsedPDF(status, pages, error_code, error_message)`. `PDFParser.parse()` must:

1. open with `pymupdf.open(path)`;
2. call `page.get_text("text", sort=True)`;
3. mark text low-quality if fewer than40 non-whitespace characters or more than30% replacement/control characters;
4. render only low-quality pages using `page.get_pixmap(dpi=200, alpha=False)`;
5. call an injected OCR callable with PNG bytes;
6. catch encrypted, truncated, and open errors into stable result codes without raising past the file boundary.

The production OCR factory must initialize `rapidocr.RapidOCR()` once per worker and join non-empty `result.txts` with newlines. It must never send images or PDF bytes to the model provider.

- [ ] **Step 4: Implement section-aware token-window chunking**

Use headings matching `摘要|abstract|方法|methods|结果|results|讨论|discussion|结论|conclusion|参考文献|references`. Count each CJK character and each alphanumeric word as one approximate token. Accumulate paragraphs to the configured target, split before900 tokens, and carry the last100 tokens into the next chunk. Build `chunk_id` from SHA256 of `document_id|file_id|page_start|page_end|section|text`; retain OCR and quality flags if any contributing page used OCR.

- [ ] **Step 5: Verify RapidOCR and run parser tests**

Run: `rapidocr check`

Expected: command exits successfully and reports the ONNX Runtime engine.

Run: `python -m pytest tests/knowledge_base/test_pdf_parser.py tests/knowledge_base/test_chunker.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit PDF extraction and chunking**

```powershell
git add knowledge_base/pdf_parser.py knowledge_base/chunker.py tests/knowledge_base/test_pdf_parser.py tests/knowledge_base/test_chunker.py
git commit -m "feat: parse and chunk literature PDFs"
```

### Task 5: Add Resilient Embeddings And Qdrant Local Collections

**Files:**
- Create: `knowledge_base/embedding_client.py`
- Create: `knowledge_base/vector_store.py`
- Create: `tests/knowledge_base/test_embedding_client.py`
- Create: `tests/knowledge_base/test_vector_store.py`

- [ ] **Step 1: Write failing transport and vector-store tests**

```python
# tests/knowledge_base/test_embedding_client.py
from knowledge_base.embedding_client import EmbeddingClient


def test_probe_uses_embeddings_endpoint_and_discovers_dimension() -> None:
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        return {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}

    client = EmbeddingClient(
        base_url="https://provider.example/v1",
        api_key="test-key",
        model="text-embedding-3-large",
        transport=transport,
        sleep=lambda seconds: None,
    )
    assert client.probe() == 3
    assert calls[0][0] == "https://provider.example/v1/embeddings"
    assert calls[0][2]["input"] == ["MPP embedding probe"]
```

```python
# tests/knowledge_base/test_vector_store.py
from pathlib import Path

from knowledge_base.vector_store import LocalVectorStore


def test_local_qdrant_upsert_and_query(tmp_path: Path) -> None:
    store = LocalVectorStore(":memory:")
    store.ensure_collections(dimension=3, model_name="test-model")
    store.upsert_metadata(
        [("doc-1", [1.0, 0.0, 0.0], {"document_id": "doc-1", "title": "RCT"})]
    )
    hits = store.query_metadata([1.0, 0.0, 0.0], limit=5)
    assert hits[0].document_id == "doc-1"
    assert hits[0].payload["title"] == "RCT"
```

- [ ] **Step 2: Run tests and verify missing modules**

Run: `python -m pytest tests/knowledge_base/test_embedding_client.py tests/knowledge_base/test_vector_store.py -v`

Expected: collection fails because both modules are absent.

- [ ] **Step 3: Implement the OpenAI-compatible embedding client**

`EmbeddingClient.embed(texts)` must send `{model, input}` to the normalized `/embeddings` URL, sort response items by `index`, reject missing/non-numeric vectors, and require one vector per input. Map HTTP 401/403 to `embedding_auth_failed`, 429 to retryable rate limiting, 5xx and network timeouts to retryable transport errors, and other 4xx responses to `embedding_request_rejected`.

Use at most four attempts with delays `1, 2, 4` seconds plus a random value from0 to0.25 seconds. Do not retry authentication or dimension errors. Never include the API key, upstream body, or source text in user-facing errors.

`probe()` must embed exactly `MPP embedding probe`, return the actual dimension, and reject an empty vector.

- [ ] **Step 4: Implement persistent and in-memory Qdrant adapters**

Initialize persistent mode with `QdrantClient(path=str(settings.qdrant_dir))` and tests with `QdrantClient(":memory:")`. Create `mpp_metadata` and `mpp_fulltext` using cosine distance and the probed dimension. Store model name and schema version in collection payload metadata through a reserved manifest point or SQLite schema record; reject an existing collection whose dimension/model differs.

Convert arbitrary record IDs to stable UUID5 point IDs. Use `upload_points` or batched `upsert`, not one request per point. Query with `query_points(..., with_payload=True).points` and convert results to `RankedHit`.

- [ ] **Step 5: Run embedding and vector tests**

Run: `python -m pytest tests/knowledge_base/test_embedding_client.py tests/knowledge_base/test_vector_store.py -v`

Expected: all tests pass without network access and without creating a persistent Qdrant directory.

- [ ] **Step 6: Commit vector infrastructure**

```powershell
git add knowledge_base/embedding_client.py knowledge_base/vector_store.py tests/knowledge_base/test_embedding_client.py tests/knowledge_base/test_vector_store.py
git commit -m "feat: add embedding and vector stores"
```

### Task 6: Build A Resumable Indexer And CLI

**Files:**
- Create: `knowledge_base/indexer.py`
- Create: `knowledge_base/cli.py`
- Create: `tests/knowledge_base/test_indexer.py`
- Create: `tests/knowledge_base/test_cli.py`

- [ ] **Step 1: Write a failing resume test**

```python
# tests/knowledge_base/test_indexer.py
from pathlib import Path

from knowledge_base.indexer import Indexer
from knowledge_base.models import ChunkRecord, DocumentRecord, FileRecord
from knowledge_base.sqlite_store import SQLiteStore


class FakeEmbeddingClient:
    def __init__(self):
        self.embedded_text_count = 0

    def embed(self, texts):
        self.embedded_text_count += len(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self):
        self.indexed_ids = []

    def upsert_metadata(self, items):
        self.indexed_ids.extend(item[0] for item in items)

    def upsert_fulltext(self, items):
        self.indexed_ids.extend(item[0] for item in items)


def test_indexer_resumes_without_reembedding_completed_items(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "manifest.sqlite3")
    store.initialize()
    store.upsert_document(DocumentRecord(
        document_id="doc-1", source_row=2, title="Trial", normalized_title="trial",
        authors=("Zhang",), year=2025, journal="Journal", doi="", normalized_doi="",
        abstract="Steroid trial", language="en",
    ))
    store.upsert_file(FileRecord(
        file_id="file-1", path=str(tmp_path / "trial.pdf"), sha256="pdf-hash",
        size_bytes=100, document_id="doc-1", match_method="source_id",
        match_confidence=1.0, status="matched",
    ))
    store.upsert_chunk(ChunkRecord(
        chunk_id="chunk-1", document_id="doc-1", file_id="file-1", section="Results",
        page_start=1, page_end=1, text="Low dose result", token_count=3,
        is_ocr=False, quality="extracted", content_hash="chunk-hash",
    ))
    embedder = FakeEmbeddingClient()
    vectors = FakeVectorStore()
    indexer = Indexer(
        store=store,
        embedding_client=embedder,
        vector_store=vectors,
        lock_path=tmp_path / "build.lock",
    )
    first = indexer.embed_pending(confirm_embedding_cost=True)
    second = indexer.embed_pending(confirm_embedding_cost=True)
    assert first.completed_embeddings == 2
    assert second.completed_embeddings == 0
    assert second.skipped_embeddings == 2
    assert embedder.embedded_text_count == 2
```

```python
# tests/knowledge_base/test_cli.py
from knowledge_base.cli import build_parser


def test_build_requires_explicit_embedding_confirmation() -> None:
    parser = build_parser()
    args = parser.parse_args(["build", "--embedding-limit", "32"])
    assert args.confirm_embedding_cost is False
    assert args.embedding_limit == 32


def test_inspect_accepts_read_only_source_override() -> None:
    parser = build_parser()
    args = parser.parse_args(["inspect", "--source", r"C:\corpus"])
    assert args.command == "inspect"
    assert args.source == r"C:\corpus"
```

- [ ] **Step 2: Run indexer/CLI tests and verify failures**

Run: `python -m pytest tests/knowledge_base/test_indexer.py tests/knowledge_base/test_cli.py -v`

Expected: collection fails because `Indexer` and `build_parser` do not exist.

- [ ] **Step 3: Implement explicit pipeline stages and checkpoints**

`Indexer.inspect()` must count source files, identify the metadata CSV/XLSX files, count CSV records, hash PDF files, and return a report without creating or modifying anything under the corpus directory.

`Indexer` receives the SQLite store, embedding client, vector store, and lock path as explicit constructor dependencies; parsing-stage collaborators are supplied to `build()` through a `BuildPipeline` record. `Indexer.build()` must execute these idempotent stages:

```text
metadata -> pdf_inventory -> pdf_match -> parse -> chunk -> lexical -> embedding -> vector
```

Define immutable `InspectionReport`, `BuildRunResult`, and `EmbeddingRunResult` records. `EmbeddingRunResult` contains `completed_embeddings`, `skipped_embeddings`, and `failed_embeddings`; `BuildRunResult` contains per-stage counters, final state, pending embedding count, and error counts. Their `as_dict()` methods are the single source for CLI and Web job JSON.

Before each stage, select only records not already completed for the current input hash and algorithm version. Commit SQLite state after each document and each embedding batch. Cache embeddings before Qdrant upsert so a failed vector write can retry without another paid call. Use `filelock.FileLock(data_dir / "build.lock")` to prevent concurrent writers.

If `confirm_embedding_cost=False`, complete all local stages, report `embedding_pending`, print the exact pending text count, and exit successfully without calling the API. Check the `paused` job flag after every PDF and embedding batch; pause only at these transaction boundaries.

- [ ] **Step 4: Implement CLI commands with JSON-capable output**

`build_parser()` must expose:

```text
inspect --source PATH --json
build --document-limit N --embedding-limit N --confirm-embedding-cost --json
status --json
pause --json
retry --stage STAGE --confirm-embedding-cost --json
query QUESTION --fulltext-only --year-from YEAR --year-to YEAR --json
```

`--document-limit` caps metadata/PDF documents processed by local stages for smoke tests. `--embedding-limit` independently caps the number of pending metadata/chunk texts sent to the paid embedding API. Omitting either flag means no limit for that stage.

`python -m knowledge_base.cli` must return exit code0 for successful/paused work,2 for configuration or validation errors,3 for model authentication errors, and4 for corrupt/incompatible indexes. Human output must not print secrets; JSON output uses stable `code`, `message`, and `details` keys.

- [ ] **Step 5: Run indexer and CLI tests**

Run: `python -m pytest tests/knowledge_base/test_indexer.py tests/knowledge_base/test_cli.py -v`

Expected: all tests pass using injected fake parser, embedding client, SQLite store, and vector store.

- [ ] **Step 6: Run a read-only corpus inspection**

Run: `python -m knowledge_base.cli inspect --source "C:\Users\LTC\Desktop\MPP" --json`

Expected report fields: `metadata_records=35408`, `pdf_files=909`, `xlsx_files=2`, `csv_files=1`, `zip_files=1`. The command must not create files under `C:\Users\LTC\Desktop\MPP`.

- [ ] **Step 7: Commit the resumable pipeline**

```powershell
git add knowledge_base/indexer.py knowledge_base/cli.py tests/knowledge_base/test_indexer.py tests/knowledge_base/test_cli.py
git commit -m "feat: add resumable knowledge base indexing"
```

### Task 7: Implement Four-List Hybrid Retrieval And Bounded Evidence Reranking

**Files:**
- Create: `knowledge_base/retriever.py`
- Create: `tests/knowledge_base/test_retriever.py`

- [ ] **Step 1: Write failing fusion, deduplication, and degradation tests**

```python
# tests/knowledge_base/test_retriever.py
from knowledge_base.models import RankedHit, SearchFilters
from knowledge_base.retriever import HybridRetriever


class FakeEmbeddingClient:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def embed(self, texts):
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeVectorStore:
    def query_metadata(self, vector, limit, filters=None):
        return [RankedHit("doc-a", "doc-a", "vector_metadata", 1, 0.9, "meta a")]

    def query_fulltext(self, vector, limit, filters=None):
        return [RankedHit("chunk-a", "doc-a", "vector_fulltext", 1, 0.95, "full a")]


class FakeLexicalStore:
    def search_metadata(self, query, limit, filters=None):
        return [RankedHit("doc-b", "doc-b", "fts_metadata", 1, -1.0, "meta b")]

    def search_fulltext(self, query, limit, filters=None):
        return [RankedHit("chunk-a", "doc-a", "fts_fulltext", 1, -0.5, "full a")]


def test_hybrid_retrieval_fuses_four_lists_and_deduplicates_documents() -> None:
    retriever = HybridRetriever(FakeEmbeddingClient(), FakeVectorStore(), FakeLexicalStore())
    result = retriever.search("低剂量激素", SearchFilters())
    assert result.mode == "hybrid"
    assert [item.document_id for item in result.documents] == ["doc-a", "doc-b"]
    assert len(result.documents[0].supporting_hits) == 2


def test_embedding_failure_returns_explicit_keyword_mode() -> None:
    retriever = HybridRetriever(FakeEmbeddingClient(fail=True), FakeVectorStore(), FakeLexicalStore())
    result = retriever.search("低剂量激素", SearchFilters())
    assert result.mode == "keyword"
    assert result.degraded_reason == "query_embedding_unavailable"
    assert [item.document_id for item in result.documents] == ["doc-b"]
```

- [ ] **Step 2: Run the tests and verify the retriever is absent**

Run: `python -m pytest tests/knowledge_base/test_retriever.py -v`

Expected: collection fails because `knowledge_base.retriever` does not exist.

- [ ] **Step 3: Implement query normalization and Reciprocal Rank Fusion**

Define `RetrievedDocument(document_id, fused_score, final_score, supporting_hits, payload)` and `RetrievalResult(mode, degraded_reason, documents, candidate_count)`.

Execute these lists independently:

```text
vector metadata: limit 40
vector full text: limit 80
FTS metadata: limit 40
FTS full text: limit 80
```

Create a `QueryContext` containing the raw question, normalized FTS query, detected population/intervention/comparator/outcome terms, and optional year/type filters. Use deterministic Chinese/English dictionaries for the known MPP disease, drug, dose, safety, and outcome terms; preserve unknown clinical terms instead of dropping them. Model-assisted PICO refinement is optional and must fall back to this deterministic context when chat access is unavailable.

For every list use one-based rank and `1 / (60 + rank)`. Merge hits first by `(source, record_id)`, then group by `document_id`; keep at most three full-text chunks per document. Normalize fused document scores by the maximum fused score in the query.

Calculate the bounded final score exactly as:

```python
final_score = (
    0.65 * relevance_score
    + 0.15 * evidence_level_score
    + 0.10 * quality_score
    + 0.05 * recency_score
    + 0.05 * source_completeness_score
)
```

Use deterministic mappings:

```python
EVIDENCE_LEVEL_SCORE = {
    "guideline": 1.0,
    "systematic_review": 0.90,
    "randomized_controlled_trial": 0.85,
    "observational_study": 0.55,
    "narrative_review": 0.35,
    "case_report": 0.20,
    "unknown": 0.10,
}
QUALITY_SCORE = {"high": 1.0, "moderate": 0.7, "low": 0.4, "very_low": 0.2, "unknown": 0.3}
```

Set source completeness to1.0 for a matched full-text chunk and0.4 for metadata-only. Normalize publication year into0.0 to1.0 over the candidate set; if all years are equal or missing, use0.5. After sorting, cap the evidence package at15 documents and retain at least the highest-relevance guideline, systematic review, RCT, and safety-focused lower-level study when present in the top30 candidates.

- [ ] **Step 4: Enforce filters before policy bonuses**

Apply `SearchFilters` to both Qdrant payload conditions and SQLite results. `fulltext_only=True` removes metadata-only documents; year bounds are inclusive; an empty evidence-type set means no type filter. Reject `year_from > year_to` with `invalid_year_range`.

If query embedding fails for authentication, timeout, or upstream availability, run both FTS searches and return `mode="keyword"`. If SQLite FTS is unavailable as well, raise `retrieval_unavailable`; do not return the static demo as if it came from the real corpus.

- [ ] **Step 5: Run retrieval tests**

Run: `python -m pytest tests/knowledge_base/test_retriever.py -v`

Expected: both tests pass, with `doc-a` first in hybrid mode and only `doc-b` in degraded mode.

- [ ] **Step 6: Commit hybrid retrieval**

```powershell
git add knowledge_base/retriever.py tests/knowledge_base/test_retriever.py
git commit -m "feat: add hybrid literature retrieval"
```

### Task 8: Classify Evidence, Extract Claims, And Build The Local Evidence Graph

**Files:**
- Create: `knowledge_base/evidence_classifier.py`
- Create: `knowledge_base/graph_builder.py`
- Create: `tests/knowledge_base/test_evidence_classifier.py`
- Create: `tests/knowledge_base/test_graph_builder.py`

- [ ] **Step 1: Write failing evidence classification tests**

```python
# tests/knowledge_base/test_evidence_classifier.py
from knowledge_base.evidence_classifier import classify_evidence


def test_classifies_major_evidence_types_from_title_and_abstract() -> None:
    cases = {
        "Clinical practice guideline for MPP": "guideline",
        "Systematic review and meta-analysis of corticosteroids": "systematic_review",
        "A multicenter randomized controlled trial": "randomized_controlled_trial",
        "A retrospective cohort study": "observational_study",
        "Narrative review of current treatment": "narrative_review",
        "Case report of pulmonary embolism": "case_report",
    }
    assert {classify_evidence(title, "").evidence_type for title in cases} == set(cases.values())
    for title, expected in cases.items():
        assert classify_evidence(title, "").evidence_type == expected


def test_unknown_classification_does_not_invent_quality() -> None:
    assessment = classify_evidence("Steroid therapy in children", "Clinical outcomes were assessed.")
    assert assessment.evidence_type == "unknown"
    assert assessment.quality == "unknown"
    assert assessment.basis
```

- [ ] **Step 2: Write failing graph relation tests**

```python
# tests/knowledge_base/test_graph_builder.py
from knowledge_base.graph_builder import EvidenceClaim, LocalGraphBuilder


def _claim(identifier, evidence_type, year, direction, outcome, safety=False, cutoff=None):
    return EvidenceClaim(
        claim_id=identifier,
        document_id=identifier,
        evidence_type=evidence_type,
        year=year,
        evidence_cutoff_year=cutoff,
        population="SMPP children",
        intervention="methylprednisolone",
        comparator="antibiotics or another dose",
        outcome=outcome,
        direction=direction,
        safety_signal=safety,
        statement="source-grounded statement",
        source_chunk_ids=(f"chunk-{identifier}",),
    )


def test_new_rct_updates_old_guideline_and_case_adds_caution() -> None:
    guideline = _claim("g", "guideline", 2023, "supports", "lung injury", cutoff=2022)
    rct = _claim("r", "randomized_controlled_trial", 2025, "supports", "lung injury")
    case = _claim("c", "case_report", 2025, "uncertain", "pulmonary embolism", safety=True)
    graph = LocalGraphBuilder().build("Should steroids be used?", [guideline, rct, case])
    relations = {(edge.source, edge.target, edge.relation) for edge in graph.edges}
    assert ("r", "g", "updates") in relations
    assert any(edge.relation == "cautions" and edge.source == "c" for edge in graph.edges)


def test_comparable_opposite_directions_create_conflict() -> None:
    first = _claim("a", "randomized_controlled_trial", 2024, "supports", "fever duration")
    second = _claim("b", "randomized_controlled_trial", 2025, "opposes", "fever duration")
    graph = LocalGraphBuilder().build("question", [first, second])
    assert any(edge.relation == "conflicts" for edge in graph.edges)
```

- [ ] **Step 3: Run the tests and verify missing classifier/graph modules**

Run: `python -m pytest tests/knowledge_base/test_evidence_classifier.py tests/knowledge_base/test_graph_builder.py -v`

Expected: collection fails because both modules are absent.

- [ ] **Step 4: Implement rule-first classification with explicit basis**

Match case-insensitive English and Chinese terms in priority order: guideline/consensus, systematic review/meta-analysis, randomized trial, cohort/case-control/cross-sectional, case report/series, narrative review. Avoid classifying a paper as an RCT merely because its abstract says it included previous RCTs; title patterns and publication type fields take precedence.

Return `EvidenceAssessment(evidence_type, confidence, quality, basis, quality_signals)`. Confidence is1.0 for explicit publication type metadata,0.9 for title patterns,0.7 for abstract-only patterns, and0.0 for unknown. Quality remains unknown unless the source text explicitly provides a signal such as multicenter randomization, a named reporting standard, sample size, adjusted analysis, or guideline evidence cutoff.

Only candidates with `evidence_type="unknown"` or confidence below0.7 are eligible for later LLM refinement.

- [ ] **Step 5: Implement claim records and deterministic relation rules**

`EvidenceClaim` must include every field used by the tests plus dose, sample size, effect measures, limitations, and source quote. Fields not supplied in the tests have empty-string or empty-tuple defaults. Reject a claim with no source chunk IDs.

Create graph nodes for the question, documents, claims, and normalized outcomes. Create only these relations: `supports`, `updates`, `supplements`, `conflicts`, `cautions`.

Apply rules in this order:

1. a safety-focused case/observational claim creates `cautions` toward the question or related recommendation;
2. a study published after a guideline evidence cutoff creates `updates` only if it is an RCT or systematic review and addresses the same population/intervention/outcome;
3. comparable claims with opposite directions create `conflicts`;
4. comparable claims with the same direction create `supports`;
5. a different dose, subgroup, endpoint, or follow-up creates `supplements`.

Every edge stores the source claim IDs and a short deterministic rationale. Do not create an edge when population/intervention overlap is insufficient.

- [ ] **Step 6: Run classifier and graph tests**

Run: `python -m pytest tests/knowledge_base/test_evidence_classifier.py tests/knowledge_base/test_graph_builder.py -v`

Expected: all tests pass and graph JSON contains no legacy `confirms` relation.

- [ ] **Step 7: Commit evidence semantics**

```powershell
git add knowledge_base/evidence_classifier.py knowledge_base/graph_builder.py tests/knowledge_base/test_evidence_classifier.py tests/knowledge_base/test_graph_builder.py
git commit -m "feat: build dynamic evidence relations"
```

### Task 9: Add Server-Side Chat, Claim Extraction, And Grounded Report Generation

**Files:**
- Create: `knowledge_base/chat_client.py`
- Create: `knowledge_base/reporter.py`
- Create: `tests/knowledge_base/test_chat_client.py`
- Create: `tests/knowledge_base/test_reporter.py`

- [ ] **Step 1: Write failing server-side model and grounding tests**

```python
# tests/knowledge_base/test_chat_client.py
from knowledge_base.chat_client import ChatClient


def test_chat_client_reads_fixed_server_configuration() -> None:
    captured = {}

    def transport(url, headers, payload, timeout):
        captured.update(url=url, headers=headers, payload=payload)
        return {"choices": [{"message": {"content": '{"status":"ok"}'}}]}

    client = ChatClient("https://provider.example/v1", "secret", "chat-model", transport=transport)
    assert client.complete_json("system", {"task": "probe"}) == {"status": "ok"}
    assert captured["url"] == "https://provider.example/v1/chat/completions"
    assert captured["payload"]["model"] == "chat-model"
```

```python
# tests/knowledge_base/test_reporter.py
import pytest

from knowledge_base.errors import KnowledgeBaseError
from knowledge_base.reporter import EvidenceBundle, EvidenceSource, GroundedReporter


class FakeChatClient:
    def __init__(self, response):
        self.response = response

    def complete_json(self, system_prompt, payload):
        return self.response


def _evidence_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        sources=(EvidenceSource(
            source_number=1,
            document_id="doc-1",
            title="Randomized trial",
            evidence_type="randomized_controlled_trial",
            year=2025,
            chunk_ids=("chunk-1",),
            snippets=("Low dose was supported in the randomized trial.",),
            page_ranges=("3",),
            fulltext=True,
        ),),
        graph={"nodes": [], "edges": []},
    )


def test_reporter_accepts_only_known_source_numbers() -> None:
    reporter = GroundedReporter(FakeChatClient({
        "analysis_steps": [{"title": "证据盘点", "body": "纳入一项RCT。[1]", "source_ids": [1]}],
        "final_answer_markdown": "推荐低剂量方案。[1]",
    }))
    report = reporter.generate("question", _evidence_bundle())
    assert report.final_answer_markdown.endswith("[1]")


def test_reporter_rejects_unknown_citation() -> None:
    reporter = GroundedReporter(FakeChatClient({
        "analysis_steps": [],
        "final_answer_markdown": "结论。[99]",
    }))
    with pytest.raises(KnowledgeBaseError) as error:
        reporter.generate("question", _evidence_bundle())
    assert error.value.code == "ungrounded_model_response"
```

- [ ] **Step 2: Run tests and verify missing chat/report modules**

Run: `python -m pytest tests/knowledge_base/test_chat_client.py tests/knowledge_base/test_reporter.py -v`

Expected: collection fails because the modules do not exist.

- [ ] **Step 3: Implement a server-only chat client**

Move the reusable HTTP behavior from `web_api.call_openai_compatible` into `ChatClient`. Normalize Base URL to `/chat/completions`, request JSON output first, retry once without `response_format` only when the provider explicitly rejects that parameter, and parse JSON with `MycoplasmaPneumonia.llm_client.extract_json_object`.

Map 401/403 to `chat_auth_failed`, provider 400 to `chat_request_rejected`, timeouts/network/5xx to `chat_unavailable`, and malformed content to `chat_response_invalid`. Redact authorization and upstream response bodies.

- [ ] **Step 4: Implement structured claim extraction and report validation**

Define immutable `EvidenceSource`, `EvidenceBundle`, and `GeneratedReport` records matching the test fields. For the final8 to15 documents, first request a JSON `claims` array. Each claim must contain `source_number`, `source_chunk_ids`, PICO fields, design, sample size, dose, outcome, direction, effect measures, safety signal, limitations, statement, and source quote. Validate that every chunk ID belongs to that numbered source; discard invalid claims and retain the original text hit for audit.

Before claim extraction, send only rule-classified candidates whose type is `unknown` or whose classification confidence is below0.7 to a `ChatEvidenceRefiner`. Accept only one of the six defined evidence types or `unknown`, require a short basis tied to title/abstract text, and fall back to the rule result if the response is missing or invalid. Never run this refinement over the full corpus.

Pass the validated evidence graph and numbered evidence bundle to a second JSON call requesting:

```json
{
  "analysis_steps": [
    {"title": "第一步：检索并盘点证据库存", "body": "可核验的分析说明。[1]", "source_ids": [1]}
  ],
  "final_answer_markdown": "综合循证回答。[1]"
}
```

The system prompt must prohibit invented studies, numbers, doses, effect sizes, confidence intervals, and citations. `GroundedReporter` must reject citation numbers outside the evidence bundle and statistical tokens absent from all cited source snippets. If the second call fails validation, return the deterministic evidence inventory and graph with `model_used=False` plus a stable model error; do not discard retrieval results.

- [ ] **Step 5: Run chat and reporting tests**

Run: `python -m pytest tests/knowledge_base/test_chat_client.py tests/knowledge_base/test_reporter.py -v`

Expected: all tests pass with injected transports and no network access.

- [ ] **Step 6: Commit grounded generation**

```powershell
git add knowledge_base/chat_client.py knowledge_base/reporter.py tests/knowledge_base/test_chat_client.py tests/knowledge_base/test_reporter.py
git commit -m "feat: generate grounded evidence reports"
```

### Task 10: Add The Application Service, Background Jobs, And HTTP API

**Files:**
- Create: `knowledge_base/service.py`
- Create: `knowledge_base/jobs.py`
- Create: `tests/knowledge_base/test_service.py`
- Create: `tests/knowledge_base/test_jobs.py`
- Create: `tests/__init__.py`
- Create: `tests/web_test_support.py`
- Create: `tests/test_web_api.py`
- Create: `tests/test_web_server.py`
- Modify: `web_api.py`
- Modify: `web_server.py`

- [ ] **Step 1: Write failing service contract tests**

```python
# tests/knowledge_base/test_service.py
from pathlib import Path

from knowledge_base.config import Settings
from knowledge_base.models import SearchFilters
from knowledge_base.service import KnowledgeBaseService


class FakeQueryPipeline:
    def run(self, question, filters):
        return {
            "question": question,
            "mode": "hybrid",
            "degraded_reason": "",
            "filters": {},
            "reasoning_steps": [{"title": "证据盘点", "body": "一项RCT。[1]", "source_ids": [1]}],
            "answer_markdown": "综合回答。[1]",
            "model_used": True,
            "model_name": "test-model",
            "model_error": None,
            "summary": {"evidence_count": 1},
            "sources": [{"source_number": 1, "document_id": "doc-1", "title": "Trial"}],
            "graph": {"nodes": [{"id": "doc-1"}], "edges": []},
            "retrieval_stats": {"candidate_count": 1},
        }


class FakeDocumentStore:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    def document_detail(self, document_id):
        return {"document_id": document_id, "title": "Trial", "pdf_path": str(self.pdf_path)}


def _service(tmp_path: Path) -> KnowledgeBaseService:
    source = tmp_path / "corpus"
    source.mkdir()
    pdf_path = source / "trial.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")
    settings = Settings.from_mapping(
        {"MPP_KB_SOURCE": str(source), "MPP_KB_DATA": str(tmp_path / "index")},
        project_root=tmp_path,
    )
    return KnowledgeBaseService(settings, FakeDocumentStore(pdf_path), FakeQueryPipeline())


def test_query_returns_two_part_report_and_sources(tmp_path: Path) -> None:
    service = _service(tmp_path)
    result = service.query("SMPP儿童是否使用糖皮质激素？", SearchFilters())
    assert result["mode"] in {"hybrid", "keyword"}
    assert result["reasoning_steps"]
    assert "answer_markdown" in result
    assert result["sources"][0]["source_number"] == 1
    assert result["graph"]["nodes"]


def test_document_detail_never_exposes_arbitrary_paths(tmp_path: Path) -> None:
    service = _service(tmp_path)
    detail = service.document_detail("doc-1")
    assert "local_path" not in detail
    assert detail["pdf_available"] is True
```

```python
# tests/knowledge_base/test_jobs.py
from threading import Event

from knowledge_base.jobs import KnowledgeBaseJobManager


class PausableIndexer:
    def __init__(self):
        self.started = Event()

    def build(self, *, should_pause, confirm_embedding_cost):
        self.started.set()
        assert confirm_embedding_cost is False
        assert should_pause.wait(timeout=5) is True
        return {"state": "paused", "completed": 1}


def test_job_manager_pauses_at_cooperative_boundary(tmp_path) -> None:
    indexer = PausableIndexer()
    manager = KnowledgeBaseJobManager(indexer=indexer, store_path=tmp_path / "jobs.sqlite3")
    job = manager.start_build(confirm_embedding_cost=False)
    assert indexer.started.wait(timeout=5) is True
    manager.pause(job.job_id)
    final = manager.wait(job.job_id, timeout=5)
    assert final.state == "paused"
    assert final.progress["completed"] == 1
```

- [ ] **Step 2: Write failing route tests**

```python
# tests/test_web_api.py
from pathlib import Path

from graph_rag_core import load_default_graph
from web_api import GraphRAGWebService


class FakeKnowledgeService:
    def health(self):
        return {"status": "ready", "documents": 1}

    def model_status(self):
        return {"configured": True, "chat_model": "test-model"}

    def query(self, question, filters):
        return {"answer_markdown": "answer", "reasoning_steps": [], "sources": []}


def _web_service() -> GraphRAGWebService:
    root = Path(__file__).resolve().parents[1]
    return GraphRAGWebService(load_default_graph(str(root)), knowledge_service=FakeKnowledgeService())


def test_health_reports_demo_and_knowledge_base_components() -> None:
    web_service = _web_service()
    health = web_service.health()
    assert health["status"] in {"ok", "degraded"}
    assert "knowledge_base" in health
    assert "model" in health


def test_query_uses_backend_model_configuration() -> None:
    web_service = _web_service()
    result = web_service.query({"question": "clinical question", "evidence_types": []})
    assert "answer_markdown" in result
```

```python
# tests/web_test_support.py
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from threading import Thread
import urllib.error
import urllib.request

from web_server import create_server


@dataclass(frozen=True)
class HTTPResult:
    status: int
    json: dict


@contextmanager
def running_server(service):
    web_dir = Path(__file__).resolve().parents[1] / "web"
    server = create_server("127.0.0.1", 0, service=service, web_dir=str(web_dir))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def get_json(base_url: str, path: str) -> HTTPResult:
    try:
        with urllib.request.urlopen(base_url + path, timeout=5) as response:
            return HTTPResult(response.status, json.loads(response.read().decode("utf-8")))
    except urllib.error.HTTPError as error:
        return HTTPResult(error.code, json.loads(error.read().decode("utf-8")))
```

```python
# tests/test_web_server.py
from web_api import APIError
from tests.web_test_support import get_json, running_server


class FakeWebService:
    def document_detail(self, document_id):
        if document_id != "doc-1":
            raise APIError("not_found", "文献不存在。", status=404)
        return {"document_id": "doc-1", "title": "Trial"}


def test_dynamic_document_route_is_parsed_without_path_traversal() -> None:
    with running_server(FakeWebService()) as base_url:
        response = get_json(base_url, "/api/documents/doc-1")
        assert response.status == 200
        assert response.json["document_id"] == "doc-1"
        rejected = get_json(base_url, "/api/documents/..%2Fsecret")
        assert rejected.status in {400, 404}
```

- [ ] **Step 3: Run service and HTTP tests to verify failures**

Run: `python -m pytest tests/knowledge_base/test_service.py tests/knowledge_base/test_jobs.py tests/test_web_api.py tests/test_web_server.py -v`

Expected: tests fail because the real service, jobs, and routes are not implemented.

- [ ] **Step 4: Implement the query/document service**

`KnowledgeBaseService(settings, store, pipeline)` receives an initialized settings object, a document-store adapter, and a query pipeline with `run(question, filters)`. The production pipeline composes the hybrid retriever, classifier/refiner, claim extractor, graph builder, and reporter. `KnowledgeBaseService.query()` must validate a non-empty question capped at2000 characters, call the pipeline, and number sources only after final deduplication. Return stable keys:

```text
question, mode, degraded_reason, filters, reasoning_steps,
answer_markdown, model_used, model_name, model_error,
summary, sources, graph, retrieval_stats
```

`document_detail(document_id)` returns bibliographic data, evidence assessment, matched snippets, pages, and `pdf_available`, but never returns the absolute file path. `resolve_pdf(document_id)` may return a `Path` internally only after loading it from the SQLite file record and verifying that its resolved path is inside `Settings.source_dir` and is one of the indexed PDF files.

- [ ] **Step 5: Implement safe background job lifecycle**

`KnowledgeBaseJobManager` owns one daemon worker thread, refuses a second simultaneous build, records `queued/running/pausing/paused/completed/failed`, and keeps progress in SQLite so a server restart can report the last state. Pass a `threading.Event` as `should_pause` to the indexer; `pause()` sets that event and the indexer checks it only at safe boundaries. `wait(job_id, timeout)` returns the latest persisted job for tests and operators. The manager must never retain API keys in job records.

- [ ] **Step 6: Replace browser-supplied model configuration and add routes**

Construct `Settings`, stores, model clients, `KnowledgeBaseService`, and `KnowledgeBaseJobManager` once in `create_server()`. Keep `GraphRAGWebService` as the compatibility facade, but remove `_validate_model_config` and model dictionaries from request bodies.

`GraphRAGWebService.query(payload)` validates the payload, builds `SearchFilters`, and delegates to `KnowledgeBaseService.query(question, filters)` when the index is queryable. If the index is not built, it returns the existing deterministic static analysis with `data_source="demo"` and a visible warning; it must not label demo nodes as retrieved corpus literature. Translate `KnowledgeBaseError` codes to `APIError` status values without exposing exception traces.

Parse every request with `urllib.parse.urlsplit`; route on the decoded path and parse query parameters separately. URL-decode document/job IDs once, reject IDs containing `/`, `\\`, `..`, or control characters, and keep the existing1 MB JSON body cap.

Expose:

```text
GET  /api/health
GET  /api/evidence                  # demo catalog only
GET  /api/kb/status
GET  /api/model/status
GET  /api/jobs/{job_id}
GET  /api/documents/{document_id}
GET  /api/documents/{document_id}/pdf
POST /api/kb/build
POST /api/kb/pause
POST /api/kb/retry
POST /api/query
POST /api/analyze                   # compatibility alias to /api/query
```

Require `{"confirm_embedding_cost": true}` before a job may enter the embedding stage. Stream PDFs with `Content-Type: application/pdf`, `Content-Disposition: inline`, `X-Content-Type-Options: nosniff`, and no arbitrary filesystem route. Return JSON errors everywhere else.

- [ ] **Step 7: Run service and HTTP tests**

Run: `python -m pytest tests/knowledge_base/test_service.py tests/knowledge_base/test_jobs.py tests/test_web_api.py tests/test_web_server.py -v`

Expected: all tests pass; route tests use fake services and temporary directories.

- [ ] **Step 8: Run legacy compatibility tests**

Run: `python demo_graph_rag.py`

Expected: the static demo still renders a complete local report without model access.

Run: `python -m disease.Task_level`

Expected: seven disease-level steps are generated.

Run: `python -m case.Case_level`

Expected: the bundled sample case completes without importing the new storage internals.

- [ ] **Step 9: Commit backend integration**

```powershell
git add knowledge_base/service.py knowledge_base/jobs.py web_api.py web_server.py tests/__init__.py tests/web_test_support.py tests/knowledge_base/test_service.py tests/knowledge_base/test_jobs.py tests/test_web_api.py tests/test_web_server.py
git commit -m "feat: expose knowledge base web API"
```

### Task 11: Rebuild The Web Workbench Around Real Corpus Status And Sources

**Files:**
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `web/styles.css`
- Create: `tests/test_web_ui.py`

- [ ] **Step 1: Add a failing Playwright smoke test**

```python
# tests/test_web_ui.py
from playwright.sync_api import sync_playwright

from tests.web_test_support import running_server


class FakeWorkbenchService:
    def health(self):
        return {"status": "ok", "knowledge_base": {"status": "ready"}, "model": {"configured": True}}

    def knowledge_base_status(self):
        return {"status": "ready", "metadata_records": 35408, "pdf_files": 909, "embedded": 1}

    def model_status(self):
        return {"configured": True, "chat_model": "test-model"}

    def evidence_catalog(self):
        return {
            "default_question": "SMPP儿童是否应使用糖皮质激素？",
            "nodes": [],
            "edges": [],
            "evidence_types": [
                {"key": "randomized_controlled_trial", "label": "随机对照试验（RCT）", "count": 1}
            ],
        }

    def query(self, payload):
        return {
            "question": payload["question"],
            "mode": "hybrid",
            "degraded_reason": "",
            "reasoning_steps": [{"title": "第一步：检索并盘点证据库存", "body": "一项RCT。[1]", "source_ids": [1]}],
            "answer_markdown": "## 推荐意见\n推荐低剂量方案。[1]",
            "model_used": True,
            "model_name": "test-model",
            "summary": {"evidence_count": 1, "consistency": "方向一致", "update_count": 0, "conclusion": "低剂量优先"},
            "sources": [{"source_number": 1, "document_id": "doc-1", "title": "Randomized trial", "evidence_type": "randomized_controlled_trial"}],
            "graph": {"nodes": [{"id": "doc-1", "title": "Randomized trial", "evidence_type": "randomized_controlled_trial"}], "edges": []},
            "retrieval_stats": {"candidate_count": 1},
        }

    def document_detail(self, document_id):
        return {
            "document_id": document_id,
            "title": "Randomized trial",
            "evidence_type": "randomized_controlled_trial",
            "quality": "high",
            "matched_snippets": [{"text": "Low dose result", "section": "Results", "page_start": 3, "page_end": 3}],
            "pdf_available": True,
        }


def test_workbench_has_no_browser_api_key_and_renders_grounded_sources():
    with running_server(FakeWorkbenchService()) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(base_url)
        page.get_by_label("临床问题").fill("SMPP儿童是否应使用糖皮质激素？")
        assert page.locator("#api-key").count() == 0
        page.get_by_role("button", name="开始循证分析").click()
        page.get_by_text("第二部分：综合循证回答").wait_for()
        assert page.locator("[data-source-number='1']").count() >= 1
        page.locator("[data-source-number='1']").first.click()
        page.get_by_text("命中的正文片段").wait_for()
        browser.close()


def test_mobile_layout_has_no_horizontal_overflow():
    with running_server(FakeWorkbenchService()) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(base_url)
        overflow = page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
        assert overflow is False
        browser.close()
```

- [ ] **Step 2: Run the UI test and verify the old settings UI fails it**

Run: `python -m pytest tests/test_web_ui.py -v`

Expected: first test fails because `#api-key` exists and the new source contract is not rendered.

- [ ] **Step 3: Replace model settings with knowledge-base controls**

Keep the clinical question as the primary first-screen action. Replace the API Key/Base URL/model inputs with:

- a compact backend model status indicator;
- metadata/PDF/matched/parsed/embedded counts;
- build, pause, continue, and retry-failed icon buttons with tooltips;
- evidence-type checkboxes, inclusive year inputs, and a `仅使用全文` checkbox;
- a progress bar whose label uses the current backend stage.

Do not put the question experience or whole page inside a decorative card. Keep existing evidence list, graph workspace, and two-column report on desktop; stack them into a logical question, report, graph, sources order below900px.

- [ ] **Step 4: Implement new API state and query flow**

On `DOMContentLoaded`, fetch `/api/health`, `/api/kb/status`, and `/api/evidence`. Use the static catalog only when knowledge-base status is `not_built`; label that mode `示例证据`.

Submit this exact query shape:

```json
{
  "question": "SMPP儿童是否应使用糖皮质激素？",
  "evidence_types": ["guideline", "systematic_review", "randomized_controlled_trial"],
  "year_from": 2015,
  "year_to": 2026,
  "fulltext_only": false
}
```

Render `reasoning_steps` as native `<details>`, `answer_markdown` through the existing safe DOM renderer, sources as buttons keyed by `data-source-number`, and graph relations including `conflicts`. A source click fetches `/api/documents/{id}` and displays bibliography, evidence type/quality, abstract/full-text status, exact matched snippet, section, and page range. Show an inline PDF button only when `pdf_available=true`.

Poll a running job no faster than once per second and stop polling on paused/completed/failed. Disable only controls whose actions conflict with the current job; querying an already-built index remains available during a later incremental build.

- [ ] **Step 5: Add stable responsive dimensions and accessible states**

Give graph, toolbars, progress, count fields, and icon buttons explicit min/max sizes so loading labels cannot shift layout. Ensure every icon-only button has `title` and `aria-label`, all status changes use an `aria-live` region, long DOI/title strings wrap with `overflow-wrap:anywhere`, and focus remains visible. Retain cards only for repeated evidence/source items and avoid card nesting.

- [ ] **Step 6: Run UI tests at desktop and mobile sizes**

Run: `python -m pytest tests/test_web_ui.py -v`

Expected: both tests pass.

Capture manual verification screenshots at `1440x900` and `390x844`, then inspect that the question, progress, report, graph, and source details do not overlap or overflow.

- [ ] **Step 7: Commit the real-corpus workbench**

```powershell
git add web/index.html web/app.js web/styles.css tests/test_web_ui.py
git commit -m "feat: add interactive literature workbench"
```

### Task 12: Validate Against The Real Corpus, Add Standard Questions, And Finish Documentation

**Files:**
- Create: `validation/mpp_questions.json`
- Create: `validation/run_validation.py`
- Create: `tests/knowledge_base/test_validation.py`
- Modify: `README.md`
- Modify: `.env.example`

- [ ] **Step 1: Add a fixed validation schema and failing evaluator test**

```python
# tests/knowledge_base/test_validation.py
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
```

- [ ] **Step 2: Run the evaluator test and verify the validation module is absent**

Run: `python -m pytest tests/knowledge_base/test_validation.py -v`

Expected: collection fails because `validation.run_validation` does not exist.

- [ ] **Step 3: Create10 to20 standard MPP questions and a reproducible evaluator**

`validation/mpp_questions.json` must contain a unique ID, Chinese clinical question, optional filters, expected evidence types, and manually reviewed title/DOI terms for each case. Cover steroid use, dose, timing, macrolide resistance, tetracyclines, fluoroquinolones, bronchoscopy, IVIG, anticoagulation complications, refractory disease, severe disease, and long-term pulmonary outcomes.

`run_validation.py` must call `KnowledgeBaseService.query()` and write a timestamped local JSON result under `.local/validation/`. Calculate per-case expected-source hit, duplicate-document count, unsupported-citation count, evidence-type coverage, and retrieval mode. Aggregate a baseline report without inventing a target accuracy threshold before manual labels are complete.

- [ ] **Step 4: Run all free local ingestion stages over the actual corpus**

Run: `python -m knowledge_base.cli inspect --source "C:\Users\LTC\Desktop\MPP" --json`

Expected: 35,408 metadata records and909 PDF files are reported.

Run: `python -m knowledge_base.cli build --json`

Expected: metadata, PDF inventory/matching, parsing, chunking, and FTS5 complete; the command stops before paid embedding work and reports the exact `embedding_pending` count. Every PDF has one of `matched`, `unmatched`, `duplicate`, or `failed`; every metadata row has a recorded state.

- [ ] **Step 5: Validate one paid batch before requesting a full embedding run**

With valid backend environment variables set, run: `python -m knowledge_base.cli build --embedding-limit 32 --confirm-embedding-cost --json`

Expected: the embedding probe succeeds, the returned dimension is persisted, no key appears in output,32 pending texts are embedded at most once, and both Qdrant collections can return a query result.

Stop here and report the total pending text count and provider model compatibility. Obtain explicit user confirmation before running all remaining paid embedding batches.

- [ ] **Step 6: Complete embeddings only after the cost checkpoint is approved**

Run: `python -m knowledge_base.cli build --confirm-embedding-cost --json`

Expected: all eligible metadata and chunks reach `indexed` or a recorded terminal failure; rerunning the same command embeds zero unchanged texts.

- [ ] **Step 7: Run the standard question baseline**

Run: `python validation/run_validation.py`

Expected: a local report lists every case, retrieval mode, expected-source hit, citation support, duplicate count, and evidence-type coverage. Manually inspect at least the steroid-use, steroid-dose, and macrolide-resistance cases against source snippets and PDF pages.

- [ ] **Step 8: Update README with exact operator workflow**

Document environment setup, dependency installation, `rapidocr check`, inspect/build/status/pause/retry commands, server startup, demo fallback, paid embedding checkpoint, local data boundaries, sharing limitations, and the medical decision-support disclaimer. State the verified corpus counts as metadata records and PDF files, never as35,408 full-text papers.

- [ ] **Step 9: Run the complete verification suite**

Run: `python -m pytest -v`

Expected: all unit, integration, API, legacy, and UI tests pass.

Run: `python -m knowledge_base.cli status --json`

Expected: valid JSON with per-stage counts, model/index version, failure counts, and no secrets.

Run: `git status --short`

Expected: only intended source, test, documentation, and validation fixture changes are present; `.local`, PDFs, SQLite, Qdrant files, environment files, logs, and generated validation reports are absent.

- [ ] **Step 10: Start the app and perform the final browser smoke test**

Run: `python web_server.py --host 127.0.0.1 --port 8765`

Expected console line: `Graph RAG web app: http://localhost:8765`.

Verify `/api/health`, one hybrid query, one source detail, one inline PDF, one keyword degradation message, and one static demo fallback. Keep the server running and provide the reachable local URL to the user.

- [ ] **Step 11: Commit documentation and validation assets**

```powershell
git add README.md .env.example validation tests/knowledge_base/test_validation.py
git commit -m "docs: add MPP corpus validation workflow"
```

## Specification Coverage

| Accepted requirement | Implementation tasks |
|---|---|
| Local paths, secrets, Git boundaries | Tasks1,10,12 |
| 35,408 metadata records and909 PDF statuses | Tasks3,6,12 |
| Exact/DOI/title deduplication and aliases | Tasks2,3 |
| Page extraction, local OCR, page-aware chunks | Task4 |
| Embedding probe, batching, cache, retry, dimension checks | Tasks5,6 |
| Qdrant local metadata/full-text collections | Tasks5,6 |
| SQLite manifest, FTS5 and Chinese tokenization | Tasks2,3 |
| Four-list hybrid retrieval, RRF, filters and degradation | Task7 |
| Evidence pyramid, quality, time and candidate-only LLM refinement | Tasks8,9 |
| Dynamic supports/updates/supplements/conflicts/cautions graph | Task8 |
| Auditable analysis plus grounded final answer | Task9 |
| Knowledge-base jobs, status, pause, retry, documents and safe PDF API | Task10 |
| Interactive desktop/mobile workbench and source inspection | Task11 |
| Real corpus inspection, paid-cost checkpoint and standard validation | Task12 |
| Static demo and disease/case compatibility | Tasks10,12 |

## Primary Technical References

- [Qdrant Python client local mode](https://github.com/qdrant/qdrant-client)
- [SQLite FTS5 and BM25](https://www.sqlite.org/fts5.html)
- [RapidOCR installation](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/install/)
- [RapidOCR Python usage](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/)
- [PyMuPDF documentation](https://pymupdf.readthedocs.io/en/latest/)

## Final Review Checklist

- [ ] Every accepted design requirement maps to at least one task above.
- [ ] The data description consistently says35,408 metadata records and909 PDF files.
- [ ] Tests never require the full private corpus or a paid API.
- [ ] Real-corpus parsing writes only under `.local/knowledge_base`.
- [ ] A full paid embedding run cannot start without explicit confirmation.
- [ ] Browser requests and storage contain no API key, Base URL, or model secret.
- [ ] Every final source is document-deduplicated and every numeric clinical claim is traceable.
- [ ] Static demo, disease workflow, and case workflow remain runnable.
- [ ] The Git repository contains code/config examples only, not private literature or generated indexes.
