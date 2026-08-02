from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import struct
from typing import Any
import zlib

from .errors import KnowledgeBaseError
from .models import (
    ChunkRecord,
    DocumentRecord,
    DocumentSource,
    EmbeddingItem,
    FileRecord,
    RankedHit,
    SearchFilters,
)


SCHEMA_VERSION = 5
_BASIC_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS schema_info (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS documents (
        document_id TEXT PRIMARY KEY,
        source_row INTEGER NOT NULL,
        source_id TEXT NOT NULL DEFAULT '',
        normalized_source_id TEXT NOT NULL DEFAULT '',
        title TEXT NOT NULL,
        normalized_title TEXT NOT NULL,
        authors_json TEXT NOT NULL,
        year INTEGER,
        journal TEXT NOT NULL,
        doi TEXT NOT NULL,
        normalized_doi TEXT NOT NULL,
        abstract TEXT NOT NULL,
        language TEXT NOT NULL,
        url TEXT NOT NULL DEFAULT '',
        evidence_type TEXT NOT NULL,
        classification_confidence REAL NOT NULL,
        classification_basis TEXT NOT NULL,
        has_fulltext INTEGER NOT NULL,
        fulltext_status TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS files (
        file_id TEXT PRIMARY KEY,
        path TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        document_id TEXT REFERENCES documents(document_id) ON DELETE SET NULL,
        match_method TEXT NOT NULL,
        match_confidence REAL NOT NULL,
        status TEXT NOT NULL,
        error_code TEXT NOT NULL,
        error_message TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS document_aliases (
        alias TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
        alias_type TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS document_sources (
        document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
        source_id TEXT NOT NULL,
        normalized_source_id TEXT NOT NULL,
        source_row INTEGER NOT NULL,
        url TEXT NOT NULL,
        PRIMARY KEY (document_id, source_row, source_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS document_sources_normalized_source_id_idx
    ON document_sources(normalized_source_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
        file_id TEXT NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
        section TEXT NOT NULL,
        page_start INTEGER NOT NULL,
        page_end INTEGER NOT NULL,
        text TEXT NOT NULL,
        token_count INTEGER NOT NULL,
        is_ocr INTEGER NOT NULL,
        quality TEXT NOT NULL,
        content_hash TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS embedding_cache (
        content_hash TEXT NOT NULL,
        model_name TEXT NOT NULL,
        dimension INTEGER NOT NULL,
        vector_blob BLOB NOT NULL,
        PRIMARY KEY (content_hash, model_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS embedding_index_state (
        record_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        model_name TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        PRIMARY KEY (record_id, kind, model_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS build_jobs (
        job_id TEXT PRIMARY KEY,
        state TEXT NOT NULL,
        progress_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS build_checkpoints (
        stage TEXT NOT NULL,
        record_id TEXT NOT NULL,
        input_hash TEXT NOT NULL,
        algorithm_version TEXT NOT NULL,
        status TEXT NOT NULL,
        counters_json TEXT NOT NULL,
        error_code TEXT NOT NULL,
        error_message TEXT NOT NULL,
        PRIMARY KEY (stage, record_id, algorithm_version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingest_errors (
        error_id INTEGER PRIMARY KEY AUTOINCREMENT,
        stage TEXT NOT NULL,
        record_id TEXT NOT NULL,
        code TEXT NOT NULL,
        message TEXT NOT NULL
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS metadata_fts USING fts5(
        document_id UNINDEXED,
        title_tokens,
        abstract_tokens
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS fulltext_fts USING fts5(
        chunk_id UNINDEXED,
        document_id UNINDEXED,
        section_tokens,
        body_tokens
    )
    """,
)


def basic_tokenizer(text: str) -> str:
    """Produce deterministic FTS tokens before a language-specific tokenizer is added."""
    return " ".join(match.group(0).lower() for match in _BASIC_TOKEN_PATTERN.finditer(text))


class SQLiteStore:
    def __init__(
        self, path: Path, tokenizer: Callable[[str], str] | None = None
    ) -> None:
        self.path = Path(path)
        self._tokenizer = tokenizer or basic_tokenizer

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("BEGIN IMMEDIATE")
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)
            previous_schema_version = self._schema_version_from_connection(connection)
            has_existing_documents = self._has_documents(connection)
            self._migrate_document_provenance(connection)
            if previous_schema_version < 3 and has_existing_documents:
                connection.execute(
                    """
                    INSERT INTO schema_info(key, value) VALUES (?, '1')
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    ("metadata_reimport_required",),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO schema_info(key, value) VALUES (?, '0')
                    ON CONFLICT(key) DO NOTHING
                    """,
                    ("metadata_reimport_required",),
                )
            connection.execute(
                """
                INSERT INTO schema_info(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ("schema_version", str(SCHEMA_VERSION)),
            )

    @staticmethod
    def _schema_version_from_connection(connection: sqlite3.Connection) -> int:
        row = connection.execute(
            "SELECT value FROM schema_info WHERE key = ?", ("schema_version",)
        ).fetchone()
        return int(row["value"]) if row is not None else 0

    @staticmethod
    def _has_documents(connection: sqlite3.Connection) -> bool:
        row = connection.execute("SELECT EXISTS(SELECT 1 FROM documents)").fetchone()
        return bool(row[0])

    @staticmethod
    def _migrate_document_provenance(connection: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(documents)").fetchall()
        }
        for name, definition in (
            ("source_id", "TEXT NOT NULL DEFAULT ''"),
            ("normalized_source_id", "TEXT NOT NULL DEFAULT ''"),
            ("url", "TEXT NOT NULL DEFAULT ''"),
        ):
            if name not in columns:
                connection.execute(f"ALTER TABLE documents ADD COLUMN {name} {definition}")
        connection.execute(
            """
            INSERT INTO document_sources(
                document_id, source_id, normalized_source_id, source_row, url
            )
            SELECT document_id, source_id, normalized_source_id, source_row, url
            FROM documents
            WHERE source_id <> '' OR normalized_source_id <> '' OR url <> ''
            ON CONFLICT(document_id, source_row, source_id) DO UPDATE SET
                normalized_source_id = excluded.normalized_source_id,
                url = excluded.url
            """
        )

    def schema_version(self) -> int:
        with self._connection() as connection:
            return self._schema_version_from_connection(connection)

    def metadata_reimport_required(self) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM schema_info WHERE key = ?",
                ("metadata_reimport_required",),
            ).fetchone()
        return row is not None and row["value"] == "1"

    def clear_metadata_reimport_required(
        self, *, imported_row_count: int, expected_row_count: int
    ) -> None:
        """Clear the flag only after a full metadata reimport rebuilt all sources.

        Future indexers must atomically rebuild ``document_sources`` from the full
        metadata input, then call this method with equal imported and expected counts.
        """
        if (
            isinstance(imported_row_count, bool)
            or isinstance(expected_row_count, bool)
            or not isinstance(imported_row_count, int)
            or not isinstance(expected_row_count, int)
            or imported_row_count < 0
            or expected_row_count < 0
        ):
            raise KnowledgeBaseError(
                "invalid_metadata_reimport_count",
                "Metadata reimport counts must be non-negative integers",
            )
        with self._connection() as connection:
            source_count = int(
                connection.execute("SELECT COUNT(*) FROM document_sources").fetchone()[0]
            )
            if (
                imported_row_count != expected_row_count
                or source_count != expected_row_count
            ):
                raise KnowledgeBaseError(
                    "metadata_reimport_incomplete",
                    "Cannot clear metadata reimport flag before all source rows are rebuilt",
                    details={
                        "imported_row_count": imported_row_count,
                        "expected_row_count": expected_row_count,
                        "recorded_source_count": source_count,
                    },
                )
            connection.execute(
                """
                INSERT INTO schema_info(key, value) VALUES (?, '0')
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ("metadata_reimport_required",),
            )

    def journal_mode(self) -> str:
        with self._connection() as connection:
            row = connection.execute("PRAGMA journal_mode").fetchone()
        return str(row[0])

    def upsert_document(self, record: DocumentRecord) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, source_row, source_id, normalized_source_id, title, normalized_title, authors_json, year,
                    journal, doi, normalized_doi, abstract, language, evidence_type,
                    classification_confidence, classification_basis, has_fulltext, fulltext_status, url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    source_row = excluded.source_row,
                    source_id = excluded.source_id,
                    normalized_source_id = excluded.normalized_source_id,
                    title = excluded.title,
                    normalized_title = excluded.normalized_title,
                    authors_json = excluded.authors_json,
                    year = excluded.year,
                    journal = excluded.journal,
                    doi = excluded.doi,
                    normalized_doi = excluded.normalized_doi,
                    abstract = excluded.abstract,
                    language = excluded.language,
                    evidence_type = excluded.evidence_type,
                    classification_confidence = excluded.classification_confidence,
                    classification_basis = excluded.classification_basis,
                    has_fulltext = excluded.has_fulltext,
                    fulltext_status = excluded.fulltext_status,
                    url = excluded.url
                """,
                self._document_values(record),
            )
            if record.source_id or record.normalized_source_id or record.url:
                connection.execute(
                    """
                    INSERT INTO document_sources(
                        document_id, source_id, normalized_source_id, source_row, url
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(document_id, source_row, source_id) DO UPDATE SET
                        normalized_source_id = excluded.normalized_source_id,
                        url = excluded.url
                    """,
                    (
                        record.document_id,
                        record.source_id,
                        record.normalized_source_id,
                        record.source_row,
                        record.url,
                    ),
                )
            if record.normalized_source_id:
                connection.execute(
                    """
                    INSERT INTO document_aliases(alias, document_id, alias_type)
                    VALUES (?, ?, ?)
                    ON CONFLICT(alias) DO NOTHING
                    """,
                    (
                        "source_id:" + record.normalized_source_id,
                        record.document_id,
                        "source_id",
                    ),
                )
            connection.execute(
                "DELETE FROM metadata_fts WHERE document_id = ?", (record.document_id,)
            )
            connection.execute(
                """
                INSERT INTO metadata_fts(document_id, title_tokens, abstract_tokens)
                VALUES (?, ?, ?)
                """,
                (
                    record.document_id,
                    self._tokenizer(record.title),
                    self._tokenizer(record.abstract),
                ),
            )

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return self._document_from_row(row) if row is not None else None

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        if row is None:
            return None
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            file_id=row["file_id"],
            section=row["section"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            text=row["text"],
            token_count=row["token_count"],
            is_ocr=bool(row["is_ocr"]),
            quality=row["quality"],
            content_hash=row["content_hash"],
        )

    def list_chunks_for_document(
        self, document_id: str, *, limit: int | None = None
    ) -> list[ChunkRecord]:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise KnowledgeBaseError(
                "invalid_chunk_limit", "Chunk limit must be positive"
            )
        sql = "SELECT * FROM chunks WHERE document_id = ? ORDER BY page_start, chunk_id"
        parameters: list[Any] = [document_id]
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)
        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [
            ChunkRecord(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                file_id=row["file_id"],
                section=row["section"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                text=row["text"],
                token_count=row["token_count"],
                is_ocr=bool(row["is_ocr"]),
                quality=row["quality"],
                content_hash=row["content_hash"],
            )
            for row in rows
        ]

    def list_document_sources(self, document_id: str) -> list[DocumentSource]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT document_id, source_id, normalized_source_id, source_row, url
                FROM document_sources
                WHERE document_id = ?
                ORDER BY source_row, source_id
                """,
                (document_id,),
            ).fetchall()
        return [
            DocumentSource(
                document_id=row["document_id"],
                source_id=row["source_id"],
                normalized_source_id=row["normalized_source_id"],
                source_row=row["source_row"],
                url=row["url"],
            )
            for row in rows
        ]

    def upsert_file(self, record: FileRecord) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO files (
                    file_id, path, sha256, size_bytes, document_id, match_method,
                    match_confidence, status, error_code, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    path = excluded.path,
                    sha256 = excluded.sha256,
                    size_bytes = excluded.size_bytes,
                    document_id = excluded.document_id,
                    match_method = excluded.match_method,
                    match_confidence = excluded.match_confidence,
                    status = excluded.status,
                    error_code = excluded.error_code,
                    error_message = excluded.error_message
                """,
                (
                    record.file_id,
                    record.path,
                    record.sha256,
                    record.size_bytes,
                    record.document_id,
                    record.match_method,
                    record.match_confidence,
                    record.status,
                    record.error_code,
                    record.error_message,
                ),
            )

    def get_file(self, file_id: str) -> FileRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM files WHERE file_id = ?", (file_id,)
            ).fetchone()
        return self._file_from_row(row) if row is not None else None

    def list_files(self, statuses: Sequence[str] | None = None) -> list[FileRecord]:
        sql = "SELECT * FROM files"
        parameters: tuple[str, ...] = ()
        if statuses:
            normalized = tuple(sorted(set(statuses)))
            placeholders = ", ".join("?" for _ in normalized)
            sql += f" WHERE status IN ({placeholders})"
            parameters = normalized
        sql += " ORDER BY path, file_id"
        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._file_from_row(row) for row in rows]

    def delete_chunks_for_file(self, file_id: str) -> int:
        with self._connection() as connection:
            chunk_rows = connection.execute(
                "SELECT chunk_id, document_id FROM chunks WHERE file_id = ?", (file_id,)
            ).fetchall()
            chunk_ids = tuple(row["chunk_id"] for row in chunk_rows)
            document_ids = tuple(sorted({row["document_id"] for row in chunk_rows}))
            connection.execute(
                """
                DELETE FROM fulltext_fts
                WHERE chunk_id IN (
                    SELECT chunk_id FROM chunks WHERE file_id = ?
                )
                """,
                (file_id,),
            )
            connection.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            if chunk_ids:
                placeholders = ", ".join("?" for _ in chunk_ids)
                connection.execute(
                    f"DELETE FROM embedding_index_state WHERE kind = 'chunk' AND record_id IN ({placeholders})",
                    chunk_ids,
                )
            for document_id in document_ids:
                remaining = connection.execute(
                    "SELECT 1 FROM chunks WHERE document_id = ? LIMIT 1",
                    (document_id,),
                ).fetchone()
                if remaining is None:
                    connection.execute(
                        """
                        UPDATE documents
                        SET has_fulltext = 0, fulltext_status = 'missing'
                        WHERE document_id = ?
                        """,
                        (document_id,),
                    )
        return len(chunk_ids)

    def update_document_fulltext(
        self, document_id: str, *, has_fulltext: bool, status: str
    ) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET has_fulltext = ?, fulltext_status = ?
                WHERE document_id = ?
                """,
                (int(has_fulltext), status, document_id),
            )
            if cursor.rowcount != 1:
                raise KnowledgeBaseError(
                    "document_not_found", "Document record was not found"
                )

    def upsert_chunk(self, record: ChunkRecord) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, file_id, section, page_start, page_end, text,
                    token_count, is_ocr, quality, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    file_id = excluded.file_id,
                    section = excluded.section,
                    page_start = excluded.page_start,
                    page_end = excluded.page_end,
                    text = excluded.text,
                    token_count = excluded.token_count,
                    is_ocr = excluded.is_ocr,
                    quality = excluded.quality,
                    content_hash = excluded.content_hash
                """,
                (
                    record.chunk_id,
                    record.document_id,
                    record.file_id,
                    record.section,
                    record.page_start,
                    record.page_end,
                    record.text,
                    record.token_count,
                    int(record.is_ocr),
                    record.quality,
                    record.content_hash,
                ),
            )
            connection.execute(
                "DELETE FROM fulltext_fts WHERE chunk_id = ?", (record.chunk_id,)
            )
            connection.execute(
                """
                INSERT INTO fulltext_fts(chunk_id, document_id, section_tokens, body_tokens)
                VALUES (?, ?, ?, ?)
                """,
                (
                    record.chunk_id,
                    record.document_id,
                    self._tokenizer(record.section),
                    self._tokenizer(record.text),
                ),
            )

    def search_metadata(
        self, query: str, limit: int, filters: SearchFilters | None = None
    ) -> list[RankedHit]:
        return self._search(query, limit, filters, fulltext=False)

    def search_fulltext(
        self, query: str, limit: int, filters: SearchFilters | None = None
    ) -> list[RankedHit]:
        return self._search(query, limit, filters, fulltext=True)

    def list_pending_embedding_items(
        self, model_name: str, limit: int | None = None
    ) -> list[EmbeddingItem]:
        if limit is not None and limit <= 0:
            raise KnowledgeBaseError(
                "invalid_embedding_limit", "Embedding item limit must be positive"
            )

        with self._connection() as connection:
            metadata_rows = connection.execute(
                """
                SELECT d.document_id, d.title, d.abstract,
                       indexed.content_hash AS indexed_content_hash
                FROM documents AS d
                LEFT JOIN embedding_index_state AS indexed
                  ON indexed.record_id = d.document_id
                 AND indexed.kind = ?
                 AND indexed.model_name = ?
                WHERE (d.title <> '' OR d.abstract <> '')
                ORDER BY d.document_id
                """,
                ("metadata", model_name),
            ).fetchall()
            chunk_rows = connection.execute(
                """
                SELECT c.chunk_id, c.document_id, c.text, c.content_hash,
                       indexed.content_hash AS indexed_content_hash
                FROM chunks AS c
                LEFT JOIN embedding_index_state AS indexed
                  ON indexed.record_id = c.chunk_id
                 AND indexed.kind = ?
                 AND indexed.model_name = ?
                ORDER BY c.chunk_id
                """,
                ("chunk", model_name),
            ).fetchall()

        items: list[EmbeddingItem] = []
        for row in metadata_rows:
            content_hash = self._metadata_content_hash(row["title"], row["abstract"])
            if row["indexed_content_hash"] == content_hash:
                continue
            items.append(
                EmbeddingItem(
                    record_id=row["document_id"],
                    document_id=row["document_id"],
                    kind="metadata",
                    text=self._metadata_text(row["title"], row["abstract"]),
                    content_hash=content_hash,
                )
            )
        for row in chunk_rows:
            if row["indexed_content_hash"] == row["content_hash"]:
                continue
            items.append(
                EmbeddingItem(
                    record_id=row["chunk_id"],
                    document_id=row["document_id"],
                    kind="chunk",
                    text=row["text"],
                    content_hash=row["content_hash"],
                )
            )
        return items if limit is None else items[:limit]

    def count_embedding_candidates(self) -> int:
        with self._connection() as connection:
            metadata_count = connection.execute(
                "SELECT COUNT(*) FROM documents WHERE title <> '' OR abstract <> ''"
            ).fetchone()[0]
            chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return int(metadata_count) + int(chunk_count)

    def statistics(self) -> dict[str, int]:
        with self._connection() as connection:
            metadata_records = connection.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
            pdf_files = connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            matched_pdf_files = connection.execute(
                "SELECT COUNT(*) FROM files WHERE document_id IS NOT NULL"
            ).fetchone()[0]
            parsed_pdf_files = connection.execute(
                "SELECT COUNT(*) FROM files WHERE status IN ('parsed', 'partial')"
            ).fetchone()[0]
            chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            embedded = connection.execute(
                "SELECT COUNT(*) FROM embedding_index_state"
            ).fetchone()[0]
            errors = connection.execute(
                "SELECT COUNT(*) FROM ingest_errors"
            ).fetchone()[0]
        return {
            "metadata_records": int(metadata_records),
            "pdf_files": int(pdf_files),
            "matched_pdf_files": int(matched_pdf_files),
            "parsed_pdf_files": int(parsed_pdf_files),
            "chunks": int(chunks),
            "embedded": int(embedded),
            "errors": int(errors),
        }

    def mark_embedding_indexed(
        self, record_id: str, kind: str, model_name: str, expected_content_hash: str
    ) -> bool:
        with self._connection() as connection:
            if kind == "metadata":
                row = connection.execute(
                    "SELECT title, abstract FROM documents WHERE document_id = ?",
                    (record_id,),
                ).fetchone()
                if row is None:
                    raise KnowledgeBaseError(
                        "embedding_record_not_found", "Embedding record was not found"
                    )
                content_hash = self._metadata_content_hash(
                    row["title"], row["abstract"]
                )
            elif kind == "chunk":
                row = connection.execute(
                    "SELECT content_hash FROM chunks WHERE chunk_id = ?", (record_id,)
                ).fetchone()
                if row is None:
                    raise KnowledgeBaseError(
                        "embedding_record_not_found", "Embedding record was not found"
                    )
                content_hash = row["content_hash"]
            else:
                raise KnowledgeBaseError(
                    "invalid_embedding_kind", "Embedding kind must be metadata or chunk"
                )
            if content_hash != expected_content_hash:
                return False
            connection.execute(
                """
                INSERT INTO embedding_index_state(record_id, kind, model_name, content_hash)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(record_id, kind, model_name) DO UPDATE SET
                    content_hash = excluded.content_hash
                """,
                (record_id, kind, model_name, content_hash),
            )
        return True

    def set_job_state(self, job_id: str, state: str, progress: dict[str, Any]) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO build_jobs(job_id, state, progress_json) VALUES (?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    state = excluded.state,
                    progress_json = excluded.progress_json
                """,
                (job_id, state, json.dumps(progress, ensure_ascii=False, sort_keys=True)),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT job_id, state, progress_json FROM build_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "job_id": row["job_id"],
            "state": row["state"],
            "progress": json.loads(row["progress_json"]),
        }

    def get_build_checkpoint(
        self, stage: str, record_id: str, algorithm_version: str
    ) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT stage, record_id, input_hash, algorithm_version, status,
                       counters_json, error_code, error_message
                FROM build_checkpoints
                WHERE stage = ? AND record_id = ? AND algorithm_version = ?
                """,
                (stage, record_id, algorithm_version),
            ).fetchone()
        if row is None:
            return None
        return {
            "stage": row["stage"],
            "record_id": row["record_id"],
            "input_hash": row["input_hash"],
            "algorithm_version": row["algorithm_version"],
            "status": row["status"],
            "counters": json.loads(row["counters_json"]),
            "error_code": row["error_code"],
            "error_message": row["error_message"],
        }

    def set_build_checkpoint(
        self,
        *,
        stage: str,
        record_id: str,
        input_hash: str,
        algorithm_version: str,
        status: str,
        counters: dict[str, int] | None = None,
        error_code: str = "",
        error_message: str = "",
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO build_checkpoints(
                    stage, record_id, input_hash, algorithm_version, status,
                    counters_json, error_code, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stage, record_id, algorithm_version) DO UPDATE SET
                    input_hash = excluded.input_hash,
                    status = excluded.status,
                    counters_json = excluded.counters_json,
                    error_code = excluded.error_code,
                    error_message = excluded.error_message
                """,
                (
                    stage,
                    record_id,
                    input_hash,
                    algorithm_version,
                    status,
                    json.dumps(counters or {}, ensure_ascii=False, sort_keys=True),
                    error_code,
                    error_message,
                ),
            )

    def record_error(self, stage: str, record_id: str, code: str, message: str) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO ingest_errors(stage, record_id, code, message)
                VALUES (?, ?, ?, ?)
                """,
                (stage, record_id, code, message),
            )

    def list_errors(self, stage: str | None = None) -> list[dict[str, str]]:
        sql = "SELECT stage, record_id, code, message FROM ingest_errors"
        parameters: Sequence[str] = ()
        if stage is not None:
            sql += " WHERE stage = ?"
            parameters = (stage,)
        sql += " ORDER BY error_id"
        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [
            {
                "stage": row["stage"],
                "record_id": row["record_id"],
                "code": row["code"],
                "message": row["message"],
            }
            for row in rows
        ]

    def get_cached_embedding(
        self, content_hash: str, model_name: str
    ) -> list[float] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT dimension, vector_blob FROM embedding_cache
                WHERE content_hash = ? AND model_name = ?
                """,
                (content_hash, model_name),
            ).fetchone()
        if row is None:
            return None

        try:
            raw_vector = zlib.decompress(row["vector_blob"])
            if len(raw_vector) % 4 != 0:
                raise ValueError("Cached embedding byte length is invalid")
            if len(raw_vector) // 4 != row["dimension"]:
                raise ValueError("Cached embedding dimension does not match")
            vector = struct.unpack(f"<{row['dimension']}f", raw_vector)
        except (ValueError, struct.error, zlib.error) as exc:
            raise KnowledgeBaseError(
                "embedding_cache_corrupt", "Cached embedding could not be decoded"
            ) from exc
        return list(vector)

    def put_cached_embedding(
        self, content_hash: str, model_name: str, vector: Sequence[float]
    ) -> None:
        values = tuple(float(value) for value in vector)
        vector_bytes = struct.pack(f"<{len(values)}f", *values)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO embedding_cache(content_hash, model_name, dimension, vector_blob)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(content_hash, model_name) DO UPDATE SET
                    dimension = excluded.dimension,
                    vector_blob = excluded.vector_blob
                """,
                (
                    content_hash,
                    model_name,
                    len(values),
                    zlib.compress(vector_bytes),
                ),
            )

    def _search(
        self,
        query: str,
        limit: int,
        filters: SearchFilters | None,
        *,
        fulltext: bool,
    ) -> list[RankedHit]:
        self._validate_search_limit(limit)
        fts_query = self._fts_query(query)
        if not fts_query:
            return []

        active_filters = filters or SearchFilters()
        fts_table = "fulltext_fts" if fulltext else "metadata_fts"
        where_parts = [f"{fts_table} MATCH ?"]
        parameters: list[Any] = [fts_query]
        if active_filters.evidence_types:
            placeholders = ", ".join("?" for _ in active_filters.evidence_types)
            where_parts.append(f"d.evidence_type IN ({placeholders})")
            parameters.extend(sorted(active_filters.evidence_types))
        if active_filters.year_from is not None:
            where_parts.append("d.year >= ?")
            parameters.append(active_filters.year_from)
        if active_filters.year_to is not None:
            where_parts.append("d.year <= ?")
            parameters.append(active_filters.year_to)
        if active_filters.fulltext_only and not fulltext:
            where_parts.append("d.has_fulltext = 1")
        parameters.append(limit)

        if fulltext:
            sql = f"""
                SELECT fulltext_fts.chunk_id, fulltext_fts.document_id,
                       c.section, c.text, bm25(fulltext_fts) AS score
                FROM fulltext_fts
                JOIN chunks AS c ON c.chunk_id = fulltext_fts.chunk_id
                JOIN documents AS d ON d.document_id = fulltext_fts.document_id
                WHERE {' AND '.join(where_parts)}
                ORDER BY rank
                LIMIT ?
            """
        else:
            sql = f"""
                SELECT metadata_fts.document_id, d.title, d.abstract, d.year,
                       d.evidence_type, bm25(metadata_fts) AS score
                FROM metadata_fts
                JOIN documents AS d ON d.document_id = metadata_fts.document_id
                WHERE {' AND '.join(where_parts)}
                ORDER BY rank
                LIMIT ?
            """

        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()

        if fulltext:
            return [
                RankedHit(
                    record_id=row["chunk_id"],
                    document_id=row["document_id"],
                    source="fulltext",
                    rank=index,
                    score=float(row["score"]),
                    text=row["text"],
                    payload={"section": row["section"]},
                )
                for index, row in enumerate(rows, start=1)
            ]
        return [
            RankedHit(
                record_id=row["document_id"],
                document_id=row["document_id"],
                source="metadata",
                rank=index,
                score=float(row["score"]),
                text=self._metadata_text(row["title"], row["abstract"]),
                payload={"year": row["year"], "evidence_type": row["evidence_type"]},
            )
            for index, row in enumerate(rows, start=1)
        ]

    @staticmethod
    def _validate_search_limit(limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise KnowledgeBaseError(
                "invalid_search_limit", "Search limit must be positive"
            )

    def _fts_query(self, query: str) -> str:
        terms = self._tokenizer(query).split()
        return " OR ".join(self._quote_fts_term(term) for term in terms)

    @staticmethod
    def _quote_fts_term(term: str) -> str:
        return '"' + term.replace('"', '""') + '"'

    @staticmethod
    def _metadata_text(title: str, abstract: str) -> str:
        return "\n\n".join(part for part in (title, abstract) if part)

    @classmethod
    def _metadata_content_hash(cls, title: str, abstract: str) -> str:
        return sha256(f"metadata\0{title}\0{abstract}".encode("utf-8")).hexdigest()

    @staticmethod
    def _document_values(record: DocumentRecord) -> tuple[Any, ...]:
        return (
            record.document_id,
            record.source_row,
            record.source_id,
            record.normalized_source_id,
            record.title,
            record.normalized_title,
            json.dumps(record.authors, ensure_ascii=False),
            record.year,
            record.journal,
            record.doi,
            record.normalized_doi,
            record.abstract,
            record.language,
            record.evidence_type,
            record.classification_confidence,
            record.classification_basis,
            int(record.has_fulltext),
            record.fulltext_status,
            record.url,
        )

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            source_row=row["source_row"],
            source_id=row["source_id"],
            normalized_source_id=row["normalized_source_id"],
            title=row["title"],
            normalized_title=row["normalized_title"],
            authors=tuple(json.loads(row["authors_json"])),
            year=row["year"],
            journal=row["journal"],
            doi=row["doi"],
            normalized_doi=row["normalized_doi"],
            abstract=row["abstract"],
            language=row["language"],
            evidence_type=row["evidence_type"],
            classification_confidence=row["classification_confidence"],
            classification_basis=row["classification_basis"],
            has_fulltext=bool(row["has_fulltext"]),
            fulltext_status=row["fulltext_status"],
            url=row["url"],
        )

    @staticmethod
    def _file_from_row(row: sqlite3.Row) -> FileRecord:
        return FileRecord(
            file_id=row["file_id"],
            path=row["path"],
            sha256=row["sha256"],
            size_bytes=row["size_bytes"],
            document_id=row["document_id"],
            match_method=row["match_method"],
            match_confidence=row["match_confidence"],
            status=row["status"],
            error_code=row["error_code"],
            error_message=row["error_message"],
        )
