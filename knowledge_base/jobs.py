"""Persistent single-writer background job management for knowledge-base builds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from threading import Event, Lock, Thread
from typing import Any, Mapping
from uuid import uuid4

from .errors import KnowledgeBaseError


_ACTIVE_STATES = {"queued", "running", "pausing"}
_TERMINAL_STATES = {
    "paused",
    "completed",
    "completed_with_errors",
    "embedding_pending",
    "failed",
}
_SENSITIVE_OPTION_MARKERS = ("key", "token", "secret", "password", "authorization")


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    state: str
    progress: dict[str, Any]
    error_code: str
    created_at: str
    updated_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "progress": dict(self.progress),
            "error_code": self.error_code,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class KnowledgeBaseJobManager:
    def __init__(self, *, indexer: Any, store_path: Path) -> None:
        self.indexer = indexer
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._pause_events: dict[str, Event] = {}
        self._threads: dict[str, Thread] = {}
        self._initialize()

    def start_build(
        self,
        *,
        confirm_embedding_cost: bool,
        options: Mapping[str, Any] | None = None,
    ) -> JobRecord:
        if not isinstance(confirm_embedding_cost, bool):
            raise KnowledgeBaseError(
                "invalid_embedding_confirmation", "Embedding confirmation is invalid"
            )
        safe_options = _safe_options(options or {})
        with self._lock:
            if self._active_job() is not None:
                raise KnowledgeBaseError(
                    "build_already_running", "A knowledge base build is already running"
                )
            job_id = "kb-" + uuid4().hex
            now = _timestamp()
            record = JobRecord(job_id, "queued", {}, "", now, now)
            self._persist(record)
            pause_event = Event()
            worker = Thread(
                target=self._run,
                args=(job_id, pause_event, confirm_embedding_cost, safe_options),
                daemon=True,
                name=f"knowledge-base-{job_id[-8:]}",
            )
            self._pause_events[job_id] = pause_event
            self._threads[job_id] = worker
            worker.start()
            return record

    def pause(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self.get(job_id)
            if record.state not in _ACTIVE_STATES:
                raise KnowledgeBaseError(
                    "job_not_running", "Knowledge base build is not running"
                )
            pause_event = self._pause_events.get(job_id)
            if pause_event is None:
                raise KnowledgeBaseError(
                    "job_not_running", "Knowledge base build is not running"
                )
            pause_event.set()
            updated = JobRecord(
                job_id=record.job_id,
                state="pausing",
                progress=record.progress,
                error_code="",
                created_at=record.created_at,
                updated_at=_timestamp(),
            )
            self._persist(updated)
            return updated

    def get(self, job_id: str) -> JobRecord:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise KnowledgeBaseError("job_not_found", "Knowledge base job was not found")
        return _job_from_row(row)

    def latest(self) -> JobRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, job_id DESC LIMIT 1"
            ).fetchone()
        return _job_from_row(row) if row is not None else None

    def wait(self, job_id: str, timeout: float | None = None) -> JobRecord:
        thread = self._threads.get(job_id)
        if thread is not None:
            thread.join(timeout=timeout)
        return self.get(job_id)

    def _run(
        self,
        job_id: str,
        pause_event: Event,
        confirm_embedding_cost: bool,
        options: dict[str, Any],
    ) -> None:
        current = self.get(job_id)
        self._persist(
            JobRecord(
                current.job_id,
                "running",
                current.progress,
                "",
                current.created_at,
                _timestamp(),
            )
        )
        try:
            result = self.indexer.build(
                should_pause=pause_event,
                confirm_embedding_cost=confirm_embedding_cost,
                **options,
            )
            progress = (
                result.as_dict()
                if hasattr(result, "as_dict")
                else dict(result)
                if isinstance(result, Mapping)
                else {"result": str(result)}
            )
            reported_state = str(
                progress.get("state", progress.get("final_state", "completed"))
            )
            state = (
                reported_state
                if reported_state
                in {"paused", "completed_with_errors", "embedding_pending"}
                else "completed"
            )
            error_code = ""
        except KnowledgeBaseError as error:
            progress = {}
            state = "failed"
            error_code = error.code
        except Exception:
            progress = {}
            state = "failed"
            error_code = "build_failed"
        current = self.get(job_id)
        self._persist(
            JobRecord(
                job_id,
                state,
                progress,
                error_code,
                current.created_at,
                _timestamp(),
            )
        )
        with self._lock:
            self._pause_events.pop(job_id, None)

    def _active_job(self) -> JobRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE state IN ('queued', 'running', 'pausing')
                ORDER BY created_at DESC, job_id DESC
                LIMIT 1
                """
            ).fetchone()
        return _job_from_row(row) if row is not None else None

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    progress_json TEXT NOT NULL,
                    error_code TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                UPDATE jobs
                SET state = 'failed', error_code = 'server_restarted', updated_at = ?
                WHERE state IN ('queued', 'running', 'pausing')
                """,
                (_timestamp(),),
            )

    def _persist(self, record: JobRecord) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO jobs(
                    job_id, state, progress_json, error_code, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    state = excluded.state,
                    progress_json = excluded.progress_json,
                    error_code = excluded.error_code,
                    updated_at = excluded.updated_at
                """,
                (
                    record.job_id,
                    record.state,
                    json.dumps(record.progress, ensure_ascii=False, sort_keys=True),
                    record.error_code,
                    record.created_at,
                    record.updated_at,
                ),
            )

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.store_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection


def _safe_options(options: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in options.items():
        normalized_key = str(key).casefold()
        if any(marker in normalized_key for marker in _SENSITIVE_OPTION_MARKERS):
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            safe[str(key)] = value
    return safe


def _job_from_row(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"],
        state=row["state"],
        progress=json.loads(row["progress_json"]),
        error_code=row["error_code"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
