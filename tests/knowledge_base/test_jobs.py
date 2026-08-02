from __future__ import annotations

import json
import sqlite3
from threading import Event

import pytest

from knowledge_base.errors import KnowledgeBaseError
from knowledge_base.jobs import KnowledgeBaseJobManager


class PausableIndexer:
    def __init__(self):
        self.started = Event()

    def build(self, *, should_pause, confirm_embedding_cost, **options):
        self.started.set()
        assert confirm_embedding_cost is False
        assert should_pause.wait(timeout=5) is True
        return {"state": "paused", "completed": 1}


def test_job_manager_pauses_at_cooperative_boundary(tmp_path) -> None:
    indexer = PausableIndexer()
    store_path = tmp_path / "jobs.sqlite3"
    manager = KnowledgeBaseJobManager(indexer=indexer, store_path=store_path)
    job = manager.start_build(confirm_embedding_cost=False)
    assert indexer.started.wait(timeout=5) is True

    pausing = manager.pause(job.job_id)
    final = manager.wait(job.job_id, timeout=5)

    assert pausing.state == "pausing"
    assert final.state == "paused"
    assert final.progress["completed"] == 1
    restarted = KnowledgeBaseJobManager(indexer=indexer, store_path=store_path)
    assert restarted.get(job.job_id).state == "paused"


class BlockingIndexer:
    def __init__(self):
        self.started = Event()
        self.release = Event()

    def build(self, *, should_pause, confirm_embedding_cost, **options):
        self.started.set()
        self.release.wait(timeout=5)
        return {"state": "completed", "completed": 1}


def test_job_manager_rejects_a_second_active_build(tmp_path) -> None:
    indexer = BlockingIndexer()
    manager = KnowledgeBaseJobManager(
        indexer=indexer, store_path=tmp_path / "jobs.sqlite3"
    )
    first = manager.start_build(confirm_embedding_cost=False)
    assert indexer.started.wait(timeout=5)

    with pytest.raises(KnowledgeBaseError) as captured:
        manager.start_build(confirm_embedding_cost=False)
    indexer.release.set()
    manager.wait(first.job_id, timeout=5)

    assert captured.value.code == "build_already_running"


def test_job_database_never_persists_secrets(tmp_path) -> None:
    indexer = BlockingIndexer()
    path = tmp_path / "jobs.sqlite3"
    manager = KnowledgeBaseJobManager(indexer=indexer, store_path=path)
    job = manager.start_build(
        confirm_embedding_cost=False,
        options={"document_limit": 2, "api_key": "must-not-be-saved"},
    )
    assert indexer.started.wait(timeout=5)
    indexer.release.set()
    manager.wait(job.job_id, timeout=5)

    with sqlite3.connect(path) as connection:
        serialized = json.dumps(connection.execute("SELECT * FROM jobs").fetchall())
    assert "must-not-be-saved" not in serialized
    assert "api_key" not in serialized
