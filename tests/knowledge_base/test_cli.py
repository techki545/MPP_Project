from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from knowledge_base.cli import build_parser, error_exit_code, main, render_error
from knowledge_base.errors import ConfigurationError, KnowledgeBaseError


def test_build_requires_explicit_embedding_confirmation() -> None:
    parser = build_parser()
    args = parser.parse_args(["build", "--embedding-limit", "32"])

    assert args.command == "build"
    assert args.confirm_embedding_cost is False
    assert args.embedding_limit == 32
    assert args.document_limit is None


def test_inspect_accepts_read_only_source_override() -> None:
    parser = build_parser()
    args = parser.parse_args(["inspect", "--source", r"C:\corpus", "--json"])

    assert args.command == "inspect"
    assert args.source == r"C:\corpus"
    assert args.json is True


def test_parser_exposes_status_pause_retry_and_query_contracts() -> None:
    parser = build_parser()
    assert parser.parse_args(["status", "--json"]).command == "status"
    assert parser.parse_args(["pause", "--json"]).command == "pause"
    retry = parser.parse_args(
        ["retry", "--stage", "parse", "--confirm-embedding-cost", "--json"]
    )
    assert retry.stage == "parse"
    assert retry.confirm_embedding_cost is True
    query = parser.parse_args(
        [
            "query",
            "儿童是否使用低剂量激素",
            "--fulltext-only",
            "--year-from",
            "2020",
            "--year-to",
            "2025",
            "--json",
        ]
    )
    assert query.question == "儿童是否使用低剂量激素"
    assert query.fulltext_only is True
    assert (query.year_from, query.year_to) == (2020, 2025)


def test_error_rendering_is_stable_and_secret_free() -> None:
    error = KnowledgeBaseError(
        "embedding_auth_failed",
        "Embedding authentication failed",
        details={"missing": ["MPP_API_KEY"]},
    )
    rendered = render_error(error, as_json=True)

    assert json.loads(rendered) == {
        "code": "embedding_auth_failed",
        "message": "Embedding authentication failed",
        "details": {"missing": ["MPP_API_KEY"]},
    }
    assert "Bearer" not in rendered
    assert error_exit_code(error) == 3
    assert error_exit_code(ConfigurationError("bad", "Bad config")) == 2
    assert error_exit_code(KnowledgeBaseError("vector_collection_mismatch", "Bad index")) == 4


def _write_tiny_corpus(source: Path) -> None:
    source.mkdir()
    (source / "metadata.csv").write_text(
        "ID,Author,Publication Year,Title,Publication Title,DOI,Url,Abstract Note,Language\n"
        "1,Zhang,2025,Steroid Trial,Journal,,,Randomized evidence,en\n",
        encoding="utf-8",
    )
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text(
        (72, 72),
        "Methods randomized evidence with enough characters for normal extraction. "
        "Results low dose treatment improved recovery without severe events.",
    )
    pdf.save(source / "1-Steroid Trial.pdf")
    pdf.close()


def test_cli_build_completes_local_stages_without_api_configuration(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "corpus"
    _write_tiny_corpus(source)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPP_KB_SOURCE", str(source))
    monkeypatch.setenv("MPP_KB_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("MPP_API_KEY", raising=False)
    monkeypatch.delenv("MPP_API_BASE", raising=False)

    exit_code = main(["build", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["final_state"] == "embedding_pending"
    assert payload["pending_embedding_count"] >= 2

    assert main(["status", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["state"] == "embedding_pending"


def test_cli_confirming_embeddings_requires_model_configuration(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "corpus"
    _write_tiny_corpus(source)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPP_KB_SOURCE", str(source))
    monkeypatch.setenv("MPP_KB_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("MPP_API_KEY", raising=False)
    monkeypatch.delenv("MPP_API_BASE", raising=False)

    exit_code = main(["build", "--confirm-embedding-cost", "--json"])
    error = json.loads(capsys.readouterr().err)

    assert exit_code == 2
    assert error["code"] == "embedding_config_missing"
