"""Command-line entry point for knowledge-base inspection and build control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from .config import Settings
from .embedding_client import EmbeddingClient
from .errors import ConfigurationError, KnowledgeBaseError
from .indexer import (
    BuildPipeline,
    Indexer,
    build_default_pipeline,
    validate_embedding_cost_gate,
)
from .models import SearchFilters
from .service import create_production_service
from .sqlite_store import SQLiteStore
from .vector_store import LocalVectorStore


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m knowledge_base.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--source")
    inspect_parser.add_argument("--json", action="store_true")

    build = commands.add_parser("build")
    build.add_argument("--document-limit", type=_positive_int)
    build.add_argument("--embedding-limit", type=_positive_int)
    build.add_argument("--confirm-embedding-cost", action="store_true")
    build.add_argument("--confirm-full-embedding-cost", action="store_true")
    build.add_argument("--json", action="store_true")

    status = commands.add_parser("status")
    status.add_argument("--json", action="store_true")

    pause = commands.add_parser("pause")
    pause.add_argument("--json", action="store_true")

    retry = commands.add_parser("retry")
    retry.add_argument(
        "--stage",
        required=True,
        choices=(*BuildPipeline.LOCAL_STAGES, "embedding", "vector"),
    )
    retry.add_argument("--confirm-embedding-cost", action="store_true")
    retry.add_argument("--confirm-full-embedding-cost", action="store_true")
    retry.add_argument("--json", action="store_true")

    query = commands.add_parser("query")
    query.add_argument("question")
    query.add_argument("--fulltext-only", action="store_true")
    query.add_argument("--year-from", type=int)
    query.add_argument("--year-to", type=int)
    query.add_argument("--json", action="store_true")
    return parser


def error_exit_code(error: KnowledgeBaseError) -> int:
    if isinstance(error, ConfigurationError) or error.code.endswith("_config_missing"):
        return 2
    if error.code in {"embedding_auth_failed", "model_auth_failed"}:
        return 3
    if error.code in {
        "vector_collection_mismatch",
        "embedding_cache_corrupt",
        "index_corrupt",
        "index_incompatible",
    }:
        return 4
    return 2


def render_error(error: KnowledgeBaseError, *, as_json: bool) -> str:
    payload = {
        "code": error.code,
        "message": error.message,
        "details": error.details,
    }
    if as_json:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return f"[{error.code}] {error.message}"


def _render_success(payload: dict, *, as_json: bool) -> str:
    if as_json:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return "\n".join(f"{key}: {value}" for key, value in payload.items())


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    as_json = bool(getattr(args, "json", False))
    try:
        settings = Settings.from_mapping(os.environ, project_root=Path.cwd())
        if args.command == "inspect":
            source = Path(args.source) if args.source else settings.source_dir
            payload = Indexer.inspect(source).as_dict()
        else:
            store = SQLiteStore(settings.sqlite_path)
            store.initialize()
            if args.command == "status":
                payload = store.get_job(Indexer.JOB_ID) or {
                    "job_id": Indexer.JOB_ID,
                    "state": "idle",
                    "progress": {},
                }
            elif args.command == "pause":
                indexer = _make_indexer(settings, store)
                payload = indexer.request_pause()
            elif args.command in {"build", "retry"}:
                full_confirmation = bool(args.confirm_full_embedding_cost)
                confirm_cost = bool(args.confirm_embedding_cost or full_confirmation)
                embedding_limit = getattr(args, "embedding_limit", None)
                validate_embedding_cost_gate(
                    store,
                    model_name=settings.embedding_model,
                    confirm_embedding_cost=confirm_cost,
                    confirm_full_embedding_cost=full_confirmation,
                    embedding_limit=embedding_limit,
                )
                vector_store: LocalVectorStore | None = None
                embedding_client: EmbeddingClient | None = None
                if confirm_cost:
                    settings.require_embedding_access()
                    embedding_client = EmbeddingClient(
                        base_url=settings.api_base,
                        api_key=settings.api_key,
                        model=settings.embedding_model,
                    )
                    dimension = embedding_client.probe()
                    vector_store = LocalVectorStore(settings.qdrant_dir)
                    vector_store.ensure_collections(
                        dimension=dimension,
                        model_name=settings.embedding_model,
                    )
                try:
                    indexer = _make_indexer(
                        settings,
                        store,
                        embedding_client=embedding_client,
                        vector_store=vector_store,
                    )
                    pipeline = build_default_pipeline(settings.source_dir, store=store)
                    if args.command == "retry" and args.stage in BuildPipeline.LOCAL_STAGES:
                        pipeline = BuildPipeline(
                            stages={args.stage: pipeline.stages[args.stage]},
                            algorithm_version=pipeline.algorithm_version,
                        )
                    result = indexer.build(
                        pipeline,
                        document_limit=getattr(args, "document_limit", None),
                        embedding_limit=embedding_limit,
                        confirm_embedding_cost=confirm_cost,
                    )
                    if (
                        confirm_cost
                        and not full_confirmation
                        and embedding_limit is not None
                        and result.stage_counters.get("embedding", {}).get("processed", 0)
                        > 0
                        and result.stage_counters.get("embedding", {}).get("failed", 0)
                        == 0
                    ):
                        store.mark_embedding_probe_completed(
                            settings.embedding_model
                        )
                    payload = result.as_dict()
                finally:
                    if vector_store is not None:
                        vector_store.close()
            elif args.command == "query":
                service = create_production_service(settings, store)
                payload = service.query(
                    args.question,
                    SearchFilters(
                        year_from=args.year_from,
                        year_to=args.year_to,
                        fulltext_only=args.fulltext_only,
                    ),
                )
            else:
                raise ConfigurationError("unknown_command", "Command is not supported")
        print(_render_success(payload, as_json=as_json))
        return 0
    except KnowledgeBaseError as error:
        print(render_error(error, as_json=as_json), file=sys.stderr)
        return error_exit_code(error)


def _make_indexer(
    settings: Settings,
    store: SQLiteStore,
    *,
    embedding_client: EmbeddingClient | None = None,
    vector_store: LocalVectorStore | None = None,
) -> Indexer:
    return Indexer(
        store=store,
        embedding_client=embedding_client,
        vector_store=vector_store,
        lock_path=settings.data_dir / "build.lock",
        model_name=settings.embedding_model,
        embedding_batch_size=settings.embedding_batch_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
