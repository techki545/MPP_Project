# -*- coding: utf-8 -*-
"""Local HTTP server for the evidence Graph RAG workbench."""

from __future__ import annotations

import argparse
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from graph_rag_core import load_default_graph
from knowledge_base.config import Settings
from knowledge_base.jobs import KnowledgeBaseJobManager
from knowledge_base.service import (
    ProductionBuildRunner,
    create_production_service,
)
from knowledge_base.sqlite_store import SQLiteStore
from web_api import APIError, GraphRAGWebService


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEB_DIR = os.path.join(BASE_DIR, "web")
MAX_BODY_BYTES = 1_000_000
STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/styles.css": "styles.css",
    "/app.js": "app.js",
}


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    service: Any | None = None,
    web_dir: str | None = None,
) -> ThreadingHTTPServer:
    app_service = service or _create_default_service()
    static_dir = os.path.abspath(web_dir or DEFAULT_WEB_DIR)
    handler = partial(
        GraphRAGRequestHandler,
        service=app_service,
        web_dir=static_dir,
    )
    return ThreadingHTTPServer((host, port), handler)


def _create_default_service() -> GraphRAGWebService:
    settings = Settings.from_mapping(os.environ, project_root=Path(BASE_DIR))
    store = SQLiteStore(settings.sqlite_path)
    store.initialize()
    knowledge_service = create_production_service(settings, store)
    runner = ProductionBuildRunner(settings, store)
    jobs = KnowledgeBaseJobManager(
        indexer=runner,
        store_path=settings.data_dir / "jobs.sqlite3",
    )
    return GraphRAGWebService(
        load_default_graph(BASE_DIR),
        knowledge_service=knowledge_service,
        job_manager=jobs,
    )


class GraphRAGRequestHandler(BaseHTTPRequestHandler):
    server_version = "GraphRAGWeb/2.0"

    def __init__(self, *args, service: Any, web_dir: str, **kwargs):
        self.service = service
        self.web_dir = os.path.abspath(web_dir)
        super().__init__(*args, **kwargs)

    def log_message(self, format_string: str, *args) -> None:
        return

    def do_GET(self) -> None:
        try:
            path = self._decoded_path()
            if path == "/api/health":
                result = self.service.health()
            elif path == "/api/evidence":
                result = self.service.evidence_catalog()
            elif path == "/api/kb/status":
                result = self.service.knowledge_base_status()
            elif path == "/api/model/status":
                result = self.service.model_status()
            elif path.startswith("/api/jobs/"):
                job_id = self._dynamic_id(path, "/api/jobs/")
                result = self.service.job_status(job_id)
            elif path.startswith("/api/documents/"):
                document_id, pdf_requested = self._document_route(path)
                if pdf_requested:
                    self._send_pdf(self.service.resolve_pdf(document_id))
                    return
                result = self.service.document_detail(document_id)
            elif path.startswith("/api/"):
                raise APIError("not_found", "接口不存在。", status=404)
            else:
                self._serve_static(path)
                return
            self._send_json(HTTPStatus.OK, result)
        except APIError as error:
            self._send_error(error)
        except Exception:
            self._send_error(
                APIError("internal_error", "服务处理请求时出现内部错误。", status=500)
            )

    def do_POST(self) -> None:
        try:
            path = self._decoded_path()
            payload = self._read_json_body()
            if path in {"/api/query", "/api/analyze"}:
                result = self.service.query(payload)
            elif path == "/api/kb/build":
                result = self.service.start_build(payload)
            elif path == "/api/kb/pause":
                result = self.service.pause_build(payload)
            elif path == "/api/kb/retry":
                result = self.service.retry_build(payload)
            else:
                raise APIError("not_found", "接口不存在。", status=404)
            self._send_json(HTTPStatus.OK, result)
        except APIError as error:
            self._send_error(error)
        except Exception:
            self._send_error(
                APIError("internal_error", "服务处理请求时出现内部错误。", status=500)
            )

    def _decoded_path(self) -> str:
        parsed = urlsplit(self.path)
        try:
            return unquote(parsed.path, encoding="utf-8", errors="strict")
        except UnicodeDecodeError:
            raise APIError("invalid_path", "请求路径编码无效。") from None

    def _document_route(self, path: str) -> tuple[str, bool]:
        prefix = "/api/documents/"
        suffix = path[len(prefix) :]
        pdf_requested = suffix.endswith("/pdf")
        if pdf_requested:
            suffix = suffix[:-4]
        return _validate_route_id(suffix), pdf_requested

    def _dynamic_id(self, path: str, prefix: str) -> str:
        return _validate_route_id(path[len(prefix) :])

    def _read_json_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            content_length = int(raw_length)
        except ValueError:
            raise APIError("invalid_request", "Content-Length 无效。") from None
        if content_length <= 0:
            raise APIError("invalid_json", "请求体必须是 JSON 对象。")
        if content_length > MAX_BODY_BYTES:
            raise APIError("request_too_large", "请求体超过允许大小。", status=413)
        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise APIError("invalid_json", "请求体不是有效 JSON。") from None
        if not isinstance(payload, dict):
            raise APIError("invalid_json", "请求体必须是 JSON 对象。")
        return payload

    def _serve_static(self, path: str) -> None:
        relative_path = STATIC_FILES.get(path)
        if relative_path is None:
            raise APIError("not_found", "页面不存在。", status=404)
        file_path = os.path.abspath(os.path.join(self.web_dir, relative_path))
        if os.path.commonpath([self.web_dir, file_path]) != self.web_dir:
            raise APIError("not_found", "页面不存在。", status=404)
        try:
            content = Path(file_path).read_bytes()
        except FileNotFoundError:
            raise APIError("not_found", "页面资源不存在。", status=404) from None
        content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {
            "application/javascript",
            "application/json",
        }:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _send_pdf(self, path: Path) -> None:
        try:
            file_path = Path(path)
            size = file_path.stat().st_size
            file = file_path.open("rb")
        except (FileNotFoundError, OSError):
            raise APIError("pdf_unavailable", "文献 PDF 不可用。", status=404) from None
        with file:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", 'inline; filename="document.pdf"')
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "private, no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            while chunk := file.read(1024 * 1024):
                self.wfile.write(chunk)

    def _send_error(self, error: APIError) -> None:
        self._send_json(error.status, error.as_dict())

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def _validate_route_id(value: str) -> str:
    if (
        not value
        or len(value) > 256
        or "/" in value
        or "\\" in value
        or ".." in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise APIError("invalid_identifier", "资源标识符无效。")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Graph RAG evidence workbench.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = create_server(args.host, args.port)
    display_host = "localhost" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host
    print(f"Graph RAG web app: http://{display_host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
