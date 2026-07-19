# -*- coding: utf-8 -*-
"""Local HTTP server for the interactive Graph RAG evidence workbench."""

from __future__ import annotations

import argparse
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from typing import Dict

from graph_rag_core import load_default_graph
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
    service: GraphRAGWebService | None = None,
    web_dir: str | None = None,
) -> ThreadingHTTPServer:
    app_service = service or GraphRAGWebService(load_default_graph(BASE_DIR))
    static_dir = os.path.abspath(web_dir or DEFAULT_WEB_DIR)
    handler = partial(
        GraphRAGRequestHandler,
        service=app_service,
        web_dir=static_dir,
    )
    return ThreadingHTTPServer((host, port), handler)


class GraphRAGRequestHandler(BaseHTTPRequestHandler):
    server_version = "GraphRAGWeb/1.0"

    def __init__(self, *args, service: GraphRAGWebService, web_dir: str, **kwargs):
        self.service = service
        self.web_dir = web_dir
        super().__init__(*args, **kwargs)

    def log_message(self, format_string: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._handle_api(self.service.health)
            return
        if self.path == "/api/evidence":
            self._handle_api(self.service.evidence_catalog)
            return
        if self.path.startswith("/api/"):
            self._send_error(APIError("not_found", "接口不存在。", status=404))
            return
        self._serve_static()

    def do_POST(self) -> None:
        try:
            payload = self._read_json_body()
            if self.path == "/api/model/test":
                result = self.service.test_model(payload.get("model"))
            elif self.path == "/api/analyze":
                result = self.service.analyze(
                    payload.get("question", ""),
                    payload.get("evidence_types"),
                    payload.get("model"),
                )
            else:
                raise APIError("not_found", "接口不存在。", status=404)
            self._send_json(HTTPStatus.OK, result)
        except APIError as exc:
            self._send_error(exc)
        except Exception:
            self._send_error(
                APIError(
                    "internal_error",
                    "服务处理请求时出现内部错误。",
                    status=500,
                )
            )

    def _handle_api(self, operation) -> None:
        try:
            self._send_json(HTTPStatus.OK, operation())
        except APIError as exc:
            self._send_error(exc)
        except Exception:
            self._send_error(
                APIError("internal_error", "证据图加载失败。", status=500)
            )

    def _read_json_body(self) -> Dict:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            content_length = int(raw_length)
        except ValueError as exc:
            raise APIError("invalid_request", "Content-Length 无效。") from exc
        if content_length <= 0:
            raise APIError("invalid_json", "请求体必须是 JSON 对象。")
        if content_length > MAX_BODY_BYTES:
            raise APIError(
                "request_too_large",
                "请求体超过允许大小。",
                status=413,
            )
        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise APIError("invalid_json", "请求体不是有效 JSON。") from exc
        if not isinstance(payload, dict):
            raise APIError("invalid_json", "请求体必须是 JSON 对象。")
        return payload

    def _serve_static(self) -> None:
        relative_path = STATIC_FILES.get(self.path)
        if relative_path is None:
            self._send_error(APIError("not_found", "页面不存在。", status=404))
            return
        file_path = os.path.abspath(os.path.join(self.web_dir, relative_path))
        if os.path.commonpath([self.web_dir, file_path]) != self.web_dir:
            self._send_error(APIError("not_found", "页面不存在。", status=404))
            return
        try:
            with open(file_path, "rb") as file:
                content = file.read()
        except FileNotFoundError:
            self._send_error(APIError("not_found", "页面资源不存在。", status=404))
            return
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

    def _send_error(self, error: APIError) -> None:
        self._send_json(error.status, error.as_dict())

    def _send_json(self, status: int, payload: Dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Graph RAG evidence web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = create_server(args.host, args.port)
    display_host = "localhost" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host
    print(
        f"Graph RAG web app: http://{display_host}:{server.server_port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
