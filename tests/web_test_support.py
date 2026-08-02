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


def request_json(base_url: str, path: str, payload: dict | None = None) -> HTTPResult:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return HTTPResult(
                response.status, json.loads(response.read().decode("utf-8"))
            )
    except urllib.error.HTTPError as error:
        return HTTPResult(
            error.code, json.loads(error.read().decode("utf-8"))
        )


def get_bytes(base_url: str, path: str):
    request = urllib.request.Request(base_url + path)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()
