from __future__ import annotations

from pathlib import Path

from web_api import APIError
from tests.web_test_support import get_bytes, request_json, running_server


class FakeWebService:
    def health(self):
        return {"status": "ok"}

    def evidence_catalog(self):
        return {"nodes": []}

    def knowledge_base_status(self):
        return {"status": "ready"}

    def model_status(self):
        return {"configured": True}

    def query(self, payload):
        return {"question": payload["question"], "answer_markdown": "answer"}

    def document_detail(self, document_id):
        if document_id != "doc-1":
            raise APIError("not_found", "文献不存在。", status=404)
        return {"document_id": "doc-1", "title": "Trial"}

    def resolve_pdf(self, document_id):
        if document_id != "doc-1":
            raise APIError("not_found", "文献不存在。", status=404)
        return self.pdf_path


def test_dynamic_document_route_is_parsed_without_path_traversal(tmp_path: Path) -> None:
    service = FakeWebService()
    service.pdf_path = tmp_path / "trial.pdf"
    service.pdf_path.write_bytes(b"%PDF-1.4 test")
    with running_server(service) as base_url:
        response = request_json(base_url, "/api/documents/doc-1")
        rejected = request_json(base_url, "/api/documents/..%2Fsecret")
        pdf_status, headers, body = get_bytes(
            base_url, "/api/documents/doc-1/pdf?inline=true"
        )

    assert response.status == 200
    assert response.json["document_id"] == "doc-1"
    assert rejected.status == 400
    assert pdf_status == 200
    assert headers["Content-Type"] == "application/pdf"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert body.startswith(b"%PDF")


def test_query_and_analyze_alias_share_the_same_backend() -> None:
    service = FakeWebService()
    service.pdf_path = Path("unused.pdf")
    with running_server(service) as base_url:
        query = request_json(base_url, "/api/query", {"question": "q"})
        analyze = request_json(base_url, "/api/analyze", {"question": "q"})

    assert query.status == analyze.status == 200
    assert query.json == analyze.json
