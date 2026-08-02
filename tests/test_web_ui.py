from __future__ import annotations

from playwright.sync_api import sync_playwright
import pytest

from tests.web_test_support import running_server


class FakeWorkbenchService:
    def __init__(self, *, probe_completed: bool = False):
        self.probe_completed = probe_completed
        self.last_build_payload = None

    def health(self):
        return {
            "status": "ok",
            "knowledge_base": {"status": "ready"},
            "model": {"configured": True},
        }

    def knowledge_base_status(self):
        return {
            "status": "ready",
            "metadata_records": 35408,
            "pdf_files": 909,
            "matched_pdf_files": 909,
            "parsed_pdf_files": 909,
            "chunks": 7699,
            "embedded": 1,
            "errors": 0,
            "embedding_probe_completed": self.probe_completed,
            "embedding_pending": 128,
        }

    def model_status(self):
        return {
            "configured": True,
            "chat_model": "test-model",
            "embedding_model": "test-embedding",
        }

    def evidence_catalog(self):
        return {
            "default_question": "SMPP儿童是否应常规使用糖皮质激素？",
            "data_source": "demo",
            "nodes": [],
            "edges": [],
            "evidence_types": [
                {"key": "guideline", "label": "指南", "count": 2},
                {"key": "systematic_review", "label": "系统综述", "count": 3},
                {
                    "key": "randomized_controlled_trial",
                    "label": "随机对照试验（RCT）",
                    "count": 1,
                },
                {"key": "observational_study", "label": "观察性研究", "count": 4},
                {"key": "narrative_review", "label": "叙述性综述", "count": 1},
                {"key": "case_report", "label": "病例报告", "count": 1},
            ],
        }

    def query(self, payload):
        return {
            "question": payload["question"],
            "mode": "hybrid",
            "degraded_reason": "",
            "reasoning_steps": [
                {
                    "title": "第一步：检索并盘点证据库存",
                    "body": "检索到一项高质量RCT。[1]",
                    "source_ids": [1],
                },
                {
                    "title": "第二步：检查证据时间关系",
                    "body": "2025年RCT更新了较早的指南证据窗口。[1]",
                    "source_ids": [1],
                },
            ],
            "answer_markdown": "## 推荐意见\n推荐优先采用低剂量方案。[1]",
            "model_used": True,
            "model_name": "test-model",
            "model_error": None,
            "summary": {
                "evidence_count": 1,
                "relation_count": 1,
                "update_count": 1,
                "conflict_count": 0,
            },
            "sources": [
                {
                    "source_number": 1,
                    "document_id": "doc-1",
                    "title": "Randomized methylprednisolone dose trial",
                    "evidence_type": "randomized_controlled_trial",
                    "year": 2025,
                    "quality": "high",
                    "journal": "Pediatric Investigation",
                    "chunk_ids": ["chunk-1"],
                    "snippets": ["Low dose result"],
                    "page_ranges": ["3"],
                    "fulltext": True,
                }
            ],
            "graph": {
                "nodes": [
                    {
                        "node_id": "question",
                        "node_type": "question",
                        "label": "是否使用糖皮质激素",
                        "payload": {},
                    },
                    {
                        "node_id": "claim-1-1",
                        "node_type": "claim",
                        "label": "低剂量方案得到支持",
                        "payload": {"evidence_type": "randomized_controlled_trial"},
                    },
                ],
                "edges": [
                    {
                        "source": "claim-1-1",
                        "target": "question",
                        "relation": "updates",
                        "rationale": "较新的RCT更新指南窗口",
                    }
                ],
            },
            "retrieval_stats": {"candidate_count": 18, "claim_count": 1},
            "data_source": "knowledge_base",
        }

    def document_detail(self, document_id):
        return {
            "document_id": document_id,
            "title": "Randomized methylprednisolone dose trial",
            "authors": ["Xu", "Zhang"],
            "year": 2025,
            "journal": "Pediatric Investigation",
            "doi": "10.1000/very-long-doi-value-for-responsive-layout-testing",
            "abstract": "A multicenter randomized controlled trial.",
            "evidence_type": "randomized_controlled_trial",
            "quality": "high",
            "has_fulltext": True,
            "fulltext_status": "parsed",
            "document_snippets": [
                {
                    "chunk_id": "chunk-1",
                    "text": "Introduction text that was not hit by this query",
                    "section": "Introduction",
                    "page_start": 1,
                    "page_end": 1,
                    "quality": "high",
                    "is_ocr": False,
                }
            ],
            "pdf_available": True,
        }

    def start_build(self, payload):
        self.last_build_payload = dict(payload)
        return {"job_id": "job-1", "state": "queued", "progress": {}}

    def job_status(self, job_id):
        return {
            "job_id": job_id,
            "state": "embedding_pending",
            "progress": {"pending_embedding_count": 128},
        }


def test_workbench_has_no_browser_api_key_and_renders_grounded_sources() -> None:
    with running_server(FakeWorkbenchService()) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(base_url)

        page.get_by_text("35,408").wait_for()
        page.get_by_label("临床问题").fill("SMPP儿童是否应常规使用糖皮质激素？")
        assert page.locator("#api-key").count() == 0
        page.get_by_role("button", name="开始循证分析").click()
        page.get_by_text("第二部分：综合循证回答").wait_for()
        page.get_by_text("推荐优先采用低剂量方案").wait_for()
        assert page.locator("[data-source-number='1']").count() >= 1
        assert page.locator("#evidence-graph [data-node-id]").count() == 2

        page.locator("[data-source-number='1']").first.click()
        page.get_by_text("本次查询命中片段").wait_for()
        page.get_by_text("Low dose result").wait_for()
        assert page.get_by_text("Introduction text that was not hit by this query").count() == 0
        assert page.get_by_role("link", name="在浏览器中查看 PDF").is_visible()
        browser.close()


def test_mobile_layout_has_no_horizontal_overflow_and_keeps_primary_action_visible() -> None:
    with running_server(FakeWorkbenchService()) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(base_url)

        assert page.get_by_role("button", name="开始循证分析").is_visible()
        overflow = page.evaluate(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth"
        )
        assert overflow is False
        assert page.locator("#api-key").count() == 0
        browser.close()


@pytest.mark.parametrize(
    ("probe_completed", "expected"),
    [
        (False, {"confirm_embedding_cost": True, "embedding_limit": 32}),
        (
            True,
            {
                "confirm_embedding_cost": True,
                "confirm_full_embedding_cost": True,
            },
        ),
    ],
)
def test_embedding_continue_uses_two_distinct_confirmations(
    probe_completed: bool, expected: dict
) -> None:
    service = FakeWorkbenchService(probe_completed=probe_completed)
    with running_server(service) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.on("dialog", lambda dialog: dialog.accept())
        page.goto(base_url)
        page.get_by_text("35,408").wait_for()

        page.locator("#continue-kb").click()
        page.locator("#job-state").get_by_text("本地索引完成，等待向量化").wait_for()

        assert service.last_build_payload == expected
        browser.close()
