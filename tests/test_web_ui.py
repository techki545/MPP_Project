from __future__ import annotations

from pathlib import Path
from threading import Event

from playwright.sync_api import sync_playwright
import pytest

from tests.web_test_support import running_server


class FakeWorkbenchService:
    def __init__(self, *, probe_completed: bool = False):
        self.probe_completed = probe_completed
        self.last_build_payload = None
        self.last_query_payload = None
        self.query_call_count = 0

    def health(self):
        return {
            "status": "ok",
            "knowledge_base": {"status": "ready"},
            "model": {"configured": True},
        }

    def knowledge_base_status(self):
        return {
            "status": "ready",
            "metadata_records": 34074,
            "source_metadata_records": 35408,
            "unique_documents": 34074,
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
        self.query_call_count += 1
        self.last_query_payload = dict(payload)
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


class DelayedStartupFailureService(FakeWorkbenchService):
    def __init__(self):
        super().__init__()
        self.health_started = Event()
        self.release_health = Event()

    def health(self):
        self.health_started.set()
        if not self.release_health.wait(timeout=5):
            raise RuntimeError("timed out waiting to release startup health check")
        raise RuntimeError("controlled startup failure")


def test_model_service_controls_are_accessible_and_key_starts_masked() -> None:
    with running_server(FakeWorkbenchService()) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(base_url)

        page.get_by_text("35,408").wait_for()
        api_base = page.get_by_label("API 地址", exact=True)
        api_key = page.get_by_label("API Key", exact=True)
        model_name = page.get_by_label("模型名", exact=True)
        toggle = page.get_by_role("button", name="显示 API Key", exact=True)

        assert api_base.get_attribute("type") == "url"
        assert api_base.get_attribute("maxlength") == "2048"
        assert api_key.get_attribute("type") == "password"
        assert api_key.get_attribute("maxlength") == "4096"
        assert model_name.get_attribute("type") == "text"
        assert model_name.get_attribute("maxlength") == "256"
        for control in (api_base, api_key, model_name):
            assert control.get_attribute("autocomplete") == "off"
            assert control.get_attribute("spellcheck") == "false"
        assert toggle.get_attribute("title") == "显示 API Key"
        assert toggle.get_attribute("aria-pressed") == "false"
        assert toggle.locator("svg").count() == 1
        assert page.locator("#model-service").evaluate(
            """section => {
                const question = document.querySelector('.question-row');
                const filters = document.querySelector('.query-options');
                return Boolean(
                    question.compareDocumentPosition(section) & Node.DOCUMENT_POSITION_FOLLOWING
                ) && Boolean(
                    section.compareDocumentPosition(filters) & Node.DOCUMENT_POSITION_FOLLOWING
                );
            }"""
        )
        assert page.evaluate(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth"
        ) is False
        browser.close()


def test_complete_browser_model_config_is_trimmed_sent_and_key_can_be_revealed() -> None:
    service = FakeWorkbenchService()
    with running_server(service) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(base_url)
        page.get_by_text("35,408").wait_for()

        page.get_by_label("API 地址", exact=True).fill(" https://models.example.invalid/v1 ")
        page.get_by_label("API Key", exact=True).fill(" test-secret-value ")
        page.get_by_label("模型名", exact=True).fill(" private-chat-model ")

        page.get_by_role("button", name="显示 API Key", exact=True).click()
        toggle = page.locator("#toggle-api-key")
        assert page.locator("#api-key").get_attribute("type") == "text"
        assert toggle.get_attribute("aria-label") == "隐藏 API Key"
        assert toggle.get_attribute("title") == "隐藏 API Key"
        assert toggle.get_attribute("aria-pressed") == "true"

        page.get_by_role("button", name="开始循证分析").click()
        page.get_by_text("分析完成，共返回 1 项来源。").wait_for()

        assert service.query_call_count == 1
        assert service.last_query_payload["model_config"] == {
            "base_url": "https://models.example.invalid/v1",
            "api_key": "test-secret-value",
            "model_name": "private-chat-model",
        }
        browser.close()


def test_partial_browser_model_config_blocks_query_and_focuses_first_missing_field() -> None:
    service = FakeWorkbenchService()
    with running_server(service) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(base_url)
        page.get_by_text("35,408").wait_for()

        page.get_by_label("API Key", exact=True).fill("partial-secret-value")
        page.get_by_label("模型名", exact=True).fill("draft-model")
        page.get_by_role("button", name="开始循证分析").click()

        assert service.query_call_count == 0
        assert page.locator("#query-status").text_content() == (
            "请完整填写 API 地址、API Key 和模型名。"
        )
        assert page.locator("#query-status").get_attribute("data-state") == "error"
        assert page.evaluate("document.activeElement.id") == "model-api-base"
        assert "partial-secret-value" not in page.locator("#query-status").text_content()
        assert "partial-secret-value" not in page.locator("#toast").text_content()
        assert page.locator("#run-query").is_enabled()
        browser.close()


def test_model_status_pill_tracks_browser_config_without_exposing_secrets() -> None:
    with running_server(FakeWorkbenchService()) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(base_url)
        page.get_by_text("模型 test-model").wait_for()

        pill = page.locator("#model-status")
        assert pill.get_attribute("data-state") == "ready"

        page.locator("#api-key").fill("pill-secret-value")
        assert pill.locator("span").text_content() == "模型配置待补全"
        assert pill.get_attribute("data-state") == "warning"

        page.locator("#model-api-base").fill("https://private.example.invalid/v1")
        page.locator("#chat-model-name").fill("browser-model")
        assert pill.locator("span").text_content() == "本次模型 browser-model"
        assert pill.get_attribute("data-state") == "ready"
        assert "pill-secret-value" not in pill.text_content()
        assert "private.example.invalid" not in pill.text_content()

        page.locator("#model-api-base").fill("")
        page.locator("#api-key").fill("")
        page.locator("#chat-model-name").fill("")
        assert pill.locator("span").text_content() == "模型 test-model"
        assert pill.get_attribute("data-state") == "ready"
        browser.close()


def test_browser_model_config_clears_on_reload_and_uses_no_browser_storage() -> None:
    with running_server(FakeWorkbenchService()) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(base_url)
        page.get_by_text("35,408").wait_for()

        page.locator("#model-api-base").fill("https://models.example.invalid/v1")
        page.locator("#api-key").fill("reload-secret-value")
        page.locator("#chat-model-name").fill("reload-model")
        assert page.evaluate(
            "() => ({local: localStorage.length, session: sessionStorage.length})"
        ) == {"local": 0, "session": 0}

        page.reload()
        page.get_by_text("35,408").wait_for()
        assert page.locator("#model-api-base").input_value() == ""
        assert page.locator("#api-key").input_value() == ""
        assert page.locator("#chat-model-name").input_value() == ""
        browser.close()

    web_dir = Path(__file__).resolve().parents[1] / "web"
    app_source = (web_dir / "app.js").read_text(encoding="utf-8")
    index_source = (web_dir / "index.html").read_text(encoding="utf-8")
    for forbidden in (
        "localStorage",
        "sessionStorage",
        "indexedDB",
        "document.cookie",
        "URLSearchParams",
        "location.search",
    ):
        assert forbidden not in app_source
    assert 'type="hidden"' not in index_source


def test_browser_model_config_clears_and_remasks_across_persisted_page_transitions() -> None:
    with running_server(FakeWorkbenchService()) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(base_url)
        page.get_by_text("模型 test-model").wait_for()

        api_base = page.locator("#model-api-base")
        api_key = page.locator("#api-key")
        model_name = page.locator("#chat-model-name")
        toggle = page.locator("#toggle-api-key")
        pill = page.locator("#model-status")

        api_base.fill("https://private.example.invalid/v1")
        api_key.fill("page-transition-secret")
        model_name.fill("page-transition-model")
        toggle.click()
        page.evaluate(
            "window.dispatchEvent(new PageTransitionEvent('pagehide', { persisted: true }))"
        )

        assert api_base.input_value() == ""
        assert api_key.input_value() == ""
        assert model_name.input_value() == ""
        assert api_key.get_attribute("type") == "password"
        assert toggle.get_attribute("aria-label") == "显示 API Key"
        assert toggle.get_attribute("title") == "显示 API Key"
        assert toggle.get_attribute("aria-pressed") == "false"
        assert "page-transition-secret" not in pill.text_content()

        api_base.fill("https://restored.example.invalid/v1")
        api_key.fill("restored-page-secret")
        model_name.fill("restored-page-model")
        toggle.click()
        page.evaluate(
            "window.dispatchEvent(new PageTransitionEvent('pageshow', { persisted: true }))"
        )

        assert api_base.input_value() == ""
        assert api_key.input_value() == ""
        assert model_name.input_value() == ""
        assert api_key.get_attribute("type") == "password"
        assert toggle.get_attribute("aria-label") == "显示 API Key"
        assert toggle.get_attribute("title") == "显示 API Key"
        assert toggle.get_attribute("aria-pressed") == "false"
        assert pill.locator("span").text_content() == "模型 test-model"
        assert "restored-page-secret" not in pill.text_content()
        assert "restored.example.invalid" not in pill.text_content()
        browser.close()


def test_complete_invalid_model_api_url_blocks_query_and_marks_the_field() -> None:
    service = FakeWorkbenchService()
    with running_server(service) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(base_url)
        page.get_by_text("模型 test-model").wait_for()

        api_base = page.locator("#model-api-base")
        api_base.fill("not a valid provider URL")
        page.locator("#api-key").fill("invalid-url-secret")
        page.locator("#chat-model-name").fill("invalid-url-model")

        assert api_base.evaluate("input => input.validity.valid") is False
        pill = page.locator("#model-status")
        assert pill.get_attribute("data-state") == "warning"
        assert pill.locator("span").text_content() == "API 地址格式有误"
        assert "not a valid provider URL" not in pill.text_content()
        assert "invalid-url-secret" not in pill.text_content()

        page.get_by_role("button", name="开始循证分析").click()

        assert service.query_call_count == 0
        assert api_base.get_attribute("aria-invalid") == "true"
        assert page.locator("#query-status").text_content() == "请输入有效的模型 API 地址。"
        assert page.locator("#query-status").get_attribute("data-state") == "error"
        assert page.evaluate("document.activeElement.id") == "model-api-base"

        api_base.fill("https://models.example.invalid/v1")
        assert api_base.get_attribute("aria-invalid") is None
        assert pill.locator("span").text_content() == "本次模型 invalid-url-model"

        api_base.fill("")
        assert api_base.get_attribute("aria-invalid") is None
        browser.close()


def test_startup_failure_preserves_completed_browser_model_status() -> None:
    service = DelayedStartupFailureService()
    with running_server(service) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(base_url)
        assert service.health_started.wait(timeout=2)

        pill = page.locator("#model-status")
        page.locator("#api-key").fill("pending-startup-secret")
        assert pill.locator("span").text_content() == "模型配置待补全"
        page.locator("#api-key").fill("")
        assert pill.locator("span").text_content() == "模型状态加载中"
        assert pill.get_attribute("data-state") == "loading"
        assert "pending-startup-secret" not in pill.text_content()

        page.locator("#model-api-base").fill("https://models.example.invalid/v1")
        page.locator("#api-key").fill("startup-secret")
        page.locator("#chat-model-name").fill("startup-browser-model")
        assert pill.locator("span").text_content() == "本次模型 startup-browser-model"

        service.release_health.set()
        page.get_by_text("知识库连接失败").wait_for()

        assert pill.locator("span").text_content() == "本次模型 startup-browser-model"
        assert pill.get_attribute("data-state") == "ready"
        assert "startup-secret" not in pill.text_content()
        assert page.locator("#query-status").text_content() == "服务处理请求时出现内部错误。"
        assert page.locator("#query-status").get_attribute("data-state") == "error"

        page.locator("#chat-model-name").fill("")
        assert pill.locator("span").text_content() == "模型配置待补全"
        assert pill.get_attribute("data-state") == "warning"

        page.locator("#model-api-base").fill("")
        page.locator("#api-key").fill("")
        assert pill.locator("span").text_content() == "后端状态未知"
        assert pill.get_attribute("data-state") == "warning"
        assert "startup-secret" not in pill.text_content()
        assert "models.example.invalid" not in pill.text_content()
        browser.close()


def test_startup_failure_preserves_partial_browser_model_status() -> None:
    service = DelayedStartupFailureService()
    with running_server(service) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(base_url)
        assert service.health_started.wait(timeout=2)

        page.locator("#api-key").fill("partial-startup-secret")
        pill = page.locator("#model-status")
        assert pill.locator("span").text_content() == "模型配置待补全"

        service.release_health.set()
        page.get_by_text("知识库连接失败").wait_for()

        assert pill.locator("span").text_content() == "模型配置待补全"
        assert pill.get_attribute("data-state") == "warning"
        assert "partial-startup-secret" not in pill.text_content()
        browser.close()


def test_workbench_renders_grounded_sources_and_omits_empty_model_config() -> None:
    service = FakeWorkbenchService()
    with running_server(service) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(base_url)

        page.get_by_text("35,408").wait_for()
        page.get_by_text("34,074").wait_for()
        page.get_by_label("临床问题").fill("SMPP儿童是否应常规使用糖皮质激素？")
        page.get_by_role("button", name="开始循证分析").click()
        page.get_by_text("第二部分：综合循证回答").wait_for()
        page.get_by_text("推荐优先采用低剂量方案").wait_for()
        assert service.query_call_count == 1
        assert "model_config" not in service.last_query_payload
        assert page.locator("[data-source-number='1']").count() >= 1
        assert page.locator("#evidence-graph [data-node-id]").count() == 2

        page.locator("[data-source-number='1']").first.click()
        page.get_by_text("本次查询命中片段").wait_for()
        page.get_by_text("Low dose result").wait_for()
        assert page.get_by_text("Introduction text that was not hit by this query").count() == 0
        assert page.get_by_role("link", name="在浏览器中查看 PDF").is_visible()
        assert page.locator(".evidence-workspace").evaluate(
            "element => getComputedStyle(element).alignItems"
        ) == "start"
        detail_metrics = page.locator("#source-detail").evaluate(
            """element => {
                const paragraph = element.querySelector('.snippet-list p');
                paragraph.textContent = 'long evidence text '.repeat(3000);
                return {
                    clientHeight: element.clientHeight,
                    scrollHeight: element.scrollHeight,
                    overflowY: getComputedStyle(element).overflowY,
                };
            }"""
        )
        assert detail_metrics["overflowY"] == "auto"
        assert detail_metrics["scrollHeight"] > detail_metrics["clientHeight"]
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
        control_boxes = [
            page.locator(selector).bounding_box()
            for selector in ("#model-api-base", "#api-key", "#chat-model-name")
        ]
        assert all(box is not None for box in control_boxes)
        assert control_boxes[0]["y"] < control_boxes[1]["y"] < control_boxes[2]["y"]
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
