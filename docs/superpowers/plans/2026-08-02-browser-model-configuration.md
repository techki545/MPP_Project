# Browser Model Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each web user supply an OpenAI-compatible API base URL, API key, and chat model for one evidence query without persisting or exposing the secret.

**Architecture:** The browser submits an optional validated `model_config` object with `/api/query`. `GraphRAGWebService` normalizes the untrusted HTTP payload, while `KnowledgeBaseService` creates a request-scoped `ChatClient` and asks `ProductionQueryPipeline` to use it for evidence refinement, claim extraction, and report generation without mutating the default pipeline. Empty browser fields preserve the existing environment-configured behavior.

**Tech Stack:** Python 3.11+, standard-library HTTP server and `urllib`, pytest, vanilla HTML/CSS/JavaScript, Playwright.

---

## File Map

- Modify `web_api.py`: validate and normalize the optional browser model configuration, reject legacy top-level credential fields, and pass sanitized configuration to the knowledge service.
- Modify `knowledge_base/service.py`: create and consume a request-scoped `ChatClient` without changing retrieval or global service state.
- Modify `web/index.html`: add accessible API base, API key, model name, and key-visibility controls.
- Modify `web/styles.css`: add responsive, stable form layout within the existing query band.
- Modify `web/app.js`: read transient fields, validate completeness, submit `model_config`, toggle secret visibility, and update non-secret status text.
- Modify `tests/test_web_api.py`: cover the HTTP-facing model-configuration contract and secret redaction.
- Modify `tests/knowledge_base/test_service.py`: cover request-scoped client construction and pipeline isolation.
- Modify `tests/test_web_ui.py`: cover payload submission, incomplete-form blocking, visibility toggle, refresh clearing, and responsive layout.
- Modify `README.md`: document browser configuration behavior and its separation from embedding configuration.

### Task 1: Validate And Forward The Query Model Configuration

**Files:**
- Modify: `tests/test_web_api.py`
- Modify: `web_api.py`

- [ ] **Step 1: Update the fake knowledge service and write acceptance tests**

Add request capture to `FakeKnowledgeService` and make its query signature accept the optional keyword argument:

```python
class FakeKnowledgeService:
    def __init__(self):
        self.received_model_config = None

    # Keep the existing health/status methods.

    def query(self, question, filters, *, model_config=None):
        self.received_model_config = model_config
        return {
            "question": question,
            "mode": "hybrid",
            "answer_markdown": "answer",
            "reasoning_steps": [],
            "sources": [],
            "received_filters": filters,
        }
```

Replace the old browser-configuration rejection test with:

```python
def test_complete_browser_model_configuration_is_normalized_and_forwarded() -> None:
    knowledge = FakeKnowledgeService()
    service = make_web_service(knowledge)

    service.query(
        {
            "question": "clinical question",
            "model_config": {
                "base_url": " https://provider.example/v1/ ",
                "api_key": " secret-token ",
                "model_name": " chat-model ",
            },
        }
    )

    assert knowledge.received_model_config == {
        "base_url": "https://provider.example/v1",
        "api_key": "secret-token",
        "model_name": "chat-model",
    }


def test_empty_browser_model_configuration_uses_server_default() -> None:
    knowledge = FakeKnowledgeService()

    make_web_service(knowledge).query({"question": "clinical question"})

    assert knowledge.received_model_config is None
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_api.py -k "browser_model or server_default" -q
```

Expected: FAIL because `model_config` is still forbidden and the fake service does not receive normalized configuration.

- [ ] **Step 3: Add malformed-input, URL-policy, legacy-field, and redaction tests**

Add parameterized coverage:

```python
@pytest.mark.parametrize(
    "model_config",
    [
        None,
        [],
        {},
        {"base_url": "https://provider.example/v1"},
        {
            "base_url": "http://provider.example/v1",
            "api_key": "secret",
            "model_name": "chat-model",
        },
        {
            "base_url": "https://user:pass@provider.example/v1",
            "api_key": "secret",
            "model_name": "chat-model",
        },
        {
            "base_url": "https://provider.example/v1?token=value",
            "api_key": "secret",
            "model_name": "chat-model",
        },
        {
            "base_url": "https://provider.example/v1",
            "api_key": "secret",
            "model_name": "chat-model",
            "unknown": "value",
        },
    ],
)
def test_invalid_browser_model_configuration_is_rejected(model_config) -> None:
    with pytest.raises(APIError) as captured:
        make_web_service(FakeKnowledgeService()).query(
            {"question": "clinical question", "model_config": model_config}
        )

    assert captured.value.code == "invalid_model_config"


@pytest.mark.parametrize(
    "base_url",
    ["http://localhost:8000/v1", "http://127.0.0.1:8000/v1", "http://[::1]:8000/v1"],
)
def test_loopback_http_model_urls_are_accepted(base_url: str) -> None:
    knowledge = FakeKnowledgeService()
    make_web_service(knowledge).query(
        {
            "question": "clinical question",
            "model_config": {
                "base_url": base_url,
                "api_key": "secret",
                "model_name": "chat-model",
            },
        }
    )
    assert knowledge.received_model_config["base_url"] == base_url


def test_invalid_model_configuration_error_never_contains_api_key() -> None:
    secret = "unique-secret-that-must-not-leak"
    with pytest.raises(APIError) as captured:
        make_web_service(FakeKnowledgeService()).query(
            {
                "question": "clinical question",
                "model_config": {
                    "base_url": "http://provider.example/v1",
                    "api_key": secret,
                    "model_name": "chat-model",
                },
            }
        )

    assert secret not in str(captured.value.as_dict())
```

Retain a test that top-level `model`, `api_key`, `base_url`, and `model_name` remain forbidden.

- [ ] **Step 4: Implement strict request normalization in `web_api.py`**

Import `ipaddress` and `urlsplit`, then add these module-level helpers:

```python
_MODEL_CONFIG_KEYS = frozenset({"base_url", "api_key", "model_name"})
_MODEL_CONFIG_LIMITS = {"base_url": 2048, "api_key": 4096, "model_name": 256}


def _request_model_config(payload: Mapping[str, Any]) -> dict[str, str] | None:
    if "model_config" not in payload:
        return None
    raw = payload["model_config"]
    if not isinstance(raw, Mapping) or set(raw) != _MODEL_CONFIG_KEYS:
        raise APIError("invalid_model_config", "请完整填写 API 地址、API Key 和模型名。")

    values: dict[str, str] = {}
    for key in _MODEL_CONFIG_KEYS:
        value = raw[key]
        if not isinstance(value, str):
            raise APIError("invalid_model_config", "模型配置格式无效。")
        clean = value.strip()
        if not clean or len(clean) > _MODEL_CONFIG_LIMITS[key]:
            raise APIError("invalid_model_config", "模型配置格式无效。")
        values[key] = clean

    values["base_url"] = _validated_model_base_url(values["base_url"])
    return values


def _validated_model_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise APIError("invalid_model_config", "模型 API 地址格式无效。")
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise APIError("invalid_model_config", "公网模型 API 地址必须使用 HTTPS。")
    return value.rstrip("/")


def _is_loopback_host(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False
```

In `GraphRAGWebService.query`, reject only legacy top-level fields, parse the optional object, and forward it:

```python
if any(key in payload for key in ("model", "api_key", "base_url", "model_name")):
    raise APIError(
        "client_model_config_forbidden",
        "请使用 model_config 提供单次查询的模型配置。",
    )
model_config = _request_model_config(payload)
# ...question and filter validation...
return self._call_knowledge(
    self.knowledge_service.query,
    question,
    filters,
    model_config=model_config,
)
```

Update `analyze` to use `model_config` instead of the legacy `model` key.

- [ ] **Step 5: Run the API tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_api.py -q
```

Expected: all tests in `tests/test_web_api.py` PASS.

- [ ] **Step 6: Commit the API contract**

```powershell
git add web_api.py tests/test_web_api.py
git commit -m "feat: accept transient query model configuration"
```

### Task 2: Use A Request-Scoped Chat Client In The Evidence Pipeline

**Files:**
- Modify: `tests/knowledge_base/test_service.py`
- Modify: `knowledge_base/service.py`

- [ ] **Step 1: Write a failing service-level client-construction test**

Update `FakeQueryPipeline.run` to accept `chat_client=None` and record it. Add:

```python
def test_query_builds_request_scoped_chat_client_without_changing_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = []

    class RequestChat:
        def __init__(self, base_url, api_key, model):
            self.base_url = base_url
            self.api_key = api_key
            self.model = model
            created.append(self)

    service = make_service(tmp_path)
    monkeypatch.setattr("knowledge_base.service.ChatClient", RequestChat)

    result = service.query(
        "clinical question",
        SearchFilters(),
        model_config={
            "base_url": "https://provider.example/v1",
            "api_key": "request-secret",
            "model_name": "request-model",
        },
    )

    assert service.pipeline.received_chat_client is created[0]
    assert result["model_name"] == "test-model"
    assert service.settings.chat_model == "chat-model"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\knowledge_base\test_service.py::test_query_builds_request_scoped_chat_client_without_changing_settings -q
```

Expected: FAIL because `KnowledgeBaseService.query` does not accept `model_config`.

- [ ] **Step 3: Add a failing component-wiring test**

Add this focused test to prove all three model-backed stages share the same
ephemeral client:

```python
def test_request_chat_components_share_one_ephemeral_client() -> None:
    request_chat = object()

    refiner, extractor, reporter = ProductionQueryPipeline._chat_components(
        request_chat
    )

    assert refiner.chat_client is request_chat
    assert extractor.chat_client is request_chat
    assert reporter.chat_client is request_chat
```

- [ ] **Step 4: Run the isolation test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\knowledge_base\test_service.py -k "request_scoped or pipeline_uses_request" -q
```

Expected: FAIL because `ProductionQueryPipeline._chat_components` does not exist.

- [ ] **Step 5: Implement request-scoped pipeline components**

Change `KnowledgeBaseService.query` to create a local client only when requested:

```python
def query(
    self,
    question: str,
    filters: SearchFilters,
    *,
    model_config: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    clean_question = " ".join(str(question or "").split())
    if not clean_question:
        raise KnowledgeBaseError("empty_question", "Clinical question is required")
    if len(clean_question) > 2000:
        raise KnowledgeBaseError(
            "question_too_long", "Clinical question exceeds 2000 characters"
        )
    if not isinstance(filters, SearchFilters):
        raise KnowledgeBaseError("invalid_filters", "Search filters are invalid")
    if (
        filters.year_from is not None
        and filters.year_to is not None
        and filters.year_from > filters.year_to
    ):
        raise KnowledgeBaseError("invalid_year_range", "Year range is invalid")

    chat_client: Any | None = None
    if model_config is not None:
        chat_client = ChatClient(
            model_config["base_url"],
            model_config["api_key"],
            model_config["model_name"],
        )
    raw_result = self.pipeline.run(
        clean_question,
        filters,
        chat_client=chat_client,
    )
```

Leave the existing `raw_result` type check and stable response normalization
immediately after this exact block.

Change `ProductionQueryPipeline.run` to select local components:

```python
@staticmethod
def _chat_components(
    chat_client: Any,
) -> tuple[ChatEvidenceRefiner, GroundedClaimExtractor, GroundedReporter]:
    return (
        ChatEvidenceRefiner(chat_client),
        GroundedClaimExtractor(chat_client),
        GroundedReporter(chat_client),
    )


def run(
    self,
    question: str,
    filters: SearchFilters,
    *,
    chat_client: Any | None = None,
) -> dict[str, Any]:
    evidence_refiner = self.evidence_refiner
    claim_extractor = self.claim_extractor
    reporter = self.reporter
    if chat_client is not None:
        evidence_refiner, claim_extractor, reporter = self._chat_components(
            chat_client
        )

    retrieval = self.retriever.search(question, filters)
    sources = tuple(
        self._evidence_source(index, item, evidence_refiner)
        for index, item in enumerate(retrieval.documents, start=1)
    )
```

In the same method, replace the two instance-component calls exactly:

```python
validated = claim_extractor.extract(question, empty_bundle)
report = reporter.generate_with_fallback(question, bundle)
```

Update `_evidence_source` to receive the selected refiner explicitly:

```python
def _evidence_source(
    self,
    number: int,
    item: Any,
    evidence_refiner: ChatEvidenceRefiner,
) -> EvidenceSource:
    # Keep current metadata assembly.
    refined = evidence_refiner.refine(title, abstract, assessment)
    # Keep current EvidenceSource construction.
```

- [ ] **Step 6: Run service and reporter tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\knowledge_base\test_service.py tests\knowledge_base\test_reporter.py -q
```

Expected: all selected tests PASS and the default server client remains unchanged after a request override.

- [ ] **Step 7: Commit request-scoped model execution**

```powershell
git add knowledge_base/service.py tests/knowledge_base/test_service.py
git commit -m "feat: scope chat model overrides to one query"
```

### Task 3: Add The Transient Browser Model Controls

**Files:**
- Modify: `tests/test_web_ui.py`
- Modify: `web/index.html`
- Modify: `web/styles.css`
- Modify: `web/app.js`

- [ ] **Step 1: Add payload capture and write the failing browser test**

In `FakeWorkbenchService.__init__`, add `self.last_query_payload = None` and set
it in `query`. Add a Playwright test:

```python
def test_browser_submits_complete_transient_model_configuration_and_clears_on_reload() -> None:
    service = FakeWorkbenchService()
    with running_server(service) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(base_url)

        page.get_by_label("API 地址").fill("https://provider.example/v1")
        page.get_by_label("API Key").fill("browser-secret")
        page.get_by_label("模型名").fill("request-model")
        page.get_by_role("button", name="显示 API Key").click()
        assert page.get_by_label("API Key").get_attribute("type") == "text"

        page.get_by_role("button", name="开始循证分析").click()
        page.locator("#query-status[data-state='ready']").wait_for()
        assert service.last_query_payload["model_config"] == {
            "base_url": "https://provider.example/v1",
            "api_key": "browser-secret",
            "model_name": "request-model",
        }

        page.reload()
        assert page.get_by_label("API 地址").input_value() == ""
        assert page.get_by_label("API Key").input_value() == ""
        assert page.get_by_label("模型名").input_value() == ""
        browser.close()
```

- [ ] **Step 2: Run the browser test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_ui.py::test_browser_submits_complete_transient_model_configuration_and_clears_on_reload -q
```

Expected: FAIL because the model inputs and visibility button do not exist.

- [ ] **Step 3: Write the failing incomplete-form browser test**

```python
def test_partial_browser_model_configuration_blocks_query() -> None:
    service = FakeWorkbenchService()
    with running_server(service) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(base_url)
        page.get_by_label("API 地址").fill("https://provider.example/v1")
        page.get_by_role("button", name="开始循证分析").click()

        assert "请完整填写" in page.locator("#query-status").inner_text()
        assert service.last_query_payload is None
        browser.close()
```

- [ ] **Step 4: Add accessible HTML controls**

Insert this unframed row between `.question-row` and `.query-options`:

```html
<section class="model-config" aria-labelledby="model-config-title">
  <h2 id="model-config-title">模型服务</h2>
  <div class="model-fields">
    <label>API 地址
      <input id="model-api-base" type="url" maxlength="2048" autocomplete="off" spellcheck="false">
    </label>
    <label>API Key
      <span class="secret-input">
        <input id="api-key" type="password" maxlength="4096" autocomplete="off" spellcheck="false">
        <button id="toggle-api-key" class="secret-toggle" type="button" aria-label="显示 API Key" aria-pressed="false" title="显示 API Key">
          <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M12 5c5.5 0 9.5 5.3 9.7 5.5l.8 1-.8 1C21.5 12.7 17.5 18 12 18S2.5 12.7 2.3 12.5l-.8-1 .8-1C2.5 10.3 6.5 5 12 5Zm0 2c-3.5 0-6.4 3-7.6 4.5C5.6 13 8.5 16 12 16s6.4-3 7.6-4.5C18.4 10 15.5 7 12 7Zm0 1.5a3 3 0 1 1 0 6 3 3 0 0 1 0-6Zm0 2a1 1 0 1 0 0 2 1 1 0 0 0 0-2Z"/></svg>
        </button>
      </span>
    </label>
    <label>模型名
      <input id="chat-model-name" type="text" maxlength="256" autocomplete="off" spellcheck="false">
    </label>
  </div>
</section>
```

- [ ] **Step 5: Add responsive CSS without nested cards**

Add stable form dimensions:

```css
.model-config {
  min-width: 0;
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid var(--line);
}

.model-config h2 {
  margin: 0 0 8px;
  color: #34433e;
  font-size: 12px;
  font-weight: 750;
}

.model-fields {
  display: grid;
  grid-template-columns: minmax(220px, 1.4fr) minmax(190px, 1fr) minmax(160px, 1fr);
  gap: 10px;
}

.model-fields label {
  min-width: 0;
  display: grid;
  gap: 4px;
  color: #34433e;
  font-size: 12px;
  font-weight: 750;
}

.model-fields input {
  width: 100%;
  height: 38px;
  min-width: 0;
  padding: 6px 10px;
  border: 1px solid #b9c5c1;
  border-radius: 4px;
  background: #fff;
  color: var(--ink);
}

.secret-input {
  position: relative;
  min-width: 0;
}

.secret-input input {
  padding-right: 42px;
}

.secret-toggle {
  position: absolute;
  top: 1px;
  right: 1px;
  width: 36px;
  height: 36px;
  display: grid;
  place-items: center;
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--muted);
  cursor: pointer;
}

.secret-toggle svg {
  width: 19px;
  height: 19px;
  fill: currentColor;
}

@media (max-width: 899px) {
  .model-fields { grid-template-columns: minmax(0, 1fr); }
}
```

- [ ] **Step 6: Implement transient field handling in JavaScript**

Cache the four controls and bind input/toggle events. Add:

```javascript
function browserModelConfig({ validate = false } = {}) {
  const config = {
    base_url: dom.modelApiBase.value.trim(),
    api_key: dom.apiKey.value.trim(),
    model_name: dom.chatModelName.value.trim(),
  };
  const entered = Object.values(config).filter(Boolean).length;
  if (entered === 0) return null;
  if (entered !== 3) {
    if (validate) {
      throw new Error("请完整填写 API 地址、API Key 和模型名。");
    }
    return null;
  }
  return config;
}


function toggleApiKeyVisibility() {
  const reveal = dom.apiKey.type === "password";
  dom.apiKey.type = reveal ? "text" : "password";
  dom.toggleApiKey.setAttribute("aria-pressed", String(reveal));
  const label = reveal ? "隐藏 API Key" : "显示 API Key";
  dom.toggleApiKey.setAttribute("aria-label", label);
  dom.toggleApiKey.title = label;
}


function renderModelInputStatus() {
  const values = [dom.modelApiBase.value, dom.apiKey.value, dom.chatModelName.value]
    .map((value) => value.trim());
  if (values.every(Boolean)) {
    setStatusPill(dom.modelStatus, "ready", `本次模型 ${values[2]}`);
  } else if (values.some(Boolean)) {
    setStatusPill(dom.modelStatus, "warning", "模型配置待补全");
  } else if (state.modelStatus) {
    renderServerModelStatus(state.modelStatus);
  }
}
```

At the start of `runQuery`, catch validation before setting busy state:

```javascript
let modelConfig;
try {
  modelConfig = browserModelConfig({ validate: true });
} catch (error) {
  setQueryStatus(error.message, "error");
  return;
}
// Build the existing payload.
if (modelConfig) payload.model_config = modelConfig;
```

Do not add any browser-storage call. Keep the API key out of status strings,
toast messages, and `state`.

- [ ] **Step 7: Run UI tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_ui.py -q
```

Expected: all UI tests PASS at desktop and mobile widths, including no horizontal overflow.

- [ ] **Step 8: Commit the browser controls**

```powershell
git add web/index.html web/styles.css web/app.js tests/test_web_ui.py
git commit -m "feat: add transient browser model controls"
```

### Task 4: Document, Integrate, And Verify The Complete Feature

**Files:**
- Modify: `README.md`
- Test: `tests/test_web_api.py`
- Test: `tests/knowledge_base/test_service.py`
- Test: `tests/test_web_ui.py`

- [ ] **Step 1: Add user-facing operation documentation**

Add a README section containing these exact operational facts:

```markdown
### 网页临时模型配置

网页中的“模型服务”可填写 OpenAI 兼容的 API 地址、API Key 和模型名。
三项信息只随本次循证查询发送，刷新页面后清空，服务器不会写入文件、
数据库或状态接口。三项全部留空时，系统继续使用 `MPP_API_BASE`、
`MPP_API_KEY` 和 `MPP_CHAT_MODEL`。

该配置只用于报告生成，不会改变 `MPP_EMBEDDING_MODEL`，也不会自动启动
付费向量化。公网 API 地址必须使用 HTTPS；本机服务可使用
`http://localhost` 或 `http://127.0.0.1`。
```

- [ ] **Step 2: Run the complete Python test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests PASS with no failures or errors.

- [ ] **Step 3: Run static syntax and whitespace checks**

Run:

```powershell
node --check web/app.js
.\.venv\Scripts\python.exe -m compileall -q knowledge_base web_api.py web_server.py
git diff --check
```

Expected: all commands exit 0. Existing line-ending notices may be reported by Git, but there must be no whitespace errors.

- [ ] **Step 4: Restart the local application from the feature worktree**

Stop only the process currently listening on port 8765, then start:

```powershell
Start-Process -WindowStyle Hidden -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "web_server.py --host 127.0.0.1 --port 8765" -WorkingDirectory (Get-Location)
```

Expected: `http://127.0.0.1:8765/api/health` returns JSON and the page loads at `http://127.0.0.1:8765/`.

- [ ] **Step 5: Perform live browser verification**

Using Playwright against the restarted server, verify:

```text
Desktop 1440x900: all three model fields fit, API Key starts masked, toggle works.
Mobile 390x844: fields stack, primary action stays visible, no horizontal overflow.
Partial config: query is blocked before POST.
Refresh: all three fields are empty.
Empty config: existing deterministic/server-configured path still works.
```

Do not submit a real paid provider request during this verification.

- [ ] **Step 6: Commit documentation and any verification-only adjustments**

```powershell
git add README.md
git commit -m "docs: explain transient web model configuration"
```

- [ ] **Step 7: Confirm final repository state**

Run:

```powershell
git status --short --branch
git log -5 --oneline
```

Expected: clean `feature/mpp-hybrid-knowledge-base` worktree with the API,
service, UI, and documentation commits visible.
