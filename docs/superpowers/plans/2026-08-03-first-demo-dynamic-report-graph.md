# Dynamic First-Demo Report and Evidence Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current excerpt-style synthesis and technical claim graph with a dynamic, source-grounded six-step report and the first demo's document-level evidence graph for every clinical question.

**Architecture:** Keep retrieval and claim-level provenance as internal pipeline data. Extend Quote-ID claims with validated closed labels, derive traceable claim relations, project those relations onto at most ten source documents, and use that document graph as the report-planning and UI contract. The frontend renders the first demo's evidence-type lanes, visible colored curved arrows, relation labels, node selection highlighting, and source-detail linkage.

**Tech Stack:** Python 3.9+, dataclasses, SQLite/Qdrant retrieval, OpenAI-compatible JSON chat API, vanilla JavaScript, SVG, CSS, pytest, Playwright.

---

## File Structure

- Modify `knowledge_base/graph_builder.py`: closed claim labels and traceable claim-to-claim relation rules.
- Create `knowledge_base/document_graph.py`: project internal claims and relations into the first-demo document graph.
- Create `knowledge_base/report_structure.py`: six-step stage contract and deterministic logical report composer.
- Modify `knowledge_base/reporter.py`: structured model prompts, validation, and six-step model/fallback output.
- Modify `knowledge_base/service.py`: build internal claim graph, public document graph, and stable summary fields.
- Modify `web/app.js`: first-demo SVG graph renderer and node/source interaction.
- Modify `web/styles.css`: first-demo node, edge, label, focus, overflow, and responsive styling.
- Modify `tests/knowledge_base/test_graph_builder.py`: closed labels and all six evidence relations.
- Create `tests/knowledge_base/test_document_graph.py`: document projection, source limits, traceability, and ordering.
- Create `tests/knowledge_base/test_report_structure.py`: exact six-step deterministic reasoning and conclusion-first answer.
- Modify `tests/knowledge_base/test_reporter.py`: exact report contract and grounded model behavior.
- Modify `tests/test_timeout_fallback.py`: timeout recovery retains six steps and document graph.
- Modify `tests/test_web_ui.py`: document graph rendering, visible relations, click highlighting, and detail linkage.
- Create `tests/test_dynamic_questions.py`: distinct questions produce distinct report and graph content.
- Modify `.gitignore`: ignore persisted visual-companion files under `.superpowers/`.

### Task 1: Preserve the Verified Timeout Baseline

**Files:**
- Modify: `.gitignore`
- Existing changes: `knowledge_base/chat_client.py`
- Existing changes: `knowledge_base/graph_builder.py`
- Existing changes: `knowledge_base/reporter.py`
- Existing changes: `knowledge_base/service.py`
- Existing tests: `tests/test_chat_client.py`
- Existing tests: `tests/test_timeout_fallback.py`
- Existing tests: `tests/knowledge_base/test_reporter.py`

- [ ] **Step 1: Ignore the visual-companion working directory**

Add this exact line to `.gitignore`:

```gitignore
.superpowers/
```

- [ ] **Step 2: Run the complete baseline suite**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Expected: `378 passed` or a larger passing count if tests were added after this plan.

- [ ] **Step 3: Confirm the baseline does not include the visual mockups**

Run:

```powershell
git status --short
```

Expected: no `.superpowers/` entry; only the timeout foundation, tests, and design/plan documents appear.

- [ ] **Step 4: Commit the timeout foundation separately**

```powershell
git add .gitignore knowledge_base/chat_client.py knowledge_base/graph_builder.py knowledge_base/reporter.py knowledge_base/service.py tests/test_chat_client.py tests/test_timeout_fallback.py tests/knowledge_base/test_reporter.py docs/superpowers/specs/2026-08-03-timeout-fallback-design.md docs/superpowers/plans/2026-08-03-timeout-fallback.md
git commit -m "fix: make grounded model queries resilient"
```

Expected: one commit that does not contain the new first-demo implementation.

### Task 2: Add Closed, Source-Grounded Claim Labels

**Files:**
- Modify: `knowledge_base/graph_builder.py`
- Modify: `knowledge_base/reporter.py`
- Modify: `tests/test_timeout_fallback.py`
- Modify: `tests/knowledge_base/test_graph_builder.py`

- [ ] **Step 1: Write failing Quote-ID label tests**

Add tests that submit a known Quote ID with closed labels and verify accepted labels while arbitrary prose fields remain ignored:

```python
def test_quote_id_claim_keeps_only_closed_reasoning_labels() -> None:
    bundle = _large_bundle(source_count=1)
    client = _PayloadCaptureClient({
        "claims": [{
            "source_number": 1,
            "source_quote_id": "quote-1-1",
            "clinical_aspect": "effectiveness",
            "direction": "supports",
            "evidence_role": "core",
            "population": "invented population",
            "dose": "999 mg/kg/day",
        }]
    })

    result = GroundedClaimExtractor(client).extract("Clinical question", bundle)

    assert result.claims[0].clinical_aspect == "effectiveness"
    assert result.claims[0].direction == "supports"
    assert result.claims[0].evidence_role == "core"
    assert result.claims[0].population == ""
    assert result.claims[0].dose == ""


def test_quote_id_claim_downgrades_unknown_labels() -> None:
    # Use the same fixture with invalid label strings.
    result = GroundedClaimExtractor(client).extract("Clinical question", bundle)
    assert result.claims[0].clinical_aspect == "other"
    assert result.claims[0].direction == "uncertain"
    assert result.claims[0].evidence_role == "supplement"
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_timeout_fallback.py -k "closed_reasoning_labels or downgrades_unknown_labels" -q
```

Expected: FAIL because `EvidenceClaim` has no `clinical_aspect` or `evidence_role`.

- [ ] **Step 3: Extend the claim contract and Quote-ID parser**

Add closed vocabularies and fields in `knowledge_base/graph_builder.py`:

```python
CLINICAL_ASPECTS = frozenset({
    "overall", "effectiveness", "dose", "timing", "safety",
    "diagnosis", "prognosis", "applicability", "other",
})
EVIDENCE_ROLES = frozenset({"core", "supplement", "boundary"})

@dataclass(frozen=True)
class EvidenceClaim:
    # Existing required fields remain unchanged.
    clinical_aspect: str = "other"
    evidence_role: str = "supplement"

    def __post_init__(self) -> None:
        # Existing validation remains.
        if self.clinical_aspect not in CLINICAL_ASPECTS:
            raise ValueError("Clinical aspect is invalid")
        if self.evidence_role not in EVIDENCE_ROLES:
            raise ValueError("Evidence role is invalid")
```

In the Quote-ID branch of `GroundedClaimExtractor._validate_claim`, normalize only closed labels:

```python
direction = _closed_label(raw_claim.get("direction"), _DIRECTIONS, "uncertain")
clinical_aspect = _closed_label(
    raw_claim.get("clinical_aspect"), CLINICAL_ASPECTS, "other"
)
evidence_role = _closed_label(
    raw_claim.get("evidence_role"), EVIDENCE_ROLES, "supplement"
)
```

Keep the exact server-resolved quote as `statement` and `source_quote`; keep arbitrary PICO, dose, sample-size, effect, and limitation fields empty in this trusted branch.

Extend the `claim(...)` helper in `tests/knowledge_base/test_graph_builder.py` with keyword-only `clinical_aspect: str = "other"` and `evidence_role: str = "supplement"` arguments, and pass both values into `EvidenceClaim`. This keeps every relation test explicit and avoids rebuilding claims inline.

- [ ] **Step 4: Update the extraction prompt**

Require `clinical_aspect`, `direction`, and `evidence_role`, define every allowed value, and state that direction means the quote's direction relative to the user's proposition. Continue requiring one supplied Quote ID per source.

- [ ] **Step 5: Run focused claim and graph tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_timeout_fallback.py tests/knowledge_base/test_graph_builder.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add knowledge_base/graph_builder.py knowledge_base/reporter.py tests/test_timeout_fallback.py tests/knowledge_base/test_graph_builder.py
git commit -m "feat: add grounded evidence reasoning labels"
```

### Task 3: Derive All First-Demo Evidence Relations

**Files:**
- Modify: `knowledge_base/graph_builder.py`
- Modify: `tests/knowledge_base/test_graph_builder.py`

- [ ] **Step 1: Write one failing test for each relation**

Use claims with the same `clinical_aspect` and explicit evidence types:

```python
@pytest.mark.parametrize(
    ("source_type", "target_type", "source_year", "target_year", "source_direction", "target_direction", "aspect", "expected"),
    [
        ("systematic_review", "guideline", 2022, 2023, "supports", "supports", "overall", "supports"),
        ("randomized_controlled_trial", "systematic_review", 2025, 2020, "supports", "supports", "dose", "confirms"),
        ("randomized_controlled_trial", "guideline", 2025, 2023, "supports", "supports", "dose", "updates"),
        ("observational_study", "systematic_review", 2024, 2022, "supports", "supports", "applicability", "supplements"),
        ("randomized_controlled_trial", "systematic_review", 2025, 2024, "opposes", "supports", "effectiveness", "conflicts"),
        ("case_report", "guideline", 2025, 2023, "uncertain", "supports", "safety", "cautions"),
    ],
)
def test_first_demo_relation_vocabulary(
    source_type: str,
    target_type: str,
    source_year: int,
    target_year: int,
    source_direction: str,
    target_direction: str,
    aspect: str,
    expected: str,
) -> None:
    source = claim(
        "source",
        source_type,
        source_year,
        source_direction,
        aspect,
        safety=expected == "cautions",
        clinical_aspect=aspect,
        evidence_role="boundary" if expected == "cautions" else "core",
    )
    target = claim(
        "target",
        target_type,
        target_year,
        target_direction,
        aspect,
        clinical_aspect=aspect,
        evidence_role="core",
    )
    graph = LocalGraphBuilder().build("question", [source, target])
    assert expected in {edge.relation for edge in graph.edges}
```

Add a separate test proving that a newer year without the same clinical aspect cannot create `updates`.

- [ ] **Step 2: Run and verify failure**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/knowledge_base/test_graph_builder.py -q
```

Expected: FAIL for `confirms` and for aspect-based relations.

- [ ] **Step 3: Implement the relation rules**

Extend `ALLOWED_RELATIONS` with `confirms`. Update `_pair_relation` so that:

```python
if _is_update(source, target):
    relation = "updates"
elif _is_caution(source, target):
    relation = "cautions"
elif same_aspect and opposite_directions:
    relation = "conflicts"
elif same_aspect and same_direction and target.evidence_type == "guideline":
    relation = "supports"
elif same_aspect and same_direction:
    relation = "confirms"
elif different_aspect or source.evidence_role == "supplement":
    relation = "supplements"
else:
    return None
```

`_is_update` must require a newer RCT/systematic review, an older guideline or review, compatible directions, and a specific aspect that is not `other`. `_is_caution` must require `safety`, `applicability`, a safety signal, or `boundary` role.

- [ ] **Step 4: Run relation tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/knowledge_base/test_graph_builder.py -q
```

Expected: all relation tests pass and every edge retains `source_claim_ids` and `rationale`.

- [ ] **Step 5: Commit**

```powershell
git add knowledge_base/graph_builder.py tests/knowledge_base/test_graph_builder.py
git commit -m "feat: derive first-demo evidence relations"
```

### Task 4: Project the Internal Graph to Document Nodes

**Files:**
- Create: `knowledge_base/document_graph.py`
- Create: `tests/knowledge_base/test_document_graph.py`
- Modify: `knowledge_base/reporter.py`

- [ ] **Step 1: Write failing projection tests**

Cover document-node shape, evidence ordering, edge collapsing, and the ten-node limit:

```python
def test_document_graph_projects_claim_edges_to_sources() -> None:
    graph = build_document_graph(sources, claims, claim_graph, limit=10)

    assert {node["node_type"] for node in graph["nodes"]} == {"document"}
    assert graph["nodes"][0]["payload"] == {
        "document_id": "guide-doc",
        "source_number": 1,
        "title": "Clinical guideline",
        "evidence_type": "guideline",
        "year": 2023,
        "quality": "moderate",
    }
    assert graph["edges"][0]["source"] == "rct-doc"
    assert graph["edges"][0]["target"] == "guide-doc"
    assert graph["edges"][0]["source_claim_ids"] == ["rct-claim", "guide-claim"]


def test_document_graph_limits_nodes_without_losing_source_list() -> None:
    graph = build_document_graph(sources, claims, claim_graph, limit=10)
    assert len(graph["nodes"]) == 10
    assert all(edge["source"] in node_ids and edge["target"] in node_ids for edge in graph["edges"])
```

- [ ] **Step 2: Run and verify failure**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/knowledge_base/test_document_graph.py -q
```

Expected: FAIL because `knowledge_base.document_graph` does not exist.

- [ ] **Step 3: Implement `build_document_graph`**

Create a focused projector with this public signature:

```python
def build_document_graph(
    sources: Sequence[EvidenceSource],
    claims: Sequence[EvidenceClaim],
    claim_graph: Mapping[str, Any],
    *,
    limit: int = 10,
) -> dict[str, Any]:
    """Return first-demo-compatible document nodes and traceable relations."""
```

Select documents in source ranking order, but ensure the first available item from each present evidence type is retained before filling remaining slots. Collapse duplicate document relations by `(source, target, relation)` and union their `source_claim_ids`. Sort nodes by evidence pyramid rank, source number, and document ID.

- [ ] **Step 4: Extend `EvidenceBundle` with internal claims**

Add a backward-compatible field:

```python
@dataclass(frozen=True)
class EvidenceBundle:
    sources: tuple[EvidenceSource, ...]
    graph: dict[str, Any]
    claims: tuple[EvidenceClaim, ...] = ()
```

Update `_graph_claims_by_document` to use `bundle.claims` first and retain its existing graph-node fallback for old fixtures.

- [ ] **Step 5: Run projection and reporter tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/knowledge_base/test_document_graph.py tests/knowledge_base/test_reporter.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add knowledge_base/document_graph.py knowledge_base/reporter.py tests/knowledge_base/test_document_graph.py tests/knowledge_base/test_reporter.py
git commit -m "feat: project evidence relations to document graph"
```

### Task 5: Enforce the First Demo's Six-Step Report

**Files:**
- Create: `knowledge_base/report_structure.py`
- Create: `tests/knowledge_base/test_report_structure.py`
- Modify: `knowledge_base/reporter.py`
- Modify: `tests/knowledge_base/test_reporter.py`
- Modify: `tests/test_timeout_fallback.py`

- [ ] **Step 1: Write failing report-structure tests**

Define the exact stage keys and verify missing evidence levels remain visible:

```python
EXPECTED_STAGE_KEYS = (
    "inventory", "guidelines", "systematic_reviews",
    "randomized_trials", "lower_level_evidence", "synthesis",
)

def test_deterministic_report_always_has_first_demo_six_steps() -> None:
    report = compose_deterministic_report("question", bundle)
    assert tuple(step["stage_key"] for step in report.analysis_steps) == EXPECTED_STAGE_KEYS
    assert report.analysis_steps[1]["title"] == "第二步：优先查看指南"
    assert "未检索到" in report.analysis_steps[1]["body"]
    assert report.final_answer_markdown.startswith("## 综合回答")


def test_deterministic_answer_is_conclusion_first() -> None:
    report = compose_deterministic_report("question", populated_bundle)
    first_paragraph = report.final_answer_markdown.split("\n\n", 2)[1]
    assert first_paragraph.startswith("**结论：")
    assert "证据链" in report.final_answer_markdown
    assert "适用边界" in report.final_answer_markdown
    assert "证据缺口" in report.final_answer_markdown
```

- [ ] **Step 2: Run and verify failure**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/knowledge_base/test_report_structure.py -q
```

Expected: FAIL because `report_structure.py` does not exist.

- [ ] **Step 3: Implement the deterministic six-step composer**

Create:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class StructuredReport:
    analysis_steps: tuple[dict[str, object], ...]
    final_answer_markdown: str

FIRST_DEMO_STAGES = (
    ("inventory", "第一步：检索并盘点证据库存"),
    ("guidelines", "第二步：优先查看指南"),
    ("systematic_reviews", "第三步：查阅系统综述，检验指南结论"),
    ("randomized_trials", "第四步：聚焦关键随机对照试验"),
    ("lower_level_evidence", "第五步：用下级证据补充安全性与边界"),
    ("synthesis", "第六步：检查一致性并形成综合判断"),
)

def compose_deterministic_report(
    question: str, bundle: EvidenceBundle
) -> StructuredReport:
    grouped = {
        evidence_type: tuple(
            source for source in bundle.sources
            if source.evidence_type == evidence_type
        )
        for evidence_type in _EVIDENCE_ORDER
    }
    relation_counts = Counter(
        edge["relation"] for edge in bundle.graph.get("edges", [])
    )
    claims_by_document = {
        claim.document_id: claim for claim in bundle.claims
    }
    steps = _compose_first_demo_steps(
        question, grouped, relation_counts, claims_by_document
    )
    answer = _compose_conclusion_first_answer(
        grouped, relation_counts, claims_by_document
    )
    return StructuredReport(steps, answer)
```

Define `_compose_first_demo_steps` in the same module to return the six dictionaries in `FIRST_DEMO_STAGES` order. Define `_compose_conclusion_first_answer` there to render exact grounded claim excerpts with citations under the five required headings. Both helpers group sources by evidence type, state absent levels explicitly, and emit no positive or negative recommendation when all directions are uncertain.

- [ ] **Step 4: Strengthen the model report contract**

Change `_MODEL_REPORT_PROMPT` and `required_output` so the model returns exactly six steps with `stage_key`, `title`, `body`, and `source_ids`. Require the final answer to contain these headings in order:

```text
## 综合回答
### 证据链
### 时间更新
### 安全性与适用边界
### 证据缺口
```

Validate known citations and quantities. Each substantive sentence must cite a document that owns at least one validated claim. If qualitative paraphrase validation fails, replace only that sentence with the cited validated claim rather than replacing the complete answer with an excerpt list.

- [ ] **Step 5: Replace the four-step timeout fallback**

Call `compose_deterministic_report` from `GroundedReporter._deterministic_fallback`. Preserve `model_used=False` and the real error code while returning six steps, a conclusion-first answer, and the document graph.

- [ ] **Step 6: Run report and timeout tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/knowledge_base/test_report_structure.py tests/knowledge_base/test_reporter.py tests/test_timeout_fallback.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add knowledge_base/report_structure.py knowledge_base/reporter.py tests/knowledge_base/test_report_structure.py tests/knowledge_base/test_reporter.py tests/test_timeout_fallback.py
git commit -m "feat: restore first-demo six-step synthesis"
```

### Task 6: Wire the Public Document Graph Through the Service

**Files:**
- Modify: `knowledge_base/service.py`
- Modify: `tests/knowledge_base/test_service.py`
- Modify: `tests/test_web_api.py`

- [ ] **Step 1: Write failing service-contract tests**

```python
def test_query_returns_document_graph_and_six_steps() -> None:
    result = pipeline.run("clinical question", SearchFilters())

    assert len(result["reasoning_steps"]) == 6
    assert {node["node_type"] for node in result["graph"]["nodes"]} == {"document"}
    assert all(
        edge["source_claim_ids"] and edge["rationale"]
        for edge in result["graph"]["edges"]
    )
    assert "confirm_count" in result["summary"]
    assert "caution_count" in result["summary"]
```

- [ ] **Step 2: Run and verify failure**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/knowledge_base/test_service.py tests/test_web_api.py -q
```

Expected: FAIL because the service still returns the technical graph.

- [ ] **Step 3: Build both internal and public graphs in the pipeline**

Use this sequence in `ProductionQueryPipeline.run`:

```python
claim_graph = self.graph_builder.build(question, validated.claims).as_dict()
public_graph = build_document_graph(
    sources, validated.claims, claim_graph, limit=10
)
bundle = EvidenceBundle(
    sources=sources,
    graph=public_graph,
    claims=validated.claims,
)
report = reporter.generate_with_fallback(question, bundle)
```

Count all six relation types in `summary`; keep all retrieved sources in `sources` even when the graph contains ten documents.

- [ ] **Step 4: Run service and API tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/knowledge_base/test_service.py tests/test_web_api.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add knowledge_base/service.py tests/knowledge_base/test_service.py tests/test_web_api.py
git commit -m "feat: expose dynamic document evidence graph"
```

### Task 7: Restore the First-Demo SVG Graph Renderer

**Files:**
- Modify: `web/app.js`
- Modify: `web/styles.css`
- Modify: `tests/test_web_ui.py`

- [ ] **Step 1: Update the UI fixture to a document graph**

Replace question/claim nodes in `FakeWorkbenchService.query` with guideline, review, RCT, and case-report document nodes. Include `supports`, `updates`, `confirms`, and `cautions` edges with `rationale` and `source_claim_ids`.

- [ ] **Step 2: Write failing Playwright assertions**

```python
def test_first_demo_document_graph_renders_and_links_to_sources() -> None:
    page.goto(base_url)
    page.get_by_role("button", name="开始循证分析").click()

    assert page.locator("#evidence-graph [data-node-type='document']").count() == 4
    assert page.locator("#evidence-graph .graph-edge").count() == 4
    assert page.locator("#evidence-graph .edge-label").count() == 4
    assert page.get_by_text("指南", exact=True).is_visible()
    assert page.get_by_text("随机对照试验（RCT）", exact=True).is_visible()

    page.locator("#evidence-graph [data-source-number='1']").click()
    page.get_by_text("本次查询命中片段").wait_for()
    assert page.locator("#evidence-graph .graph-node.is-selected").count() == 1
    assert page.locator("#evidence-graph .graph-node.is-muted").count() >= 1
```

Add desktop `1440x900` and mobile `390x844` checks proving the graph scroll container has no clipped node text and the page itself has no unintended horizontal overflow.

- [ ] **Step 3: Run and verify failure**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_web_ui.py -k "first_demo_document_graph" -q
```

Expected: FAIL because the renderer still uses question/document/claim/outcome layers.

- [ ] **Step 4: Port and adapt the first-demo renderer**

Add state and style maps:

```javascript
const TYPE_STYLES = {
  guideline: { color: "#176b4d", soft: "#e8f3ed" },
  systematic_review: { color: "#315f75", soft: "#eaf1f5" },
  randomized_controlled_trial: { color: "#a56713", soft: "#fff3dd" },
  observational_study: { color: "#6f5b7c", soft: "#f1edf4" },
  narrative_review: { color: "#68746e", soft: "#f0f3f1" },
  case_report: { color: "#a3443d", soft: "#faecea" },
};
const RELATION_STYLES = {
  supports: { color: "#176b4d", label: "支持" },
  supplements: { color: "#315f75", label: "补充" },
  updates: { color: "#a56713", label: "更新" },
  confirms: { color: "#6f5b7c", label: "确认" },
  conflicts: { color: "#b0303b", label: "冲突" },
  cautions: { color: "#a3443d", label: "警示" },
};
```

Replace `graphLayout` and the four technical layers with first-demo evidence lanes. Keep all edges visible by default. Route short edges directly and assign long edges alternating upper/lower channel indices so paths do not pass through node rectangles. Render a marker and Chinese label for every edge.

- [ ] **Step 5: Connect graph nodes to source details**

On click, set `state.selectedGraphNodeId`, rerender graph highlighting direct neighbors, and call:

```javascript
const sourceNumber = Number(node.payload.source_number);
if (Number.isInteger(sourceNumber)) selectSource(sourceNumber);
```

Use Enter and Space keyboard handlers and include source title in `aria-label`.

- [ ] **Step 6: Implement stable responsive styles**

Port `.graph-edge`, `.edge-label`, `.graph-node`, `.node-type`, `.node-title`, and `.node-year` from the first demo. Keep cards at `rx=5`, stable `134x78` dimensions, a minimum SVG width based on lane count, and horizontal scrolling inside the graph area only.

- [ ] **Step 7: Run all UI tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_web_ui.py -q
```

Expected: all Playwright tests pass.

- [ ] **Step 8: Commit**

```powershell
git add web/app.js web/styles.css tests/test_web_ui.py
git commit -m "feat: restore first-demo evidence graph UI"
```

### Task 8: Verify Dynamic Behavior Across Questions

**Files:**
- Create: `tests/test_dynamic_questions.py`
- Modify: `validation/mpp_questions.json` only if an existing validation entry is malformed.

- [ ] **Step 1: Write two deterministic dynamic-query tests**

Use fixtures for a treatment question and a diagnostic/prognostic question:

```python
def test_distinct_questions_produce_distinct_reports_and_graphs() -> None:
    treatment = treatment_pipeline.run("Should steroids be used?", SearchFilters())
    prognosis = prognosis_pipeline.run("Which findings predict refractory MPP?", SearchFilters())

    assert treatment["answer_markdown"] != prognosis["answer_markdown"]
    assert {node["node_id"] for node in treatment["graph"]["nodes"]} != {
        node["node_id"] for node in prognosis["graph"]["nodes"]
    }
    assert len(treatment["reasoning_steps"]) == 6
    assert len(prognosis["reasoning_steps"]) == 6
```

Add a missing-guideline fixture and assert that step two says `未检索到指南` rather than inventing one.

- [ ] **Step 2: Run and verify failure**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_dynamic_questions.py -q
```

Expected: FAIL until the service and report contract are fully dynamic.

- [ ] **Step 3: Enforce query-independent composition**

Search `knowledge_base/document_graph.py`, `knowledge_base/report_structure.py`, and `knowledge_base/reporter.py` for fixed steroid/SMPP language; remove any query-specific clinical text from executable code. Pass the submitted question, retrieved sources, validated claims, and derived relations into the report composer and document-graph projector. Generate each missing-evidence message from the corresponding empty evidence group, so a missing guideline, RCT, or lower-level group is described accurately for every query without special cases.

```powershell
rg -n "支原体|糖皮质激素|甲泼尼龙|SMPP|steroid|methylprednisolone" knowledge_base/document_graph.py knowledge_base/report_structure.py knowledge_base/reporter.py
```

Expected: no clinical case-specific constants; only generic prompt instructions or test fixtures may contain such wording.

- [ ] **Step 4: Run the dynamic tests and complete suite**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_dynamic_questions.py -q
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Expected: both commands pass.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_dynamic_questions.py validation/mpp_questions.json
git commit -m "test: verify dynamic reports across clinical questions"
```

### Task 9: Live Model and Visual Acceptance

**Files:**
- Create: `validation/first_demo_acceptance.json` containing no API key.
- Modify: `README.md`

- [ ] **Step 1: Restart the local service with the latest code**

Stop only the process listening on `127.0.0.1:8765`, verify its command belongs to this worktree, and restart:

```powershell
& '.\.venv\Scripts\python.exe' web_server.py --host 127.0.0.1 --port 8765
```

Expected: `Graph RAG web app: http://localhost:8765`.

- [ ] **Step 2: Run one authorized DeepSeek query without logging the key**

Submit the standard steroid/SMPP question through `/api/query` with the user-provided per-request model configuration. Record only this sanitized acceptance data:

```json
{
  "model_used": true,
  "model_error": null,
  "reasoning_step_count": 6,
  "answer_nonempty": true,
  "document_node_count": 8,
  "relation_count_greater_than_zero": true
}
```

Do not store the request body, Authorization header, API key, or raw provider response.

- [ ] **Step 3: Capture desktop and mobile screenshots**

Use Playwright at `1440x900` and `390x844`. Verify all document nodes render, arrows do not cross node rectangles, relation labels remain readable, and clicking a node highlights its direct evidence neighborhood and opens source details.

- [ ] **Step 4: Run final verification**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
git diff --check
Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/health'
```

Expected: full suite passes, `git diff --check` prints nothing, and health returns `status: ok`.

- [ ] **Step 5: Document operation and interpretation**

Update `README.md` to explain the fixed six-step output, dynamic document graph, six relation meanings, ten-node display limit, degraded retrieval status, and the distinction between graph display nodes and internal claim provenance.

- [ ] **Step 6: Commit**

```powershell
git add README.md validation/first_demo_acceptance.json
git commit -m "docs: record first-demo acceptance workflow"
```
