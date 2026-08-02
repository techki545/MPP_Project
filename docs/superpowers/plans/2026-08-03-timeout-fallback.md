# Model Timeout Evidence Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure a timed-out model request still produces a traceable evidence synthesis and a non-empty relation graph.

**Architecture:** Add a deterministic, extractive claim fallback beside the existing model claim extractor. Feed those uncertain claims through the existing graph builder and make the deterministic reporter summarize evidence hierarchy, chronology, and grounded excerpts without inventing clinical direction.

**Tech Stack:** Python 3, dataclasses, pytest, existing Graph RAG reporter and graph builder, vanilla JavaScript UI.

---

### Task 1: Reproduce the missing-output failure

**Files:**
- Create: `tests/test_timeout_fallback.py`

- [ ] Write a pipeline test with a chat client that raises `chat_timeout`.
- [ ] Assert the current pipeline returns zero claims, zero graph edges, and the placeholder answer.
- [ ] Run `python -m pytest tests/test_timeout_fallback.py -q` and verify the assertions for desired rich output fail.

### Task 2: Build extractive fallback claims

**Files:**
- Modify: `knowledge_base/reporter.py`
- Modify: `knowledge_base/service.py`
- Test: `tests/test_timeout_fallback.py`

- [ ] Add a helper that selects up to eight grounded source excerpts and creates `uncertain` claims with source chunk identifiers.
- [ ] Use the helper when model claim extraction raises an error.
- [ ] Preserve the original model error code for the UI warning.
- [ ] Run the focused test and verify claim assertions pass.

### Task 3: Render fallback relations and synthesis

**Files:**
- Modify: `knowledge_base/graph_builder.py`
- Modify: `knowledge_base/reporter.py`
- Test: `tests/test_timeout_fallback.py`

- [ ] Connect uncertain claims to the question with a traceable `supplements` edge.
- [ ] Generate evidence-pyramid counts, publication chronology, grounded excerpts, and a bounded synthesis statement.
- [ ] Verify the report contains citations and the graph contains visible edges.

### Task 4: Verify the complete application

**Files:**
- Test: `tests/test_timeout_fallback.py`
- Test: `tests/test_web_api.py`
- Test: `tests/test_web_ui.py`

- [ ] Run the focused fallback tests.
- [ ] Run the complete pytest suite.
- [ ] Restart `web_server.py` on port 8765.
- [ ] Submit a real local query without a model and verify the API and browser both show a synthesis and relation graph.
