"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";

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
  updates: { color: "#a56713", label: "更新" },
  supplements: { color: "#315f75", label: "补充" },
  confirms: { color: "#6f5b7c", label: "确认" },
  cautions: { color: "#a3443d", label: "警示" },
};

const state = {
  catalog: null,
  visibleTypes: new Set(),
  selectedNodeId: null,
  analysis: null,
  busy: false,
  toastTimer: null,
};

const dom = {};

document.addEventListener("DOMContentLoaded", initializeApp);

async function initializeApp() {
  bindDom();
  bindEvents();
  restoreModelSettings();
  updateQuestionCount();
  try {
    const catalog = await apiRequest("/api/evidence");
    state.catalog = catalog;
    state.visibleTypes = new Set(catalog.evidence_types.map((item) => item.key));
    dom.question.value = catalog.default_question;
    updateQuestionCount();
    renderFilters();
    renderGraph();
    renderEvidenceList();
    renderInitialSummary();
    const firstNode = visibleNodes()[0];
    if (firstNode) {
      selectEvidence(firstNode.id);
    }
  } catch (error) {
    dom.graphMeta.textContent = "证据图加载失败";
    showToast(error.message);
  }
}

function bindDom() {
  dom.question = document.getElementById("clinical-question");
  dom.questionCount = document.getElementById("question-count");
  dom.generateButton = document.getElementById("generate-answer");
  dom.progress = document.getElementById("analysis-progress");
  dom.filters = document.getElementById("evidence-filters");
  dom.toggleAllTypes = document.getElementById("toggle-all-types");
  dom.modelSettings = document.getElementById("model-settings");
  dom.focusSettings = document.getElementById("focus-settings");
  dom.apiKey = document.getElementById("api-key");
  dom.baseUrl = document.getElementById("base-url");
  dom.modelName = document.getElementById("model-name");
  dom.testModelButton = document.getElementById("test-model");
  dom.toggleKeyVisibility = document.getElementById("toggle-key-visibility");
  dom.modelStatus = document.getElementById("model-status");
  dom.graph = document.getElementById("evidence-graph");
  dom.graphMeta = document.getElementById("graph-meta");
  dom.graphEmpty = document.getElementById("graph-empty");
  dom.analysisChips = document.getElementById("analysis-chips");
  dom.decisionSummary = document.getElementById("decision-summary");
  dom.evidenceList = document.getElementById("evidence-list");
  dom.evidenceCount = document.getElementById("visible-evidence-count");
  dom.evidenceDetail = document.getElementById("evidence-detail");
  dom.reasoningSteps = document.getElementById("reasoning-steps");
  dom.finalAnswer = document.getElementById("final-answer");
  dom.reportModel = document.getElementById("report-model");
  dom.answerGrounding = document.getElementById("answer-grounding");
  dom.toast = document.getElementById("toast");
}

function bindEvents() {
  dom.question.addEventListener("input", handleQuestionInput);
  dom.generateButton.addEventListener("click", submitAnalysis);
  dom.testModelButton.addEventListener("click", testModelConnection);
  dom.toggleAllTypes.addEventListener("click", toggleAllEvidenceTypes);
  dom.focusSettings.addEventListener("click", () => {
    dom.modelSettings.open = true;
    dom.modelSettings.scrollIntoView({ behavior: "smooth", block: "center" });
    dom.apiKey.focus({ preventScroll: true });
  });
  dom.toggleKeyVisibility.addEventListener("click", () => {
    dom.apiKey.type = dom.apiKey.type === "password" ? "text" : "password";
  });
  dom.baseUrl.addEventListener("change", persistModelSettings);
  dom.modelName.addEventListener("change", persistModelSettings);
}

function updateQuestionCount() {
  dom.questionCount.textContent = `${dom.question.value.length} / 500`;
}

function handleQuestionInput() {
  updateQuestionCount();
  if (invalidateReport("临床问题已变更，当前报告已失效")) {
    renderGraph();
    renderEvidenceList();
    renderEvidenceDetail();
  }
}

function restoreModelSettings() {
  const savedBaseUrl = sessionStorage.getItem("graphRagBaseUrl");
  const savedModelName = sessionStorage.getItem("graphRagModelName");
  if (savedBaseUrl) {
    dom.baseUrl.value = savedBaseUrl;
  }
  if (savedModelName) {
    dom.modelName.value = savedModelName;
  }
}

function persistModelSettings() {
  sessionStorage.setItem("graphRagBaseUrl", dom.baseUrl.value.trim());
  sessionStorage.setItem("graphRagModelName", dom.modelName.value.trim());
}

function renderFilters() {
  dom.filters.replaceChildren();
  if (!state.catalog) {
    return;
  }
  state.catalog.evidence_types.forEach((evidenceType) => {
    const label = document.createElement("label");
    label.className = "filter-row";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.visibleTypes.has(evidenceType.key);
    checkbox.dataset.evidenceType = evidenceType.key;
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        state.visibleTypes.add(evidenceType.key);
      } else {
        state.visibleTypes.delete(evidenceType.key);
      }
      handleFilterChange();
    });

    const swatch = document.createElement("span");
    swatch.className = "filter-swatch";
    swatch.style.setProperty("--swatch", typeStyle(evidenceType.key).color);

    const name = document.createElement("span");
    name.textContent = evidenceType.label;

    const count = document.createElement("span");
    count.className = "filter-count";
    count.textContent = String(evidenceType.count);

    label.append(checkbox, swatch, name, count);
    dom.filters.append(label);
  });
  updateToggleAllLabel();
}

function toggleAllEvidenceTypes() {
  if (!state.catalog) {
    return;
  }
  const allKeys = state.catalog.evidence_types.map((item) => item.key);
  if (state.visibleTypes.size === allKeys.length) {
    state.visibleTypes.clear();
  } else {
    state.visibleTypes = new Set(allKeys);
  }
  renderFilters();
  handleFilterChange();
}

function updateToggleAllLabel() {
  const total = state.catalog ? state.catalog.evidence_types.length : 0;
  dom.toggleAllTypes.textContent = state.visibleTypes.size === total ? "全不选" : "全选";
}

function handleFilterChange() {
  invalidateReport("证据筛选已变更，当前报告已失效");
  const nodes = visibleNodes();
  if (!nodes.some((node) => node.id === state.selectedNodeId)) {
    state.selectedNodeId = nodes[0] ? nodes[0].id : null;
  }
  renderGraph();
  renderEvidenceList();
  renderEvidenceDetail();
  updateToggleAllLabel();
}

function visibleNodes() {
  return activeNodes().filter((node) => state.visibleTypes.has(node.evidence_type));
}

function visibleEdges(nodes) {
  const ids = new Set(nodes.map((node) => node.id));
  return activeEdges().filter(
    (edge) => ids.has(edge.source) && ids.has(edge.target),
  );
}

function activeNodes() {
  if (state.analysis && Array.isArray(state.analysis.evidence)) {
    return state.analysis.evidence;
  }
  return state.catalog ? state.catalog.nodes : [];
}

function activeEdges() {
  if (state.analysis && Array.isArray(state.analysis.edges)) {
    return state.analysis.edges;
  }
  return state.catalog ? state.catalog.edges : [];
}

function renderGraph() {
  const nodes = visibleNodes();
  const edges = visibleEdges(nodes);
  dom.graph.replaceChildren();
  dom.graphEmpty.hidden = nodes.length > 0;
  dom.graphMeta.textContent = `${nodes.length} 个节点 · ${edges.length} 条关系 · 按证据层级布局`;
  dom.evidenceCount.textContent = String(nodes.length);
  if (!nodes.length) {
    dom.graph.setAttribute("viewBox", "0 0 760 360");
    return;
  }

  const orderedTypes = state.catalog.evidence_types
    .map((item) => item.key)
    .filter((key) => nodes.some((node) => node.evidence_type === key));
  const viewWidth = Math.max(760, orderedTypes.length * 170 + 70);
  const viewHeight = 360;
  dom.graph.setAttribute("viewBox", `0 0 ${viewWidth} ${viewHeight}`);

  const defs = svgElement("defs");
  Object.entries(RELATION_STYLES).forEach(([relation, style]) => {
    const marker = svgElement("marker", {
      id: `arrow-${relation}`,
      viewBox: "0 0 8 8",
      refX: "7",
      refY: "4",
      markerWidth: "6",
      markerHeight: "6",
      orient: "auto-start-reverse",
    });
    marker.append(svgElement("path", { d: "M 0 0 L 8 4 L 0 8 z", fill: style.color }));
    defs.append(marker);
  });
  dom.graph.append(defs);

  const positions = new Map();
  orderedTypes.forEach((evidenceType, laneIndex) => {
    const laneNodes = nodes.filter((node) => node.evidence_type === evidenceType);
    const x = 35 + laneIndex * 170;
    const laneTitle = svgElement("text", {
      x: String(x + 67),
      y: "26",
      class: "node-type",
      fill: typeStyle(evidenceType).color,
      "text-anchor": "middle",
    });
    laneTitle.textContent = typeLabel(evidenceType);
    dom.graph.append(laneTitle);
    laneNodes.forEach((node, rowIndex) => {
      positions.set(node.id, { x, y: 48 + rowIndex * 128, width: 134, height: 78 });
    });
  });

  const selectedConnections = connectedNodeIds(state.selectedNodeId, edges);
  edges.forEach((edge) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) {
      return;
    }
    const relationStyle = relationStyleFor(edge.relation);
    const x1 = source.x + source.width;
    const y1 = source.y + source.height / 2;
    const x2 = target.x;
    const y2 = target.y + target.height / 2;
    const path = svgElement("path", {
      d: curvedPath(x1, y1, x2, y2),
      class: "graph-edge",
      stroke: relationStyle.color,
      "marker-end": `url(#arrow-${edge.relation})`,
      "data-source": edge.source,
      "data-target": edge.target,
    });
    if (
      state.selectedNodeId
      && edge.source !== state.selectedNodeId
      && edge.target !== state.selectedNodeId
    ) {
      path.classList.add("is-muted");
    }
    dom.graph.append(path);

    const label = svgElement("text", {
      x: String((x1 + x2) / 2),
      y: String((y1 + y2) / 2 - 5),
      class: "edge-label",
      "text-anchor": "middle",
    });
    label.textContent = relationStyle.label;
    dom.graph.append(label);
  });

  nodes.forEach((node) => {
    const position = positions.get(node.id);
    const style = typeStyle(node.evidence_type);
    const group = svgElement("g", {
      class: "graph-node",
      role: "button",
      tabindex: "0",
      "aria-label": node.citation,
      "data-node-id": node.id,
    });
    if (node.id === state.selectedNodeId) {
      group.classList.add("is-selected");
    } else if (state.selectedNodeId && !selectedConnections.has(node.id)) {
      group.classList.add("is-muted");
    }
    group.append(svgElement("rect", {
      x: String(position.x),
      y: String(position.y),
      width: String(position.width),
      height: String(position.height),
      rx: "5",
      fill: style.soft,
      stroke: style.color,
    }));

    const typeText = svgElement("text", {
      x: String(position.x + 10),
      y: String(position.y + 17),
      class: "node-type",
      fill: style.color,
    });
    typeText.textContent = typeLabel(node.evidence_type);
    group.append(typeText);

    const yearText = svgElement("text", {
      x: String(position.x + position.width - 9),
      y: String(position.y + 17),
      class: "node-year",
    });
    yearText.textContent = String(node.year);
    group.append(yearText);

    const titleLines = splitTitle(node.title, 12, 2);
    const titleText = svgElement("text", {
      x: String(position.x + 10),
      y: String(position.y + 40),
      class: "node-title",
    });
    titleLines.forEach((line, index) => {
      const tspan = svgElement("tspan", {
        x: String(position.x + 10),
        dy: index === 0 ? "0" : "16",
      });
      tspan.textContent = line;
      titleText.append(tspan);
    });
    group.append(titleText);

    group.addEventListener("click", () => selectEvidence(node.id));
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectEvidence(node.id);
      }
    });
    dom.graph.append(group);
  });
}

function renderEvidenceList() {
  const nodes = visibleNodes();
  dom.evidenceList.replaceChildren();
  dom.evidenceCount.textContent = String(nodes.length);
  if (!nodes.length) {
    dom.evidenceList.append(createParagraph("empty-state", "当前筛选条件下没有证据"));
    return;
  }
  nodes.forEach((node) => {
    const style = typeStyle(node.evidence_type);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "evidence-item";
    button.dataset.nodeId = node.id;
    button.setAttribute("aria-pressed", String(node.id === state.selectedNodeId));
    button.style.setProperty("--item-color", style.color);
    button.style.setProperty("--item-soft", style.soft);
    if (node.id === state.selectedNodeId) {
      button.classList.add("is-selected");
    }

    const bar = document.createElement("span");
    bar.className = "evidence-item-bar";
    const content = document.createElement("span");
    const meta = document.createElement("span");
    meta.className = "evidence-item-meta";
    const type = document.createElement("span");
    type.textContent = typeLabel(node.evidence_type);
    const year = document.createElement("span");
    year.textContent = `${node.year} · ${qualityLabel(node.quality)}`;
    meta.append(type, year);
    const title = document.createElement("span");
    title.className = "evidence-item-title";
    title.textContent = node.title;
    content.append(meta, title);
    button.append(bar, content);
    button.addEventListener("click", () => selectEvidence(node.id));
    dom.evidenceList.append(button);
  });
}

function selectEvidence(nodeId) {
  state.selectedNodeId = nodeId;
  renderGraph();
  renderEvidenceList();
  renderEvidenceDetail();
}

function renderEvidenceDetail() {
  const node = activeNodes().find((item) => item.id === state.selectedNodeId);
  dom.evidenceDetail.replaceChildren();
  const eyebrow = createParagraph("eyebrow", "SELECTED EVIDENCE");
  const heading = document.createElement("h2");
  heading.id = "detail-title";
  heading.className = "section-title";
  heading.textContent = "证据详情";
  dom.evidenceDetail.append(eyebrow, heading);
  if (!node) {
    dom.evidenceDetail.append(createParagraph("empty-state", "尚未选择证据"));
    return;
  }

  const citation = createParagraph("detail-citation", node.citation);
  const details = document.createElement("dl");
  details.className = "detail-grid";
  appendDetail(
    details,
    "发表时间",
    node.month ? `${node.year} 年 ${node.month} 月` : `${node.year} 年`,
  );
  appendDetail(details, "来源", node.source);
  appendDetail(details, "研究对象", node.population);
  appendDetail(details, "干预", node.intervention);
  appendDetail(details, "对照", node.comparator);
  appendDetail(details, "主要发现", node.main_findings);
  appendDetail(details, "证据质量", qualityLabel(node.quality));
  dom.evidenceDetail.append(citation, details);

  if (Array.isArray(node.safety_signals) && node.safety_signals.length) {
    const safetyTitle = createParagraph("field-label", "安全性与局限");
    const safetyList = document.createElement("div");
    safetyList.className = "safety-list";
    node.safety_signals.forEach((signal) => {
      const tag = document.createElement("span");
      tag.className = "safety-tag";
      tag.textContent = signal;
      safetyList.append(tag);
    });
    dom.evidenceDetail.append(safetyTitle, safetyList);
  }
}

function appendDetail(container, term, description) {
  const wrapper = document.createElement("div");
  wrapper.className = "detail-row";
  const dt = document.createElement("dt");
  dt.textContent = term;
  const dd = document.createElement("dd");
  dd.textContent = description || "未提供";
  wrapper.append(dt, dd);
  container.append(wrapper);
}

function renderInitialSummary() {
  if (!state.catalog) {
    return;
  }
  dom.analysisChips.replaceChildren(
    createChip(`${state.catalog.nodes.length} 条本地证据`, "neutral"),
    createChip("证据金字塔已加载", "good"),
    createChip("等待时间关系分析", "update"),
  );
  dom.decisionSummary.textContent = "证据图已加载。生成后将合成证据一致性、时间更新与安全性判断。";
}

function renderAnalysis(result) {
  const nodes = visibleNodes();
  if (!nodes.some((node) => node.id === state.selectedNodeId)) {
    state.selectedNodeId = nodes[0] ? nodes[0].id : null;
  }
  renderGraph();
  renderEvidenceList();
  renderEvidenceDetail();
  dom.analysisChips.replaceChildren(
    createChip(`${result.summary.evidence_count} 条证据`, "neutral"),
    createChip(result.summary.consistency, "good"),
    createChip(`${result.summary.update_count} 条时间更新`, "update"),
  );
  dom.decisionSummary.textContent = result.summary.conclusion;
  renderReasoningSteps(result.reasoning_steps);
  dom.answerGrounding.classList.remove("is-grounded", "is-local");
  if (result.model_used) {
    renderSafeMarkdown(dom.finalAnswer, result.answer_markdown);
    dom.reportModel.textContent = `模型：${result.model_name}`;
    dom.answerGrounding.textContent = "已基于证据图生成";
    dom.answerGrounding.classList.add("is-grounded");
  } else {
    dom.finalAnswer.replaceChildren(
      createParagraph("empty-state", "大模型综合回答未生成，本地循证分析已保留。"),
    );
    dom.reportModel.textContent = `本地分析 · ${result.model_name}`;
    dom.answerGrounding.textContent = "本地分析已保留";
    dom.answerGrounding.classList.add("is-local");
  }
}

function invalidateReport(reason) {
  if (!state.analysis) {
    return false;
  }
  state.analysis = null;
  dom.reasoningSteps.replaceChildren(
    createParagraph("empty-state", "当前报告已失效"),
  );
  dom.finalAnswer.replaceChildren(
    createParagraph("empty-state", "当前综合回答已失效"),
  );
  dom.reportModel.textContent = "报告已失效";
  dom.answerGrounding.textContent = "报告已失效";
  dom.answerGrounding.classList.remove("is-grounded", "is-local");
  renderInitialSummary();
  dom.decisionSummary.textContent = reason;
  dom.progress.textContent = reason;
  return true;
}

function renderReasoningSteps(steps) {
  dom.reasoningSteps.replaceChildren();
  steps.forEach((step, index) => {
    const details = document.createElement("details");
    details.className = "reasoning-step";
    details.open = index === 0 || index === steps.length - 1;
    const summary = document.createElement("summary");
    summary.textContent = step.title;
    const body = document.createElement("div");
    body.className = "reasoning-step-body";
    body.append(document.createTextNode(step.body));

    if (Array.isArray(step.node_ids) && step.node_ids.length) {
      const references = document.createElement("div");
      references.className = "reasoning-references";
      step.node_ids.forEach((nodeId) => {
        const node = activeNodes().find((item) => item.id === nodeId);
        if (!node) {
          return;
        }
        const button = document.createElement("button");
        button.type = "button";
        button.className = "reference-button";
        button.textContent = node.id;
        button.title = node.citation;
        button.addEventListener("click", () => selectEvidence(node.id));
        references.append(button);
      });
      body.append(references);
    }
    details.append(summary, body);
    dom.reasoningSteps.append(details);
  });
}

function renderSafeMarkdown(container, markdown) {
  container.replaceChildren();
  const lines = String(markdown || "").replace(/\r/g, "").split("\n");
  let list = null;
  lines.forEach((rawLine) => {
    const line = rawLine.trim();
    if (!line) {
      list = null;
      return;
    }
    if (line.startsWith("### ") || line.startsWith("## ")) {
      list = null;
      const heading = document.createElement(line.startsWith("### ") ? "h3" : "h2");
      appendInlineMarkdown(heading, line.replace(/^###?\s+/, ""));
      container.append(heading);
      return;
    }
    if (line.startsWith("- ")) {
      if (!list) {
        list = document.createElement("ul");
        container.append(list);
      }
      const item = document.createElement("li");
      appendInlineMarkdown(item, line.slice(2));
      list.append(item);
      return;
    }
    list = null;
    const paragraph = document.createElement("p");
    appendInlineMarkdown(paragraph, line);
    container.append(paragraph);
  });
}

function appendInlineMarkdown(container, text) {
  const parts = String(text).split(/(\*\*[^*]+\*\*)/g);
  parts.forEach((part) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      const strong = document.createElement("strong");
      strong.textContent = part.slice(2, -2);
      container.append(strong);
    } else {
      container.append(document.createTextNode(part));
    }
  });
}

async function submitAnalysis() {
  if (state.busy) {
    return;
  }
  const question = dom.question.value.trim();
  if (!question) {
    dom.question.focus();
    showToast("请输入临床问题。");
    return;
  }
  if (!state.visibleTypes.size) {
    showToast("至少保留一种证据类型。 ");
    return;
  }
  const model = readModelConfig();
  if (!model) {
    return;
  }

  setBusy(true, "正在检索证据、检查时间关系并调用大模型...");
  try {
    const result = await apiRequest("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        evidence_types: Array.from(state.visibleTypes),
        model,
      }),
    });
    state.analysis = result;
    renderAnalysis(result);
    persistModelSettings();
    if (result.model_used) {
      setModelStatus("connected", `模型已连接 · ${result.model_name}`);
      dom.progress.textContent = "循证分析已完成";
    } else {
      setModelStatus("error", "模型请求失败 · 本地分析已保留");
      dom.progress.textContent = "本地循证分析已完成，模型回答未生成";
      showToast(result.model_error.message);
    }
    document.querySelector(".report-section").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setModelStatus("error", "模型请求失败");
    dom.progress.textContent = "本地证据图保持可用";
    showToast(error.message);
  } finally {
    setBusy(false);
  }
}

async function testModelConnection() {
  if (state.busy) {
    return;
  }
  const model = readModelConfig();
  if (!model) {
    return;
  }
  setBusy(true, "正在测试模型连接...");
  try {
    const result = await apiRequest("/api/model/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    });
    setModelStatus("connected", `模型已连接 · ${result.model_name}`);
    dom.progress.textContent = "模型连接测试成功";
    persistModelSettings();
  } catch (error) {
    setModelStatus("error", "模型连接失败");
    dom.progress.textContent = "模型连接测试失败";
    showToast(error.message);
  } finally {
    setBusy(false);
  }
}

function readModelConfig() {
  const model = {
    api_key: dom.apiKey.value.trim(),
    base_url: dom.baseUrl.value.trim(),
    model_name: dom.modelName.value.trim(),
  };
  if (!model.api_key || !model.base_url || !model.model_name) {
    dom.modelSettings.open = true;
    const missingField = !model.api_key ? dom.apiKey : (!model.base_url ? dom.baseUrl : dom.modelName);
    missingField.focus();
    showToast("请填写 API Key、Base URL 和模型名称。 ");
    return null;
  }
  return model;
}

function setBusy(busy, message) {
  state.busy = busy;
  dom.generateButton.disabled = busy;
  dom.testModelButton.disabled = busy;
  if (message) {
    dom.progress.textContent = message;
  }
}

function setModelStatus(status, message) {
  dom.modelStatus.dataset.state = status;
  dom.modelStatus.lastChild.textContent = message;
}

async function apiRequest(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    throw new Error("无法连接本地 Graph RAG 服务。", { cause: error });
  }
  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error("服务返回了无法解析的数据。", { cause: error });
  }
  if (!response.ok) {
    throw new Error(payload.error && payload.error.message ? payload.error.message : "请求失败。 ");
  }
  return payload;
}

function showToast(message) {
  window.clearTimeout(state.toastTimer);
  dom.toast.textContent = message;
  dom.toast.hidden = false;
  state.toastTimer = window.setTimeout(() => {
    dom.toast.hidden = true;
  }, 4800);
}

function createChip(text, variant) {
  const chip = document.createElement("span");
  chip.className = `analysis-chip ${variant}`;
  chip.textContent = text;
  return chip;
}

function createParagraph(className, text) {
  const paragraph = document.createElement("p");
  paragraph.className = className;
  paragraph.textContent = text;
  return paragraph;
}

function svgElement(tagName, attributes = {}) {
  const element = document.createElementNS(SVG_NS, tagName);
  Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, value));
  return element;
}

function curvedPath(x1, y1, x2, y2) {
  const direction = x2 >= x1 ? 1 : -1;
  const bend = Math.max(34, Math.abs(x2 - x1) * 0.42) * direction;
  return `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`;
}

function connectedNodeIds(selectedId, edges) {
  const ids = new Set();
  if (!selectedId) {
    return ids;
  }
  ids.add(selectedId);
  edges.forEach((edge) => {
    if (edge.source === selectedId) {
      ids.add(edge.target);
    }
    if (edge.target === selectedId) {
      ids.add(edge.source);
    }
  });
  return ids;
}

function splitTitle(title, maxLength, maxLines) {
  const characters = Array.from(String(title));
  const lines = [];
  for (let index = 0; index < characters.length && lines.length < maxLines; index += maxLength) {
    let line = characters.slice(index, index + maxLength).join("");
    if (index + maxLength < characters.length && lines.length === maxLines - 1) {
      line = `${line.slice(0, Math.max(1, maxLength - 1))}…`;
    }
    lines.push(line);
  }
  return lines;
}

function typeStyle(evidenceType) {
  return TYPE_STYLES[evidenceType] || { color: "#627069", soft: "#f0f3f1" };
}

function relationStyleFor(relation) {
  return RELATION_STYLES[relation] || { color: "#627069", label: relation };
}

function typeLabel(evidenceType) {
  if (!state.catalog) {
    return evidenceType;
  }
  const match = state.catalog.evidence_types.find((item) => item.key === evidenceType);
  return match ? match.label : evidenceType;
}

function qualityLabel(quality) {
  return {
    high: "高质量",
    moderate: "中等质量",
    low: "低质量",
    very_low: "极低质量",
  }[quality] || quality;
}
