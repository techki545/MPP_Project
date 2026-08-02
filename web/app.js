"use strict";

const TYPE_LABELS = {
  guideline: "指南",
  systematic_review: "系统综述",
  randomized_controlled_trial: "随机对照试验（RCT）",
  observational_study: "观察性研究",
  narrative_review: "叙述性综述",
  case_report: "病例报告",
  unknown: "未分类",
};

const TYPE_ORDER = [
  "guideline",
  "systematic_review",
  "randomized_controlled_trial",
  "observational_study",
  "narrative_review",
  "case_report",
];

const RELATION_LABELS = {
  supports: "支持",
  updates: "更新",
  supplements: "补充",
  conflicts: "冲突",
  cautions: "警示",
};

const STAGE_LABELS = {
  metadata: "导入元数据",
  pdf_inventory: "盘点 PDF",
  pdf_match: "匹配文献与 PDF",
  parse: "解析 PDF",
  chunk: "语义分块",
  lexical: "建立关键词索引",
  embedding: "生成向量",
  vector: "写入向量库",
};

const state = {
  catalog: null,
  kbStatus: null,
  modelStatus: null,
  result: null,
  sources: new Map(),
  activeSource: null,
  currentJobId: null,
  pollTimer: null,
};

const dom = {};

document.addEventListener("DOMContentLoaded", initialize);

async function initialize() {
  cacheDom();
  bindEvents();
  updateQuestionCount();
  setDefaultYears();
  try {
    const [health, kbStatus, modelStatus, catalog] = await Promise.all([
      fetchJson("/api/health"),
      fetchJson("/api/kb/status"),
      fetchJson("/api/model/status"),
      fetchJson("/api/evidence"),
    ]);
    state.kbStatus = kbStatus;
    state.modelStatus = modelStatus;
    state.catalog = catalog;
    renderSystemStatus(health, kbStatus, modelStatus);
    renderCorpusMetrics(kbStatus);
    renderEvidenceFilters(catalog.evidence_types || []);
    if (!dom.question.value.trim()) {
      dom.question.value = catalog.default_question || "";
      updateQuestionCount();
    }
    const demoMode = kbStatus.status !== "ready";
    dom.dataSourceLabel.textContent = demoMode ? "示例证据模式" : "真实本地语料";
    dom.dataSourceLabel.dataset.mode = demoMode ? "demo" : "knowledge_base";
    if (demoMode && Array.isArray(catalog.nodes)) {
      renderGraph({ nodes: catalog.nodes, edges: catalog.edges || [] });
    }
  } catch (error) {
    renderStartupFailure(error);
  }
}

function cacheDom() {
  Object.assign(dom, {
    question: document.querySelector("#clinical-question"),
    questionCount: document.querySelector("#question-count"),
    runQuery: document.querySelector("#run-query"),
    queryStatus: document.querySelector("#query-status"),
    filters: document.querySelector("#evidence-filters"),
    yearFrom: document.querySelector("#year-from"),
    yearTo: document.querySelector("#year-to"),
    fulltextOnly: document.querySelector("#fulltext-only"),
    kbStatus: document.querySelector("#kb-status"),
    modelStatus: document.querySelector("#model-status"),
    dataSourceLabel: document.querySelector("#data-source-label"),
    stats: {
      metadata_records: document.querySelector("#stat-metadata"),
      pdf_files: document.querySelector("#stat-pdf"),
      matched_pdf_files: document.querySelector("#stat-matched"),
      parsed_pdf_files: document.querySelector("#stat-parsed"),
      chunks: document.querySelector("#stat-chunks"),
      embedded: document.querySelector("#stat-embedded"),
    },
    buildKb: document.querySelector("#build-kb"),
    pauseKb: document.querySelector("#pause-kb"),
    continueKb: document.querySelector("#continue-kb"),
    retryKb: document.querySelector("#retry-kb"),
    retryStage: document.querySelector("#retry-stage"),
    jobProgress: document.querySelector("#job-progress"),
    jobStage: document.querySelector("#job-stage"),
    jobState: document.querySelector("#job-state"),
    reasoningSteps: document.querySelector("#reasoning-steps"),
    finalAnswer: document.querySelector("#final-answer"),
    retrievalMode: document.querySelector("#retrieval-mode"),
    reportModel: document.querySelector("#report-model"),
    groundingStatus: document.querySelector("#grounding-status"),
    graph: document.querySelector("#evidence-graph"),
    graphEmpty: document.querySelector("#graph-empty"),
    graphMeta: document.querySelector("#graph-meta"),
    sourceList: document.querySelector("#source-list"),
    sourceCount: document.querySelector("#source-count"),
    sourceDetail: document.querySelector("#source-detail"),
    toast: document.querySelector("#toast"),
  });
}

function bindEvents() {
  dom.question.addEventListener("input", updateQuestionCount);
  dom.runQuery.addEventListener("click", runQuery);
  dom.buildKb.addEventListener("click", () => startBuild(false));
  dom.continueKb.addEventListener("click", () => {
    if (window.confirm("继续向量化将调用服务器配置的嵌入模型，是否确认？")) {
      startBuild(true);
    }
  });
  dom.pauseKb.addEventListener("click", pauseBuild);
  dom.retryKb.addEventListener("click", retryBuild);
  dom.sourceList.addEventListener("click", handleSourceClick);
  dom.reasoningSteps.addEventListener("click", handleSourceClick);
  dom.finalAnswer.addEventListener("click", handleSourceClick);
}

function setDefaultYears() {
  dom.yearFrom.value = "2015";
  dom.yearTo.value = String(new Date().getFullYear());
}

function updateQuestionCount() {
  dom.questionCount.textContent = `${dom.question.value.length} / 2000`;
}

function renderSystemStatus(health, kbStatus, modelStatus) {
  const ready = kbStatus.status === "ready";
  setStatusPill(
    dom.kbStatus,
    ready ? "ready" : "warning",
    ready ? "知识库可查询" : "知识库未构建"
  );
  const configured = Boolean(modelStatus.configured);
  setStatusPill(
    dom.modelStatus,
    configured ? "ready" : "warning",
    configured ? `模型 ${modelStatus.chat_model || "已配置"}` : "模型未配置"
  );
  if (health.status === "degraded" && ready) {
    setStatusPill(dom.kbStatus, "warning", "知识库降级可用");
  }
}

function setStatusPill(element, status, label) {
  element.dataset.state = status;
  const labelNode = element.querySelector("span");
  if (labelNode) labelNode.textContent = label;
}

function renderCorpusMetrics(status) {
  const numberFormat = new Intl.NumberFormat("zh-CN");
  Object.entries(dom.stats).forEach(([key, element]) => {
    const value = Number(status[key] || 0);
    element.textContent = numberFormat.format(Number.isFinite(value) ? value : 0);
  });
}

function renderEvidenceFilters(items) {
  const catalogByType = new Map(items.map((item) => [item.key, item]));
  dom.filters.replaceChildren();
  TYPE_ORDER.forEach((type) => {
    const catalogItem = catalogByType.get(type) || {};
    const label = document.createElement("label");
    label.className = "evidence-filter";
    label.dataset.type = type;
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = type;
    checkbox.checked = true;
    const swatch = document.createElement("i");
    swatch.className = "type-swatch";
    swatch.setAttribute("aria-hidden", "true");
    const text = document.createElement("span");
    text.textContent = catalogItem.label || TYPE_LABELS[type];
    label.append(checkbox, swatch, text);
    dom.filters.append(label);
  });
}

async function runQuery() {
  const question = dom.question.value.trim();
  if (!question) {
    setQueryStatus("请输入临床问题。", "error");
    dom.question.focus();
    return;
  }
  const yearFrom = optionalInteger(dom.yearFrom.value);
  const yearTo = optionalInteger(dom.yearTo.value);
  if (yearFrom !== null && yearTo !== null && yearFrom > yearTo) {
    setQueryStatus("起始年份不能晚于结束年份。", "error");
    return;
  }
  const evidenceTypes = Array.from(
    dom.filters.querySelectorAll("input[type='checkbox']:checked")
  ).map((input) => input.value);
  const payload = {
    question,
    evidence_types: evidenceTypes,
    year_from: yearFrom,
    year_to: yearTo,
    fulltext_only: dom.fulltextOnly.checked,
  };

  setQueryBusy(true);
  setQueryStatus("正在执行四路检索、证据重排与关系构建…", "loading");
  try {
    const result = await fetchJson("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.result = result;
    indexSources(result.sources || []);
    renderReport(result);
    renderSources(result.sources || []);
    renderGraph(result.graph || { nodes: [], edges: [] });
    const warning = result.warning ? ` ${result.warning}` : "";
    setQueryStatus(`分析完成，共返回 ${result.sources?.length || 0} 项来源。${warning}`, result.warning ? "warning" : "ready");
    dom.dataSourceLabel.textContent = result.data_source === "demo" ? "示例证据模式" : "真实本地语料";
    dom.dataSourceLabel.dataset.mode = result.data_source || "knowledge_base";
    document.querySelector("#report-workspace").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setQueryStatus(error.message, "error");
    showToast(error.message);
  } finally {
    setQueryBusy(false);
  }
}

function setQueryBusy(busy) {
  dom.runQuery.disabled = busy;
  const label = dom.runQuery.querySelector("span");
  label.textContent = busy ? "正在分析" : "开始循证分析";
}

function setQueryStatus(message, status) {
  dom.queryStatus.textContent = message;
  dom.queryStatus.dataset.state = status;
}

function renderReport(result) {
  dom.retrievalMode.textContent = retrievalModeLabel(result.mode, result.degraded_reason);
  dom.reportModel.textContent = result.model_used
    ? `模型：${result.model_name || "已配置"}`
    : "确定性回退";
  dom.groundingStatus.textContent = result.model_used ? "引用已校验" : "模型结果未采用";
  dom.groundingStatus.dataset.state = result.model_used ? "ready" : "warning";

  dom.reasoningSteps.replaceChildren();
  const steps = Array.isArray(result.reasoning_steps) ? result.reasoning_steps : [];
  if (!steps.length) {
    dom.reasoningSteps.append(emptyState("本次没有可展示的循证步骤。"));
  } else {
    steps.forEach((step, index) => {
      const details = document.createElement("details");
      details.className = "reasoning-step";
      details.open = index === 0;
      const summary = document.createElement("summary");
      summary.textContent = step.title || `步骤 ${index + 1}`;
      const body = document.createElement("div");
      body.className = "reasoning-body";
      appendInlineContent(body, String(step.body || ""));
      details.append(summary, body);
      dom.reasoningSteps.append(details);
    });
  }
  renderMarkdown(dom.finalAnswer, result.answer_markdown || "暂无综合回答。\n");
}

function retrievalModeLabel(mode, reason) {
  const labels = {
    hybrid: "混合检索",
    keyword: "关键词降级检索",
    vector: "向量检索",
    demo: "示例图谱",
  };
  const label = labels[mode] || String(mode || "未知模式");
  return reason ? `${label} · ${reason}` : label;
}

function renderMarkdown(container, markdown) {
  container.replaceChildren();
  const lines = String(markdown).replace(/\r/g, "").split("\n");
  let list = null;
  lines.forEach((rawLine) => {
    const line = rawLine.trim();
    if (!line) {
      list = null;
      return;
    }
    if (line.startsWith("### ")) {
      const heading = document.createElement("h3");
      appendInlineContent(heading, line.slice(4));
      container.append(heading);
      list = null;
      return;
    }
    if (line.startsWith("## ")) {
      const heading = document.createElement("h2");
      appendInlineContent(heading, line.slice(3));
      container.append(heading);
      list = null;
      return;
    }
    if (/^[-*]\s+/.test(line)) {
      if (!list) {
        list = document.createElement("ul");
        container.append(list);
      }
      const item = document.createElement("li");
      appendInlineContent(item, line.replace(/^[-*]\s+/, ""));
      list.append(item);
      return;
    }
    list = null;
    const paragraph = document.createElement("p");
    appendInlineContent(paragraph, line);
    container.append(paragraph);
  });
}

function appendInlineContent(container, text) {
  const tokenPattern = /(\[\d+\]|\*\*[^*]+\*\*)/g;
  let cursor = 0;
  for (const match of text.matchAll(tokenPattern)) {
    if (match.index > cursor) {
      container.append(document.createTextNode(text.slice(cursor, match.index)));
    }
    const token = match[0];
    if (/^\[\d+\]$/.test(token)) {
      const number = Number(token.slice(1, -1));
      container.append(citationButton(number));
    } else {
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      container.append(strong);
    }
    cursor = match.index + token.length;
  }
  if (cursor < text.length) {
    container.append(document.createTextNode(text.slice(cursor)));
  }
}

function citationButton(number) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "citation-button";
  button.dataset.sourceNumber = String(number);
  button.setAttribute("aria-label", `查看来源 ${number}`);
  button.textContent = `[${number}]`;
  return button;
}

function indexSources(sources) {
  state.sources.clear();
  sources.forEach((source) => {
    state.sources.set(Number(source.source_number), source);
  });
}

function renderSources(sources) {
  dom.sourceList.replaceChildren();
  dom.sourceCount.textContent = String(sources.length);
  if (!sources.length) {
    dom.sourceList.append(emptyState("本次检索没有返回来源。"));
    renderEmptySourceDetail();
    return;
  }
  sources.forEach((source) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "source-item";
    button.dataset.sourceNumber = String(source.source_number);
    button.setAttribute("aria-label", `查看来源 ${source.source_number}：${source.title}`);
    const number = document.createElement("span");
    number.className = "source-number";
    number.textContent = String(source.source_number);
    const copy = document.createElement("span");
    copy.className = "source-copy";
    const title = document.createElement("span");
    title.className = "source-title";
    title.textContent = source.title || source.document_id;
    const meta = document.createElement("span");
    meta.className = "source-meta";
    meta.append(
      textSpan(TYPE_LABELS[source.evidence_type] || source.evidence_type || "未分类"),
      textSpan(source.year ? String(source.year) : "年份未知"),
      textSpan(source.fulltext ? "全文命中" : "元数据命中")
    );
    copy.append(title, meta);
    button.append(number, copy);
    dom.sourceList.append(button);
  });
}

function textSpan(text) {
  const span = document.createElement("span");
  span.textContent = text;
  return span;
}

function handleSourceClick(event) {
  const trigger = event.target.closest("[data-source-number]");
  if (!trigger) return;
  const sourceNumber = Number(trigger.dataset.sourceNumber);
  if (Number.isInteger(sourceNumber)) selectSource(sourceNumber);
}

async function selectSource(sourceNumber) {
  const source = state.sources.get(sourceNumber);
  if (!source) {
    showToast(`来源 ${sourceNumber} 不在本次结果中。`);
    return;
  }
  state.activeSource = sourceNumber;
  document.querySelectorAll(".source-item").forEach((item) => {
    item.setAttribute(
      "aria-current",
      item.dataset.sourceNumber === String(sourceNumber) ? "true" : "false"
    );
  });
  renderSourceLoading(source);
  try {
    const detail = await fetchJson(`/api/documents/${encodeURIComponent(source.document_id)}`);
    renderSourceDetail(detail, sourceNumber);
    dom.sourceDetail.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    renderSourceError(source, error.message);
  }
}

function renderSourceLoading(source) {
  dom.sourceDetail.replaceChildren();
  const heading = document.createElement("h3");
  heading.id = "source-detail-title";
  heading.textContent = source.title || "文献详情";
  dom.sourceDetail.append(heading, emptyState("正在加载文献详情…"));
}

function renderSourceDetail(detail, sourceNumber) {
  dom.sourceDetail.replaceChildren();
  const heading = document.createElement("h3");
  heading.id = "source-detail-title";
  heading.textContent = detail.title || `来源 ${sourceNumber}`;
  const bibliography = document.createElement("dl");
  bibliography.className = "bibliography";
  addBibliographyRow(bibliography, "作者", formatAuthors(detail.authors));
  addBibliographyRow(bibliography, "期刊 / 年份", [detail.journal, detail.year].filter(Boolean).join(" · ") || "未知");
  addBibliographyRow(bibliography, "DOI", detail.doi || "未提供");
  addBibliographyRow(bibliography, "证据类型", TYPE_LABELS[detail.evidence_type] || detail.evidence_type || "未分类");
  addBibliographyRow(bibliography, "质量", qualityLabel(detail.quality));
  addBibliographyRow(bibliography, "全文状态", detail.has_fulltext ? `可用 · ${detail.fulltext_status || "已解析"}` : "仅元数据");
  dom.sourceDetail.append(heading, bibliography);

  if (detail.abstract) {
    const abstractHeading = document.createElement("h4");
    abstractHeading.className = "snippet-heading";
    abstractHeading.textContent = "摘要";
    const abstract = document.createElement("p");
    abstract.className = "empty-state";
    abstract.textContent = detail.abstract;
    dom.sourceDetail.append(abstractHeading, abstract);
  }

  const snippetHeading = document.createElement("h4");
  snippetHeading.className = "snippet-heading";
  snippetHeading.textContent = "命中的正文片段";
  const snippetList = document.createElement("div");
  snippetList.className = "snippet-list";
  const snippets = Array.isArray(detail.matched_snippets) ? detail.matched_snippets : [];
  if (!snippets.length) {
    snippetList.append(emptyState("该文献当前没有可展示的全文片段。"));
  } else {
    snippets.forEach((snippet) => {
      const item = document.createElement("article");
      item.className = "snippet";
      const text = document.createElement("p");
      text.textContent = snippet.text || "";
      const meta = document.createElement("small");
      meta.textContent = `${snippet.section || "未标注章节"} · 第 ${pageRange(snippet.page_start, snippet.page_end)} 页${snippet.is_ocr ? " · OCR" : ""}`;
      item.append(text, meta);
      snippetList.append(item);
    });
  }
  dom.sourceDetail.append(snippetHeading, snippetList);

  if (detail.pdf_available) {
    const link = document.createElement("a");
    link.className = "pdf-link";
    link.href = `/api/documents/${encodeURIComponent(detail.document_id)}/pdf`;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = "在浏览器中查看 PDF";
    dom.sourceDetail.append(link);
  }
}

function renderSourceError(source, message) {
  dom.sourceDetail.replaceChildren();
  const heading = document.createElement("h3");
  heading.id = "source-detail-title";
  heading.textContent = source.title || "文献详情";
  dom.sourceDetail.append(heading, emptyState(message));
}

function renderEmptySourceDetail() {
  dom.sourceDetail.replaceChildren();
  const heading = document.createElement("h3");
  heading.id = "source-detail-title";
  heading.textContent = "文献详情";
  dom.sourceDetail.append(heading, emptyState("选择来源后查看书目信息和命中正文。"));
}

function addBibliographyRow(list, label, value) {
  const term = document.createElement("dt");
  term.textContent = label;
  const description = document.createElement("dd");
  description.textContent = String(value || "未知");
  list.append(term, description);
}

function formatAuthors(authors) {
  return Array.isArray(authors) && authors.length ? authors.join("，") : "未提供";
}

function qualityLabel(value) {
  return {
    high: "高",
    moderate: "中等",
    low: "低",
    very_low: "极低",
    unknown: "未知",
  }[value] || value || "未知";
}

function pageRange(start, end) {
  if (!start && !end) return "未知";
  return start === end || !end ? String(start) : `${start}–${end}`;
}

function renderGraph(graph) {
  const rawNodes = Array.isArray(graph.nodes) ? graph.nodes : [];
  const rawEdges = Array.isArray(graph.edges) ? graph.edges : [];
  const nodes = rawNodes.slice(0, 24).map(normalizeGraphNode);
  const visibleIds = new Set(nodes.map((node) => node.id));
  const edges = rawEdges
    .map((edge) => ({ ...edge, relation: edge.relation === "confirms" ? "supports" : edge.relation }))
    .filter((edge) => visibleIds.has(String(edge.source)) && visibleIds.has(String(edge.target)));
  dom.graph.replaceChildren();
  dom.graphMeta.textContent = `${nodes.length} 个节点 · ${edges.length} 条关系`;
  if (!nodes.length) {
    dom.graphEmpty.hidden = false;
    return;
  }
  dom.graphEmpty.hidden = true;
  const layerCounts = new Map();
  nodes.forEach((node) => layerCounts.set(node.type, (layerCounts.get(node.type) || 0) + 1));
  const widestRow = Math.min(5, Math.max(...layerCounts.values()));
  const compactViewport = dom.graph.clientWidth < 500;
  const viewWidth = widestRow <= 2 ? (compactViewport ? 480 : 640) : widestRow === 3 ? 760 : 1000;
  const positions = graphLayout(nodes, viewWidth);
  const height = Math.max(420, Math.max(...Array.from(positions.values()).map((value) => value.y)) + 90);
  dom.graph.setAttribute("viewBox", `0 0 ${viewWidth} ${height}`);
  dom.graph.setAttribute("preserveAspectRatio", "xMidYMid meet");

  edges.forEach((edge) => {
    const source = positions.get(String(edge.source));
    const target = positions.get(String(edge.target));
    if (!source || !target) return;
    const line = svgElement("line");
    line.classList.add("graph-edge");
    line.dataset.relation = edge.relation || "supports";
    line.setAttribute("x1", source.x);
    line.setAttribute("y1", source.y);
    line.setAttribute("x2", target.x);
    line.setAttribute("y2", target.y);
    const title = svgElement("title");
    title.textContent = `${RELATION_LABELS[edge.relation] || edge.relation || "关系"}：${edge.rationale || ""}`;
    line.append(title);
    dom.graph.append(line);
  });

  nodes.forEach((node) => {
    const position = positions.get(node.id);
    const group = svgElement("g");
    group.classList.add("graph-node");
    group.dataset.nodeId = node.id;
    group.dataset.nodeType = node.type;
    group.setAttribute("transform", `translate(${position.x - 76} ${position.y - 27})`);
    const rect = svgElement("rect");
    rect.setAttribute("width", "152");
    rect.setAttribute("height", "54");
    const label = svgElement("text");
    label.setAttribute("x", "76");
    label.setAttribute("y", "22");
    label.setAttribute("text-anchor", "middle");
    wrapSvgLabel(label, node.label);
    const title = svgElement("title");
    title.textContent = node.label;
    group.append(rect, label, title);
    dom.graph.append(group);
  });
}

function normalizeGraphNode(node) {
  const id = String(node.node_id ?? node.id ?? "unknown");
  const payload = node.payload || {};
  return {
    id,
    type: String(node.node_type || payload.node_type || (id === "question" ? "question" : "document")),
    label: String(node.label || node.title || payload.title || id),
  };
}

function graphLayout(nodes, viewWidth) {
  const layers = ["question", "document", "claim", "outcome"];
  const grouped = new Map(layers.map((layer) => [layer, []]));
  nodes.forEach((node) => {
    const layer = grouped.has(node.type) ? node.type : "document";
    grouped.get(layer).push(node);
  });
  const positions = new Map();
  let y = 55;
  layers.forEach((layer) => {
    const items = grouped.get(layer);
    if (!items.length) return;
    for (let start = 0; start < items.length; start += 5) {
      const row = items.slice(start, start + 5);
      const spacing = (viewWidth - 120) / row.length;
      row.forEach((node, index) => {
        positions.set(node.id, { x: 60 + spacing * (index + 0.5), y });
      });
      y += 92;
    }
  });
  return positions;
}

function wrapSvgLabel(textNode, label) {
  const clean = String(label).replace(/\s+/g, " ").trim();
  const first = clean.slice(0, 18);
  const second = clean.length > 18 ? `${clean.slice(18, 34)}${clean.length > 34 ? "…" : ""}` : "";
  const firstLine = svgElement("tspan");
  firstLine.setAttribute("x", "76");
  firstLine.textContent = first;
  textNode.append(firstLine);
  if (second) {
    const secondLine = svgElement("tspan");
    secondLine.setAttribute("x", "76");
    secondLine.setAttribute("dy", "16");
    secondLine.textContent = second;
    textNode.append(secondLine);
  }
}

function svgElement(name) {
  return document.createElementNS("http://www.w3.org/2000/svg", name);
}

async function startBuild(confirmEmbeddingCost) {
  setJobControls(true);
  try {
    const job = await fetchJson("/api/kb/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm_embedding_cost: confirmEmbeddingCost }),
    });
    beginJobPolling(job);
    showToast(confirmEmbeddingCost ? "已开始本地构建与向量化。" : "已开始免费本地构建阶段。");
  } catch (error) {
    setJobControls(false);
    showToast(error.message);
  }
}

async function pauseBuild() {
  if (!state.currentJobId) return;
  try {
    const job = await fetchJson("/api/kb/pause", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: state.currentJobId }),
    });
    renderJob(job);
  } catch (error) {
    showToast(error.message);
  }
}

async function retryBuild() {
  const stage = dom.retryStage.value;
  const paidStage = stage === "embedding" || stage === "vector";
  const confirmed = !paidStage || window.confirm("重试该阶段可能调用嵌入模型，是否确认？");
  if (!confirmed) return;
  setJobControls(true);
  try {
    const job = await fetchJson("/api/kb/retry", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stage, confirm_embedding_cost: paidStage }),
    });
    beginJobPolling(job);
  } catch (error) {
    setJobControls(false);
    showToast(error.message);
  }
}

function beginJobPolling(job) {
  state.currentJobId = job.job_id;
  renderJob(job);
  if (state.pollTimer) clearTimeout(state.pollTimer);
  pollJob();
}

async function pollJob() {
  if (!state.currentJobId) return;
  try {
    const job = await fetchJson(`/api/jobs/${encodeURIComponent(state.currentJobId)}`);
    renderJob(job);
    if (["paused", "completed", "failed"].includes(job.state)) {
      state.currentJobId = null;
      setJobControls(false);
      await refreshKbStatus();
      return;
    }
  } catch (error) {
    showToast(error.message);
    state.currentJobId = null;
    setJobControls(false);
    return;
  }
  state.pollTimer = window.setTimeout(pollJob, 1100);
}

function renderJob(job) {
  const progress = job.progress || {};
  const stage = progress.stage || progress.current_stage || "";
  dom.jobStage.textContent = stage ? STAGE_LABELS[stage] || stage : jobStateLabel(job.state);
  dom.jobState.textContent = jobStateLabel(job.state);
  const terminal = ["paused", "completed", "failed"].includes(job.state);
  if (terminal) {
    dom.jobProgress.value = job.state === "completed" ? 100 : 0;
  } else {
    dom.jobProgress.removeAttribute("value");
  }
  dom.pauseKb.disabled = !["queued", "running", "pausing"].includes(job.state);
}

function jobStateLabel(value) {
  return {
    queued: "等待执行",
    running: "正在构建",
    pausing: "正在安全暂停",
    paused: "已暂停",
    completed: "已完成",
    failed: "构建失败",
  }[value] || "空闲";
}

function setJobControls(active) {
  dom.buildKb.disabled = active;
  dom.continueKb.disabled = active;
  dom.retryKb.disabled = active;
  dom.retryStage.disabled = active;
  dom.pauseKb.disabled = !active;
}

async function refreshKbStatus() {
  try {
    const status = await fetchJson("/api/kb/status");
    state.kbStatus = status;
    renderCorpusMetrics(status);
    setStatusPill(
      dom.kbStatus,
      status.status === "ready" ? "ready" : "warning",
      status.status === "ready" ? "知识库可查询" : "知识库未构建"
    );
  } catch (error) {
    showToast(error.message);
  }
}

function renderStartupFailure(error) {
  setStatusPill(dom.kbStatus, "error", "知识库连接失败");
  setStatusPill(dom.modelStatus, "error", "后端状态未知");
  setQueryStatus(error.message, "error");
  showToast(error.message);
}

function optionalInteger(value) {
  const clean = String(value).trim();
  if (!clean) return null;
  const parsed = Number(clean);
  return Number.isInteger(parsed) ? parsed : null;
}

function emptyState(message) {
  const paragraph = document.createElement("p");
  paragraph.className = "empty-state";
  paragraph.textContent = message;
  return paragraph;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error("服务器返回了无法解析的响应。");
  }
  if (!response.ok) {
    const error = payload?.error || {};
    throw new Error(error.message || `请求失败（HTTP ${response.status}）`);
  }
  return payload;
}

let toastTimer = null;
function showToast(message) {
  dom.toast.textContent = message;
  dom.toast.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    dom.toast.hidden = true;
  }, 4200);
}
