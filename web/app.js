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
  confirms: "确认",
  conflicts: "冲突",
  cautions: "警示",
};

const APPLICABILITY_LABELS = {
  direct: "直接证据",
  indirect_rmpp: "RMPP 间接外推",
  indirect_smpp: "SMPP 间接外推",
  general_mpp: "一般 MPP 证据",
  unclear: "人群不明确",
  not_assessed: "未评估适用性",
};

const TYPE_STYLES = {
  guideline: { color: "#176b4d", soft: "#e8f3ed" },
  systematic_review: { color: "#315f75", soft: "#eaf1f5" },
  randomized_controlled_trial: { color: "#a56713", soft: "#fff3dd" },
  observational_study: { color: "#6f5b7c", soft: "#f1edf4" },
  narrative_review: { color: "#68746e", soft: "#f0f3f1" },
  case_report: { color: "#a3443d", soft: "#faecea" },
  unknown: { color: "#59645f", soft: "#f0f3f1" },
};

const RELATION_STYLES = {
  supports: { color: "#176b4d", label: "支持" },
  updates: { color: "#a56713", label: "更新" },
  supplements: { color: "#315f75", label: "补充" },
  confirms: { color: "#6f5b7c", label: "确认" },
  conflicts: { color: "#b0303b", label: "冲突" },
  cautions: { color: "#a3443d", label: "警示" },
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

const MODEL_CONFIG_ERROR = "请完整填写 API 地址、API Key 和模型名。";
const MODEL_API_URL_ERROR = "请输入有效的模型 API 地址。";
const MODEL_ERROR_MESSAGES = Object.freeze({
  chat_auth_failed: "模型服务拒绝了 API Key，请检查令牌是否有效以及当前模型访问权限。",
  chat_model_not_found: "模型不存在或当前令牌无权访问，请检查模型名。",
  chat_timeout: "模型服务响应超时，请稍后重试。",
  chat_request_rejected: "模型服务拒绝了请求，请检查 API 地址、模型名及接口兼容性。",
  chat_unavailable: "模型服务暂时不可用，请稍后重试。",
  chat_response_invalid: "模型响应格式无法解析，已使用本地证据综合。",
  ungrounded_model_response: "模型回答未通过引用校验，已使用本地证据综合。",
  model_generation_failed: "模型未能完成综合，已使用本地证据综合。",
  no_relevant_evidence: "未形成可验证的相关声明，已使用本地证据综合。",
  model_config_missing: "未配置模型，已使用本地证据综合。",
});

const state = {
  catalog: null,
  kbStatus: null,
  modelStatus: null,
  startupFailed: false,
  result: null,
  graph: { nodes: [], edges: [] },
  selectedGraphNodeId: null,
  sources: new Map(),
  activeSource: null,
  currentJobId: null,
  pollTimer: null,
};

const dom = {};

document.addEventListener("DOMContentLoaded", initialize);

async function initialize() {
  cacheDom();
  clearBrowserModelConfig();
  bindEvents();
  updateQuestionCount();
  setDefaultYears();
  state.startupFailed = false;
  try {
    const [health, kbStatus, modelStatus, catalog] = await Promise.all([
      fetchJson("/api/health"),
      fetchJson("/api/kb/status"),
      fetchJson("/api/model/status"),
      fetchJson("/api/evidence"),
    ]);
    state.kbStatus = kbStatus;
    state.modelStatus = modelStatus;
    state.startupFailed = false;
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
    state.startupFailed = true;
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
    toggleAllTypes: document.querySelector("#toggle-all-types"),
    yearFrom: document.querySelector("#year-from"),
    yearTo: document.querySelector("#year-to"),
    fulltextOnly: document.querySelector("#fulltext-only"),
    modelApiBase: document.querySelector("#model-api-base"),
    apiKey: document.querySelector("#api-key"),
    chatModelName: document.querySelector("#chat-model-name"),
    toggleApiKey: document.querySelector("#toggle-api-key"),
    modelSettings: document.querySelector("#model-service"),
    focusSettings: document.querySelector("#focus-settings"),
    kbStatus: document.querySelector("#kb-status"),
    modelStatus: document.querySelector("#model-status"),
    dataSourceLabel: document.querySelector("#data-source-label"),
    stats: {
      source_metadata_records: document.querySelector("#stat-metadata"),
      unique_documents: document.querySelector("#stat-unique"),
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
    analysisChips: document.querySelector("#analysis-chips"),
    decisionSummary: document.querySelector("#decision-summary"),
    sourceList: document.querySelector("#source-list"),
    sourceCount: document.querySelector("#source-count"),
    sourceDetail: document.querySelector("#source-detail"),
    toast: document.querySelector("#toast"),
  });
}

function bindEvents() {
  dom.question.addEventListener("input", updateQuestionCount);
  [dom.modelApiBase, dom.apiKey, dom.chatModelName].forEach((input) => {
    input.addEventListener("input", handleModelConfigInput);
  });
  dom.toggleApiKey.addEventListener("click", toggleApiKeyVisibility);
  dom.toggleAllTypes.addEventListener("click", toggleAllEvidenceTypes);
  dom.filters.addEventListener("change", syncToggleAllTypesLabel);
  dom.focusSettings.addEventListener("click", focusModelSettings);
  dom.runQuery.addEventListener("click", runQuery);
  dom.buildKb.addEventListener("click", () => startBuild());
  dom.continueKb.addEventListener("click", continueEmbedding);
  dom.pauseKb.addEventListener("click", pauseBuild);
  dom.retryKb.addEventListener("click", retryBuild);
  dom.sourceList.addEventListener("click", handleSourceClick);
  dom.reasoningSteps.addEventListener("click", handleSourceClick);
  dom.finalAnswer.addEventListener("click", handleSourceClick);
  window.addEventListener("pagehide", resetBrowserModelConfig);
  window.addEventListener("pageshow", (event) => {
    if (event.persisted) resetBrowserModelConfig();
  });
}

function setDefaultYears() {
  dom.yearFrom.value = "2015";
  dom.yearTo.value = String(new Date().getFullYear());
}

function updateQuestionCount() {
  dom.questionCount.textContent = `${dom.question.value.length} / 2000`;
}

function clearBrowserModelConfig() {
  dom.modelApiBase.value = "";
  dom.modelApiBase.removeAttribute("aria-invalid");
  dom.apiKey.value = "";
  dom.chatModelName.value = "";
  dom.apiKey.type = "password";
  dom.toggleApiKey.setAttribute("aria-label", "显示 API Key");
  dom.toggleApiKey.setAttribute("title", "显示 API Key");
  dom.toggleApiKey.setAttribute("aria-pressed", "false");
}

function resetBrowserModelConfig() {
  clearBrowserModelConfig();
  renderModelStatus();
}

function handleModelConfigInput() {
  if (!dom.modelApiBase.value.trim() || dom.modelApiBase.validity.valid) {
    dom.modelApiBase.removeAttribute("aria-invalid");
  }
  renderModelStatus();
}

function browserModelConfig({ validate = false } = {}) {
  const baseUrl = dom.modelApiBase.value.trim();
  const apiKey = dom.apiKey.value.trim();
  const modelName = dom.chatModelName.value.trim();
  const completed = [baseUrl, apiKey, modelName].map(Boolean);
  if (completed.every((value) => !value)) return null;
  if (completed.every(Boolean)) {
    if (!dom.modelApiBase.validity.valid) {
      if (validate) throw new Error(MODEL_API_URL_ERROR);
      return null;
    }
    return { base_url: baseUrl, api_key: apiKey, model_name: modelName };
  }
  if (validate) throw new Error(MODEL_CONFIG_ERROR);
  return null;
}

function browserModelConfigState() {
  const completed = [dom.modelApiBase, dom.apiKey, dom.chatModelName].map(
    (input) => Boolean(input.value.trim())
  );
  if (completed.every(Boolean)) {
    return dom.modelApiBase.validity.valid ? "complete" : "invalid";
  }
  if (completed.some(Boolean)) return "partial";
  return "empty";
}

function focusFirstMissingModelField() {
  const missing = [dom.modelApiBase, dom.apiKey, dom.chatModelName].find(
    (input) => !input.value.trim()
  );
  if (missing) missing.focus();
}

function toggleApiKeyVisibility() {
  const reveal = dom.apiKey.type === "password";
  const label = reveal ? "隐藏 API Key" : "显示 API Key";
  dom.apiKey.type = reveal ? "text" : "password";
  dom.toggleApiKey.setAttribute("aria-label", label);
  dom.toggleApiKey.setAttribute("title", label);
  dom.toggleApiKey.setAttribute("aria-pressed", String(reveal));
  dom.apiKey.focus({ preventScroll: true });
}

function toggleAllEvidenceTypes() {
  const checkboxes = Array.from(dom.filters.querySelectorAll("input[type='checkbox']"));
  const shouldSelectAll = checkboxes.some((checkbox) => !checkbox.checked);
  checkboxes.forEach((checkbox) => {
    checkbox.checked = shouldSelectAll;
  });
  syncToggleAllTypesLabel();
}

function syncToggleAllTypesLabel() {
  const checkboxes = Array.from(dom.filters.querySelectorAll("input[type='checkbox']"));
  dom.toggleAllTypes.textContent = checkboxes.length && checkboxes.every((checkbox) => checkbox.checked)
    ? "全不选"
    : "全选";
}

function focusModelSettings() {
  dom.modelSettings.open = true;
  dom.modelSettings.scrollIntoView({ behavior: "smooth", block: "center" });
  dom.modelApiBase.focus({ preventScroll: true });
}

function renderSystemStatus(health, kbStatus, modelStatus) {
  const ready = kbStatus.status === "ready";
  setStatusPill(
    dom.kbStatus,
    ready ? "ready" : "warning",
    ready ? "知识库可查询" : "知识库未构建"
  );
  renderModelStatus();
  if (health.status === "degraded" && ready) {
    setStatusPill(dom.kbStatus, "warning", "知识库降级可用");
  }
}

function renderModelStatus() {
  const browserState = browserModelConfigState();
  if (browserState === "invalid") {
    setStatusPill(dom.modelStatus, "warning", "API 地址格式有误");
    return;
  }
  if (browserState === "complete") {
    const config = browserModelConfig({ validate: false });
    setStatusPill(dom.modelStatus, "ready", `本次模型 ${config.model_name}`);
    return;
  }
  if (browserState === "partial") {
    setStatusPill(dom.modelStatus, "warning", "模型配置待补全");
    return;
  }
  if (state.modelStatus) {
    const configured = Boolean(state.modelStatus.configured);
    setStatusPill(
      dom.modelStatus,
      configured ? "ready" : "warning",
      configured ? `模型 ${state.modelStatus.chat_model || "已配置"}` : "模型未配置"
    );
    return;
  }
  if (state.startupFailed) {
    setStatusPill(dom.modelStatus, "warning", "后端状态未知");
    return;
  }
  setStatusPill(dom.modelStatus, "loading", "模型状态加载中");
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
  syncToggleAllTypesLabel();
}

async function runQuery() {
  let modelConfig;
  try {
    modelConfig = browserModelConfig({ validate: true });
  } catch (error) {
    const invalidUrl = error.message === MODEL_API_URL_ERROR;
    setQueryStatus(invalidUrl ? MODEL_API_URL_ERROR : MODEL_CONFIG_ERROR, "error");
    if (invalidUrl) {
      dom.modelApiBase.setAttribute("aria-invalid", "true");
      dom.modelApiBase.focus();
    } else {
      focusFirstMissingModelField();
    }
    return;
  }
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
  if (modelConfig) payload.model_config = modelConfig;

  setQueryBusy(true);
  setQueryStatus("正在执行向量检索、证据重排与文献结论综合…", "loading");
  try {
    const result = await fetchJson("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.result = result;
    state.selectedGraphNodeId = null;
    indexSources(result.sources || []);
    renderReport(result);
    renderSources(result.sources || []);
    renderGraph(result.graph || { nodes: [], edges: [] });
    const warnings = queryResultWarnings(result);
    const warningSuffix = warnings.length ? ` ${warnings.join(" ")}` : "";
    setQueryStatus(
      `综合完成，共返回 ${result.sources?.length || 0} 项来源。${warningSuffix}`,
      warnings.length ? "warning" : "ready"
    );
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
  label.textContent = busy ? "正在综合" : "开始循证分析";
}

function setQueryStatus(message, status) {
  dom.queryStatus.textContent = message;
  dom.queryStatus.dataset.state = status;
}

function modelErrorMessage(errorCode) {
  return MODEL_ERROR_MESSAGES[errorCode] || "";
}

function queryResultWarnings(result) {
  const modelWarning = result.model_used === false
    ? modelErrorMessage(result.model_error)
    : "";
  return [...new Set([result.warning, modelWarning]
    .filter((message) => typeof message === "string")
    .map((message) => message.trim())
    .filter(Boolean))];
}

function renderReport(result) {
  dom.retrievalMode.textContent = retrievalModeLabel(result.mode, result.degraded_reason);
  dom.reportModel.textContent = result.model_used
    ? `模型：${result.model_name || "已配置"}`
    : "本地证据综合";
  dom.groundingStatus.textContent = result.model_used ? "模型综合已采用" : "引用已校验";
  dom.groundingStatus.dataset.state = "ready";
  renderAnalysisSummary(result);

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

function renderAnalysisSummary(result) {
  const summary = result.summary && typeof result.summary === "object" ? result.summary : {};
  const queryContext = result.query_context && typeof result.query_context === "object"
    ? result.query_context
    : {};
  const evidenceCount = Number(summary.evidence_count ?? result.sources?.length ?? 0);
  const relationCount = Number(summary.relation_count ?? result.graph?.edges?.length ?? 0);
  const updateCount = Number(summary.update_count ?? 0);
  const conflictCount = Number(summary.conflict_count ?? 0);
  const chips = [
    ...queryUnderstandingChips(queryContext),
    { text: `${evidenceCount} 项证据`, className: "good" },
    { text: `${relationCount} 条关系`, className: "neutral" },
    { text: `${updateCount} 项时间更新`, className: updateCount ? "update" : "neutral" },
    { text: `${conflictCount} 项冲突`, className: conflictCount ? "warning" : "neutral" },
  ];
  dom.analysisChips.replaceChildren();
  chips.forEach((item) => {
    const chip = document.createElement("span");
    chip.className = `analysis-chip ${item.className}`;
    chip.textContent = item.text;
    dom.analysisChips.append(chip);
  });

  dom.decisionSummary.textContent = [
    queryContext.question_type
      ? `识别为${questionTypeLabel(queryContext.question_type)}`
      : "未限定临床问题类型",
    Array.isArray(queryContext.subquestions) && queryContext.subquestions.length
      ? `已拆分 ${queryContext.subquestions.length} 个补充向量查询`
      : "使用原问题执行向量检索",
    `已完成 ${evidenceCount} 项来源的分层检索`,
    `识别 ${relationCount} 条证据关系`,
    updateCount ? `其中 ${updateCount} 项较新证据更新既有结论` : "未识别到明确的时间更新信号",
    conflictCount ? `并发现 ${conflictCount} 项冲突，需在报告中重点核验。` : "各层证据方向未见明确冲突。",
  ].join("；");
}

function queryUnderstandingChips(context) {
  const chips = [];
  if (context.question_type) {
    chips.push({ text: questionTypeLabel(context.question_type), className: "good" });
  }
  const pico = context.pico && typeof context.pico === "object" ? context.pico : {};
  [
    ["P", pico.population],
    ["I", pico.intervention],
    ["C", pico.comparator],
    ["O", pico.outcome],
  ].forEach(([label, values]) => {
    if (!Array.isArray(values) || !values.length) return;
    chips.push({ text: `${label} · ${values.slice(0, 2).join(" / ")}`, className: "neutral" });
  });
  return chips;
}

function questionTypeLabel(value) {
  return {
    treatment: "治疗问题",
    diagnosis: "诊断问题",
    prognosis: "预后问题",
    etiology: "病因问题",
    safety: "安全性问题",
    prevention: "预防问题",
    general: "一般临床问题",
  }[value] || "临床问题";
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
  const source = state.sources.get(number);
  const firstPage = Array.isArray(source?.page_ranges) ? source.page_ranges.find(Boolean) : "";
  button.title = firstPage ? `来源 ${number} · 第 ${firstPage} 页` : `查看来源 ${number}`;
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
      textSpan(source.fulltext ? "全文命中" : "元数据命中"),
      textSpan(APPLICABILITY_LABELS[source.population_applicability] || "适用性未知")
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
    renderSourceDetail(detail, source);
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

function renderSourceDetail(detail, source) {
  dom.sourceDetail.replaceChildren();
  const heading = document.createElement("h3");
  heading.id = "source-detail-title";
  heading.textContent = detail.title || source.title || `来源 ${source.source_number}`;
  const bibliography = document.createElement("dl");
  bibliography.className = "bibliography";
  addBibliographyRow(bibliography, "作者", formatAuthors(detail.authors));
  addBibliographyRow(bibliography, "期刊 / 年份", [detail.journal, detail.year].filter(Boolean).join(" · ") || "未知");
  addBibliographyRow(bibliography, "DOI", detail.doi || "未提供");
  addBibliographyRow(bibliography, "证据类型", TYPE_LABELS[detail.evidence_type] || detail.evidence_type || "未分类");
  addBibliographyRow(
    bibliography,
    "人群适用性",
    APPLICABILITY_LABELS[source.population_applicability] || "适用性未知"
  );
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
  snippetHeading.textContent = "本次查询命中片段";
  const snippetList = document.createElement("div");
  snippetList.className = "snippet-list";
  const snippets = querySourceSnippets(source);
  if (!snippets.length) {
    snippetList.append(emptyState("本次查询没有可展示的来源片段。"));
  } else {
    snippets.forEach((snippet) => {
      const item = document.createElement("article");
      item.className = "snippet";
      const text = document.createElement("p");
      text.textContent = snippet.text || "";
      const meta = document.createElement("small");
      meta.textContent = `${snippet.section || "未标注章节"} · 第 ${pageRange(snippet.page_start, snippet.page_end)} 页${snippet.is_ocr ? " · OCR" : ""}`;
      const footer = document.createElement("div");
      footer.className = "snippet-footer";
      footer.append(meta);
      if (detail.pdf_available && snippet.page_start) {
        const locator = document.createElement("a");
        locator.className = "snippet-page-link";
        locator.href = pdfPageHref(detail.document_id, snippet.page_start);
        locator.target = "_blank";
        locator.rel = "noopener";
        locator.textContent = `PDF · 第 ${snippet.page_start} 页 ↗`;
        locator.title = `在 PDF 中定位到第 ${snippet.page_start} 页`;
        footer.append(locator);
      }
      item.append(text, footer);
      snippetList.append(item);
    });
  }
  dom.sourceDetail.append(snippetHeading, snippetList);

  if (detail.pdf_available) {
    const link = document.createElement("a");
    link.className = "pdf-link";
    const firstPage = snippets.find((snippet) => snippet.page_start)?.page_start;
    link.href = pdfPageHref(detail.document_id, firstPage);
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = "在浏览器中查看 PDF";
    dom.sourceDetail.append(link);
  }
}

function pdfPageHref(documentId, page) {
  const base = `/api/documents/${encodeURIComponent(documentId)}/pdf`;
  return page ? `${base}#page=${Number(page)}` : base;
}

function querySourceSnippets(source) {
  const texts = Array.isArray(source.snippets) ? source.snippets : [];
  const ranges = Array.isArray(source.page_ranges) ? source.page_ranges : [];
  const chunkIds = Array.isArray(source.chunk_ids) ? source.chunk_ids : [];
  return texts
    .filter((text) => typeof text === "string" && text.trim())
    .map((text, index) => {
      const [pageStart, pageEnd] = parsePageRange(ranges[index]);
      return {
        chunk_id: chunkIds[index] || "",
        text,
        section: source.fulltext ? "全文命中" : "题录或摘要命中",
        page_start: pageStart,
        page_end: pageEnd,
        is_ocr: false,
      };
    });
}

function parsePageRange(value) {
  const match = String(value || "").match(/^(\d+)(?:-(\d+))?$/);
  if (!match) return [null, null];
  return [Number(match[1]), Number(match[2] || match[1])];
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

function renderGraph(graph = state.graph) {
  renderLaneGraph(graph);
}

function renderRelationChainGraph(rawGraph) {
  state.graph = rawGraph;
  const nodes = (Array.isArray(rawGraph.nodes) ? rawGraph.nodes : [])
    .slice(0, 10)
    .map(normalizeGraphNode)
    .filter((node) => node.type === "document");
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const allEdges = (Array.isArray(rawGraph.edges) ? rawGraph.edges : [])
    .map((edge) => ({
      ...edge,
      source: String(edge.source),
      target: String(edge.target),
      relation: String(edge.relation || "supplements"),
    }))
    .filter((edge) => (
      nodeById.has(edge.source)
      && nodeById.has(edge.target)
      && RELATION_STYLES[edge.relation]
    ));
  const edges = selectCoreGraphEdges(allEdges, 6);

  dom.graph.replaceChildren();
  dom.graphMeta.textContent = edges.length < allEdges.length
    ? `${nodes.length} 篇文献 · 展示 ${edges.length}/${allEdges.length} 条核心关系`
    : `${nodes.length} 篇文献 · ${edges.length} 条核心关系`;
  if (!edges.length) {
    renderLaneGraph(rawGraph);
    return;
  }
  dom.graphEmpty.hidden = true;

  const viewWidth = 760;
  const rowHeight = 92;
  const topOffset = 30;
  const nodeWidth = 216;
  const nodeHeight = 62;
  const leftX = 24;
  const rightX = viewWidth - leftX - nodeWidth;
  const viewHeight = Math.max(360, topOffset * 2 + edges.length * rowHeight);
  dom.graph.setAttribute("viewBox", `0 0 ${viewWidth} ${viewHeight}`);
  dom.graph.setAttribute("preserveAspectRatio", "xMinYMin meet");
  dom.graph.style.minWidth = `${viewWidth}px`;
  dom.graph.style.height = `${viewHeight}px`;

  const connected = connectedNodeIds(state.selectedGraphNodeId, allEdges);
  edges.forEach((edge, index) => {
    const source = nodeById.get(edge.source);
    const target = nodeById.get(edge.target);
    const style = relationStyleFor(edge.relation);
    const y = topOffset + index * rowHeight;
    const centerY = y + nodeHeight / 2;
    const row = svgElement("g", {
      class: "relation-row",
      "data-relation": edge.relation,
    });
    row.append(svgElement("line", {
      x1: "12",
      x2: String(viewWidth - 12),
      y1: String(y + nodeHeight + 15),
      y2: String(y + nodeHeight + 15),
      class: "relation-row-divider",
    }));
    const path = svgElement("path", {
      d: `M ${leftX + nodeWidth + 12} ${centerY} L ${rightX - 12} ${centerY}`,
      class: "graph-edge chain-edge",
      stroke: style.color,
      "data-relation": edge.relation,
      "data-source": edge.source,
      "data-target": edge.target,
    });
    const isRelated = Boolean(
      state.selectedGraphNodeId
      && (edge.source === state.selectedGraphNodeId || edge.target === state.selectedGraphNodeId)
    );
    path.classList.add(isRelated ? "is-highlighted" : "is-hidden");
    const pathTitle = svgElement("title");
    pathTitle.textContent = `${style.label}：${edge.rationale || "文献间证据关系"}`;
    path.append(pathTitle);
    row.append(path);

    const labelBackground = svgElement("rect", {
      x: "337",
      y: String(centerY - 13),
      width: "86",
      height: "26",
      rx: "4",
      class: "edge-label-background",
      stroke: style.color,
    });
    labelBackground.classList.add(isRelated ? "is-highlighted" : "is-hidden");
    row.append(labelBackground);
    const label = svgElement("text", {
      x: "380",
      y: String(centerY + 4),
      class: "edge-label chain-label",
      fill: style.color,
      "text-anchor": "middle",
      "data-relation": edge.relation,
    });
    label.classList.add(isRelated ? "is-highlighted" : "is-hidden");
    label.textContent = style.label;
    row.append(label);
    row.append(
      relationChainNode(source, leftX, y, nodeWidth, nodeHeight, connected),
      relationChainNode(target, rightX, y, nodeWidth, nodeHeight, connected),
    );
    dom.graph.append(row);
  });
}

function selectCoreGraphEdges(edges, limit) {
  const priority = {
    updates: 0,
    conflicts: 1,
    cautions: 2,
    supports: 3,
    confirms: 4,
    supplements: 5,
  };
  return [...edges]
    .sort((left, right) => (
      (priority[left.relation] ?? 99) - (priority[right.relation] ?? 99)
      || left.source.localeCompare(right.source)
      || left.target.localeCompare(right.target)
    ))
    .slice(0, limit);
}

function relationChainNode(node, x, y, width, height, connected) {
  const style = typeStyle(node.evidenceType);
  const group = svgElement("g", {
    class: "graph-node chain-node",
    role: "button",
    tabindex: "0",
    "aria-label": `${TYPE_LABELS[node.evidenceType] || "文献"}：${node.title}`,
    "data-node-id": node.id,
    "data-node-type": "document",
  });
  if (Number.isInteger(node.sourceNumber)) {
    group.dataset.sourceNumber = String(node.sourceNumber);
  }
  if (node.id === state.selectedGraphNodeId) {
    group.classList.add("is-selected");
  } else if (state.selectedGraphNodeId && !connected.has(node.id)) {
    group.classList.add("is-muted");
  }
  group.append(svgElement("rect", {
    x: String(x),
    y: String(y),
    width: String(width),
    height: String(height),
    rx: "5",
    fill: style.soft,
    stroke: style.color,
  }));
  const typeText = svgElement("text", {
    x: String(x + 10),
    y: String(y + 17),
    class: "node-type",
    fill: style.color,
  });
  typeText.textContent = compactTypeLabel(node.evidenceType);
  group.append(typeText);
  const yearText = svgElement("text", {
    x: String(x + width - 9),
    y: String(y + 17),
    class: "node-year",
  });
  yearText.textContent = node.year || "年份未知";
  group.append(yearText);
  const titleText = svgElement("text", {
    x: String(x + 10),
    y: String(y + 39),
    class: "node-title",
  });
  splitTitle(node.title, 20, 2).forEach((line, index) => {
    const tspan = svgElement("tspan", {
      x: String(x + 10),
      dy: index === 0 ? "0" : "14",
    });
    tspan.textContent = line;
    titleText.append(tspan);
  });
  group.append(titleText);
  const title = svgElement("title");
  title.textContent = node.title;
  group.append(title);

  const activate = () => {
    state.selectedGraphNodeId = node.id;
    renderGraph();
    if (Number.isInteger(node.sourceNumber)) selectSource(node.sourceNumber);
  };
  group.addEventListener("click", activate);
  group.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      activate();
    }
  });
  return group;
}

function renderLaneGraph(graph = state.graph) {
  const rawGraph = graph && typeof graph === "object" ? graph : { nodes: [], edges: [] };
  state.graph = rawGraph;
  const nodes = (Array.isArray(rawGraph.nodes) ? rawGraph.nodes : [])
    .slice(0, 10)
    .map(normalizeGraphNode)
    .filter((node) => node.type === "document");
  const visibleIds = new Set(nodes.map((node) => node.id));
  const edges = (Array.isArray(rawGraph.edges) ? rawGraph.edges : [])
    .map((edge) => ({
      ...edge,
      source: String(edge.source),
      target: String(edge.target),
      relation: String(edge.relation || "supplements"),
    }))
    .filter((edge) => (
      visibleIds.has(edge.source)
      && visibleIds.has(edge.target)
      && RELATION_STYLES[edge.relation]
    ));
  const activeEdges = state.selectedGraphNodeId
    ? edges.filter((edge) => (
      edge.source === state.selectedGraphNodeId || edge.target === state.selectedGraphNodeId
    ))
    : [];

  dom.graph.replaceChildren();
  dom.graphMeta.textContent = `${nodes.length} 篇文献 · ${edges.length} 条关系 · 按证据层级布局`;
  if (!nodes.length) {
    dom.graphEmpty.hidden = false;
    dom.graph.removeAttribute("style");
    dom.graph.setAttribute("viewBox", "0 0 760 360");
    return;
  }
  dom.graphEmpty.hidden = true;

  const evidenceTypes = TYPE_ORDER
    .concat(["unknown"])
    .filter((type) => nodes.some((node) => node.evidenceType === type));
  const laneWidth = 190;
  const nodeWidth = 142;
  const nodeHeight = 78;
  const laneGap = 46;
  const rowGap = 48;
  const maxRows = Math.max(...evidenceTypes.map((type) => (
    nodes.filter((node) => node.evidenceType === type).length
  )));
  const sameLaneEdgeIndexes = activeEdges
    .map((edge, edgeIndex) => {
      const source = nodes.find((node) => node.id === edge.source);
      const target = nodes.find((node) => node.id === edge.target);
      return source && target && source.evidenceType === target.evidenceType
        ? edgeIndex
        : null;
    })
    .filter((edgeIndex) => edgeIndex !== null);
  const sameLaneRouteByEdge = new Map(sameLaneEdgeIndexes.map((edgeIndex, index) => [
    edgeIndex,
    { channel: index },
  ]));
  const topOffset = 74;
  const contentBottom = topOffset + maxRows * nodeHeight + Math.max(0, maxRows - 1) * rowGap;
  const viewWidth = Math.max(760, evidenceTypes.length * laneWidth + laneGap * 2);
  const viewHeight = Math.max(360, contentBottom + 44);
  dom.graph.setAttribute("viewBox", `0 0 ${viewWidth} ${viewHeight}`);
  dom.graph.setAttribute("preserveAspectRatio", "xMinYMin meet");
  dom.graph.style.minWidth = `${viewWidth}px`;
  dom.graph.style.height = `${viewHeight}px`;

  const positions = new Map();
  evidenceTypes.forEach((evidenceType, laneIndex) => {
    const x = laneGap + laneIndex * laneWidth;
    const laneTitle = svgElement("text", {
      x: String(x + nodeWidth / 2),
      y: "28",
      class: "lane-title",
      fill: typeStyle(evidenceType).color,
      "text-anchor": "middle",
    });
    laneTitle.textContent = TYPE_LABELS[evidenceType] || "未分类";
    dom.graph.append(laneTitle);
    nodes
      .filter((node) => node.evidenceType === evidenceType)
      .sort((left, right) => left.sourceNumber - right.sourceNumber || left.id.localeCompare(right.id))
      .forEach((node, rowIndex) => {
        positions.set(node.id, {
          x,
          y: topOffset + rowIndex * (nodeHeight + rowGap),
          width: nodeWidth,
          height: nodeHeight,
          laneIndex,
        });
      });
  });

  const connected = connectedNodeIds(state.selectedGraphNodeId, edges);
  const edgePlans = activeEdges.map((edge, edgeIndex) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) return null;
    const sameLane = source.laneIndex === target.laneIndex;
    const movingLeft = source.laneIndex > target.laneIndex;
    return {
      edge,
      edgeIndex,
      source,
      target,
      sourceSide: sameLane ? "right" : (movingLeft ? "left" : "right"),
      targetSide: sameLane ? "right" : (movingLeft ? "right" : "left"),
      outerRoute: sameLaneRouteByEdge.get(edgeIndex) || null,
    };
  }).filter(Boolean);
  const endpointGroups = new Map();
  edgePlans.forEach((plan) => {
    [
      { role: "source", nodeId: plan.edge.source, side: plan.sourceSide },
      { role: "target", nodeId: plan.edge.target, side: plan.targetSide },
    ].forEach((endpoint) => {
      const key = `${endpoint.nodeId}:${endpoint.side}`;
      if (!endpointGroups.has(key)) endpointGroups.set(key, []);
      endpointGroups.get(key).push({ plan, role: endpoint.role });
    });
  });
  endpointGroups.forEach((endpoints) => {
    endpoints
      .sort((left, right) => {
        const leftOther = left.role === "source" ? left.plan.target : left.plan.source;
        const rightOther = right.role === "source" ? right.plan.target : right.plan.source;
        return leftOther.y - rightOther.y || left.plan.edgeIndex - right.plan.edgeIndex;
      })
      .forEach((endpoint, slot) => {
        endpoint.plan[`${endpoint.role}Port`] = { slot, total: endpoints.length };
      });
  });

  const labelLayer = svgElement("g", { class: "edge-label-layer" });
  const reservedLabels = [];
  edgePlans.forEach((plan) => {
    const { edge, source, target } = plan;
    const style = relationStyleFor(edge.relation);
    const sourceY = graphPortY(source, plan.sourcePort);
    const targetY = graphPortY(target, plan.targetPort);
    const geometry = edgeGeometry(source, target, {
      sourceSide: plan.sourceSide,
      targetSide: plan.targetSide,
      sourceY,
      targetY,
      sourceStemOffset: graphStemOffset(plan.sourcePort),
      targetStemOffset: graphStemOffset(plan.targetPort),
      route: plan.outerRoute,
    });
    const path = svgElement("path", {
      d: geometry.path,
      class: "graph-edge",
      stroke: style.color,
      "data-relation": edge.relation,
      "data-source": edge.source,
      "data-target": edge.target,
    });
    const isRelated = Boolean(
      state.selectedGraphNodeId
      && (edge.source === state.selectedGraphNodeId || edge.target === state.selectedGraphNodeId)
    );
    path.classList.add(isRelated ? "is-highlighted" : "is-hidden");
    const title = svgElement("title");
    title.textContent = `${style.label}：${edge.rationale || "文献间证据关系"}`;
    path.append(title);
    dom.graph.append(path);

    const labelPosition = reserveGraphLabel(
      geometry.labelX,
      geometry.labelY,
      style.label,
      positions,
      reservedLabels,
      viewWidth,
      viewHeight,
    );
    const labelGroup = svgElement("g", {
      class: "edge-label-group",
      "data-relation": edge.relation,
    });
    labelGroup.classList.add(isRelated ? "is-highlighted" : "is-hidden");
    labelGroup.append(svgElement("rect", {
      x: String(labelPosition.x - labelPosition.width / 2),
      y: String(labelPosition.y - 12),
      width: String(labelPosition.width),
      height: "18",
      rx: "3",
      class: "edge-label-bg",
      stroke: style.color,
    }));
    const label = svgElement("text", {
      x: String(labelPosition.x),
      y: String(labelPosition.y + 1),
      class: "edge-label",
      "text-anchor": "middle",
      "data-relation": edge.relation,
    });
    label.textContent = style.label;
    labelGroup.append(label);
    labelLayer.append(labelGroup);
  });

  nodes.forEach((node) => {
    const position = positions.get(node.id);
    if (!position) return;
    const style = typeStyle(node.evidenceType);
    const group = svgElement("g", {
      class: "graph-node",
      role: "button",
      tabindex: "0",
      "aria-pressed": node.id === state.selectedGraphNodeId ? "true" : "false",
      "aria-label": `${TYPE_LABELS[node.evidenceType] || "文献"}：${node.title}`,
      "data-node-id": node.id,
      "data-node-type": "document",
    });
    if (Number.isInteger(node.sourceNumber)) {
      group.dataset.sourceNumber = String(node.sourceNumber);
    }
    if (node.id === state.selectedGraphNodeId) {
      group.classList.add("is-selected");
    } else if (state.selectedGraphNodeId && !connected.has(node.id)) {
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
      y: String(position.y + 18),
      class: "node-type",
      fill: style.color,
    });
    typeText.textContent = compactTypeLabel(node.evidenceType);
    group.append(typeText);

    const yearText = svgElement("text", {
      x: String(position.x + position.width - 9),
      y: String(position.y + 18),
      class: "node-year",
    });
    yearText.textContent = node.year || "年份未知";
    group.append(yearText);

    const titleText = svgElement("text", {
      x: String(position.x + 10),
      y: String(position.y + 42),
      class: "node-title",
    });
    splitTitle(node.title, 13, 2).forEach((line, index) => {
      const tspan = svgElement("tspan", {
        x: String(position.x + 10),
        dy: index === 0 ? "0" : "16",
      });
      tspan.textContent = line;
      titleText.append(tspan);
    });
    group.append(titleText);
    const title = svgElement("title");
    title.textContent = node.title;
    group.append(title);

    const activate = () => {
      const deselecting = state.selectedGraphNodeId === node.id;
      state.selectedGraphNodeId = deselecting ? null : node.id;
      renderGraph();
      if (!deselecting && Number.isInteger(node.sourceNumber)) selectSource(node.sourceNumber);
    };
    group.addEventListener("click", activate);
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate();
      }
    });
    dom.graph.append(group);
  });
  dom.graph.append(labelLayer);
}

function normalizeGraphNode(node) {
  const id = String(node.node_id ?? node.id ?? "unknown");
  const payload = node.payload && typeof node.payload === "object" ? node.payload : {};
  return {
    id,
    type: String(node.node_type || payload.node_type || "document"),
    title: String(payload.title || node.label || node.title || id),
    evidenceType: String(payload.evidence_type || node.evidence_type || "unknown"),
    year: payload.year ?? node.year ?? "",
    quality: String(payload.quality || node.quality || "unknown"),
    sourceNumber: Number(payload.source_number ?? node.source_number),
  };
}

function typeStyle(evidenceType) {
  return TYPE_STYLES[evidenceType] || TYPE_STYLES.unknown;
}

function relationStyleFor(relation) {
  return RELATION_STYLES[relation] || RELATION_STYLES.supplements;
}

function compactTypeLabel(evidenceType) {
  return {
    randomized_controlled_trial: "RCT",
    observational_study: "观察性",
    narrative_review: "叙述综述",
    systematic_review: "系统综述",
  }[evidenceType] || TYPE_LABELS[evidenceType] || "未分类";
}

function connectedNodeIds(selectedId, edges) {
  const connected = new Set(selectedId ? [selectedId] : []);
  if (!selectedId) return connected;
  edges.forEach((edge) => {
    if (edge.source === selectedId) connected.add(edge.target);
    if (edge.target === selectedId) connected.add(edge.source);
  });
  return connected;
}

function initialCurvedPath(x1, y1, x2, y2) {
  const direction = x2 >= x1 ? 1 : -1;
  const bend = Math.max(34, Math.abs(x2 - x1) * 0.42) * direction;
  return `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`;
}

function graphPortY(position, port = { slot: 0, total: 1 }) {
  const usableHeight = position.height - 30;
  return position.y + 19 + usableHeight * ((port.slot + 1) / (port.total + 1));
}

function graphStemOffset(port = { slot: 0, total: 1 }) {
  return 10 + 26 * ((port.slot + 1) / (port.total + 1));
}

function edgeGeometry(source, target, options) {
  const {
    sourceSide,
    targetSide,
    sourceY,
    targetY,
    sourceStemOffset,
    targetStemOffset,
    route,
  } = options;
  const x1 = sourceSide === "left" ? source.x : source.x + source.width;
  const x2 = targetSide === "left" ? target.x : target.x + target.width;
  const sameLane = source.laneIndex === target.laneIndex;
  if (!sameLane) {
    const direction = x2 >= x1 ? 1 : -1;
    const bend = Math.min(110, Math.max(34, Math.abs(x2 - x1) * 0.34));
    return {
      path: `M ${x1} ${sourceY} C ${x1 + direction * bend} ${sourceY}, ${x2 - direction * bend} ${targetY}, ${x2} ${targetY}`,
      labelX: (x1 + x2) / 2,
      labelY: (sourceY + targetY) / 2 - 9,
    };
  }
  const sourceDirection = sourceSide === "left" ? -1 : 1;
  const channel = route?.channel || 0;
  const loopOffset = Math.max(sourceStemOffset, targetStemOffset) + channel * 8;
  const loopX = x1 + loopOffset * sourceDirection;
  return {
    path: `M ${x1} ${sourceY} C ${loopX} ${sourceY}, ${loopX} ${targetY}, ${x2} ${targetY}`,
    labelX: loopX,
    labelY: (sourceY + targetY) / 2 - 7,
  };
}

function reserveGraphLabel(candidateX, candidateY, text, positions, reserved, viewWidth, viewHeight) {
  const width = Math.max(34, String(text || "").length * 10 + 12);
  const height = 18;
  const offsets = [0, -20, 20, -40, 40, -60, 60];
  const nodeBoxes = Array.from(positions.values());
  const overlaps = (box, other, padding = 0) => !(
    box.x + box.width + padding <= other.x
    || other.x + other.width + padding <= box.x
    || box.y + box.height + padding <= other.y
    || other.y + other.height + padding <= box.y
  );
  for (const yOffset of offsets) {
    for (const xOffset of offsets) {
      const x = Math.min(viewWidth - width / 2 - 4, Math.max(width / 2 + 4, candidateX + xOffset));
      const y = Math.min(viewHeight - 8, Math.max(18, candidateY + yOffset));
      const box = { x: x - width / 2, y: y - 12, width, height };
      if (nodeBoxes.some((node) => overlaps(box, node, 6))) continue;
      if (reserved.some((label) => overlaps(box, label, 5))) continue;
      reserved.push(box);
      return { x, y, width };
    }
  }
  const fallbackY = Math.min(viewHeight - 8, Math.max(18, candidateY + reserved.length * 20));
  const fallback = { x: candidateX - width / 2, y: fallbackY - 12, width, height };
  reserved.push(fallback);
  return { x: candidateX, y: fallbackY, width };
}

function splitTitle(value, maxCharacters, maxLines) {
  const clean = String(value || "未命名文献").replace(/\s+/g, " ").trim();
  const lines = [];
  let remaining = clean;
  while (remaining && lines.length < maxLines) {
    if (remaining.length <= maxCharacters) {
      lines.push(remaining);
      remaining = "";
      break;
    }
    let splitAt = remaining.lastIndexOf(" ", maxCharacters);
    if (splitAt < Math.floor(maxCharacters * 0.55)) splitAt = maxCharacters;
    lines.push(remaining.slice(0, splitAt).trim());
    remaining = remaining.slice(splitAt).trim();
  }
  if (remaining && lines.length) {
    lines[lines.length - 1] = `${lines[lines.length - 1].slice(0, maxCharacters - 1)}…`;
  }
  return lines.length ? lines : ["未命名文献"];
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

async function continueEmbedding() {
  const pending = Number(state.kbStatus?.embedding_pending || 0);
  const countLabel = pending > 0 ? `约 ${pending.toLocaleString("zh-CN")} 条文本` : "待处理文本";
  showToast(`已开始在本机向量化${countLabel}。`);
  await startBuild({
    confirmEmbeddingCost: true,
    confirmFullEmbeddingCost: true,
    stage: "embedding",
  });
}

async function startBuild({
  confirmEmbeddingCost = false,
  confirmFullEmbeddingCost = false,
  embeddingLimit = null,
  stage = null,
} = {}) {
  setJobControls(true);
  try {
    const payload = {
      confirm_embedding_cost: confirmEmbeddingCost,
      ...(confirmFullEmbeddingCost ? { confirm_full_embedding_cost: true } : {}),
      ...(embeddingLimit ? { embedding_limit: embeddingLimit } : {}),
      ...(stage ? { stage } : {}),
    };
    const job = await fetchJson("/api/kb/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    beginJobPolling(job);
    showToast(confirmEmbeddingCost ? "已提交本地向量化任务。" : "已开始本地构建阶段。");
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
  if (paidStage) {
    await startBuild({
      confirmEmbeddingCost: true,
      confirmFullEmbeddingCost: true,
      stage,
    });
    return;
  }
  setJobControls(true);
  try {
    const job = await fetchJson("/api/kb/retry", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stage, confirm_embedding_cost: false }),
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
    if (["paused", "completed", "completed_with_errors", "embedding_pending", "failed"].includes(job.state)) {
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
  const embedding = progress.stage_counters?.embedding || {};
  const processed = Number(embedding.processed || 0);
  const total = Number(progress.total_embedding_count || 0);
  const stageLabel = stage ? STAGE_LABELS[stage] || stage : jobStateLabel(job.state);
  dom.jobStage.textContent = stage === "embedding" && total > 0
    ? `${stageLabel} ${processed.toLocaleString("zh-CN")} / ${total.toLocaleString("zh-CN")}`
    : stageLabel;
  dom.jobState.textContent = jobStateLabel(job.state);
  const terminal = ["paused", "completed", "completed_with_errors", "embedding_pending", "failed"].includes(job.state);
  if (terminal) {
    if (["completed", "completed_with_errors"].includes(job.state)) {
      dom.jobProgress.value = 100;
    } else if (job.state === "embedding_pending") {
      dom.jobProgress.removeAttribute("value");
    } else {
      dom.jobProgress.value = 0;
    }
  } else {
    if (stage === "embedding" && total > 0) {
      dom.jobProgress.value = Math.min(100, Math.max(0, (processed / total) * 100));
    } else {
      dom.jobProgress.removeAttribute("value");
    }
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
    completed_with_errors: "完成，存在失败记录",
    embedding_pending: "本地索引完成，等待向量化",
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
  renderModelStatus();
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
