"use strict";

// ----------------------------------------------------------------------------
// State + DOM
// ----------------------------------------------------------------------------
const els = {};
const state = {
  outputDir: "outputs",
  unifiedRoot: "data/unified",
  dataset: "",
  taskId: "",
  trajectories: [],
  data: null,
  model: null,
};

// Timeline lifecycle classes (derived from phase2 instance_states). These drive
// the timeline bar colors and the trajectory step border.
const LIFE_ORDER = ["clean", "costly", "manifest", "latent"];
const LIFE_LABEL = {
  clean: "Clean Resolution（干净修复）",
  costly: "Costly Resolution（代价修复）",
  manifest: "Manifest Active（显性未修复）",
  latent: "Latent Active（隐性未修复）",
  gt_only: "真值（未检出）",
};

// classification key -> CSS color variable
const CLS_VAR = {
  // report buckets
  critical: "--c-critical",
  cascade: "--c-cascade",
  independent_chain: "--c-chain",
  fixed: "--c-fixed",
  dormant: "--c-dormant",
  suppressed: "--c-suppressed",
  // lifecycle classes
  clean: "--c-clean",
  costly: "--c-costly",
  manifest: "--c-manifest",
  latent: "--c-latent",
  gt_only: "--c-gt-only",
  flagged: "--border-strong",
};
function clsColor(cls) { return `var(${CLS_VAR[cls] || "--border-strong"})`; }

document.addEventListener("DOMContentLoaded", init);

async function init() {
  els.dataset = document.getElementById("dataset-select");
  els.traj = document.getElementById("trajectory-select");
  els.output = document.getElementById("output-input");
  els.reload = document.getElementById("reload-btn");
  els.status = document.getElementById("status");
  els.timeline = document.getElementById("timeline");
  els.legend = document.getElementById("legend");
  els.trajectory = document.getElementById("trajectory");
  els.report = document.getElementById("report");
  els.expandObs = document.getElementById("expand-obs");
  els.currentTask = document.getElementById("current-task");

  els.dataset.addEventListener("change", onDatasetChange);
  els.traj.addEventListener("change", onTrajectoryChange);
  els.reload.addEventListener("click", () => loadConfig(true));
  els.expandObs.addEventListener("change", onToggleObs);

  renderLegend();
  await loadConfig(false);
}

function setStatus(msg, isError) {
  els.status.textContent = msg || "";
  els.status.classList.toggle("error", !!isError);
}

// ----------------------------------------------------------------------------
// Data loading
// ----------------------------------------------------------------------------
async function api(path, params) {
  const qs = new URLSearchParams(params).toString();
  const res = await fetch(`${path}?${qs}`);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return res.json();
}

async function loadConfig(useInputOutput) {
  try {
    if (useInputOutput) state.outputDir = els.output.value.trim() || state.outputDir;
    const cfg = await api("/api/config", {
      output_dir: state.outputDir,
      unified_root: state.unifiedRoot,
    });
    state.outputDir = cfg.output_dir;
    state.unifiedRoot = cfg.unified_root;
    els.output.value = state.outputDir;

    els.dataset.innerHTML = "";
    cfg.datasets.forEach((d) => {
      const o = document.createElement("option");
      o.value = d; o.textContent = d;
      els.dataset.appendChild(o);
    });
    if (!cfg.datasets.length) {
      setStatus("在输出目录下没有发现数据集（找不到 <name>_report / _phase1）", true);
      return;
    }
    state.dataset = cfg.default_dataset && cfg.datasets.includes(cfg.default_dataset)
      ? cfg.default_dataset : cfg.datasets[0];
    els.dataset.value = state.dataset;
    await loadTrajectories();
  } catch (e) {
    setStatus("加载配置失败：" + e.message, true);
  }
}

async function onDatasetChange() {
  state.dataset = els.dataset.value;
  await loadTrajectories();
}

async function loadTrajectories() {
  setStatus("正在扫描轨迹列表…");
  try {
    const list = await api("/api/trajectories", {
      dataset: state.dataset,
      output_dir: state.outputDir,
      unified_root: state.unifiedRoot,
    });
    state.trajectories = list;
    els.traj.innerHTML = "";
    list.forEach((t) => {
      const o = document.createElement("option");
      o.value = t.task_id;
      let mark = "";
      if (t.has_report && t.predicted_step != null) {
        mark = t.gt_step == null ? "  · 预测" + t.predicted_step
          : (t.hit ? "  · ✓" : "  · ✗") + ` 预测${t.predicted_step}/真值${t.gt_step}`;
      }
      o.textContent = midTruncate(t.task_id, 44) + mark;
      o.title = t.task_id + mark;
      els.traj.appendChild(o);
    });
    if (!list.length) {
      setStatus("该数据集下没有轨迹文件", true);
      els.trajectory.innerHTML = '<div class="empty">无数据</div>';
      els.report.innerHTML = "";
      els.timeline.innerHTML = "";
      return;
    }
    state.taskId = list[0].task_id;
    els.traj.value = state.taskId;
    await loadTrajectory();
  } catch (e) {
    setStatus("加载轨迹列表失败：" + e.message, true);
  }
}

async function onTrajectoryChange() {
  state.taskId = els.traj.value;
  await loadTrajectory();
}

async function loadTrajectory() {
  setStatus("正在加载轨迹…");
  try {
    const data = await api("/api/trajectory", {
      dataset: state.dataset,
      task_id: state.taskId,
      output_dir: state.outputDir,
      unified_root: state.unifiedRoot,
    });
    state.data = data;
    state.model = buildModel(data);
    if (els.currentTask) {
      els.currentTask.textContent = data.task_id || state.taskId || "";
      els.currentTask.title = data.task_id || state.taskId || "";
    }
    renderAll();
    const m = state.model;
    let s = `${data.messages.length} 步`;
    if (!data.phase1) s += " · 缺少 Phase1";
    if (!data.report) s += " · 缺少 Report";
    setStatus(s);
  } catch (e) {
    setStatus("加载轨迹失败：" + e.message, true);
    els.trajectory.innerHTML = `<div class="empty">${escapeHtml(e.message)}</div>`;
    els.report.innerHTML = "";
    els.timeline.innerHTML = "";
  }
}

// ----------------------------------------------------------------------------
// Model derivation
// ----------------------------------------------------------------------------
function fixedAtStep(fixStatus) {
  if (typeof fixStatus !== "string") return null;
  const m = fixStatus.match(/fixed_at_step_(\d+)/);
  return m ? parseInt(m[1], 10) : null;
}

// Classify an instance's lifecycle from its phase2 instance_states record into
// one of four chain classes, and decide where its timeline bar should end.
//   (1) Clean Resolution  : fixed, not on a chain, span within cap        → fixed_at_step_N
//   (2) Costly Resolution : fixed, but on a chain OR span exceeds cap      → resource last_referencing_step
//   (3) Manifest Active   : not fixed, on a chain OR span exceeds cap      → resource last_referencing_step
//   (4) Latent Active     : not fixed, not on a chain, span within cap; or exploration_suppressed
// `end` may be null here; the caller fills a fallback (last_trigger_step / origin).
function classifyLifecycle(s) {
  const fixedStep = fixedAtStep(s.fix_status);
  const isFixed = fixedStep != null || s.semantic_status === "fixed";
  const chain = s.chain_membership === true;
  const cap = s.span_exceeds_cap === true;
  const suppressed = s.exploration_suppressed === true;
  const lastRef = s.resource_effect && s.resource_effect.last_referencing_step != null
    ? s.resource_effect.last_referencing_step : null;

  if (suppressed) return { cls: "latent", end: lastRef };
  if (isFixed && !chain && !cap) return { cls: "clean", end: fixedStep };
  if (isFixed) return { cls: "costly", end: lastRef != null ? lastRef : fixedStep };
  if (chain || cap) return { cls: "manifest", end: lastRef };
  return { cls: "latent", end: lastRef };
}

function buildModel(data) {
  const messages = data.messages || [];
  const report = data.report;
  const phase1 = data.phase1;
  const phase2 = data.phase2;
  const annotation = data.annotation || {};
  const total = (report && report.total_steps)
    || (phase2 && phase2.total_steps)
    || messages.length;

  // per-step triggers (phase1 if present, else phase2 carries the same array)
  const triggerSrc = (phase1 && Array.isArray(phase1.step_triggers)) ? phase1.step_triggers
    : (phase2 && Array.isArray(phase2.step_triggers)) ? phase2.step_triggers : [];
  const triggersByStep = {};
  triggerSrc.forEach((t) => {
    (triggersByStep[t.step] = triggersByStep[t.step] || []).push(t);
  });

  // instances/timeline. Preferred source = phase2 (instances + instance_states),
  // which carries the fields needed for the four-class chain classification.
  // Fall back to report.error_timeline / phase1.instances when phase2 is absent.
  let instances = [];
  if (phase2 && Array.isArray(phase2.instances)) {
    const statesById = {};
    (phase2.instance_states || []).forEach((s) => { statesById[s.instance_id] = s; });
    instances = phase2.instances.map((it) => {
      const s = statesById[it.instance_id] || {};
      const lc = classifyLifecycle(s);
      const origin = it.origin_step;
      let end = lc.end;
      if (end == null) end = it.last_trigger_step != null ? it.last_trigger_step : origin;
      if (end < origin) end = origin;
      return {
        instance_id: it.instance_id,
        start: origin,
        end,
        cls: lc.cls,
        category: it.category,
        taxonomy_tag: s.qualified_origin_taxonomy_tag || null,
        error_content: it.error_content,
        fix_status: s.fix_status,
        state: s.state,
        terminal_connection: s.terminal_connection,
        chain_membership: s.chain_membership,
        span_exceeds_cap: s.span_exceeds_cap,
        wasted_steps: s.resource_effect ? s.resource_effect.wasted_step_count : null,
      };
    });
  } else if (report && Array.isArray(report.error_timeline)) {
    instances = report.error_timeline.map((it) => ({
      instance_id: it.instance_id,
      start: it.origin_step,
      end: fixedAtStep(it.fix_status) != null ? fixedAtStep(it.fix_status) : (total - 1),
      cls: it.classification || "flagged",
      category: it.category,
      taxonomy_tag: it.taxonomy_tag,
      error_content: it.error_content,
      fix_status: it.fix_status,
      state: it.state,
      terminal_connection: it.terminal_connection,
      chain_membership: it.chain_membership,
      wasted_steps: it.wasted_steps,
    }));
  } else if (phase1 && Array.isArray(phase1.instances)) {
    instances = phase1.instances.map((it) => ({
      instance_id: it.instance_id,
      start: it.origin_step,
      end: it.last_trigger_step != null ? it.last_trigger_step : it.origin_step,
      cls: "flagged",
      category: it.category,
      taxonomy_tag: null,
      error_content: it.error_content,
    }));
  }

  // markers
  const predicted = report && report.critical_error_analysis
    ? report.critical_error_analysis.critical_step : null;
  const gt = annotation.critical_error_step != null ? annotation.critical_error_step : null;
  const alts = report && Array.isArray(report.alternative_critical_steps)
    ? report.alternative_critical_steps.map((a) => a.step) : [];

  // Is the predicted critical step represented in any detected error (Phase-1
  // trigger or instance)? If not, we draw a dedicated timeline bar for it.
  const detectedSteps = new Set();
  instances.forEach((it) => detectedSteps.add(it.start));
  Object.keys(triggersByStep).forEach((s) => detectedSteps.add(parseInt(s, 10)));
  const criticalMissing = predicted != null && !detectedSteps.has(predicted);

  // Predicted critical step & alternative critical steps may not have a Phase-1
  // trigger at that step. If so, synthesize a trigger from the report so the
  // conflict still renders inline on the trajectory at that step.
  const addSyntheticTrigger = (step, t) => {
    if (step == null || triggersByStep[step]) return;
    (triggersByStep[step] = triggersByStep[step] || []).push(t);
  };
  if (report && report.critical_error_analysis) {
    const ca = report.critical_error_analysis;
    addSyntheticTrigger(ca.critical_step, {
      step: ca.critical_step,
      taxonomy_tag: ca.taxonomy_tag,
      category: ca.conflict_type,
      wrong_content_quote: ca.wrong_content_quote,
      reference_quote: ca.reference_quote,
      confidence_reasoning: ca.error_explanation || ca.conflict_type_reason,
      synthetic: true,
      synthetic_kind: "critical",
    });
  }
  if (report && Array.isArray(report.alternative_critical_steps)) {
    report.alternative_critical_steps.forEach((a) => {
      addSyntheticTrigger(a.step, {
        step: a.step,
        taxonomy_tag: a.taxonomy_tag,
        category: null,
        wrong_content_quote: "",
        reference_quote: "",
        confidence_reasoning: a.reason,
        hint_sentence: a.hint_sentence,
        synthetic: true,
        synthetic_kind: "alt",
      });
    });
  }

  // index instances by origin step
  const instByStep = {};
  instances.forEach((it) => {
    (instByStep[it.start] = instByStep[it.start] || []).push(it);
  });

  // lane assignment for timeline bars
  const sorted = instances.slice().sort((a, b) => a.start - b.start || a.end - b.end);
  const laneEnds = [];
  sorted.forEach((it) => {
    let lane = laneEnds.findIndex((e) => e < it.start);
    if (lane === -1) { lane = laneEnds.length; laneEnds.push(it.end); }
    else laneEnds[lane] = it.end;
    it.lane = lane;
  });
  const lanes = Math.max(laneEnds.length, 1);

  return {
    total, instances, instByStep, triggersByStep,
    predicted, gt, alts, criticalMissing, lanes,
  };
}

// ----------------------------------------------------------------------------
// Rendering
// ----------------------------------------------------------------------------
function renderAll() {
  renderTimeline();
  renderTrajectory();
  renderReport();
}

function renderLegend() {
  els.legend.innerHTML = [
    ...LIFE_ORDER.map((c) => `<span class="lg"><span class="sw dot-c-${c}"></span>${LIFE_LABEL[c].split("（")[0]}</span>`),
    `<span class="lg"><span class="mk" style="border-left-color:var(--m-pred)"></span>预测</span>`,
    `<span class="lg"><span class="mk" style="border-left-color:var(--m-gt);border-left-style:dashed"></span>真值</span>`,
    `<span class="lg"><span class="mk" style="border-left-color:var(--m-alt);border-left-style:dotted"></span>备选</span>`,
  ].join("");
}

function renderTimeline() {
  const m = state.model;
  const denom = Math.max(m.total - 1, 1);
  const pct = (s) => (clamp(s, 0, denom) / denom) * 100;
  // A predicted critical step that was not detected in Phase 1 gets its own
  // dashed bar on an extra bottom lane; it is purely a timeline overlay (not
  // part of m.instances / buckets / triggers).
  const extraLane = m.criticalMissing ? 1 : 0;
  const height = 8 + (m.lanes + extraLane) * 16 + 22;
  els.timeline.style.height = height + "px";

  let html = "";

  // lifecycle bars
  m.instances.forEach((it) => {
    const left = pct(it.start);
    const w = Math.max(pct(it.end) - left, 0.8);
    const top = 8 + it.lane * 16;
    const tip = `#${it.instance_id} ${it.cls} · 起源步 ${it.start}` +
      (it.fix_status ? ` · ${it.fix_status}` : "") +
      (it.error_content ? `\n${it.error_content}` : "");
    html += `<div class="tl-bar cls-${it.cls}" data-step="${it.start}" title="${escapeAttr(tip)}"
      style="left:${left}%;width:${w}%;top:${top}px;background:${clsColor(it.cls)}"></div>`;
  });

  // predicted critical step missing from Phase 1: blue dashed bar from that step
  // to the last step.
  if (m.criticalMissing && m.predicted != null) {
    const left = pct(m.predicted);
    const w = Math.max(pct(m.total - 1) - left, 0.8);
    const top = 8 + m.lanes * 16;
    html += `<div class="tl-bar missing-bar" data-step="${m.predicted}"
      style="left:${left}%;width:${w}%;top:${top}px"></div>`;
  }

  // markers
  const addMarker = (step, label, style, capCls) => {
    if (step == null) return "";
    const left = pct(step);
    return `<div class="tl-marker" style="left:${left}%">
      <div class="ln" style="border-left-color:${style.color};border-left-style:${style.dash}"></div>
      <div class="cap ${capCls || ""}" data-step="${step}" style="${style.cap || `background:${style.color}`}">${label}${step}</div>
    </div>`;
  };
  m.alts.forEach((s) => { html += addMarker(s, "备选", { color: "var(--m-alt)", dash: "dotted" }); });
  // When prediction is correct (predicted == gt) show a single combined marker
  // instead of two overlapping lines.
  if (m.predicted != null && m.predicted === m.gt) {
    html += addMarker(m.gt, "预测｜真值 ", {
      color: "var(--m-pred)",
      dash: "solid",
      cap: "background:linear-gradient(90deg,var(--m-pred) 0 50%,var(--m-gt) 50% 100%)",
    }, "cap-hit");
  } else {
    html += addMarker(m.gt, "真值", { color: "var(--m-gt)", dash: "dashed" });
    html += addMarker(m.predicted, "预测", { color: "var(--m-pred)", dash: "solid" });
  }

  // axis ticks
  html += `<div class="tl-axis" style="left:2px">0</div>`;
  html += `<div class="tl-axis" style="right:2px">${m.total - 1}</div>`;

  els.timeline.innerHTML = html;
  els.timeline.querySelectorAll("[data-step]").forEach((el) => {
    el.addEventListener("click", () => jumpToStep(parseInt(el.dataset.step, 10)));
  });
}

function renderTrajectory() {
  const data = state.data;
  const m = state.model;
  const frag = document.createDocumentFragment();

  data.messages.forEach((msg, idx) => {
    const step = msg.step != null ? msg.step : idx;
    const role = msg.role || "?";
    const isAssistant = role === "assistant";
    // user / tool / system are environment or context, not agent reasoning
    const isContext = !isAssistant;

    const card = document.createElement("div");
    card.className = `step role-${role}` + (isContext ? " obs" : "");
    card.id = `step-${step}`;
    if (isContext) card.classList.add("collapsed");

    // classification border from instance origin
    const insts = m.instByStep[step] || [];
    if (insts.length) {
      card.classList.add("flagged", `cls-${insts[0].cls}`);
    } else if (m.triggersByStep[step]) {
      card.classList.add("flagged");
    }

    // ribbons
    const ribbons = [];
    if (step === m.predicted) ribbons.push(`<span class="ribbon pred">预测关键步</span>`);
    if (step === m.gt) ribbons.push(`<span class="ribbon gt">真值关键步</span>`);
    if (m.alts.includes(step)) ribbons.push(`<span class="ribbon alt">备选</span>`);

    const tagBadge = insts.length && insts[0].taxonomy_tag
      ? `<span class="badge">${escapeHtml(insts[0].taxonomy_tag)}</span>` : "";

    let summary;
    if (isAssistant) {
      const action = parseAssistant(msg.content).action;
      summary = action ? action.slice(0, 90) : summarizeObs(msg.content);
    } else {
      summary = summarizeObs(msg.content);
    }

    const head = document.createElement("div");
    head.className = "step-head";
    head.innerHTML =
      `<span class="step-num">Step ${step}</span>` +
      `<span class="step-role ${role}">${escapeHtml(role)}</span>` +
      tagBadge +
      `<div class="ribbons">${ribbons.join("")}</div>` +
      `<span class="step-summary">${escapeHtml(summary)}</span>`;
    head.addEventListener("click", () => card.classList.toggle("collapsed"));
    card.appendChild(head);

    const body = document.createElement("div");
    body.className = "step-body";
    body.innerHTML = isAssistant ? renderAssistantBody(msg.content) : renderUserBody(msg.content);

    // triggers + instance notes
    const extra = renderStepErrors(step, m);
    if (extra) body.insertAdjacentHTML("beforeend", extra);

    card.appendChild(body);
    frag.appendChild(card);
  });

  els.trajectory.innerHTML = "";
  els.trajectory.appendChild(frag);
  els.trajectory.querySelectorAll(".inst-jump").forEach((el) => {
    el.addEventListener("click", (e) => { e.stopPropagation(); jumpToStep(parseInt(el.dataset.step, 10)); });
  });
}

function renderUserBody(content) {
  return `<div class="obs-full">${escapeHtml(content || "")}</div>`;
}

function hasReasoningTags(content) {
  return /<(memory|reflection|plan|action)>[\s\S]*?<\/\1>/i.test(content || "");
}

function renderAssistantBody(content) {
  // Only the ALFWorld-style scaffold uses <memory>/<reflection>/<plan>/<action>.
  // Other datasets (tau2bench, swebenchpro, whoandwhen, ...) are free-form text,
  // tool calls, or JSON ledgers — render them as-is.
  if (!hasReasoningTags(content)) {
    return `<div class="obs-full">${escapeHtml(content || "")}</div>`;
  }
  const p = parseAssistant(content);
  let html = "";
  const block = (key, label, cls) => {
    if (!p[key]) return "";
    return `<div class="subblock ${cls || key}"><div class="lbl">${label}</div>` +
      `<div class="txt">${escapeHtml(p[key])}</div></div>`;
  };
  html += block("memory", "Memory");
  html += block("reflection", "Reflection");
  html += block("plan", "Plan");
  html += block("action", "Action", "action");
  if (!html) html = `<div class="obs-full">${escapeHtml(content || "")}</div>`;
  return html;
}

function renderStepErrors(step, m) {
  let html = "";
  const insts = m.instByStep[step] || [];
  insts.forEach((it) => {
    const life = [];
    if (it.fix_status) life.push(it.fix_status);
    if (it.state) life.push(it.state);
    if (it.terminal_connection && it.terminal_connection !== "none") life.push("TC:" + it.terminal_connection);
    if (it.chain_membership) life.push("chain");
    if (it.span_exceeds_cap) life.push("span>cap");
    const label = LIFE_LABEL[it.cls] ? LIFE_LABEL[it.cls].split("（")[0] : it.cls;
    html += `<div class="inst-note"><span class="badge" style="color:#fff;border:none;background:${clsColor(it.cls)}">${escapeHtml(label)}</span> ` +
      `<span class="ec">实例#${it.instance_id}</span> ${escapeHtml(it.error_content || "")}` +
      (life.length ? ` <span style="color:var(--muted)">[${escapeHtml(life.join(" · "))}]</span>` : "") + `</div>`;
  });

  const trigs = m.triggersByStep[step] || [];
  if (trigs.length) {
    html += `<div class="triggers">`;
    trigs.forEach((t) => {
      const conf = (t.confidence || "").toLowerCase();
      const hasConflict = t.wrong_content_quote || t.reference_quote;
      html += `<div class="trigger">` +
        `<div class="thead">` +
        `<span class="badge">${escapeHtml(t.taxonomy_tag || "?")}</span>` +
        `<span class="step-role">${escapeHtml(t.category || "")}</span>` +
        (t.confidence ? `<span class="conf ${conf}">${escapeHtml(t.confidence)}</span>` : "") +
        `</div>` +
        (hasConflict ? `<div class="conflict">` +
          `<div class="cell wrong"><span class="k">错误承诺</span><span class="q">${escapeHtml(t.wrong_content_quote || "")}</span></div>` +
          `<div class="cell ref"><span class="k">违反的参照</span><span class="q">${escapeHtml(t.reference_quote || "")}</span></div>` +
          `</div>` : "") +
        (t.confidence_reasoning ? `<div class="why">${escapeHtml(t.confidence_reasoning)}</div>` : "") +
        (t.hint_sentence ? `<div class="why"><b>Hint：</b>${escapeHtml(t.hint_sentence)}</div>` : "") +
        `</div>`;
    });
    html += `</div>`;
  }
  return html;
}

function renderReport() {
  const data = state.data;
  const m = state.model;
  const r = data.report;
  const out = [];

  if (!r) {
    els.report.innerHTML = '<div class="empty">该轨迹没有 Report v2 文件。' +
      (data.phase1 ? "已根据 Phase 1 渲染左侧错误。" : "也没有 Phase 1。") + "</div>";
    return;
  }

  // 1. verdict banner
  const ca = r.critical_error_analysis || {};
  const hit = m.gt != null && m.predicted != null && m.gt === m.predicted;
  out.push(`<div class="card verdict">
    <div class="verdict-row">
      <span class="big" style="color:var(--m-pred)">预测 ${fmt(m.predicted)}</span>
      <span class="big" style="color:var(--m-gt)">真值 ${m.gt == null ? "—" : m.gt}</span>
      ${m.gt == null ? "" : `<span class="pill ${hit ? "hit" : "miss"}">${hit ? "命中 ✓" : "未命中 ✗"}</span>`}
    </div>
    <div class="verdict-row">
      ${ca.taxonomy_tag ? `<span class="pill tag">${escapeHtml(ca.taxonomy_tag)}</span>` : ""}
      ${r.task_outcome ? `<span class="pill outcome">${escapeHtml(r.task_outcome)}</span>` : ""}
      ${ca.unified_source ? `<span class="kv"><span class="v">来源：${escapeHtml(ca.unified_source)}</span></span>` : ""}
    </div>
  </div>`);

  // 2. critical error card
  out.push(`<div class="card">
    <h3>关键错误 · <span class="step-link" data-step="${ca.critical_step}">Step ${fmt(ca.critical_step)}</span></h3>
    ${ca.conflict_type ? `<div class="kv"><span class="k">冲突类型</span><span class="v">${escapeHtml(ca.conflict_type)} — ${escapeHtml(ca.conflict_type_description || "")}</span></div>` : ""}
    ${ca.conflict_type_reason ? `<div class="explain"><span class="lbl">冲突原因</span>${escapeHtml(ca.conflict_type_reason)}</div>` : ""}
    <div class="conflict">
      <div class="cell wrong"><span class="k">错误承诺</span><span class="q">${escapeHtml(ca.wrong_content_quote || "")}</span></div>
      <div class="cell ref"><span class="k">违反的参照</span><span class="q">${escapeHtml(ca.reference_quote || "")}</span></div>
    </div>
    ${ca.error_explanation ? `<div class="explain"><span class="lbl">错误解释</span>${escapeHtml(ca.error_explanation)}</div>` : ""}
    ${ca.chain_explanation ? `<div class="explain"><span class="lbl">因果链</span>${escapeHtml(ca.chain_explanation)}</div>` : ""}
  </div>`);

  // 3. fix suggestion
  const fs = r.fix_suggestion;
  if (fs) {
    out.push(`<div class="card">
      <h3>修复建议</h3>
      ${fs.hint_sentence ? `<div class="explain"><span class="lbl">Hint</span>${escapeHtml(fs.hint_sentence)}</div>` : ""}
      ${fs.narrative_summary ? `<div class="explain"><span class="lbl">叙述摘要</span>${escapeHtml(fs.narrative_summary)}</div>` : ""}
    </div>`);
  }

  // 4. statistics
  const st = r.error_statistics;
  if (st) {
    const stat = (n, t) => `<div class="stat"><div class="n">${n}</div><div class="t">${t}</div></div>`;
    out.push(`<div class="card">
      <h3>错误统计</h3>
      <div class="stats">
        ${stat(fmt(st.total_instances), "实例总数")}
        ${stat(fmt(st.active_instances), "活跃")}
        ${stat(fmt(st.fixed_instances), "已修复")}
        ${stat(fmt(st.dormant_instances), "休眠")}
        ${stat(fmt(st.chain_instances), "链上")}
        ${stat(fmt(st.total_wasted_steps), "浪费步数")}
      </div>
      ${st.wasted_step_ratio != null ? `<div class="kv" style="margin-top:8px"><span class="k">浪费占比</span><span class="v">${(st.wasted_step_ratio * 100).toFixed(0)}%</span></div>` : ""}
    </div>`);
  }

  // 5. classification buckets — group the timeline instances by their four
  // lifecycle classes (same source/colors as the timeline), not the report's
  // own error_classification field.
  const byCls = {};
  m.instances.forEach((it) => { (byCls[it.cls] = byCls[it.cls] || []).push(it); });
  const bucketOrder = LIFE_ORDER.concat(byCls.gt_only ? ["gt_only"] : []);
  let buckets = `<div class="card"><h3>错误分类（链）</h3>`;
  bucketOrder.forEach((key) => {
    const arr = byCls[key] || [];
    const open = (key === "manifest" || key === "costly") ? "open" : "";
    buckets += `<details class="bucket" ${open}>
      <summary><span class="dot dot-c-${key}"></span>${LIFE_LABEL[key]}<span class="count">${arr.length}</span></summary>
      <div class="items">`;
    arr.forEach((it) => {
      const isCritical = m.predicted != null && it.start === m.predicted;
      const tags =
        (isCritical ? `<span class="rowtag critical">★ 关键</span>` : "") +
        (it.chain_membership ? `<span class="rowtag chain">链</span>` : "") +
        (it.span_exceeds_cap ? `<span class="rowtag cap">span&gt;cap</span>` : "");
      buckets += `<div class="inst-row inst-jump${isCritical ? " is-critical" : ""}" data-step="${it.start}">
        <span class="os">Step ${fmt(it.start)}→${fmt(it.end)}</span>
        <span class="cat">${escapeHtml(it.taxonomy_tag || it.category || "")}</span>
        ${tags}
        <span class="ec">${escapeHtml(it.error_content || "")}</span>
      </div>`;
    });
    if (!arr.length) buckets += `<div class="inst-row" style="color:var(--muted);cursor:default">无</div>`;
    buckets += `</div></details>`;
  });
  buckets += `</div>`;
  out.push(buckets);

  // 6. alternative steps
  const alts = r.alternative_critical_steps || [];
  if (alts.length) {
    let a = `<div class="card"><h3>备选关键步</h3>`;
    alts.forEach((it) => {
      a += `<div class="alt-row">
        <div class="h"><span class="step-link" data-step="${it.step}">Step ${fmt(it.step)}</span>
        ${it.taxonomy_tag ? `<span class="badge">${escapeHtml(it.taxonomy_tag)}</span>` : ""}</div>
        ${it.reason ? `<div class="hint">${escapeHtml(it.reason)}</div>` : ""}
        ${it.hint_sentence ? `<div class="hint"><b>Hint:</b> ${escapeHtml(it.hint_sentence)}</div>` : ""}
      </div>`;
    });
    a += `</div>`;
    out.push(a);
  }

  els.report.innerHTML = out.join("");
  els.report.querySelectorAll("[data-step]").forEach((el) => {
    el.addEventListener("click", () => jumpToStep(parseInt(el.dataset.step, 10)));
  });
}

// ----------------------------------------------------------------------------
// Interactions + helpers
// ----------------------------------------------------------------------------
function jumpToStep(step) {
  const card = document.getElementById(`step-${step}`);
  if (!card) return;
  card.classList.remove("collapsed");
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.remove("jump-hl");
  void card.offsetWidth; // restart animation
  card.classList.add("jump-hl");
}

function onToggleObs() {
  const expand = els.expandObs.checked;
  els.trajectory.querySelectorAll(".step.obs").forEach((c) => {
    c.classList.toggle("collapsed", !expand);
  });
}

function parseAssistant(content) {
  content = content || "";
  const grab = (tag) => {
    const m = content.match(new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`, "i"));
    return m ? m[1].trim() : "";
  };
  return {
    memory: grab("memory"),
    reflection: grab("reflection"),
    plan: grab("plan"),
    action: grab("action"),
  };
}

function summarizeObs(content) {
  content = content || "";
  let m = content.match(/your current observation is:\s*([^\n]+)/i);
  if (m) return m[1].trim().slice(0, 200);
  m = content.match(/Your task is to:\s*([^\n]+)/i);
  if (m) return "任务：" + m[1].trim().slice(0, 160);
  return content.replace(/\s+/g, " ").trim().slice(0, 160);
}

// Shorten a long string by eliding the middle, keeping head and tail readable.
function midTruncate(s, max) {
  s = String(s == null ? "" : s);
  max = max || 44;
  if (s.length <= max) return s;
  const keep = max - 1;
  const head = Math.ceil(keep / 2);
  const tail = Math.floor(keep / 2);
  return s.slice(0, head) + "…" + s.slice(s.length - tail);
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function fmt(v) { return v == null ? "—" : v; }

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
function escapeAttr(s) { return escapeHtml(s).replace(/\n/g, "&#10;"); }
