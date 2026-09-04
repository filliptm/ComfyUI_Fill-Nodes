import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const NODE_CLASS = "FL_GenerateText";
const LAYOUT_VERSION = 2;
const DEFAULT_NODE_SIZE = [1080, 520];
const MIN_NODE_SIZE = [760, 440];
const INSTANCES = new Map();

const BACKEND_FIELDS = [
  "system_prompt",
  "prompt",
  "max_length",
  "sampling",
  "temperature",
  "top_k",
  "top_p",
  "min_p",
  "repetition_penalty",
  "presence_penalty",
  "seed",
  "thinking",
];
const UI_FIELDS = [...BACKEND_FIELDS, "control_after_generate"];
const INTEGER_FIELDS = new Set(["max_length", "top_k", "seed"]);
const FLOAT_FIELDS = new Set(["temperature", "top_p", "min_p", "repetition_penalty", "presence_penalty"]);

const STYLES = `
  .flgt-host {
    container-name: flgt-node;
    container-type: inline-size;
    height: 100%;
    min-height: 410px;
    width: 100%;
  }
  .flgt-console {
    --flgt-bg: var(--comfy-menu-bg, #111318);
    --flgt-panel: var(--comfy-input-bg, #181b22);
    --flgt-border: var(--border-color, #343946);
    --flgt-text: var(--input-text, #edf0f7);
    --flgt-muted: var(--descrip-text, #8f98aa);
    background:
      radial-gradient(circle at 15% 0%, rgba(124, 58, 237, .13), transparent 27%),
      radial-gradient(circle at 85% 0%, rgba(37, 99, 235, .11), transparent 29%),
      var(--flgt-bg);
    border: 1px solid var(--flgt-border);
    border-radius: 12px;
    box-sizing: border-box;
    color: var(--flgt-text);
    display: grid;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    gap: 10px;
    grid-template-rows: auto minmax(230px, 1fr) auto;
    height: 100%;
    min-height: 410px;
    overflow: hidden;
    padding: 11px;
    position: relative;
    width: 100%;
  }
  .flgt-console * { box-sizing: border-box; }
  .flgt-console button,
  .flgt-console input,
  .flgt-console select,
  .flgt-console textarea,
  .flgt-modal button,
  .flgt-modal input,
  .flgt-modal select,
  .flgt-modal textarea { font: inherit; }
  .flgt-header {
    align-items: center;
    display: flex;
    gap: 10px;
    min-height: 38px;
  }
  .flgt-brand {
    align-items: center;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    border-radius: 8px;
    box-shadow: 0 5px 18px rgba(37, 99, 235, .24);
    color: white;
    display: flex;
    font-size: 12px;
    font-weight: 800;
    height: 30px;
    justify-content: center;
    letter-spacing: .05em;
    width: 34px;
  }
  .flgt-heading { min-width: 150px; }
  .flgt-title { font-size: 13px; font-weight: 750; line-height: 1.1; }
  .flgt-subtitle { color: var(--flgt-muted); font-size: 9px; margin-top: 3px; }
  .flgt-badge {
    background: rgba(59, 130, 246, .12);
    border: 1px solid rgba(96, 165, 250, .34);
    border-radius: 999px;
    color: #bfdbfe;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: .06em;
    padding: 4px 7px;
    text-transform: uppercase;
  }
  .flgt-status {
    align-items: center;
    color: var(--flgt-muted);
    display: flex;
    font-size: 10px;
    gap: 6px;
    margin-left: auto;
    min-width: 96px;
  }
  .flgt-status::before {
    background: #71717a;
    border-radius: 50%;
    box-shadow: 0 0 0 3px rgba(113, 113, 122, .13);
    content: "";
    height: 7px;
    width: 7px;
  }
  .flgt-status.ready::before { background: #60a5fa; box-shadow: 0 0 0 3px rgba(96, 165, 250, .13); }
  .flgt-status.running { color: #fde68a; }
  .flgt-status.running::before {
    animation: flgt-pulse 1s ease-in-out infinite;
    background: #fbbf24;
    box-shadow: 0 0 0 3px rgba(251, 191, 36, .14);
  }
  .flgt-status.complete { color: #86efac; }
  .flgt-status.complete::before { background: #4ade80; box-shadow: 0 0 0 3px rgba(74, 222, 128, .14); }
  .flgt-status.error { color: #fca5a5; }
  .flgt-status.error::before { background: #f87171; box-shadow: 0 0 0 3px rgba(248, 113, 113, .14); }
  .flgt-elapsed { color: var(--flgt-muted); font-variant-numeric: tabular-nums; min-width: 34px; }
  @keyframes flgt-pulse { 50% { opacity: .45; transform: scale(.82); } }
  .flgt-actions { display: flex; gap: 6px; }
  .flgt-button {
    align-items: center;
    background: rgba(39, 39, 42, .86);
    border: 1px solid #4b5160;
    border-radius: 7px;
    color: #f4f4f5;
    cursor: pointer;
    display: inline-flex;
    font-size: 10px;
    font-weight: 650;
    gap: 5px;
    justify-content: center;
    min-height: 27px;
    padding: 5px 9px;
  }
  .flgt-button:hover:not(:disabled) { background: #343945; border-color: #687083; }
  .flgt-button:disabled { cursor: default; opacity: .45; }
  .flgt-button.primary {
    background: linear-gradient(135deg, #6d28d9, #2563eb);
    border-color: #6366f1;
    box-shadow: 0 4px 14px rgba(67, 56, 202, .22);
  }
  .flgt-button.primary:hover:not(:disabled) { background: linear-gradient(135deg, #7c3aed, #3b82f6); }
  .flgt-button.small { min-height: 22px; padding: 3px 7px; }
  .flgt-workspace {
    display: grid;
    gap: 9px;
    grid-template-columns: minmax(210px, .8fr) minmax(260px, 1.12fr) minmax(290px, 1.2fr);
    min-height: 0;
  }
  .flgt-card {
    background: rgba(19, 22, 28, .9);
    border: 1px solid var(--flgt-border);
    border-radius: 9px;
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
    position: relative;
  }
  .flgt-card::before { content: ""; height: 2px; left: 0; position: absolute; right: 0; top: 0; }
  .flgt-card.system::before { background: linear-gradient(90deg, #a855f7, #7c3aed); }
  .flgt-card.user::before { background: linear-gradient(90deg, #3b82f6, #06b6d4); }
  .flgt-card.output::before { background: linear-gradient(90deg, #10b981, #84cc16); }
  .flgt-card-head {
    align-items: center;
    border-bottom: 1px solid rgba(82, 82, 91, .45);
    display: flex;
    gap: 7px;
    min-height: 34px;
    padding: 7px 9px 6px;
  }
  .flgt-role {
    border-radius: 5px;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: .08em;
    padding: 3px 5px;
    text-transform: uppercase;
  }
  .system .flgt-role { background: rgba(168, 85, 247, .14); color: #d8b4fe; }
  .user .flgt-role { background: rgba(59, 130, 246, .14); color: #bfdbfe; }
  .output .flgt-role { background: rgba(16, 185, 129, .14); color: #a7f3d0; }
  .flgt-count { color: var(--flgt-muted); font-size: 9px; margin-left: auto; }
  .flgt-card textarea {
    background: transparent;
    border: 0;
    color: var(--flgt-text);
    flex: 1 1 auto;
    font-size: 11px;
    line-height: 1.5;
    min-height: 0;
    outline: none;
    padding: 10px;
    resize: none;
    width: 100%;
  }
  .flgt-card textarea::placeholder { color: #626979; }
  .flgt-card:focus-within { border-color: #6366f1; box-shadow: 0 0 0 1px rgba(99, 102, 241, .35); }
  .flgt-output-body {
    color: var(--flgt-text);
    flex: 1 1 auto;
    font-size: 11px;
    line-height: 1.5;
    min-height: 0;
    overflow: auto;
    padding: 10px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .flgt-output-body.empty { color: var(--flgt-muted); font-style: italic; }
  .flgt-output-foot {
    align-items: center;
    border-top: 1px solid rgba(82, 82, 91, .35);
    color: var(--flgt-muted);
    display: flex;
    font-size: 8px;
    justify-content: space-between;
    min-height: 24px;
    padding: 4px 8px;
  }
  .flgt-output-actions { display: flex; gap: 4px; margin-left: 4px; }
  .flgt-deck {
    background: rgba(19, 22, 28, .92);
    border: 1px solid var(--flgt-border);
    border-radius: 9px;
    padding: 7px 9px 8px;
  }
  .flgt-deck-head {
    align-items: center;
    color: var(--flgt-muted);
    display: flex;
    font-size: 8px;
    font-weight: 700;
    justify-content: space-between;
    letter-spacing: .08em;
    margin-bottom: 5px;
    text-transform: uppercase;
  }
  .flgt-deck-row { display: grid; gap: 10px; grid-template-columns: minmax(390px, 1.2fr) minmax(540px, 1.8fr); }
  .flgt-control-group { display: grid; gap: 6px; }
  .flgt-control-group.primary { grid-template-columns: 1fr .8fr .92fr .78fr .8fr; }
  .flgt-control-group.sampling { grid-template-columns: repeat(6, minmax(64px, 1fr)); transition: opacity .15s ease; }
  .flgt-control-group.sampling.disabled { opacity: .38; }
  .flgt-control { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
  .flgt-control span { color: var(--flgt-muted); font-size: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .flgt-control input,
  .flgt-control select,
  .flgt-modal input,
  .flgt-modal select,
  .flgt-modal textarea {
    background: var(--comfy-input-bg, #101218);
    border: 1px solid #3c4250;
    border-radius: 5px;
    color: var(--flgt-text, #edf0f7);
    outline: none;
  }
  .flgt-control input,
  .flgt-control select { height: 25px; min-width: 0; padding: 2px 5px; width: 100%; }
  .flgt-control input:focus,
  .flgt-control select:focus,
  .flgt-modal input:focus,
  .flgt-modal select:focus,
  .flgt-modal textarea:focus { border-color: #6366f1; box-shadow: 0 0 0 1px rgba(99, 102, 241, .42); }
  .flgt-control input:disabled { color: #737b8d; }
  .flgt-modal-backdrop {
    align-items: center;
    background: rgba(3, 5, 9, .78);
    display: flex;
    inset: 0;
    justify-content: center;
    padding: 24px;
    position: fixed;
    z-index: 10000;
  }
  .flgt-modal {
    --flgt-text: var(--input-text, #edf0f7);
    background: var(--comfy-menu-bg, #14171d);
    border: 1px solid var(--border-color, #424856);
    border-radius: 12px;
    box-shadow: 0 24px 80px rgba(0, 0, 0, .58);
    color: var(--flgt-text);
    display: flex;
    flex-direction: column;
    gap: 13px;
    max-height: calc(100vh - 48px);
    max-width: 1120px;
    overflow: auto;
    padding: 16px;
    width: min(1120px, calc(100vw - 48px));
  }
  .flgt-modal-head, .flgt-modal-actions { align-items: center; display: flex; justify-content: space-between; }
  .flgt-modal h2 { font-size: 16px; margin: 0; }
  .flgt-modal-subtitle { color: var(--descrip-text, #8f98aa); font-size: 10px; margin-top: 3px; }
  .flgt-modal-editor { display: grid; gap: 10px; grid-template-columns: minmax(0, .82fr) minmax(0, 1.18fr); }
  .flgt-modal-field { display: flex; flex-direction: column; gap: 5px; min-height: 320px; }
  .flgt-modal-label { color: var(--descrip-text, #8f98aa); font-size: 9px; font-weight: 750; letter-spacing: .08em; text-transform: uppercase; }
  .flgt-modal-field textarea { flex: 1 1 auto; line-height: 1.5; min-height: 285px; padding: 10px; resize: vertical; width: 100%; }
  .flgt-modal-controls { display: grid; gap: 10px; grid-template-columns: minmax(390px, 1.2fr) minmax(520px, 1.8fr); }
  .flgt-modal-actions { border-top: 1px solid var(--border-color, #343946); gap: 7px; justify-content: flex-end; padding-top: 11px; }
  @container flgt-node (max-width: 940px) {
    .flgt-console { grid-template-rows: auto minmax(370px, 1fr) auto; }
    .flgt-workspace { grid-template-columns: minmax(210px, .85fr) minmax(300px, 1.15fr); }
    .flgt-card.output { grid-column: 1 / -1; min-height: 150px; }
    .flgt-deck-row { grid-template-columns: 1fr; }
  }
  @container flgt-node (max-width: 680px) {
    .flgt-header { flex-wrap: wrap; }
    .flgt-status { margin-left: 0; }
    .flgt-actions { margin-left: auto; }
    .flgt-workspace { grid-template-columns: 1fr; }
    .flgt-card.output { grid-column: auto; }
    .flgt-control-group.primary { grid-template-columns: repeat(3, minmax(70px, 1fr)); }
    .flgt-control-group.sampling { grid-template-columns: repeat(3, minmax(70px, 1fr)); }
  }
  @media (max-width: 760px) {
    .flgt-modal-editor { grid-template-columns: 1fr; }
    .flgt-modal-field { min-height: 210px; }
    .flgt-modal-field textarea { min-height: 180px; }
    .flgt-modal-controls { grid-template-columns: 1fr; }
  }
`;

function injectStyles() {
  if (document.getElementById("flgt-styles")) return;
  const style = document.createElement("style");
  style.id = "flgt-styles";
  style.textContent = STYLES;
  document.head.appendChild(style);
}

function nodeKey(id) {
  return String(id ?? "");
}

function eventNode(detail) {
  if (detail && typeof detail === "object") return detail.node ?? detail.node_id;
  return detail;
}

function findWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function hideWidget(widget) {
  if (!widget) return;
  widget.computeSize = () => [0, -4];
  widget.draw = () => {};
  if (widget.element) widget.element.style.display = "none";
}

function setWidgetValue(node, widget, value) {
  if (!widget || Object.is(widget.value, value)) return;
  widget.value = value;
  widget.callback?.call(widget, value);
  node.graph?.change?.();
  node.setDirtyCanvas?.(true, false);
}

function executionText(message) {
  const value = message?.generated_text ?? message?.ui?.generated_text;
  if (Array.isArray(value)) return value.length ? String(value[0] ?? "") : "";
  return value == null ? "" : String(value);
}

function selectOptions(name) {
  if (name === "sampling") return '<option value="on">On</option><option value="off">Off</option>';
  if (name === "thinking") return '<option value="false">Off</option><option value="true">On</option>';
  if (name === "control_after_generate") {
    return '<option value="fixed">Fixed</option><option value="increment">Increment</option><option value="decrement">Decrement</option><option value="randomize">Randomize</option>';
  }
  return "";
}

function controlMarkup(name, label, attributes = "") {
  const select = ["sampling", "thinking", "control_after_generate"].includes(name);
  const field = select
    ? `<select data-field="${name}">${selectOptions(name)}</select>`
    : `<input data-field="${name}" type="number" ${attributes}>`;
  return `<label class="flgt-control"><span title="${label}">${label}</span>${field}</label>`;
}

function primaryControlsMarkup() {
  return `<div class="flgt-control-group primary">
    ${controlMarkup("max_length", "Max tokens", 'min="1" max="32768" step="1"')}
    ${controlMarkup("sampling", "Sampling")}
    ${controlMarkup("seed", "Seed", 'min="0" step="1"')}
    ${controlMarkup("control_after_generate", "After run")}
    ${controlMarkup("thinking", "Thinking")}
  </div>`;
}

function samplingControlsMarkup() {
  return `<div class="flgt-control-group sampling" data-role="sampling-controls">
    ${controlMarkup("temperature", "Temperature", 'min="0.01" max="2" step="0.01" data-sampling-control')}
    ${controlMarkup("top_k", "Top K", 'min="0" max="1000" step="1" data-sampling-control')}
    ${controlMarkup("top_p", "Top P", 'min="0" max="1" step="0.01" data-sampling-control')}
    ${controlMarkup("min_p", "Min P", 'min="0" max="1" step="0.01" data-sampling-control')}
    ${controlMarkup("repetition_penalty", "Repetition", 'min="0" max="5" step="0.01" data-sampling-control')}
    ${controlMarkup("presence_penalty", "Presence", 'min="0" max="5" step="0.01" data-sampling-control')}
  </div>`;
}

function modalControlsMarkup() {
  return `<div class="flgt-modal-controls">
    ${primaryControlsMarkup()}
    ${samplingControlsMarkup()}
  </div>`;
}

function readControl(element, name) {
  if (name === "thinking") return element.value === "true";
  if (INTEGER_FIELDS.has(name)) return Number.parseInt(element.value, 10);
  if (FLOAT_FIELDS.has(name)) return Number.parseFloat(element.value);
  return element.value;
}

async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch (_error) {
    }
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

function applyWideLayout(node, force = false) {
  node.min_size = [...MIN_NODE_SIZE];
  window.requestAnimationFrame(() => {
    if (!node.graph) return;
    if (force || node.size[0] < MIN_NODE_SIZE[0]) {
      node.setSize([
        Math.max(DEFAULT_NODE_SIZE[0], node.size[0]),
        Math.max(DEFAULT_NODE_SIZE[1], Math.min(node.size[1], 620)),
      ]);
    }
  });
}

function registerInstance(panel) {
  for (const [key, instance] of INSTANCES) {
    if (instance === panel) INSTANCES.delete(key);
  }
  INSTANCES.set(nodeKey(panel.node.id), panel);
}

function instanceForNode(id) {
  const key = nodeKey(id);
  const direct = INSTANCES.get(key);
  if (direct) return direct;
  for (const panel of INSTANCES.values()) {
    if (nodeKey(panel.node.id) === key) {
      registerInstance(panel);
      return panel;
    }
  }
  return null;
}

class GenerateTextPanel {
  constructor(node, host, widgets) {
    this.node = node;
    this.host = host;
    this.widgets = widgets;
    this.output = "";
    this.modal = null;
    this.active = false;
    this.startedAt = null;
    this.timer = null;
    this.cleanups = [];
    this.build();
    this.bind();
    this.syncFromWidgets();
    this.updateConnection();
    this.observeSize();
  }

  build() {
    this.host.className = "flgt-host";
    this.host.innerHTML = `<section class="flgt-console" data-layout="wide">
      <header class="flgt-header">
        <span class="flgt-brand">FL</span>
        <div class="flgt-heading"><div class="flgt-title">Generate Text</div><div class="flgt-subtitle">Local system + user chat workspace</div></div>
        <span class="flgt-badge">Qwen3 chat</span>
        <span class="flgt-badge" title="Requires a complete language model checkpoint">Full LM required</span>
        <span class="flgt-status" data-role="status" title="Connect a generation-capable CLIP">Connect CLIP</span>
        <span class="flgt-elapsed" data-role="elapsed"></span>
        <div class="flgt-actions">
          <button class="flgt-button" data-action="focus" type="button" title="Open a larger editing workspace">Focus</button>
          <button class="flgt-button primary" data-action="generate" type="button" title="Queue the current ComfyUI workflow">Generate</button>
        </div>
      </header>
      <div class="flgt-workspace">
        <section class="flgt-card system">
          <div class="flgt-card-head"><span class="flgt-role">System</span><span class="flgt-count" data-count="system_prompt">0 chars</span></div>
          <textarea data-field="system_prompt" aria-label="System prompt" spellcheck="true" placeholder="Define the model's role and rules."></textarea>
        </section>
        <section class="flgt-card user">
          <div class="flgt-card-head"><span class="flgt-role">User</span><span class="flgt-count" data-count="prompt">0 chars</span></div>
          <textarea data-field="prompt" aria-label="User prompt" spellcheck="true" placeholder="What should the model generate?"></textarea>
        </section>
        <section class="flgt-card output">
          <div class="flgt-card-head">
            <span class="flgt-role">Output</span>
            <span class="flgt-count" data-count="output">0 chars</span>
            <div class="flgt-output-actions">
              <button class="flgt-button small" data-action="copy" type="button" disabled>Copy</button>
              <button class="flgt-button small" data-action="clear" type="button" disabled>Clear</button>
            </div>
          </div>
          <div class="flgt-output-body empty" data-role="output">Run the workflow to generate text.</div>
          <div class="flgt-output-foot"><span>Ephemeral preview</span><span>STRING output remains workflow-owned</span></div>
        </section>
      </div>
      <section class="flgt-deck">
        <div class="flgt-deck-head"><span>Generation controls</span><span data-role="sampling-hint">Sampling enabled</span></div>
        <div class="flgt-deck-row">${primaryControlsMarkup()}${samplingControlsMarkup()}</div>
      </section>
    </section>`;
    this.root = this.host.firstElementChild;
    this.statusEl = this.root.querySelector('[data-role="status"]');
    this.elapsedEl = this.root.querySelector('[data-role="elapsed"]');
    this.outputEl = this.root.querySelector('[data-role="output"]');
    this.copyButton = this.root.querySelector('[data-action="copy"]');
    this.clearButton = this.root.querySelector('[data-action="clear"]');
    this.generateButton = this.root.querySelector('[data-action="generate"]');
    this.controls = new Map(UI_FIELDS.map((name) => [name, this.root.querySelector(`[data-field="${name}"]`)]));
  }

  listen(element, event, handler) {
    if (!element) return;
    element.addEventListener(event, handler);
    this.cleanups.push(() => element.removeEventListener(event, handler));
  }

  bind() {
    this.listen(this.root, "pointerdown", (event) => event.stopPropagation());
    this.listen(this.root, "wheel", (event) => event.stopPropagation());
    this.listen(this.root, "keydown", (event) => event.stopPropagation());

    for (const [name, element] of this.controls) {
      const eventName = name === "system_prompt" || name === "prompt" ? "input" : "change";
      this.listen(element, eventName, () => {
        const value = readControl(element, name);
        if (typeof value === "number" && !Number.isFinite(value)) return;
        setWidgetValue(this.node, this.widgets[name], value);
        if (name === "system_prompt" || name === "prompt") this.updateCount(name, value);
        if (name === "sampling") this.updateSamplingControls();
      });
    }

    this.listen(this.root.querySelector('[data-action="focus"]'), "click", () => this.openFocusMode());
    this.listen(this.generateButton, "click", () => this.queueGeneration());
    this.listen(this.copyButton, "click", async () => {
      if (!this.output) return;
      await copyText(this.output);
      this.copyButton.textContent = "Copied";
      window.setTimeout(() => { this.copyButton.textContent = "Copy"; }, 900);
    });
    this.listen(this.clearButton, "click", () => this.clearOutput());
  }

  observeSize() {
    if (typeof ResizeObserver === "undefined") return;
    this.resizeObserver = new ResizeObserver(([entry]) => {
      const width = entry.contentRect.width;
      this.root.dataset.layout = width >= 940 ? "wide" : width >= 680 ? "medium" : "compact";
    });
    this.resizeObserver.observe(this.host);
  }

  syncFromWidgets() {
    for (const [name, element] of this.controls) {
      if (!element) continue;
      const value = this.widgets[name]?.value;
      element.value = name === "thinking" ? String(Boolean(value)) : String(value ?? "");
    }
    this.updateCount("system_prompt", this.widgets.system_prompt?.value ?? "");
    this.updateCount("prompt", this.widgets.prompt?.value ?? "");
    this.updateSamplingControls();
  }

  updateCount(name, value) {
    const count = this.root.querySelector(`[data-count="${name}"]`);
    if (count) count.textContent = `${String(value).length} chars`;
  }

  updateSamplingControls(root = this.root) {
    const enabled = root.querySelector('[data-field="sampling"]')?.value === "on";
    const group = root.querySelector('[data-role="sampling-controls"]');
    group?.classList.toggle("disabled", !enabled);
    for (const control of root.querySelectorAll("[data-sampling-control]")) control.disabled = !enabled;
    const hint = root.querySelector('[data-role="sampling-hint"]');
    if (hint) hint.textContent = enabled ? "Sampling enabled" : "Deterministic / sampling disabled";
  }

  clipConnected() {
    return this.node.inputs?.find((input) => input.name === "clip")?.link != null;
  }

  updateConnection() {
    if (this.active) return;
    if (this.clipConnected()) this.setStatus("Ready", "ready");
    else this.setStatus("Connect CLIP");
  }

  setStatus(text, state = "") {
    this.statusEl.textContent = text;
    this.statusEl.title = text;
    this.statusEl.className = `flgt-status${state ? ` ${state}` : ""}`;
  }

  startTimer(reset = false) {
    if (reset || this.startedAt == null) this.startedAt = Date.now();
    window.clearInterval(this.timer);
    const update = () => {
      const elapsed = Math.max(0, Date.now() - this.startedAt);
      this.elapsedEl.textContent = `${(elapsed / 1000).toFixed(1)}s`;
    };
    update();
    this.timer = window.setInterval(update, 100);
  }

  stopTimer() {
    window.clearInterval(this.timer);
    this.timer = null;
  }

  setActive(active) {
    this.active = active;
    this.generateButton.disabled = active;
    this.generateButton.textContent = active ? "Working..." : "Generate";
  }

  async queueGeneration() {
    if (this.active) return;
    if (!this.clipConnected()) {
      this.fail("Connect a generation-capable CLIP first");
      return;
    }
    this.setActive(true);
    this.setStatus("Queued", "running");
    this.startTimer(true);
    try {
      await app.queuePrompt(0, 1);
    } catch (error) {
      this.fail(error?.message || "Could not queue workflow");
    }
  }

  beginExecution() {
    this.setActive(true);
    this.setStatus("Generating", "running");
    this.startTimer(false);
  }

  markCached() {
    this.setActive(false);
    this.stopTimer();
    this.setStatus("Cached", "complete");
  }

  fail(message) {
    this.setActive(false);
    this.stopTimer();
    this.setStatus(message || "Generation failed", "error");
  }

  showOutput(text) {
    this.output = text;
    this.setActive(false);
    this.stopTimer();
    if (text) {
      this.outputEl.textContent = text;
      this.outputEl.classList.remove("empty");
      this.copyButton.disabled = false;
      this.clearButton.disabled = false;
    } else {
      this.outputEl.textContent = "The model returned an empty response.";
      this.outputEl.classList.add("empty");
      this.copyButton.disabled = true;
      this.clearButton.disabled = false;
    }
    this.updateCount("output", text);
    this.setStatus("Complete", "complete");
  }

  clearOutput() {
    this.output = "";
    this.outputEl.textContent = "Run the workflow to generate text.";
    this.outputEl.classList.add("empty");
    this.copyButton.disabled = true;
    this.clearButton.disabled = true;
    this.updateCount("output", "");
    this.elapsedEl.textContent = "";
    this.updateConnection();
  }

  openFocusMode() {
    if (this.modal) return;
    const backdrop = document.createElement("div");
    backdrop.className = "flgt-modal-backdrop";
    backdrop.innerHTML = `<div class="flgt-modal" role="dialog" aria-modal="true" aria-label="FL Generate Text Focus Mode">
      <div class="flgt-modal-head">
        <div><h2>FL Generate Text</h2><div class="flgt-modal-subtitle">Focused Qwen3 system and user workspace</div></div>
        <button class="flgt-button" data-action="cancel" type="button">Close</button>
      </div>
      <div class="flgt-modal-editor">
        <label class="flgt-modal-field"><span class="flgt-modal-label">System</span><textarea data-field="system_prompt" aria-label="Focus system prompt" spellcheck="true"></textarea></label>
        <label class="flgt-modal-field"><span class="flgt-modal-label">User</span><textarea data-field="prompt" aria-label="Focus user prompt" spellcheck="true"></textarea></label>
      </div>
      ${modalControlsMarkup()}
      <div class="flgt-modal-actions">
        <button class="flgt-button" data-action="cancel" type="button">Cancel</button>
        <button class="flgt-button" data-action="apply" type="button">Apply</button>
        <button class="flgt-button primary" data-action="apply-generate" type="button">Apply + Generate</button>
      </div>
    </div>`;
    document.body.appendChild(backdrop);
    this.modal = backdrop;
    const dialog = backdrop.firstElementChild;
    const controls = new Map(UI_FIELDS.map((name) => [name, dialog.querySelector(`[data-field="${name}"]`)]));
    for (const [name, element] of controls) {
      if (!element) continue;
      const value = this.widgets[name]?.value;
      element.value = name === "thinking" ? String(Boolean(value)) : String(value ?? "");
    }
    this.updateSamplingControls(dialog);

    const close = () => this.closeFocusMode();
    const apply = () => {
      for (const [name, element] of controls) {
        if (!element) continue;
        const value = readControl(element, name);
        if (typeof value === "number" && !Number.isFinite(value)) continue;
        setWidgetValue(this.node, this.widgets[name], value);
      }
      this.syncFromWidgets();
    };
    for (const button of dialog.querySelectorAll('[data-action="cancel"]')) button.addEventListener("click", close);
    dialog.querySelector('[data-field="sampling"]')?.addEventListener("change", () => this.updateSamplingControls(dialog));
    dialog.querySelector('[data-action="apply"]').addEventListener("click", () => { apply(); close(); });
    dialog.querySelector('[data-action="apply-generate"]').addEventListener("click", () => { apply(); close(); this.queueGeneration(); });
    backdrop.addEventListener("pointerdown", (event) => {
      event.stopPropagation();
      if (event.target === backdrop) close();
    });
    dialog.addEventListener("keydown", (event) => event.stopPropagation());
    this.modalKeyHandler = (event) => {
      if (event.key === "Escape") close();
    };
    document.addEventListener("keydown", this.modalKeyHandler, true);
    controls.get("prompt")?.focus();
  }

  closeFocusMode() {
    if (!this.modal) return;
    document.removeEventListener("keydown", this.modalKeyHandler, true);
    this.modal.remove();
    this.modal = null;
    this.modalKeyHandler = null;
  }

  dispose() {
    this.closeFocusMode();
    this.stopTimer();
    this.resizeObserver?.disconnect();
    for (const cleanup of this.cleanups) cleanup();
    this.cleanups = [];
    this.host.remove();
  }
}

function removeInstance(node) {
  const panel = node._flGenerateTextPanel;
  if (!panel) return;
  node._flGenerateTextPanel = null;
  for (const [key, instance] of INSTANCES) {
    if (instance === panel) INSTANCES.delete(key);
  }
  panel.dispose();
}

app.registerExtension({
  name: "ComfyUI.FL_GenerateText",
  nodeCreated(node) {
    const comfyClass = node.constructor?.comfyClass || "";
    if (comfyClass !== NODE_CLASS) return;

    injectStyles();
    const previousLayout = Number(node.properties?.flGenerateTextLayoutVersion || 0);
    node.properties = node.properties || {};
    node.properties.flGenerateTextLayoutVersion = LAYOUT_VERSION;

    const widgets = Object.fromEntries(UI_FIELDS.map((name) => [name, findWidget(node, name)]));
    for (const widget of Object.values(widgets)) hideWidget(widget);

    const host = document.createElement("div");
    const domWidget = node.addDOMWidget("fl_generate_text_console", "fl-generate-text", host, {
      getMinHeight: () => 410,
      hideOnZoom: false,
      serialize: false,
    });

    const panel = new GenerateTextPanel(node, host, widgets);
    node._flGenerateTextPanel = panel;
    window.setTimeout(() => {
      if (node._flGenerateTextPanel !== panel) return;
      if (!node.graph) {
        removeInstance(node);
        return;
      }
      registerInstance(panel);
      applyWideLayout(node, previousLayout < LAYOUT_VERSION);
    }, 0);

    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      originalOnExecuted?.apply(this, arguments);
      panel.showOutput(executionText(message));
    };

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
      const result = originalOnConfigure?.apply(this, arguments);
      this.properties = this.properties || {};
      const needsMigration = Number(this.properties.flGenerateTextLayoutVersion || 0) < LAYOUT_VERSION;
      this.properties.flGenerateTextLayoutVersion = LAYOUT_VERSION;
      for (const widget of Object.values(widgets)) hideWidget(widget);
      window.setTimeout(() => {
        panel.syncFromWidgets();
        registerInstance(panel);
        applyWideLayout(this, needsMigration);
      }, 0);
      return result;
    };

    const originalOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function () {
      const result = originalOnConnectionsChange?.apply(this, arguments);
      panel.updateConnection();
      return result;
    };

    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
      removeInstance(this);
      return originalOnRemoved?.apply(this, arguments);
    };
    domWidget.onRemove = () => removeInstance(node);
  },
});

api.addEventListener("executing", (event) => {
  instanceForNode(eventNode(event.detail))?.beginExecution();
});

api.addEventListener("execution_cached", (event) => {
  const nodes = Array.isArray(event.detail?.nodes) ? event.detail.nodes : [];
  for (const nodeId of nodes) instanceForNode(nodeId)?.markCached();
});

api.addEventListener("execution_error", (event) => {
  const detail = event.detail || {};
  instanceForNode(eventNode(detail))?.fail(detail.exception_message || detail.exception_type);
});

api.addEventListener("execution_interrupted", () => {
  for (const panel of INSTANCES.values()) {
    if (panel.active) panel.fail("Interrupted");
  }
});
