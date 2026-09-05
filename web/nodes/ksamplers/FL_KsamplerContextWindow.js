import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const STYLES = `
  .flks-context-widget {
    background: #17181c;
    border: 1px solid #2a2d34;
    border-radius: 8px;
    color: #f4f4f5;
    display: flex;
    flex-direction: column;
    font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
    gap: 5px;
    padding: 8px;
    box-sizing: border-box;
  }
  .flks-context-widget * { box-sizing: border-box; }
  .flks-context-header {
    align-items: center;
    display: flex;
    justify-content: space-between;
    gap: 8px;
  }
  .flks-context-title {
    font-size: 11px;
    font-weight: 650;
    line-height: 1.2;
  }
  .flks-context-badge {
    background: #06b6d4;
    border-radius: 999px;
    color: white;
    font-size: 10px;
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    line-height: 1;
    padding: 4px 7px;
    white-space: nowrap;
  }
  .flks-context-bar {
    background: #27272a;
    border-radius: 999px;
    height: 9px;
    overflow: hidden;
    width: 100%;
  }
  .flks-context-fill {
    background: linear-gradient(90deg, #06b6d4, #22c55e);
    height: 100%;
    transition: width 120ms linear;
    width: 0%;
  }
  .flks-context-meta {
    color: #cbd5e1;
    display: grid;
    gap: 4px;
    grid-template-columns: 1fr 1fr;
    font-size: 10px;
    font-variant-numeric: tabular-nums;
    line-height: 1.25;
  }
  .flks-context-window {
    color: #94a3b8;
    font-size: 10px;
    line-height: 1.25;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .flks-context-advanced {
    background: transparent;
    border: 0;
    color: #cbd5e1;
    cursor: pointer;
    font: inherit;
    font-size: 10px;
    padding: 0;
    text-align: left;
  }
`;

const ADVANCED_DEFAULTS = {
  context_schedule: "standard_static",
  context_stride: 1,
  fuse_method: "pyramid",
  temporal_unit: "auto",
  closed_loop: false,
  freenoise: false,
  causal_window_fix: true,
  temporal_dim: 2,
  cond_retain_index_list: "",
  split_conds_to_windows: false,
};

const findWidget = (node, name) => node.widgets?.find((widget) => widget.name === name);
const linkedInput = (node, name) => node.inputs?.some((input) => input.name === name && input.link != null);

export function visibleContextControls(node, expanded) {
  const schedule = findWidget(node, "context_schedule")?.value;
  const scheduleLinked = linkedInput(node, "context_schedule");
  return Object.fromEntries(Object.keys(ADVANCED_DEFAULTS).map((name) => {
    let relevant = true;
    if (name === "context_stride") relevant = scheduleLinked || schedule?.endsWith("_uniform");
    if (name === "closed_loop") relevant = scheduleLinked || schedule === "looped_uniform";
    return [name, Boolean(expanded && relevant)];
  }));
}

export function advancedOverrides(node) {
  return Object.entries(ADVANCED_DEFAULTS).filter(([name, value]) =>
    linkedInput(node, name) || (findWidget(node, name) && findWidget(node, name).value !== value)
  ).map(([name]) => name);
}

const HIDDEN_WIDGETS = new WeakMap();

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  if (widget.type === "converted-widget" && !HIDDEN_WIDGETS.has(widget)) return;
  if (!visible && !HIDDEN_WIDGETS.has(widget)) {
    HIDDEN_WIDGETS.set(widget, { type: widget.type, computeSize: widget.computeSize, hidden: widget.hidden });
    widget.type = "converted-widget";
    widget.computeSize = () => [0, -4];
    widget.hidden = true;
    if (widget.element) widget.element.style.display = "none";
  } else if (visible && HIDDEN_WIDGETS.has(widget)) {
    Object.assign(widget, HIDDEN_WIDGETS.get(widget));
    HIDDEN_WIDGETS.delete(widget);
    if (widget.element) widget.element.style.display = "";
  }
}

class ContextWindowProgressWidget {
  constructor({ container, node }) {
    this.container = container;
    this.node = node;
    this.injectStyles();
    this.element = document.createElement("div");
    this.element.className = "flks-context-widget";
    this.element.innerHTML = `
      <div class="flks-context-header">
        <span class="flks-context-title" data-role="model">Auto · resolves when sampling</span>
        <span class="flks-context-badge" data-role="percent">idle</span>
      </div>
      <div class="flks-context-bar">
        <div class="flks-context-fill" data-role="fill"></div>
      </div>
      <div class="flks-context-meta">
        <span data-role="step">step - / -</span>
        <span data-role="window">window - / -</span>
      </div>
      <div class="flks-context-window" data-role="indices">Window settings use video frames in Auto mode.</div>
      <button type="button" class="flks-context-advanced" data-role="advanced">▸ Advanced</button>
    `;
    this.percentEl = this.element.querySelector('[data-role="percent"]');
    this.fillEl = this.element.querySelector('[data-role="fill"]');
    this.stepEl = this.element.querySelector('[data-role="step"]');
    this.windowEl = this.element.querySelector('[data-role="window"]');
    this.indicesEl = this.element.querySelector('[data-role="indices"]');
    this.modelEl = this.element.querySelector('[data-role="model"]');
    this.advancedEl = this.element.querySelector('[data-role="advanced"]');
    this.advancedEl.addEventListener("click", () => {
      this.node.properties.fl_context_advanced = !this.node.properties.fl_context_advanced;
      this.refreshControls();
      this.node.graph?.change();
    });
    this.container.appendChild(this.element);
  }

  refreshControls() {
    const expanded = Boolean(this.node.properties.fl_context_advanced);
    for (const [name, visible] of Object.entries(visibleContextControls(this.node, expanded))) {
      // Converted inputs keep their socket and their frontend-managed widget state.
      if (!linkedInput(this.node, name)) {
        setWidgetVisible(findWidget(this.node, name), visible);
      }
    }
    const batched = findWidget(this.node, "context_schedule")?.value === "batched" && !linkedInput(this.node, "context_schedule");
    if (!linkedInput(this.node, "context_overlap")) {
      setWidgetVisible(findWidget(this.node, "context_overlap"), !batched);
    }
    const overrides = advancedOverrides(this.node);
    this.advancedEl.textContent = `${expanded ? "▾" : "▸"} Advanced${overrides.length ? ` · ${overrides.length} overrides` : ""}`;
    this.advancedEl.title = overrides.join(", ");
    this.advancedEl.setAttribute("aria-expanded", String(expanded));
    const size = this.node.computeSize();
    this.node.setSize([Math.max(this.node.size[0], 330), size[1]]);
    this.node.setDirtyCanvas(true, true);
  }

  markStale() {
    const mode = findWidget(this.node, "temporal_unit")?.value || "auto";
    this.modelEl.textContent = this.settings ? "Settings changed · run to resolve" : `${mode === "auto" ? "Auto" : mode} · resolves when sampling`;
    this.modelEl.title = "Model detection is performed by the backend when this node runs.";
    this.percentEl.textContent = "idle";
    this.fillEl.style.width = "0%";
    this.stepEl.textContent = "step - / -";
    this.windowEl.textContent = "window - / -";
    this.indicesEl.textContent = "Run to resolve window settings.";
  }

  injectStyles() {
    const id = "flks-context-window-styles";
    if (document.getElementById(id)) return;
    const style = document.createElement("style");
    style.id = id;
    style.textContent = STYLES;
    document.head.appendChild(style);
  }

  reset() {
    this.settings = null;
    this.modelEl.textContent = "Resolving model and window…";
    this.percentEl.textContent = "0%";
    this.fillEl.style.width = "0%";
    this.stepEl.textContent = "step 0 / -";
    this.windowEl.textContent = "window 0 / -";
    this.indicesEl.textContent = "Waiting for first context window...";
  }

  update(detail) {
    if (detail.status === "resolved") {
      this.settings = detail;
      const names = { wan: "Wan", ltx: "LTX", minimax_h3: "MiniMax H3", video_frames_4n_plus_1: "Legacy 4n+1", latent_frames: "Latent units" };
      const units = detail.video_frames == null ? "" : `${detail.video_frames} frames / `;
      this.modelEl.textContent = `${names[detail.profile] || detail.profile} · ${units}${detail.latent_length} latents`;
      this.modelEl.title = `Requested window: ${detail.requested_length}; overlap: ${detail.requested_overlap}. Effective overlap: ${detail.latent_overlap} latents${detail.overlap_frames == null ? "" : ` (up to ${detail.overlap_frames} frames)`}. Additional causal anchor: ${detail.anchor_latents} latent. Total clip: ${detail.total_latents} video latents. Window frame spans can vary at temporal boundaries.`;
      this.indicesEl.textContent = detail.context_active ? `Overlap ${detail.latent_overlap} latents · ${detail.schedule}${detail.anchor_latents ? " · +1 anchor" : ""}` : "Single window · clip duration preserved";
      return;
    }
    if (detail.status === "error") {
      this.percentEl.textContent = "error";
      this.indicesEl.textContent = detail.message || "Sampling failed. See the node error for details.";
      return;
    }
    const value = Number(detail.value || 0);
    const max = Math.max(1, Number(detail.max || 1));
    const pct = Math.max(0, Math.min(100, (value / max) * 100));
    this.percentEl.textContent = detail.status === "done" ? "done" : `${pct.toFixed(1)}%`;
    this.fillEl.style.width = `${pct}%`;
    this.stepEl.textContent = `step ${detail.step ?? "-"} / ${detail.total_steps ?? "-"}`;
    this.windowEl.textContent = `window ${detail.window_index ?? "-"} / ${detail.total_windows ?? "-"}`;

    const indices = Array.isArray(detail.window) ? detail.window : [];
    if (indices.length) {
      const first = indices[0];
      const last = indices[indices.length - 1];
      this.indicesEl.textContent = `latent frames ${first}-${last} (${indices.length})`;
    } else if (detail.status === "done") {
      this.indicesEl.textContent = "Sampling completed.";
      if (this.settings) this.modelEl.textContent = `Last run: ${this.modelEl.textContent}`;
    }
  }

  dispose() {
    this.element?.remove();
  }
}

const INSTANCES = new Map();
const nodeKey = (value) => String(value);

app.registerExtension({
  name: "ComfyUI.FL_KsamplerContextWindow",
  nodeCreated(node) {
    const comfyClass = (node.constructor && node.constructor.comfyClass) || "";
    if (comfyClass !== "FL_KsamplerContextWindow") return;

    const container = document.createElement("div");
    container.style.width = "100%";

    const widget = node.addDOMWidget(
      "context_progress",
      "flks-context-window-progress",
      container,
      {
        getMinHeight: () => 104,
        getMaxHeight: () => 104,
        hideOnZoom: false,
        serialize: false,
      }
    );

    node.properties ||= {};
    let inst;
    let registeredKey;
    const initTimer = setTimeout(() => {
      inst = new ContextWindowProgressWidget({ container, node });
      INSTANCES.set(nodeKey(node.id), inst);
      registeredKey = nodeKey(node.id);
      inst.markStale();
      inst.refreshControls();
    }, 50);

    for (const control of node.widgets || []) {
      if (control === widget) continue;
      const callback = control.callback;
      control.callback = function (...args) {
        const result = callback?.apply(this, args);
        inst?.markStale();
        inst?.refreshControls();
        return result;
      };
    }
    const onConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const result = onConfigure?.apply(this, args);
      if (inst) {
        if (INSTANCES.get(registeredKey) === inst) INSTANCES.delete(registeredKey);
        registeredKey = nodeKey(node.id);
        INSTANCES.set(registeredKey, inst);
        inst.markStale();
        inst.refreshControls();
      }
      return result;
    };
    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
      const result = onConnectionsChange?.apply(this, args);
      inst?.markStale();
      inst?.refreshControls();
      return result;
    };

    widget.onRemove = () => {
      clearTimeout(initTimer);
      if (inst) {
        inst.dispose();
        if (INSTANCES.get(registeredKey) === inst) INSTANCES.delete(registeredKey);
      }
    };
  },
});

api.addEventListener("executing", (event) => {
    const detail = event.detail;
  const id = detail?.node ?? detail;
  if (id == null) return;
  const inst = INSTANCES.get(nodeKey(id));
  if (inst) inst.reset();
});

api.addEventListener("execution_error", (event) => {
  const detail = event.detail;
  INSTANCES.get(nodeKey(detail?.node_id))?.update({ status: "error", message: detail?.exception_message });
});

api.addEventListener("fl_context_window_progress", (event) => {
  const detail = event.detail;
  if (!detail) return;
  const inst = INSTANCES.get(nodeKey(detail.node));
  if (!inst) return;
  inst.update(detail);
});
