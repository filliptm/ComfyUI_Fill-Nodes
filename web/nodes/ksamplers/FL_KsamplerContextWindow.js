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
    gap: 8px;
    min-height: 104px;
    padding: 10px;
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
`;

class ContextWindowProgressWidget {
  constructor({ container }) {
    this.container = container;
    this.injectStyles();
    this.element = document.createElement("div");
    this.element.className = "flks-context-widget";
    this.element.innerHTML = `
      <div class="flks-context-header">
        <span class="flks-context-title">Context Windows</span>
        <span class="flks-context-badge" data-role="percent">idle</span>
      </div>
      <div class="flks-context-bar">
        <div class="flks-context-fill" data-role="fill"></div>
      </div>
      <div class="flks-context-meta">
        <span data-role="step">step - / -</span>
        <span data-role="window">window - / -</span>
      </div>
      <div class="flks-context-window" data-role="indices">Run to see context-window progress.</div>
    `;
    this.percentEl = this.element.querySelector('[data-role="percent"]');
    this.fillEl = this.element.querySelector('[data-role="fill"]');
    this.stepEl = this.element.querySelector('[data-role="step"]');
    this.windowEl = this.element.querySelector('[data-role="window"]');
    this.indicesEl = this.element.querySelector('[data-role="indices"]');
    this.container.appendChild(this.element);
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
    this.percentEl.textContent = "0%";
    this.fillEl.style.width = "0%";
    this.stepEl.textContent = "step 0 / -";
    this.windowEl.textContent = "window 0 / -";
    this.indicesEl.textContent = "Waiting for first context window...";
  }

  update(detail) {
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
    }
  }

  dispose() {
    this.element?.remove();
  }
}

const INSTANCES = new Map();

app.registerExtension({
  name: "ComfyUI.FL_KsamplerContextWindow",
  nodeCreated(node) {
    const comfyClass = (node.constructor && node.constructor.comfyClass) || "";
    if (comfyClass !== "FL_KsamplerContextWindow") return;

    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.minHeight = "104px";

    const widget = node.addDOMWidget(
      "context_progress",
      "flks-context-window-progress",
      container,
      {
        getMinHeight: () => 130,
        hideOnZoom: false,
        serialize: false,
      }
    );

    const [oldW, oldH] = node.size;
    node.setSize([Math.max(oldW, 330), Math.max(oldH, 730)]);

    setTimeout(() => {
      const inst = new ContextWindowProgressWidget({ container });
      INSTANCES.set(node.id, inst);
    }, 50);

    widget.onRemove = () => {
      const inst = INSTANCES.get(node.id);
      if (inst) {
        inst.dispose();
        INSTANCES.delete(node.id);
      }
    };
  },
});

api.addEventListener("executing", (event) => {
  const detail = event.detail;
  if (!detail || !detail.node) return;
  const nodeId = parseInt(detail.node, 10);
  const inst = INSTANCES.get(nodeId);
  if (inst) inst.reset();
});

api.addEventListener("fl_context_window_progress", (event) => {
  const detail = event.detail;
  if (!detail) return;
  const nodeId = parseInt(detail.node, 10);
  const inst = INSTANCES.get(nodeId);
  if (!inst) return;
  inst.update(detail);
});
