import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const FLKS_STYLES = `
  .flks-widget {
    --primary: #06b6d4;
    --primary-glow: rgba(6, 182, 212, 0.4);
    --secondary: #8b5cf6;
    --success: #22c55e;
    --danger: #ef4444;
    --bg-dark: #0f0f12;
    --bg-card: #18181b;
    --bg-elevated: #1f1f23;
    --border: #27272a;
    --border-hover: #3f3f46;
    --text-primary: #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;

    background: var(--bg-card);
    border-radius: 10px;
    border: 1px solid var(--border);
    overflow: hidden;
    position: relative;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
    box-sizing: border-box;
    height: 100%;
    min-height: 260px;
    display: flex;
    flex-direction: column;
  }
  .flks-widget * { box-sizing: border-box; }

  .flks-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 9px;
    background: var(--bg-elevated);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    gap: 8px;
  }
  .flks-title {
    font-size: 11px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .flks-toggles {
    display: flex;
    gap: 6px;
    align-items: center;
  }
  .flks-toggle {
    background: var(--bg-dark);
    color: var(--text-secondary);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 9px;
    font-weight: 600;
    cursor: pointer;
    user-select: none;
    transition: all 0.15s ease;
  }
  .flks-toggle:hover { border-color: var(--border-hover); color: var(--text-primary); }
  .flks-toggle.on { background: var(--primary); color: white; border-color: var(--primary); }

  .flks-canvas-wrap {
    flex: 1;
    position: relative;
    background: var(--bg-dark);
    overflow: hidden;
    min-height: 160px;
  }
  .flks-canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: block;
    cursor: crosshair;
  }

  .flks-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 10px;
    background: var(--bg-elevated);
    border-top: 1px solid var(--border);
    font-size: 10px;
    color: var(--text-secondary);
    gap: 12px;
    flex-shrink: 0;
  }
  .flks-stat {
    display: flex;
    gap: 4px;
    align-items: baseline;
  }
  .flks-stat-label {
    font-size: 9px;
    color: var(--text-muted);
    text-transform: uppercase;
  }
  .flks-stat-value {
    font-variant-numeric: tabular-nums;
    color: var(--primary);
    font-weight: 600;
  }
  .flks-hint {
    color: var(--text-muted);
    font-size: 9px;
    font-style: italic;
  }
`;

// Fallback when no hint widgets are wired yet.
const FALLBACK_SIGMA_MAX = 14.6146;
const FALLBACK_SIGMA_MIN = 0.0292;

// ---------------------------------------------------------------------------
// Curve editor widget
// ---------------------------------------------------------------------------

class SigmaCurveEditor {
  constructor({ node, container, curveWidget, stepsWidget, sigmaMaxWidget, sigmaMinWidget }) {
    this.node = node;
    this.container = container;
    this.curveWidget = curveWidget;
    this.stepsWidget = stepsWidget;
    this.sigmaMaxWidget = sigmaMaxWidget;
    this.sigmaMinWidget = sigmaMinWidget;

    // Control points are stored as [t, sigma] with t in [0,1].
    const initMax = this.getAxisMax();
    this.points = [
      [0.0, initMax],
      [0.5, initMax * 0.3],
      [1.0, 0.0],
    ];
    this.logY = true;
    this.dragging = null;
    this.hoveredIdx = -1;

    this.resizeObserver = null;
    this.resizeTimeout = null;
    this.pendingRAF = null;

    this.element = document.createElement("div");
    this.element.className = "flks-widget";

    this.injectStyles();
    this.buildUI();
    this.container.appendChild(this.element);

    this.loadFromWidget();
    this.serialize();
    this.scheduleRedraw();

    if (this.stepsWidget) {
      const origCallback = this.stepsWidget.callback;
      this.stepsWidget.callback = (v) => {
        if (origCallback) origCallback.call(this.stepsWidget, v);
        this.scheduleRedraw();
      };
    }
    if (this.sigmaMaxWidget) {
      this.prevAxisMax = this.getAxisMax();
      const origCallback = this.sigmaMaxWidget.callback;
      this.sigmaMaxWidget.callback = (v) => {
        if (origCallback) origCallback.call(this.sigmaMaxWidget, v);
        this.onAxisMaxChanged();
      };
    }
    if (this.sigmaMinWidget) {
      const origCallback = this.sigmaMinWidget.callback;
      this.sigmaMinWidget.callback = (v) => {
        if (origCallback) origCallback.call(this.sigmaMinWidget, v);
        this.scheduleRedraw();
      };
    }
  }

  // When the user changes sigma_max_hint, rescale every point's sigma
  // proportionally so the curve shape is preserved and no point falls off
  // the visible axis. The first (top) endpoint is pinned to exactly the new
  // max; the last endpoint stays at 0 (the sampling invariant).
  onAxisMaxChanged() {
    const newMax = this.getAxisMax();
    const oldMax = this.prevAxisMax || newMax;
    if (oldMax <= 1e-6 || Math.abs(newMax - oldMax) < 1e-9) {
      this.prevAxisMax = newMax;
      this.scheduleRedraw();
      return;
    }
    const k = newMax / oldMax;
    for (let i = 0; i < this.points.length; i++) {
      const [t, s] = this.points[i];
      this.points[i] = [t, s * k];
    }
    // Pin endpoints cleanly.
    if (this.points.length > 0) {
      this.points[0] = [0, newMax];
      this.points[this.points.length - 1] = [1, 0];
    }
    // Safety clamp in case float drift pushed something above max.
    for (let i = 0; i < this.points.length; i++) {
      const [t, s] = this.points[i];
      this.points[i] = [t, Math.max(0, Math.min(newMax, s))];
    }
    this.prevAxisMax = newMax;
    this.serialize();
    this.scheduleRedraw();
  }

  getAxisMax() {
    if (this.sigmaMaxWidget && typeof this.sigmaMaxWidget.value === "number") {
      return Math.max(1e-3, this.sigmaMaxWidget.value);
    }
    return FALLBACK_SIGMA_MAX;
  }
  getAxisMin() {
    if (this.sigmaMinWidget && typeof this.sigmaMinWidget.value === "number") {
      return Math.max(0, this.sigmaMinWidget.value);
    }
    return FALLBACK_SIGMA_MIN;
  }

  injectStyles() {
    const styleId = "flks-widget-styles";
    if (!document.getElementById(styleId)) {
      const s = document.createElement("style");
      s.id = styleId;
      s.textContent = FLKS_STYLES;
      document.head.appendChild(s);
    }
  }

  buildUI() {
    this.element.innerHTML = `
      <div class="flks-header">
        <div class="flks-title">
          <span>Sigma Schedule</span>
        </div>
        <div class="flks-toggles">
          <div class="flks-toggle on" data-role="logy-toggle">Log Y</div>
          <div class="flks-toggle" data-role="reset-btn">Reset</div>
        </div>
      </div>
      <div class="flks-canvas-wrap">
        <canvas class="flks-canvas"></canvas>
      </div>
      <div class="flks-footer">
        <div class="flks-stat">
          <span class="flks-stat-label">σmax</span>
          <span class="flks-stat-value" data-role="sigma-max">-</span>
        </div>
        <div class="flks-stat">
          <span class="flks-stat-label">σmin</span>
          <span class="flks-stat-value" data-role="sigma-min">-</span>
        </div>
        <div class="flks-stat">
          <span class="flks-stat-label">Pts</span>
          <span class="flks-stat-value" data-role="pt-count">-</span>
        </div>
        <div class="flks-hint">drag • R-click: add/remove • Shift: fine</div>
      </div>
    `;

    this.canvas = this.element.querySelector(".flks-canvas");
    this.logToggle = this.element.querySelector('[data-role="logy-toggle"]');
    this.resetBtn = this.element.querySelector('[data-role="reset-btn"]');
    this.sigmaMaxEl = this.element.querySelector('[data-role="sigma-max"]');
    this.sigmaMinEl = this.element.querySelector('[data-role="sigma-min"]');
    this.ptCountEl = this.element.querySelector('[data-role="pt-count"]');

    this.logToggle.addEventListener("click", () => {
      this.logY = !this.logY;
      this.logToggle.classList.toggle("on", this.logY);
      this.scheduleRedraw();
    });

    this.resetBtn.addEventListener("click", () => {
      const m = this.getAxisMax();
      this.points = [
        [0.0, m],
        [0.5, m * 0.3],
        [1.0, 0.0],
      ];
      this.serialize();
      this.scheduleRedraw();
    });

    this.canvas.addEventListener("pointerdown", (e) => this.onPointerDown(e));
    this.canvas.addEventListener("pointermove", (e) => this.onPointerMove(e));
    this.canvas.addEventListener("pointerup", (e) => this.onPointerUp(e));
    this.canvas.addEventListener("pointercancel", (e) => this.onPointerUp(e));
    this.canvas.addEventListener("pointerleave", () => {
      this.hoveredIdx = -1;
      this.scheduleRedraw();
    });
    this.canvas.addEventListener("contextmenu", (e) => this.onContextMenu(e));

    this.resizeObserver = new ResizeObserver(() => {
      if (this.resizeTimeout) clearTimeout(this.resizeTimeout);
      this.resizeTimeout = window.setTimeout(() => this.scheduleRedraw(), 16);
    });
    this.resizeObserver.observe(this.element.querySelector(".flks-canvas-wrap"));
  }

  getSteps() {
    if (this.stepsWidget && typeof this.stepsWidget.value === "number") {
      return Math.max(1, Math.round(this.stepsWidget.value));
    }
    return 20;
  }

  loadFromWidget() {
    if (!this.curveWidget) return;
    const raw = this.curveWidget.value;
    if (!raw) return;
    try {
      const parsed = JSON.parse(raw);
      const pts = parsed.points || parsed;
      if (Array.isArray(pts) && pts.length >= 2) {
        this.points = pts.map((p) => {
          if (Array.isArray(p)) return [p[0], p[1]];
          return [p.t, p.sigma];
        });
      }
    } catch (e) {
      // leave defaults
    }
  }

  serialize() {
    if (!this.curveWidget) return;
    const data = {
      points: this.points.map(([t, s]) => [Number(t.toFixed(6)), Number(s.toFixed(6))]),
      interpolation: "monotone_cubic",
    };
    this.curveWidget.value = JSON.stringify(data);
    if (this.node.graph && this.node.graph.change) this.node.graph.change();
  }

  // Coordinate mapping. The axis top is driven by the sigma_max_hint widget
  // (user-set), not by the highest control point — so the axis stays stable
  // as you drag and reflects the model's expected sigma range.
  sigmaMax() {
    return this.getAxisMax();
  }
  // Actual curve extrema — used for the footer readouts only.
  curveMaxValue() {
    let m = 0;
    for (const [, s] of this.points) m = Math.max(m, s);
    return m;
  }
  curveMinValue() {
    let m = Infinity;
    for (const [, s] of this.points) m = Math.min(m, s);
    return m === Infinity ? 0 : m;
  }
  yFromSigma(s, h, padTop, padBottom) {
    const top = padTop;
    const bottom = h - padBottom;
    const usable = bottom - top;
    const sMax = this.sigmaMax();
    if (this.logY) {
      const eps = 1e-3;
      const logMax = Math.log(sMax + eps);
      const logMin = Math.log(eps);
      const norm = (Math.log(Math.max(s, 0) + eps) - logMin) / (logMax - logMin);
      return bottom - norm * usable;
    }
    return bottom - (s / sMax) * usable;
  }
  sigmaFromY(y, h, padTop, padBottom) {
    const top = padTop;
    const bottom = h - padBottom;
    const usable = bottom - top;
    const sMax = this.sigmaMax();
    const norm = Math.max(0, Math.min(1, (bottom - y) / usable));
    if (this.logY) {
      const eps = 1e-3;
      const logMax = Math.log(sMax + eps);
      const logMin = Math.log(eps);
      return Math.exp(logMin + norm * (logMax - logMin)) - eps;
    }
    return norm * sMax;
  }
  xFromT(t, w, padLeft, padRight) {
    return padLeft + t * (w - padLeft - padRight);
  }
  tFromX(x, w, padLeft, padRight) {
    return Math.max(0, Math.min(1, (x - padLeft) / (w - padLeft - padRight)));
  }

  // ---- Pointer handling ----
  getLocal(e) {
    const rect = this.canvas.getBoundingClientRect();
    const cssW = this.canvas.clientWidth || rect.width;
    const cssH = this.canvas.clientHeight || rect.height;
    const sx = rect.width > 0 ? cssW / rect.width : 1;
    const sy = rect.height > 0 ? cssH / rect.height : 1;
    const x = (e.clientX - rect.left) * sx;
    const y = (e.clientY - rect.top) * sy;
    return { x, y, w: cssW, h: cssH };
  }
  findPoint(x, y, w, h) {
    const padL = 40, padR = 10, padT = 12, padB = 22;
    const hitRadiusSq = 12 * 12;
    let best = -1;
    let bestD = Infinity;
    for (let i = 0; i < this.points.length; i++) {
      const [t, s] = this.points[i];
      const px = this.xFromT(t, w, padL, padR);
      const py = this.yFromSigma(s, h, padT, padB);
      const d = (px - x) * (px - x) + (py - y) * (py - y);
      if (d <= hitRadiusSq && d < bestD) {
        best = i;
        bestD = d;
      }
    }
    return best;
  }
  onPointerDown(e) {
    const { x, y, w, h } = this.getLocal(e);
    const idx = this.findPoint(x, y, w, h);
    if (idx !== -1) {
      this.dragging = { idx, shift: e.shiftKey };
      this.canvas.setPointerCapture(e.pointerId);
      e.preventDefault();
    }
  }
  onPointerMove(e) {
    const { x, y, w, h } = this.getLocal(e);
    if (this.dragging) {
      const padL = 40, padR = 10, padT = 12, padB = 22;
      let t = this.tFromX(x, w, padL, padR);
      let s = this.sigmaFromY(y, h, padT, padB);
      const { idx } = this.dragging;
      if (idx === 0) t = 0;
      if (idx === this.points.length - 1) t = 1;
      if (idx > 0) t = Math.max(t, this.points[idx - 1][0] + 1e-4);
      if (idx < this.points.length - 1) t = Math.min(t, this.points[idx + 1][0] - 1e-4);
      if (idx === this.points.length - 1) s = 0;
      s = Math.max(0, s);
      if (e.shiftKey) {
        const prev = this.points[idx][1];
        s = prev + (s - prev) * 0.15;
      }
      this.points[idx] = [t, s];
      this.serialize();
      this.scheduleRedraw();
      e.preventDefault();
      return;
    }
    const prevHover = this.hoveredIdx;
    this.hoveredIdx = this.findPoint(x, y, w, h);
    if (prevHover !== this.hoveredIdx) this.scheduleRedraw();
  }
  onPointerUp(e) {
    if (this.dragging) {
      try { this.canvas.releasePointerCapture(e.pointerId); } catch (_) {}
      this.dragging = null;
      this.serialize();
      this.scheduleRedraw();
    }
  }
  onContextMenu(e) {
    e.preventDefault();
    const { x, y, w, h } = this.getLocal(e);
    const idx = this.findPoint(x, y, w, h);
    if (idx !== -1) {
      if (idx === 0 || idx === this.points.length - 1) return;
      this.points.splice(idx, 1);
    } else {
      const padL = 40, padR = 10, padT = 12, padB = 22;
      const t = this.tFromX(x, w, padL, padR);
      const s = this.sigmaFromY(y, h, padT, padB);
      const newPoint = [t, Math.max(0, s)];
      let insertAt = this.points.length;
      for (let i = 0; i < this.points.length; i++) {
        if (this.points[i][0] > t) { insertAt = i; break; }
      }
      this.points.splice(insertAt, 0, newPoint);
    }
    this.serialize();
    this.scheduleRedraw();
  }

  // ---- Drawing ----
  scheduleRedraw() {
    if (this.pendingRAF) return;
    this.pendingRAF = requestAnimationFrame(() => {
      this.pendingRAF = null;
      this.draw();
    });
  }

  interpolate(pts) {
    if (pts.length <= 2) return pts.slice();
    const out = [];
    const density = 24;
    for (let i = 0; i < pts.length - 1; i++) {
      const [t0, s0] = pts[i];
      const [t1, s1] = pts[i + 1];
      for (let k = 0; k < density; k++) {
        const u = k / density;
        const smoothU = u * u * (3 - 2 * u);
        out.push([t0 + (t1 - t0) * u, s0 + (s1 - s0) * smoothU]);
      }
    }
    out.push(pts[pts.length - 1]);
    return out;
  }

  draw() {
    if (!this.canvas) return;
    const cssW = this.canvas.clientWidth;
    const cssH = this.canvas.clientHeight;
    if (cssW <= 0 || cssH <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    if (this.canvas.width !== cssW * dpr || this.canvas.height !== cssH * dpr) {
      this.canvas.width = cssW * dpr;
      this.canvas.height = cssH * dpr;
    }
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const padL = 40, padR = 10, padT = 12, padB = 22;
    const w = cssW, h = cssH;

    ctx.fillStyle = "#0f0f12";
    ctx.fillRect(0, 0, w, h);

    const sMax = this.sigmaMax();
    const curveMax = this.curveMaxValue();
    const curveMin = this.curveMinValue();

    ctx.strokeStyle = "#27272a";
    ctx.lineWidth = 0.5;
    ctx.fillStyle = "#71717a";
    ctx.font = "9px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    const gridCount = 4;
    for (let i = 0; i <= gridCount; i++) {
      const frac = i / gridCount;
      const sigVal = this.logY
        ? Math.exp(Math.log(sMax + 1e-3) * (1 - frac) + Math.log(1e-3) * frac) - 1e-3
        : sMax * (1 - frac);
      const y = this.yFromSigma(sigVal, h, padT, padB);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
      ctx.fillText(sigVal.toFixed(sigVal < 1 ? 3 : 2), padL - 4, y);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i <= gridCount; i++) {
      const frac = i / gridCount;
      const x = this.xFromT(frac, w, padL, padR);
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, h - padB);
      ctx.stroke();
      const steps = this.getSteps();
      const stepIdx = Math.round(frac * steps);
      ctx.fillText(String(stepIdx), x, h - padB + 4);
    }

    ctx.save();
    ctx.translate(10, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#52525b";
    ctx.font = "9px Inter, sans-serif";
    ctx.fillText("σ" + (this.logY ? " (log)" : ""), 0, 0);
    ctx.restore();

    const samples = this.interpolate(this.points);
    this.drawCurve(ctx, samples, w, h, padL, padR, padT, padB, {
      stroke: "#06b6d4",
      lineWidth: 2,
      fill: "rgba(6, 182, 212, 0.18)",
    });

    for (let i = 0; i < this.points.length; i++) {
      const [t, s] = this.points[i];
      const x = this.xFromT(t, w, padL, padR);
      const y = this.yFromSigma(s, h, padT, padB);
      const isHover = i === this.hoveredIdx;
      const isDrag = this.dragging && this.dragging.idx === i;
      const isEndpoint = i === 0 || i === this.points.length - 1;
      ctx.beginPath();
      ctx.arc(x, y, isDrag ? 6 : (isHover ? 5 : 4), 0, Math.PI * 2);
      ctx.fillStyle = isEndpoint ? "#fafafa" : (isDrag ? "#06b6d4" : (isHover ? "#22d3ee" : "#fafafa"));
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = isDrag ? "#22d3ee" : "#06b6d4";
      ctx.stroke();
    }

    if (this.sigmaMaxEl) this.sigmaMaxEl.textContent = curveMax.toFixed(3);
    if (this.sigmaMinEl) this.sigmaMinEl.textContent = curveMin.toFixed(4);
    if (this.ptCountEl) this.ptCountEl.textContent = String(this.points.length);
  }

  drawCurve(ctx, pts, w, h, padL, padR, padT, padB, { stroke, lineWidth, fill }) {
    if (!pts || pts.length < 2) return;
    ctx.save();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const [t, s] = pts[i];
      const x = this.xFromT(t, w, padL, padR);
      const y = this.yFromSigma(s, h, padT, padB);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    if (fill) {
      ctx.lineTo(this.xFromT(pts[pts.length - 1][0], w, padL, padR), h - padB);
      ctx.lineTo(this.xFromT(pts[0][0], w, padL, padR), h - padB);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
    }

    ctx.restore();
  }

  dispose() {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    if (this.resizeTimeout) {
      clearTimeout(this.resizeTimeout);
      this.resizeTimeout = null;
    }
    if (this.pendingRAF) {
      cancelAnimationFrame(this.pendingRAF);
      this.pendingRAF = null;
    }
    if (this.element && this.element.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }
  }
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

const EDITOR_INSTANCES = new Map();

function findWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name) || null;
}

function hideWidget(widget) {
  if (!widget) return;
  widget.origType = widget.type;
  widget.origComputeSize = widget.computeSize;
  widget.computeSize = () => [0, -4];
  widget.type = "converted-widget";
  if (widget.element) widget.element.style.display = "none";
}

app.registerExtension({
  name: "ComfyUI.FL_KsamplerSigma",

  nodeCreated(node) {
    const comfyClass = (node.constructor && node.constructor.comfyClass) || "";
    if (comfyClass !== "FL_KsamplerSigma") return;

    const curveWidget = findWidget(node, "schedule_curve");
    const stepsWidget = findWidget(node, "steps");
    const sigmaMaxWidget = findWidget(node, "sigma_max_hint");
    const sigmaMinWidget = findWidget(node, "sigma_min_hint");

    hideWidget(curveWidget);

    const container = document.createElement("div");
    container.id = `flks-editor-${node.id}`;
    container.style.width = "100%";
    container.style.height = "100%";
    container.style.minHeight = "320px";
    container.style.marginTop = "4px";

    const widget = node.addDOMWidget(
      "sigma_editor",
      "flks-sigma-editor",
      container,
      {
        getMinHeight: () => 340,
        hideOnZoom: false,
        serialize: false,
      }
    );

    const [oldWidth, oldHeight] = node.size;
    node.setSize([Math.max(oldWidth, 440), Math.max(oldHeight, 560)]);

    setTimeout(() => {
      const editor = new SigmaCurveEditor({
        node,
        container,
        curveWidget,
        stepsWidget,
        sigmaMaxWidget,
        sigmaMinWidget,
      });
      EDITOR_INSTANCES.set(node.id, editor);
    }, 50);

    widget.onRemove = () => {
      const ed = EDITOR_INSTANCES.get(node.id);
      if (ed) {
        ed.dispose();
        EDITOR_INSTANCES.delete(node.id);
      }
    };

    const origOnConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const r = origOnConfigure ? origOnConfigure.apply(this, args) : undefined;
      const ed = EDITOR_INSTANCES.get(node.id);
      if (ed) {
        ed.loadFromWidget();
        ed.scheduleRedraw();
      }
      return r;
    };
  },
});
