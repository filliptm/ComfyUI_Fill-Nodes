import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const STYLES = `
  .fl-region-png {
    height: 100%;
    min-height: 360px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border: 1px solid #2f3137;
    border-radius: 8px;
    background: #15161a;
    color: #f4f4f5;
    font: 12px Inter, system-ui, sans-serif;
  }
  .fl-region-png-toolbar {
    display: flex;
    gap: 6px;
    align-items: center;
    padding: 7px;
    border-bottom: 1px solid #2f3137;
    background: #202127;
  }
  .fl-region-png-toolbar button {
    height: 24px;
    border: 1px solid #3f424a;
    border-radius: 5px;
    background: #2b2d34;
    color: #f4f4f5;
    padding: 0 8px;
    cursor: pointer;
    font-size: 11px;
  }
  .fl-region-png-toolbar button:hover { background: #383b44; }
  .fl-region-png-count {
    margin-left: auto;
    color: #a1a1aa;
    font-size: 11px;
  }
  .fl-region-png-stage {
    flex: 1;
    min-height: 240px;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0f1013;
  }
  .fl-region-png-stage canvas {
    max-width: 100%;
    max-height: 100%;
    cursor: crosshair;
  }
  .fl-region-png-empty {
    color: #71717a;
    padding: 20px;
    text-align: center;
    line-height: 1.4;
  }
  .fl-region-png-editor {
    display: flex;
    flex-direction: column;
    gap: 5px;
    max-height: 150px;
    overflow-y: auto;
    padding: 7px;
    border-top: 1px solid #2f3137;
    background: #202127;
  }
  .fl-region-png-dir-row {
    display: grid;
    grid-template-columns: 58px 1fr 28px;
    gap: 6px;
    align-items: center;
  }
  .fl-region-png-dir-row.is-selected label {
    color: #38bdf8;
  }
  .fl-region-png-editor label {
    color: #a1a1aa;
    font-size: 11px;
  }
  .fl-region-png-editor input {
    min-width: 0;
    height: 24px;
    border: 1px solid #3f424a;
    border-radius: 5px;
    background: #111216;
    color: #f4f4f5;
    padding: 0 7px;
    font-size: 11px;
  }
  .fl-region-png-dir-delete {
    width: 26px;
    height: 24px;
    border: 1px solid #4a2f35;
    border-radius: 5px;
    background: #2a171b;
    color: #fca5a5;
    cursor: pointer;
    line-height: 1;
    font-size: 15px;
  }
  .fl-region-png-dir-delete:hover {
    background: #3b1d24;
    border-color: #7f1d1d;
  }
  .fl-region-png-dir-empty {
    color: #71717a;
    font-size: 11px;
  }
`;

function injectStyles() {
  const id = "fl-region-png-overlay-styles";
  if (document.getElementById(id)) return;
  const style = document.createElement("style");
  style.id = id;
  style.textContent = STYLES;
  document.head.appendChild(style);
}

class RegionPNGOverlayWidget {
  constructor(node, container) {
    this.node = node;
    this.container = container;
    this.regionsWidget = node.widgets?.find((w) => w.name === "regions_json");
    this.widgetByName = new Map((node.widgets || []).map((w) => [w.name, w]));
    this.regions = this.readRegions();
    this.selected = this.regions.length ? 0 : -1;
    this.image = null;
    this.sourceSize = null;
    this.drag = null;

    injectStyles();
    this.build();
    this.bind();
    this.bindWidgetRedraws();
    this.render();
  }

  build() {
    this.root = document.createElement("div");
    this.root.className = "fl-region-png";
    this.root.innerHTML = `
      <div class="fl-region-png-toolbar">
        <button type="button" data-action="delete">Delete</button>
        <button type="button" data-action="clear">Clear</button>
        <span class="fl-region-png-count" data-role="count">0 regions</span>
      </div>
      <div class="fl-region-png-stage" data-role="stage">
        <div class="fl-region-png-empty">Run once to load the image preview.</div>
      </div>
      <div class="fl-region-png-editor" data-role="editor"></div>
    `;
    this.stage = this.root.querySelector('[data-role="stage"]');
    this.countEl = this.root.querySelector('[data-role="count"]');
    this.editor = this.root.querySelector('[data-role="editor"]');
    this.container.appendChild(this.root);
  }

  bind() {
    this.root.querySelector('[data-action="delete"]').addEventListener("click", () => this.deleteSelected());
    this.root.querySelector('[data-action="clear"]').addEventListener("click", () => {
      this.regions = [];
      this.selected = -1;
      this.commit();
    });
    this.editor.addEventListener("input", (event) => {
      const input = event.target.closest("input[data-index]");
      if (!input) return;
      const index = parseInt(input.dataset.index, 10);
      if (!Number.isFinite(index) || !this.regions[index]) return;
      this.regions[index].directory = input.value;
      this.commit(false);
    });
    this.editor.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-delete-index]");
      if (!button) return;
      const index = parseInt(button.dataset.deleteIndex, 10);
      if (!Number.isFinite(index) || !this.regions[index]) return;
      this.deleteRegion(index);
    });
    this.editor.addEventListener("focusin", (event) => {
      const input = event.target.closest("input[data-index]");
      if (!input) return;
      const index = parseInt(input.dataset.index, 10);
      if (!Number.isFinite(index) || !this.regions[index]) return;
      if (this.selected === index) return;
      this.selected = index;
      this.render();
    });
  }

  bindWidgetRedraws() {
    for (const name of [
      "rotation_min",
      "rotation_max",
      "vignette_enabled",
      "vignette_diameter",
      "vignette_skew",
      "vignette_opacity",
      "vignette_offset_x",
      "vignette_offset_y",
      "vignette_rotation_offset",
    ]) {
      const widget = this.widgetByName.get(name);
      if (!widget) continue;
      const original = widget.callback;
      widget.callback = (...args) => {
        if (original) original.apply(widget, args);
        this.render();
      };
    }
  }

  setImage(dataUri, sourceSize, regionsFromServer) {
    this.sourceSize = sourceSize;
    if (Array.isArray(regionsFromServer) && !this.regions.length) {
      this.regions = regionsFromServer;
      this.selected = this.regions.length ? 0 : -1;
      this.commit(false);
    }

    const img = new Image();
    img.onload = () => {
      this.image = img;
      this.ensureCanvas();
      this.resizeCanvas();
      this.render();
    };
    img.src = dataUri;
  }

  ensureCanvas() {
    if (this.canvas) return;
    this.stage.innerHTML = "";
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.stage.appendChild(this.canvas);

    this.canvas.addEventListener("pointerdown", (e) => this.onPointerDown(e));
    this.canvas.addEventListener("pointermove", (e) => this.onPointerMove(e));
    window.addEventListener("pointerup", () => this.onPointerUp());
    window.addEventListener("resize", () => {
      this.resizeCanvas();
      this.render();
    });
  }

  resizeCanvas() {
    if (!this.canvas || !this.image) return;
    const sourceW = this.sourceSize?.[0] || this.image.width;
    const sourceH = this.sourceSize?.[1] || this.image.height;
    const maxW = Math.max(100, this.stage.clientWidth - 12);
    const maxH = Math.max(100, this.stage.clientHeight - 12);
    const scale = Math.min(maxW / sourceW, maxH / sourceH);
    this.viewScale = scale;
    this.canvas.width = Math.max(1, Math.round(sourceW * scale));
    this.canvas.height = Math.max(1, Math.round(sourceH * scale));
  }

  onPointerDown(event) {
    if (!this.image) return;
    if (event.button === 2) return;
    const p = this.eventToImage(event);
    const vignetteHit = this.hitVignetteCenter(p.x, p.y);
    const handleHit = vignetteHit < 0 ? this.hitHandle(p.x, p.y) : -1;
    const hit = vignetteHit >= 0 ? vignetteHit : (handleHit >= 0 ? handleHit : this.hitRegion(p.x, p.y));
    if (hit >= 0) {
      this.selected = hit;
      const r = this.regions[hit];
      const edge = handleHit >= 0 ? this.hitResizeEdge(r, p.x, p.y) : null;
      this.drag = { mode: vignetteHit >= 0 ? "vignette" : (edge || "move"), start: p, original: { ...r } };
    } else {
      const region = { x: p.x, y: p.y, w: 0, h: 0, directory: "", vignette_offset_x: 0, vignette_offset_y: 0 };
      this.regions.push(region);
      this.selected = this.regions.length - 1;
      this.drag = { mode: "draw", start: p, original: { ...region } };
    }
    this.updateEditor();
    this.render();
    event.preventDefault();
  }

  onPointerMove(event) {
    if (!this.drag || this.selected < 0) return;
    const p = this.eventToImage(event);
    const r = this.regions[this.selected];
    const dx = p.x - this.drag.start.x;
    const dy = p.y - this.drag.start.y;
    const o = this.drag.original;

    if (this.drag.mode === "draw") {
      this.setRegionRect(r, o.x, o.y, dx, dy);
    } else if (this.drag.mode === "vignette") {
      r.vignette_offset_x = Math.round((o.vignette_offset_x || 0) + dx);
      r.vignette_offset_y = Math.round((o.vignette_offset_y || 0) + dy);
    } else if (this.drag.mode === "move") {
      const { w: sourceW, h: sourceH } = this.sourceDims();
      r.x = this.clamp(o.x + dx, 0, sourceW - r.w);
      r.y = this.clamp(o.y + dy, 0, sourceH - r.h);
      r.vignette_offset_x = o.vignette_offset_x || 0;
      r.vignette_offset_y = o.vignette_offset_y || 0;
    } else {
      this.resizeRegion(r, o, dx, dy, this.drag.mode);
    }
    this.render();
    event.preventDefault();
  }

  onPointerUp() {
    if (!this.drag) return;
    const r = this.regions[this.selected];
    if (r && (r.w < 3 || r.h < 3)) {
      this.regions.splice(this.selected, 1);
      this.selected = this.regions.length ? Math.max(0, this.regions.length - 1) : -1;
    }
    this.drag = null;
    this.commit();
  }

  setRegionRect(region, x, y, w, h) {
    region.x = Math.round(w < 0 ? x + w : x);
    region.y = Math.round(h < 0 ? y + h : y);
    region.w = Math.round(Math.abs(w));
    region.h = Math.round(Math.abs(h));
    this.clampRegion(region);
  }

  resizeRegion(region, original, dx, dy, edge) {
    let x1 = original.x;
    let y1 = original.y;
    let x2 = original.x + original.w;
    let y2 = original.y + original.h;
    if (edge.includes("l")) x1 += dx;
    if (edge.includes("r")) x2 += dx;
    if (edge.includes("t")) y1 += dy;
    if (edge.includes("b")) y2 += dy;
    this.setRegionRect(region, x1, y1, x2 - x1, y2 - y1);
  }

  render() {
    if (!this.canvas || !this.ctx || !this.image) {
      this.updateEditor();
      return;
    }
    this.resizeCanvas();
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.drawImage(this.image, 0, 0, this.canvas.width, this.canvas.height);

    this.regions.forEach((r, index) => {
      this.drawVignetteGuide(ctx, r, index === this.selected);
    });

    this.regions.forEach((r, index) => {
      const x = r.x * this.viewScale;
      const y = r.y * this.viewScale;
      const w = r.w * this.viewScale;
      const h = r.h * this.viewScale;
      const selected = index === this.selected;
      ctx.fillStyle = selected ? "rgba(14, 165, 233, 0.25)" : "rgba(250, 204, 21, 0.20)";
      ctx.strokeStyle = selected ? "#38bdf8" : "#facc15";
      ctx.lineWidth = selected ? 2 : 1.5;
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = selected ? "#38bdf8" : "#facc15";
      ctx.font = "12px sans-serif";
      ctx.fillText(String(index + 1), x + 5, y + 15);
      if (selected) this.drawHandles(ctx, x, y, w, h);
    });

    this.updateEditor();
    this.node.setDirtyCanvas(true);
  }

  drawHandles(ctx, x, y, w, h) {
    ctx.fillStyle = "#e0f2fe";
    const pts = [[x, y], [x + w, y], [x, y + h], [x + w, y + h]];
    ctx.strokeStyle = "#0284c7";
    ctx.lineWidth = 1;
    for (const [px, py] of pts) {
      ctx.fillRect(px - 6, py - 6, 12, 12);
      ctx.strokeRect(px - 6, py - 6, 12, 12);
    }
  }

  drawVignetteGuide(ctx, region, selected) {
    if (!this.getWidgetValue("vignette_enabled", true)) return;
    const diameter = Math.max(1, Math.min(region.w, region.h) * this.getWidgetValue("vignette_diameter", 0.85));
    const skew = Math.max(0.05, this.getWidgetValue("vignette_skew", 0.35));
    const opacity = Math.max(0, Math.min(1, this.getWidgetValue("vignette_opacity", 0.45)));
    const angle = (
      (this.getWidgetValue("rotation_min", -5) + this.getWidgetValue("rotation_max", 5)) / 2
      + this.getWidgetValue("vignette_rotation_offset", 0)
    ) * Math.PI / 180;
    const cx = (
      region.x + region.w / 2
      + (region.vignette_offset_x || 0)
      + this.getWidgetValue("vignette_offset_x", 0)
    ) * this.viewScale;
    const cy = (
      region.y + region.h / 2
      + (region.vignette_offset_y || 0)
      + this.getWidgetValue("vignette_offset_y", 0)
    ) * this.viewScale;
    const rx = (diameter / 2) * this.viewScale;
    const ry = (diameter * skew / 2) * this.viewScale;

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, Math.max(rx, ry));
    gradient.addColorStop(0, `rgba(0, 0, 0, ${opacity * 0.55})`);
    gradient.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.scale(1, ry / Math.max(1, rx));
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(0, 0, rx, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.strokeStyle = selected ? "#f97316" : "rgba(249, 115, 22, 0.7)";
    ctx.lineWidth = selected ? 2 : 1;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = selected ? "#fed7aa" : "#f97316";
    ctx.beginPath();
    ctx.arc(0, 0, selected ? 6 : 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  eventToImage(event) {
    const rect = this.canvas.getBoundingClientRect();
    const { w, h } = this.sourceDims();
    const x = ((event.clientX - rect.left) / Math.max(1, rect.width)) * w;
    const y = ((event.clientY - rect.top) / Math.max(1, rect.height)) * h;
    return {
      x: this.clamp(Math.round(x), 0, w),
      y: this.clamp(Math.round(y), 0, h),
    };
  }

  hitRegion(x, y) {
    for (let i = this.regions.length - 1; i >= 0; i--) {
      const r = this.regions[i];
      if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) return i;
    }
    return -1;
  }

  hitHandle(x, y) {
    for (let i = this.regions.length - 1; i >= 0; i--) {
      if (this.hitResizeEdge(this.regions[i], x, y)) return i;
    }
    return -1;
  }

  hitVignetteCenter(x, y) {
    if (!this.getWidgetValue("vignette_enabled", true)) return -1;
    const tol = Math.max(14, 16 / Math.max(0.1, this.viewScale));
    for (let i = this.regions.length - 1; i >= 0; i--) {
      const r = this.regions[i];
      const cx = r.x + r.w / 2 + (r.vignette_offset_x || 0) + this.getWidgetValue("vignette_offset_x", 0);
      const cy = r.y + r.h / 2 + (r.vignette_offset_y || 0) + this.getWidgetValue("vignette_offset_y", 0);
      if (Math.hypot(x - cx, y - cy) <= tol) return i;
    }
    return -1;
  }

  hitResizeEdge(region, x, y) {
    const tol = Math.max(14, 16 / Math.max(0.1, this.viewScale));
    const left = Math.abs(x - region.x) <= tol;
    const right = Math.abs(x - (region.x + region.w)) <= tol;
    const top = Math.abs(y - region.y) <= tol;
    const bottom = Math.abs(y - (region.y + region.h)) <= tol;
    if (left && top) return "lt";
    if (right && top) return "rt";
    if (left && bottom) return "lb";
    if (right && bottom) return "rb";
    return null;
  }

  deleteSelected() {
    if (this.selected < 0) return;
    this.deleteRegion(this.selected);
  }

  deleteRegion(index) {
    this.regions.splice(index, 1);
    if (!this.regions.length) {
      this.selected = -1;
    } else if (this.selected === index) {
      this.selected = Math.min(index, this.regions.length - 1);
    } else if (this.selected > index) {
      this.selected -= 1;
    }
    this.commit();
  }

  clampRegion(region) {
    const { w, h } = this.sourceDims();
    region.x = this.clamp(region.x, 0, w);
    region.y = this.clamp(region.y, 0, h);
    region.w = this.clamp(region.w, 0, w - region.x);
    region.h = this.clamp(region.h, 0, h - region.y);
  }

  sourceDims() {
    return {
      w: this.sourceSize?.[0] || this.image?.width || 1,
      h: this.sourceSize?.[1] || this.image?.height || 1,
    };
  }

  readRegions() {
    try {
      const parsed = JSON.parse(this.regionsWidget?.value || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  commit(redraw = true) {
    if (this.regionsWidget) {
      this.regionsWidget.value = JSON.stringify(this.regions);
    }
    if (redraw) this.render();
  }

  updateEditor() {
    if (this.countEl) this.countEl.textContent = `${this.regions.length} region${this.regions.length === 1 ? "" : "s"}`;
    if (!this.editor) return;

    const active = document.activeElement;
    const activeIndex = active?.dataset?.index;
    const activeSelection = active && active.tagName === "INPUT"
      ? [active.selectionStart, active.selectionEnd]
      : null;

    if (!this.regions.length) {
      this.editor.innerHTML = `<div class="fl-region-png-dir-empty">Draw boxes to add directory rows.</div>`;
      return;
    }

    this.editor.innerHTML = this.regions.map((region, index) => `
      <div class="fl-region-png-dir-row ${index === this.selected ? "is-selected" : ""}">
        <label>Box ${index + 1}</label>
        <input data-index="${index}" value="${this.escapeAttr(region.directory || "")}" placeholder="Paste PNG folder path" />
        <button class="fl-region-png-dir-delete" type="button" data-delete-index="${index}" title="Delete box ${index + 1}">×</button>
      </div>
    `).join("");

    if (activeIndex !== undefined) {
      const next = this.editor.querySelector(`input[data-index="${activeIndex}"]`);
      if (next) {
        next.focus();
        if (activeSelection) next.setSelectionRange(activeSelection[0], activeSelection[1]);
      }
    }
  }

  escapeAttr(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll('"', "&quot;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  getWidgetValue(name, fallback) {
    const widget = this.widgetByName.get(name);
    return widget ? widget.value : fallback;
  }

  dispose() {
    this.root?.remove();
  }
}

const INSTANCES = new Map();

app.registerExtension({
  name: "ComfyUI.FL_RegionPNGOverlay",
  nodeCreated(node) {
    const comfyClass = (node.constructor && node.constructor.comfyClass) || node.comfyClass || "";
    if (comfyClass !== "FL_RegionPNGOverlay") return;

    const regionsWidget = node.widgets?.find((w) => w.name === "regions_json");
    if (regionsWidget) {
      regionsWidget.hidden = true;
      regionsWidget.computeSize = () => [0, -4];
    }

    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.height = "100%";
    container.style.minHeight = "360px";

    const widget = node.addDOMWidget("region_png_overlay", "fl-region-png-overlay", container, {
      getMinHeight: () => 420,
      hideOnZoom: false,
      serialize: false,
    });

    const [oldW, oldH] = node.size;
    node.setSize([Math.max(oldW, 460), Math.max(oldH, 640)]);

    setTimeout(() => {
      const inst = new RegionPNGOverlayWidget(node, container);
      INSTANCES.set(node.id, inst);
    }, 50);

    widget.onRemove = () => {
      const inst = INSTANCES.get(node.id);
      if (inst) inst.dispose();
      INSTANCES.delete(node.id);
    };
  },
});

api.addEventListener("fl_region_png_overlay_canvas", (event) => {
  const detail = event.detail;
  if (!detail) return;
  const nodeId = parseInt(detail.node, 10);
  const inst = INSTANCES.get(nodeId);
  if (!inst) return;
  inst.setImage(detail.image, detail.source_size, detail.regions);
});
