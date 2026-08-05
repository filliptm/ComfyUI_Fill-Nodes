import { app } from "../../../../scripts/app.js";

const STYLE_ID = "fl-beat-prompt-sequencer-styles";
const INSTANCES = new Map();
const HEADER_RE = /^\s*\[\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)(?:\s*\|\s*(.*?))?\s*\]\s*$/;
const EPSILON = 1e-6;
const FORMAT_VERSION = 2;
const MIN_NODE_WIDTH = 680;
const MIN_NODE_HEIGHT = 900;
const TIMELINE_LEFT = 42;
const TIMELINE_RIGHT = 12;

const STYLES = `
  .flbps-root {
    height: 100%;
    min-height: 650px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    color: #e4e4e7;
    background: #151518;
    border: 1px solid #303036;
    border-radius: 10px;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    box-sizing: border-box;
  }
  .flbps-root * { box-sizing: border-box; }
  .flbps-header, .flbps-toolbar, .flbps-actions, .flbps-footer {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 7px 9px;
    border-bottom: 1px solid #2b2b31;
    background: #1c1c20;
  }
  .flbps-header { justify-content: space-between; }
  .flbps-title { font-size: 12px; font-weight: 700; color: #fafafa; }
  .flbps-status {
    max-width: 72%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    padding: 3px 7px;
    border-radius: 9px;
    color: #a1a1aa;
    background: #27272a;
    font-size: 9px;
  }
  .flbps-status.fresh { color: #d1fae5; background: #065f46; }
  .flbps-status.cached { color: #fef3c7; background: #713f12; }
  .flbps-status.error { color: #fee2e2; background: #7f1d1d; }
  .flbps-toolbar { flex-wrap: wrap; padding-top: 6px; padding-bottom: 6px; }
  .flbps-control {
    display: flex;
    align-items: center;
    gap: 4px;
    color: #a1a1aa;
    font-size: 9px;
  }
  .flbps-control select, .flbps-inspector input,
  .flbps-inspector textarea, .flbps-raw textarea {
    color: #f4f4f5;
    background: #252529;
    border: 1px solid #3f3f46;
    border-radius: 5px;
    outline: none;
    font: inherit;
  }
  .flbps-control select {
    height: 23px;
    min-width: 66px;
    padding: 2px 5px;
    font-size: 9px;
  }
  .flbps-control select:focus, .flbps-inspector input:focus,
  .flbps-inspector textarea:focus, .flbps-raw textarea:focus { border-color: #22d3ee; }
  .flbps-canvas-wrap {
    position: relative;
    flex: 1 1 auto;
    min-height: 245px;
    overflow: hidden;
    background: #101013;
  }
  .flbps-canvas { width: 100%; height: 100%; display: block; touch-action: none; }
  .flbps-empty {
    position: absolute;
    left: 50%;
    top: 58%;
    transform: translate(-50%, -50%);
    color: #71717a;
    font-size: 11px;
    pointer-events: none;
  }
  .flbps-actions {
    border-top: 1px solid #2b2b31;
    border-bottom: 1px solid #2b2b31;
  }
  .flbps-button {
    height: 24px;
    padding: 3px 8px;
    color: #d4d4d8;
    background: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 5px;
    font-size: 9px;
    cursor: pointer;
  }
  .flbps-button:hover { color: #fff; border-color: #52525b; background: #303036; }
  .flbps-button.primary { color: #ecfeff; border-color: #0e7490; background: #155e75; }
  .flbps-button.active { color: #cffafe; border-color: #0891b2; background: #164e63; }
  .flbps-button.danger:hover { border-color: #b91c1c; background: #7f1d1d; }
  .flbps-button:disabled { opacity: .4; cursor: default; }
  .flbps-spacer { flex: 1; }
  .flbps-inspector {
    flex: 0 0 auto;
    padding: 8px 9px;
    background: #19191d;
    border-bottom: 1px solid #2b2b31;
  }
  .flbps-inspector.disabled { opacity: 0.45; pointer-events: none; }
  .flbps-inspector-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 6px;
    margin-bottom: 7px;
  }
  .flbps-field { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
  .flbps-field label { color: #8b8b95; font-size: 8px; text-transform: uppercase; letter-spacing: .04em; }
  .flbps-field input { width: 100%; height: 24px; padding: 3px 5px; font-size: 10px; }
  .flbps-prompt-label {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 4px;
    color: #8b8b95;
    font-size: 8px;
    text-transform: uppercase;
    letter-spacing: .04em;
  }
  .flbps-prompt-meta {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: #67e8f9;
    text-transform: none;
    letter-spacing: 0;
  }
  .flbps-inspector textarea { width: 100%; height: 68px; resize: vertical; padding: 6px; font-size: 10px; line-height: 1.35; }
  .flbps-raw { display: none; flex: 0 0 auto; padding: 8px 9px; background: #17171a; border-bottom: 1px solid #2b2b31; }
  .flbps-raw.open { display: block; }
  .flbps-raw-label { margin-bottom: 5px; color: #a1a1aa; font-size: 9px; }
  .flbps-raw textarea { width: 100%; height: 130px; resize: vertical; padding: 7px; font-family: "Cascadia Mono", Consolas, monospace; font-size: 9px; line-height: 1.35; }
  .flbps-raw-actions { display: flex; gap: 6px; margin-top: 6px; justify-content: flex-end; }
  .flbps-footer {
    justify-content: space-between;
    border-bottom: 0;
    color: #71717a;
    font-size: 8px;
  }
  .flbps-summary { color: #a1a1aa; }
  .flbps-error {
    display: none;
    flex: 0 0 auto;
    padding: 6px 9px;
    color: #fecaca;
    background: #450a0a;
    border-bottom: 1px solid #7f1d1d;
    font-size: 9px;
  }
  .flbps-error.open { display: block; }
`;

function injectStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = STYLES;
  document.head.appendChild(style);
}

function findWidget(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name) || null;
}

function hideWidget(widget) {
  if (!widget) return;
  if (!widget.origType) widget.origType = widget.type;
  if (!widget.origComputeSize) widget.origComputeSize = widget.computeSize;
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
  widget.computedHeight = 0;
  widget.type = "converted-widget";
  if (widget.element) widget.element.style.display = "none";
}

function enforceMinimumNodeSize(node) {
  node.min_size = [
    Math.max(node.min_size?.[0] || 0, MIN_NODE_WIDTH),
    Math.max(node.min_size?.[1] || 0, MIN_NODE_HEIGHT),
  ];
  const width = Math.max(node.size[0], MIN_NODE_WIDTH);
  const height = Math.max(node.size[1], MIN_NODE_HEIGHT);
  if (width !== node.size[0] || height !== node.size[1]) node.setSize([width, height]);
}

function finiteNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function formatClock(seconds) {
  const value = Math.max(0, finiteNumber(seconds));
  const minutes = Math.floor(value / 60);
  const remainder = value - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(3).padStart(6, "0")}`;
}

function parseOptions(raw, defaultFadeIn, defaultFadeOut, lineNumber) {
  const values = {
    fadeIn: finiteNumber(defaultFadeIn),
    fadeOut: finiteNumber(defaultFadeOut),
  };
  if (!raw) return values;
  for (const part of raw.split("|")) {
    const separator = part.indexOf("=");
    if (separator < 0) {
      throw new Error(`Line ${lineNumber}: options must use fade_in=value or fade_out=value.`);
    }
    const name = part.slice(0, separator).trim();
    const value = Number(part.slice(separator + 1).trim());
    if (name !== "fade_in" && name !== "fade_out") {
      throw new Error(`Line ${lineNumber}: unknown option '${name}'.`);
    }
    if (!Number.isFinite(value) || value < 0) {
      throw new Error(`Line ${lineNumber}: ${name} must be zero or greater.`);
    }
    if (name === "fade_in") values.fadeIn = value;
    else values.fadeOut = value;
  }
  return values;
}

function parseTimeline(text, defaultFadeIn, defaultFadeOut) {
  const clips = [];
  let current = null;
  let promptLines = [];

  const finish = () => {
    if (!current) return;
    const prompt = promptLines.join("\n").trim();
    if (!prompt) throw new Error(`Line ${current.line}: section prompt is empty.`);
    clips.push({ ...current, prompt });
  };

  const lines = String(text || "").split(/\r?\n/);
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    const match = line.match(HEADER_RE);
    if (match) {
      finish();
      promptLines = [];
      const start = Number(match[1]);
      const end = Number(match[2]);
      if (!(end > start)) throw new Error(`Line ${index + 1}: section end must be after its start.`);
      const options = parseOptions(match[3], defaultFadeIn, defaultFadeOut, index + 1);
      if (options.fadeIn + options.fadeOut > end - start + EPSILON) {
        throw new Error(`Line ${index + 1}: fades exceed the section duration.`);
      }
      current = {
        line: index + 1,
        start,
        end,
        fadeIn: options.fadeIn,
        fadeOut: options.fadeOut,
      };
      continue;
    }
    if (line.trimStart().startsWith("[")) {
      throw new Error(`Line ${index + 1}: invalid schedule header.`);
    }
    if (!current) {
      if (line.trim()) throw new Error(`Line ${index + 1}: prompt text needs a schedule header.`);
      continue;
    }
    promptLines.push(line);
  }
  finish();
  if (!clips.length) throw new Error("The prompt schedule has no sections.");
  for (let index = 1; index < clips.length; index++) {
    if (clips[index].start < clips[index - 1].end - EPSILON) {
      throw new Error(`Line ${clips[index].line}: section overlaps the previous section.`);
    }
  }
  return clips;
}

function validateFrameClips(clips) {
  for (const clip of clips) {
    for (const [name, value] of Object.entries({
      start: clip.start,
      end: clip.end,
      fade_in: clip.fadeIn,
      fade_out: clip.fadeOut,
    })) {
      if (Math.abs(value - Math.round(value)) > EPSILON) {
        throw new Error(`Line ${clip.line}: ${name} must be a whole frame.`);
      }
    }
    clip.start = Math.round(clip.start);
    clip.end = Math.round(clip.end);
    clip.fadeIn = Math.round(clip.fadeIn);
    clip.fadeOut = Math.round(clip.fadeOut);
  }
  return clips;
}

function serializeTimeline(clips) {
  return clips.map((clip) => (
    `[${Math.round(clip.start)} - ${Math.round(clip.end)} | ` +
    `fade_in=${Math.round(clip.fadeIn)} | fade_out=${Math.round(clip.fadeOut)}]\n` +
    clip.prompt.trim()
  )).join("\n\n");
}

function niceFrameStep(range, width, fps) {
  const target = Math.max(1, range / Math.max(2, width / 80));
  const candidates = new Set([1]);
  for (let power = 1; power <= Math.max(target * 10, 100); power *= 10) {
    candidates.add(power);
    candidates.add(2 * power);
    candidates.add(5 * power);
  }
  for (const multiple of [0.25, 0.5, 1, 2, 5, 10, 30, 60]) {
    candidates.add(Math.max(1, Math.round(fps * multiple)));
  }
  const ordered = [...candidates].sort((a, b) => a - b);
  return ordered.find((value) => value >= target) || ordered[ordered.length - 1];
}

function normalizeWaveformPreview(value) {
  if (!value || value.version !== 1 || !Array.isArray(value.peaks) || value.peaks.length < 2 ||
      value.peaks.length % 2 !== 0) {
    return null;
  }
  const duration = finiteNumber(value.duration);
  const scale = finiteNumber(value.scale);
  if (!(duration > 0) || !(scale > 0)) return null;
  const peaks = value.peaks.map((peak) => finiteNumber(peak));
  return { version: 1, duration, scale, peaks };
}

class BeatPromptSequencer {
  constructor({ node, container, widgets }) {
    this.node = node;
    this.container = container;
    this.widgets = widgets;
    this.clips = [];
    this.selectedIndex = -1;
    this.playheadFrame = null;
    this.snapGuideFrame = null;
    this.drag = null;
    this.clipRects = [];
    this.pendingFrame = null;
    this.resizeObserver = null;
    this.callbackRestorers = [];
    this.rawInvalid = false;
    this.migrationPending = false;
    this.hover = null;

    const saved = node.properties?.flBeatPromptSequencer || {};
    this.beatData = saved.beatData || null;
    if (this.beatData) {
      this.beatData.waveformPreview = normalizeWaveformPreview(this.beatData.waveformPreview);
    }
    this.dataFresh = false;
    this.viewStart = saved.formatVersion === FORMAT_VERSION ? finiteNumber(saved.viewStart, 0) : 0;
    this.viewEnd = saved.formatVersion === FORMAT_VERSION ? finiteNumber(saved.viewEnd, 0) : 0;
    this.snapMode = ["beat", "frame", "off"].includes(saved.snapMode) ? saved.snapMode : "beat";
    this.waveformVisible = saved.waveformVisible !== false;

    injectStyles();
    this.build();
    this.bindWidgetCallbacks();
    this.loadTimeline();
    this.refreshBeatStatus();
    if (!(this.viewEnd > this.viewStart)) this.zoomToFit(false);
    this.scheduleDraw();
  }

  fps() {
    return Math.max(1, finiteNumber(this.widgets.fps?.value, 24));
  }

  configuredFrameCount() {
    return Math.max(0, Math.round(finiteNumber(this.widgets.sequenceDuration?.value, 0)));
  }

  defaultFadeIn() {
    return Math.max(0, finiteNumber(this.widgets.defaultFadeIn?.value, 0));
  }

  defaultFadeOut() {
    return Math.max(0, finiteNumber(this.widgets.defaultFadeOut?.value, 0));
  }

  build() {
    this.root = document.createElement("div");
    this.root.className = "flbps-root";
    this.root.tabIndex = 0;
    this.root.innerHTML = `
      <div class="flbps-header">
        <span class="flbps-title">Audio Beat Prompt Sequencer</span>
        <span class="flbps-status" data-role="status">Run once to load exact beat markers</span>
      </div>
      <div class="flbps-toolbar">
        <label class="flbps-control">Snap
          <select data-role="snap" title="Beat snaps edits to detected beats. Frame uses single-frame steps. Off disables beat snapping.">
            <option value="beat">Beat</option>
            <option value="frame">Frame</option>
            <option value="off">Off</option>
          </select>
        </label>
        <button class="flbps-button" data-action="zoom-out" title="Show more frames">Zoom -</button>
        <button class="flbps-button" data-action="zoom-in" title="Show fewer frames">Zoom +</button>
        <button class="flbps-button" data-action="fit" title="Show the complete frame range">Fit</button>
        <button class="flbps-button" data-action="waveform" title="Show or hide the aligned audio waveform">Waveform</button>
        <span class="flbps-spacer"></span>
        <span class="flbps-control">Frame-native timeline</span>
      </div>
      <div class="flbps-error" data-role="error"></div>
      <div class="flbps-canvas-wrap">
        <canvas class="flbps-canvas"></canvas>
        <div class="flbps-empty" data-role="empty"></div>
      </div>
      <div class="flbps-actions">
        <button class="flbps-button primary" data-action="add">+ Prompt</button>
        <button class="flbps-button" data-action="split">Split</button>
        <button class="flbps-button" data-action="duplicate">Duplicate</button>
        <button class="flbps-button danger" data-action="delete">Delete</button>
        <span class="flbps-spacer"></span>
        <button class="flbps-button" data-action="raw">Raw frames</button>
      </div>
      <div class="flbps-inspector disabled" data-role="inspector">
        <div class="flbps-inspector-grid">
          <div class="flbps-field"><label>Start frame</label><input data-field="start" type="number" min="0" step="1"></div>
          <div class="flbps-field"><label>End frame</label><input data-field="end" type="number" min="1" step="1"></div>
          <div class="flbps-field"><label>Duration</label><input data-field="clip-duration" type="text" readonly></div>
          <div class="flbps-field"><label>Fade in frames</label><input data-field="fade-in" type="number" min="0" step="1"></div>
          <div class="flbps-field"><label>Fade out frames</label><input data-field="fade-out" type="number" min="0" step="1"></div>
        </div>
        <div class="flbps-prompt-label">
          <span>Prompt</span><span class="flbps-prompt-meta" data-role="prompt-meta"></span>
        </div>
        <textarea data-field="prompt" placeholder="Describe what should happen during this frame range."></textarea>
      </div>
      <div class="flbps-raw" data-role="raw-panel">
        <div class="flbps-raw-label">Advanced frame schedule. All positions and fades must be integer frames.</div>
        <textarea data-role="raw-text"></textarea>
        <div class="flbps-raw-actions">
          <button class="flbps-button" data-action="raw-cancel">Close</button>
          <button class="flbps-button primary" data-action="raw-apply">Apply frames</button>
        </div>
      </div>
      <div class="flbps-footer">
        <span class="flbps-summary" data-role="summary"></span>
        <span>drag edges/handles · wheel zoom · Shift+wheel pan · Shift bypasses beat snap</span>
      </div>
    `;
    this.container.appendChild(this.root);

    this.statusEl = this.root.querySelector('[data-role="status"]');
    this.errorEl = this.root.querySelector('[data-role="error"]');
    this.emptyEl = this.root.querySelector('[data-role="empty"]');
    this.canvas = this.root.querySelector(".flbps-canvas");
    this.inspector = this.root.querySelector('[data-role="inspector"]');
    this.rawPanel = this.root.querySelector('[data-role="raw-panel"]');
    this.rawText = this.root.querySelector('[data-role="raw-text"]');
    this.summaryEl = this.root.querySelector('[data-role="summary"]');
    this.promptMetaEl = this.root.querySelector('[data-role="prompt-meta"]');
    this.controls = {
      snap: this.root.querySelector('[data-role="snap"]'),
    };
    this.fields = {
      start: this.root.querySelector('[data-field="start"]'),
      end: this.root.querySelector('[data-field="end"]'),
      duration: this.root.querySelector('[data-field="clip-duration"]'),
      fadeIn: this.root.querySelector('[data-field="fade-in"]'),
      fadeOut: this.root.querySelector('[data-field="fade-out"]'),
      prompt: this.root.querySelector('[data-field="prompt"]'),
    };
    this.editButtons = [
      ...this.root.querySelectorAll('[data-action="add"], [data-action="split"], [data-action="duplicate"], [data-action="delete"]'),
    ];

    this.controls.snap.value = this.snapMode;
    this.waveformButton = this.root.querySelector('[data-action="waveform"]');
    this.waveformButton.classList.toggle("active", this.waveformVisible);
    this.controls.snap.addEventListener("change", () => {
      this.snapMode = this.controls.snap.value;
      this.saveViewState();
      this.scheduleDraw();
    });

    this.root.querySelector('[data-action="zoom-out"]').addEventListener("click", () => this.zoom(1.5));
    this.root.querySelector('[data-action="zoom-in"]').addEventListener("click", () => this.zoom(0.65));
    this.root.querySelector('[data-action="fit"]').addEventListener("click", () => this.zoomToFit());
    this.waveformButton.addEventListener("click", () => {
      this.waveformVisible = !this.waveformVisible;
      this.waveformButton.classList.toggle("active", this.waveformVisible);
      this.saveViewState();
      this.scheduleDraw();
    });
    this.root.querySelector('[data-action="add"]').addEventListener("click", () => this.addClip());
    this.root.querySelector('[data-action="split"]').addEventListener("click", () => this.splitClip());
    this.root.querySelector('[data-action="duplicate"]').addEventListener("click", () => this.duplicateClip());
    this.root.querySelector('[data-action="delete"]').addEventListener("click", () => this.deleteClip());
    this.root.querySelector('[data-action="raw"]').addEventListener("click", () => this.toggleRaw());
    this.root.querySelector('[data-action="raw-cancel"]').addEventListener("click", () => this.toggleRaw(false));
    this.root.querySelector('[data-action="raw-apply"]').addEventListener("click", () => this.applyRaw());

    for (const name of ["start", "end", "fadeIn", "fadeOut"]) {
      this.fields[name].addEventListener("change", () => this.applyInspectorTiming());
    }
    this.fields.prompt.addEventListener("input", () => {
      const clip = this.selectedClip();
      if (!clip) return;
      clip.prompt = this.fields.prompt.value;
      this.serialize();
      this.scheduleDraw();
    });

    this.canvas.addEventListener("pointerdown", (event) => this.onPointerDown(event));
    this.canvas.addEventListener("pointermove", (event) => this.onPointerMove(event));
    this.canvas.addEventListener("pointerleave", () => {
      if (!this.drag) {
        this.hover = null;
        this.canvas.style.cursor = "default";
        this.scheduleDraw();
      }
    });
    this.canvas.addEventListener("pointerup", (event) => this.onPointerUp(event));
    this.canvas.addEventListener("pointercancel", (event) => this.onPointerUp(event));
    this.canvas.addEventListener("dblclick", (event) => this.addClipAtPointer(event));
    this.canvas.addEventListener("wheel", (event) => this.onWheel(event), { passive: false });
    this.root.addEventListener("keydown", (event) => this.onKeyDown(event));

    this.resizeObserver = new ResizeObserver(() => this.scheduleDraw());
    this.resizeObserver.observe(this.canvas.parentElement);
  }

  bindWidgetCallbacks() {
    const bind = (widget, callback) => {
      if (!widget) return;
      const original = widget.callback;
      widget.callback = (value) => {
        original?.call(widget, value);
        callback(value);
      };
      this.callbackRestorers.push(() => {
        widget.callback = original;
      });
    };

    bind(this.widgets.timeUnit, () => this.loadTimeline());
    bind(this.widgets.fps, () => {
      this.syncInspector();
      this.zoomToFit();
      this.markDirty();
    });
    bind(this.widgets.sequenceDuration, () => {
      this.zoomToFit();
      this.markDirty();
    });
    bind(this.widgets.defaultFadeIn, () => this.markDirty());
    bind(this.widgets.defaultFadeOut, () => this.markDirty());
    bind(this.widgets.curve, () => this.markDirty());
  }

  markDirty() {
    this.node.graph?.change?.();
  }

  saveViewState() {
    this.node.properties = this.node.properties || {};
    this.node.properties.flBeatPromptSequencer = {
      ...(this.node.properties.flBeatPromptSequencer || {}),
      formatVersion: FORMAT_VERSION,
      beatData: this.beatData,
      viewStart: this.viewStart,
      viewEnd: this.viewEnd,
      snapMode: this.snapMode,
      waveformVisible: this.waveformVisible,
    };
    this.markDirty();
  }

  sourceUnit() {
    return this.widgets.timeUnit?.value || "frames";
  }

  sourceToSeconds(value, unit) {
    if (unit === "seconds") return value;
    if (unit === "frames") return value / this.fps();
    const beats = this.beatData?.beatTimes;
    const duration = this.beatData?.audioDuration;
    if (!beats?.length || !(duration > 0)) return null;
    if (value > beats.length + EPSILON) return null;
    if (value >= beats.length - EPSILON) return duration;
    const index = Math.max(0, Math.floor(value));
    const amount = value - index;
    const start = beats[index];
    const end = index + 1 < beats.length ? beats[index + 1] : duration;
    return start + (end - start) * amount;
  }

  migrateLegacyClips(clips, unit) {
    if (unit === "beats" && !this.beatData?.beatTimes?.length) return null;
    const fps = this.fps();
    return clips.map((clip) => {
      const startSeconds = this.sourceToSeconds(clip.start, unit);
      const endSeconds = this.sourceToSeconds(clip.end, unit);
      const fadeInSeconds = this.sourceToSeconds(clip.start + clip.fadeIn, unit);
      const fadeOutSeconds = this.sourceToSeconds(clip.end - clip.fadeOut, unit);
      if ([startSeconds, endSeconds, fadeInSeconds, fadeOutSeconds].some((value) => value == null)) {
        return null;
      }
      const start = Math.max(0, Math.round(startSeconds * fps));
      const end = Math.max(start + 1, Math.round(endSeconds * fps));
      const fadeInEnd = clamp(Math.round(fadeInSeconds * fps), start, end);
      const fadeOutStart = clamp(Math.round(fadeOutSeconds * fps), start, end);
      return {
        ...clip,
        start,
        end,
        fadeIn: fadeInEnd - start,
        fadeOut: end - fadeOutStart,
      };
    });
  }

  finishMigration(clips) {
    if (clips.some((clip) => !clip)) {
      throw new Error("The legacy schedule could not be converted to frames.");
    }
    validateFrameClips(clips);
    for (let index = 1; index < clips.length; index++) {
      if (clips[index].start < clips[index - 1].end) {
        throw new Error(`Line ${clips[index].line}: frame conversion makes this section overlap the previous section.`);
      }
    }
    this.clips = clips;
    this.widgets.timeUnit.value = "frames";
    this.widgets.defaultFadeIn.value = 0;
    this.widgets.defaultFadeOut.value = 0;
    this.migrationPending = false;
    this.rawInvalid = false;
    this.clearError();
    this.serialize();
    this.saveViewState();
  }

  loadTimeline() {
    const raw = this.widgets.timeline?.value || "";
    const unit = this.sourceUnit();
    try {
      const clips = parseTimeline(raw, this.defaultFadeIn(), this.defaultFadeOut());
      if (unit === "frames") {
        this.clips = validateFrameClips(clips);
        this.migrationPending = false;
        this.rawInvalid = false;
        this.clearError();
        this.saveViewState();
      } else {
        const migrated = this.migrateLegacyClips(clips, unit);
        if (!migrated) {
          this.clips = [];
          this.selectedIndex = -1;
          this.migrationPending = true;
          this.rawInvalid = false;
          this.showMigration();
        } else {
          this.finishMigration(migrated);
        }
      }
      if (this.selectedIndex >= this.clips.length) this.selectedIndex = -1;
    } catch (error) {
      this.clips = [];
      this.selectedIndex = -1;
      this.migrationPending = false;
      this.rawInvalid = true;
      this.showError(error.message);
      this.rawText.value = raw;
      this.toggleRaw(true);
    }
    this.setEditorEnabled(!this.migrationPending && !this.rawInvalid);
    this.syncInspector();
    this.scheduleDraw();
  }

  frameClipsFromPayload(sections) {
    return sections.map((section) => ({
      line: finiteNumber(section.line, 0),
      start: Math.round(finiteNumber(section.start_frame)),
      end: Math.round(finiteNumber(section.end_frame)),
      fadeIn: Math.round(finiteNumber(section.fade_in_frames)),
      fadeOut: Math.round(finiteNumber(section.fade_out_frames)),
      prompt: String(section.prompt || ""),
    }));
  }

  updateFromExecution(message) {
    const values = message?.fl_prompt_sequencer ?? message?.ui?.fl_prompt_sequencer;
    const payload = Array.isArray(values) ? values[0] : values;
    if (!payload || !Array.isArray(payload.beat_times)) return;
    this.beatData = {
      bpm: finiteNumber(payload.bpm),
      beatTimes: payload.beat_times.map((value) => finiteNumber(value)),
      audioDuration: finiteNumber(payload.audio_duration),
      fps: finiteNumber(payload.fps, this.fps()),
      waveformPreview: normalizeWaveformPreview(payload.waveform_preview),
    };
    this.dataFresh = true;

    const sourceUnit = payload.source_unit || payload.time_unit || this.sourceUnit();
    if (sourceUnit !== "frames" && Array.isArray(payload.frame_sections)) {
      try {
        this.finishMigration(this.frameClipsFromPayload(payload.frame_sections));
      } catch (error) {
        this.showError(error.message);
      }
    } else {
      this.loadTimeline();
    }
    this.saveViewState();
    this.zoomToFit();
    this.refreshBeatStatus();
  }

  markBeatDataCached() {
    if (!this.beatData) return;
    this.dataFresh = false;
    this.refreshBeatStatus();
  }

  setEditorEnabled(enabled) {
    for (const button of this.editButtons) button.disabled = !enabled;
    this.inspector.classList.toggle("disabled", !enabled || !this.selectedClip());
  }

  showMigration() {
    this.errorEl.textContent = "Run this node once to convert the legacy beat schedule to integer frames. The resolved timing is preserved.";
    this.errorEl.classList.add("open");
    this.statusEl.className = "flbps-status error";
    this.statusEl.textContent = "Legacy beat schedule needs one run";
  }

  refreshBeatStatus() {
    if (this.migrationPending) {
      this.showMigration();
      return;
    }
    this.statusEl.className = "flbps-status";
    if (!this.beatData) {
      this.statusEl.textContent = "Run once to load exact beat markers";
      return;
    }
    const count = this.beatData.beatTimes?.length || 0;
    const text = `${finiteNumber(this.beatData.bpm).toFixed(2)} BPM · ${count} beats · ` +
      `${finiteNumber(this.beatData.audioDuration).toFixed(2)} sec`;
    if (this.dataFresh) {
      this.statusEl.classList.add("fresh");
      this.statusEl.textContent = text;
    } else {
      this.statusEl.classList.add("cached");
      this.statusEl.textContent = `${text} · cached`;
    }
    this.updateSummary();
  }

  showError(message) {
    this.errorEl.textContent = message;
    this.errorEl.classList.add("open");
    this.statusEl.className = "flbps-status error";
    this.statusEl.textContent = "Schedule source needs attention";
  }

  clearError() {
    this.errorEl.textContent = "";
    this.errorEl.classList.remove("open");
    this.refreshBeatStatus();
  }

  selectedClip() {
    return this.selectedIndex >= 0 ? this.clips[this.selectedIndex] : null;
  }

  select(index) {
    this.selectedIndex = index >= 0 && index < this.clips.length ? index : -1;
    this.syncInspector();
    this.scheduleDraw();
  }

  nearestBeatLabel(frame) {
    const frames = this.beatFrames();
    if (!frames.length) return "unavailable";
    let nearest = 0;
    for (let index = 1; index < frames.length; index++) {
      if (Math.abs(frames[index] - frame) < Math.abs(frames[nearest] - frame)) nearest = index;
    }
    return `B${nearest}`;
  }

  syncInspector() {
    const clip = this.selectedClip();
    this.inspector.classList.toggle("disabled", this.migrationPending || !clip);
    if (!clip) {
      for (const field of Object.values(this.fields)) field.value = "";
      this.promptMetaEl.textContent = "";
      return;
    }
    this.fields.start.value = String(clip.start);
    this.fields.end.value = String(clip.end);
    this.fields.fadeIn.value = String(clip.fadeIn);
    this.fields.fadeOut.value = String(clip.fadeOut);
    this.fields.prompt.value = clip.prompt;
    const frames = clip.end - clip.start;
    this.fields.duration.value = `${frames} frames / ${(frames / this.fps()).toFixed(3)}s`;
    this.promptMetaEl.textContent =
      `frames ${clip.start}–${clip.end} · ${formatClock(clip.start / this.fps())}–${formatClock(clip.end / this.fps())} · ` +
      `beats ${this.nearestBeatLabel(clip.start)}–${this.nearestBeatLabel(clip.end)}`;
  }

  applyInspectorTiming() {
    const clip = this.selectedClip();
    if (!clip || this.migrationPending) return;
    const start = Math.max(0, Math.round(finiteNumber(this.fields.start.value, clip.start)));
    const end = Math.round(finiteNumber(this.fields.end.value, clip.end));
    const fadeIn = Math.max(0, Math.round(finiteNumber(this.fields.fadeIn.value, clip.fadeIn)));
    const fadeOut = Math.max(0, Math.round(finiteNumber(this.fields.fadeOut.value, clip.fadeOut)));
    if (!(end > start)) {
      this.showError("The selected prompt must end after it starts.");
      this.syncInspector();
      return;
    }
    if (fadeIn + fadeOut > end - start) {
      this.showError("Fade in and fade out exceed the selected prompt duration.");
      this.syncInspector();
      return;
    }
    const previous = this.clips[this.selectedIndex - 1];
    const next = this.clips[this.selectedIndex + 1];
    const maximum = this.maximumFrame();
    if ((previous && start < previous.end) || (next && end > next.start) ||
        (Number.isFinite(maximum) && end > maximum)) {
      this.showError("Prompt clips cannot overlap or extend beyond the configured length.");
      this.syncInspector();
      return;
    }
    Object.assign(clip, { start, end, fadeIn, fadeOut });
    this.rawInvalid = false;
    this.clearError();
    this.serialize();
    this.syncInspector();
    this.scheduleDraw();
  }

  toggleRaw(force) {
    const open = typeof force === "boolean" ? force : !this.rawPanel.classList.contains("open");
    if (open) this.rawText.value = this.widgets.timeline?.value || "";
    this.rawPanel.classList.toggle("open", open);
  }

  applyRaw() {
    try {
      const clips = validateFrameClips(parseTimeline(
        this.rawText.value,
        Math.round(this.defaultFadeIn()),
        Math.round(this.defaultFadeOut()),
      ));
      this.widgets.timeUnit.value = "frames";
      this.clips = clips;
      this.selectedIndex = clips.length ? 0 : -1;
      this.migrationPending = false;
      this.rawInvalid = false;
      this.clearError();
      this.serialize();
      this.toggleRaw(false);
      this.setEditorEnabled(true);
      this.zoomToFit();
      this.syncInspector();
    } catch (error) {
      this.rawInvalid = true;
      this.showError(error.message);
    }
  }

  serialize() {
    if (this.rawInvalid || this.migrationPending || !this.widgets.timeline) return;
    this.widgets.timeline.value = serializeTimeline(this.clips);
    this.rawText.value = this.widgets.timeline.value;
    this.markDirty();
    this.updateSummary();
  }

  beatFrames() {
    return (this.beatData?.beatTimes || []).map((seconds) => Math.round(seconds * this.fps()));
  }

  sequenceFrameCount() {
    const configured = this.configuredFrameCount();
    if (configured > 0) return configured;
    if (this.beatData?.audioDuration > 0) return Math.max(1, Math.round(this.beatData.audioDuration * this.fps()));
    const last = this.clips[this.clips.length - 1];
    return Math.max(1, last?.end || Math.round(this.fps() * 8));
  }

  maximumFrame() {
    const configured = this.configuredFrameCount();
    if (configured > 0) return configured;
    if (this.beatData?.audioDuration > 0) return Math.max(1, Math.round(this.beatData.audioDuration * this.fps()));
    return Infinity;
  }

  snapFrame(value, bypassBeat = false) {
    const frame = Math.max(0, Math.round(value));
    if (bypassBeat || this.snapMode !== "beat") return frame;
    const beats = this.beatFrames();
    if (!beats.length) return frame;
    let nearest = beats[0];
    for (let index = 1; index < beats.length; index++) {
      if (Math.abs(beats[index] - frame) < Math.abs(nearest - frame)) nearest = beats[index];
    }
    return nearest;
  }

  defaultClipLength() {
    return Math.max(1, Math.round(this.fps() * 2));
  }

  addClip(startOverride = null) {
    if (this.migrationPending) return;
    const previousEnd = this.clips.length ? this.clips[this.clips.length - 1].end : 0;
    const start = startOverride == null ? previousEnd : this.snapFrame(startOverride);
    let end = start + this.defaultClipLength();
    const maximum = this.maximumFrame();
    if (Number.isFinite(maximum)) {
      if (start >= maximum) {
        this.showError("There is no room for another prompt inside the configured frame length.");
        return;
      }
      end = Math.min(end, maximum);
    }
    const duration = end - start;
    const fadeIn = Math.min(Math.round(this.defaultFadeIn()), duration);
    const fadeOut = Math.min(Math.round(this.defaultFadeOut()), duration - fadeIn);
    const clip = { start, end, fadeIn, fadeOut, prompt: "Describe this prompt section." };
    let index = this.clips.findIndex((item) => item.start > start);
    if (index < 0) index = this.clips.length;
    const previous = this.clips[index - 1];
    const next = this.clips[index];
    if ((previous && start < previous.end) || (next && end > next.start)) {
      this.showError("The new prompt would overlap an existing clip.");
      return;
    }
    this.clips.splice(index, 0, clip);
    this.rawInvalid = false;
    this.clearError();
    this.select(index);
    this.serialize();
    this.scheduleDraw();
  }

  addClipAtPointer(event) {
    if (this.migrationPending) return;
    this.addClip(this.frameAtEvent(event));
  }

  deleteClip() {
    if (!this.selectedClip()) return;
    this.clips.splice(this.selectedIndex, 1);
    this.selectedIndex = Math.min(this.selectedIndex, this.clips.length - 1);
    this.serialize();
    this.syncInspector();
    this.scheduleDraw();
  }

  duplicateClip() {
    const clip = this.selectedClip();
    if (!clip) return;
    const duration = clip.end - clip.start;
    const start = clip.end;
    const end = start + duration;
    const next = this.clips[this.selectedIndex + 1];
    const maximum = this.maximumFrame();
    if ((next && end > next.start) || (Number.isFinite(maximum) && end > maximum)) {
      this.showError("There is not enough room after this prompt to duplicate it.");
      return;
    }
    this.clips.splice(this.selectedIndex + 1, 0, { ...clip, start, end });
    this.select(this.selectedIndex + 1);
    this.clearError();
    this.serialize();
  }

  splitClip() {
    const clip = this.selectedClip();
    if (!clip) return;
    let split = this.playheadFrame == null ? Math.round((clip.start + clip.end) / 2) : this.playheadFrame;
    split = this.snapFrame(split);
    if (split <= clip.start || split >= clip.end) {
      this.showError("Place the playhead inside the selected prompt before splitting.");
      return;
    }
    const first = {
      ...clip,
      end: split,
      fadeIn: Math.min(clip.fadeIn, split - clip.start),
      fadeOut: 0,
    };
    const second = {
      ...clip,
      start: split,
      fadeIn: 0,
      fadeOut: Math.min(clip.fadeOut, clip.end - split),
    };
    this.clips.splice(this.selectedIndex, 1, first, second);
    this.select(this.selectedIndex + 1);
    this.clearError();
    this.serialize();
  }

  frameAtX(x) {
    const width = Math.max(1, this.canvas.clientWidth);
    const right = width - TIMELINE_RIGHT;
    const clampedX = clamp(x, TIMELINE_LEFT, right);
    const ratio = (clampedX - TIMELINE_LEFT) / Math.max(1, right - TIMELINE_LEFT);
    return this.viewStart + ratio * (this.viewEnd - this.viewStart);
  }

  frameAtEvent(event) {
    return this.frameAtX(this.eventPosition(event).x);
  }

  eventPosition(event) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * (this.canvas.clientWidth / Math.max(1, rect.width)),
      y: (event.clientY - rect.top) * (this.canvas.clientHeight / Math.max(1, rect.height)),
    };
  }

  onPointerDown(event) {
    if (this.migrationPending || this.rawInvalid) return;
    this.root.focus({ preventScroll: true });
    const { x, y } = this.eventPosition(event);
    const hit = this.hitTest(x, y);
    if (!hit) {
      this.playheadFrame = this.snapFrame(this.frameAtX(x), event.shiftKey);
      this.scheduleDraw();
      return;
    }

    this.select(hit.index);
    const clip = this.selectedClip();
    this.drag = {
      type: hit.type,
      pointerStart: this.snapFrame(this.frameAtX(x), event.shiftKey),
      pointerStartRaw: this.snapFrame(this.frameAtX(x), true),
      pointerStartX: x,
      pointerStartY: y,
      pointerX: x,
      pointerY: y,
      original: { ...clip },
      active: false,
    };
    this.canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  panDuringDrag(x) {
    const width = Math.max(1, this.canvas.clientWidth);
    const range = this.viewEnd - this.viewStart;
    const margin = 28;
    let shift = 0;
    if (x < TIMELINE_LEFT + margin) shift = -Math.max(1, range * 0.025);
    if (x > width - TIMELINE_RIGHT - margin) shift = Math.max(1, range * 0.025);
    if (!shift) return;
    const duration = this.sequenceFrameCount();
    this.viewStart = clamp(this.viewStart + shift, 0, Math.max(0, duration - range));
    this.viewEnd = Math.min(duration, this.viewStart + range);
  }

  updateDrag(x, shiftKey) {
    const clip = this.selectedClip();
    if (!this.drag || !clip) return;
    this.panDuringDrag(x);
    const current = this.snapFrame(this.frameAtX(x), shiftKey);
    const pointerStart = shiftKey ? this.drag.pointerStartRaw : this.drag.pointerStart;
    const delta = current - pointerStart;
    const original = this.drag.original;
    const previous = this.clips[this.selectedIndex - 1];
    const next = this.clips[this.selectedIndex + 1];
    const maximum = this.maximumFrame();

    if (this.drag.type === "move") {
      const duration = original.end - original.start;
      let start = Math.max(0, original.start + delta);
      if (previous) start = Math.max(start, previous.end);
      if (next) start = Math.min(start, next.start - duration);
      if (Number.isFinite(maximum)) start = Math.min(start, maximum - duration);
      clip.start = Math.round(start);
      clip.end = clip.start + duration;
    } else if (this.drag.type === "start") {
      let start = Math.min(original.end - 1, original.start + delta);
      start = Math.max(previous?.end || 0, start);
      clip.start = Math.round(start);
      clip.fadeIn = Math.min(original.fadeIn, clip.end - clip.start - clip.fadeOut);
    } else if (this.drag.type === "end") {
      let end = Math.max(original.start + 1, original.end + delta);
      if (next) end = Math.min(end, next.start);
      if (Number.isFinite(maximum)) end = Math.min(end, maximum);
      clip.end = Math.round(end);
      clip.fadeOut = Math.min(original.fadeOut, clip.end - clip.start - clip.fadeIn);
    } else if (this.drag.type === "fade-in") {
      clip.fadeIn = clamp(current - clip.start, 0, clip.end - clip.start - clip.fadeOut);
    } else if (this.drag.type === "fade-out") {
      clip.fadeOut = clamp(clip.end - current, 0, clip.end - clip.start - clip.fadeIn);
    }

    clip.fadeIn = Math.max(0, Math.round(clip.fadeIn));
    clip.fadeOut = Math.max(0, Math.round(clip.fadeOut));
    this.snapGuideFrame = current;
    this.syncInspector();
    this.scheduleDraw();
  }

  onPointerMove(event) {
    const { x, y } = this.eventPosition(event);
    this.hover = { x, y };
    if (!this.drag || !this.selectedClip()) {
      const hit = this.hitTest(x, y);
      this.canvas.style.cursor = hit
        ? hit.type === "move" ? "grab" : "ew-resize"
        : "default";
      this.scheduleDraw();
      return;
    }

    this.drag.pointerX = x;
    this.drag.pointerY = y;
    if (!this.drag.active) {
      const distance = Math.hypot(x - this.drag.pointerStartX, y - this.drag.pointerStartY);
      if (Number.isFinite(distance) && distance < 3) return;
      this.drag.active = true;
    }
    this.canvas.style.cursor = this.drag.type === "move" ? "grabbing" : "ew-resize";
    this.updateDrag(x, event.shiftKey);
    event.preventDefault();
  }

  onPointerUp(event) {
    if (!this.drag) return;
    const changed = this.drag.active;
    this.drag = null;
    this.snapGuideFrame = null;
    this.canvas.style.cursor = "default";
    if (this.canvas.hasPointerCapture(event.pointerId)) this.canvas.releasePointerCapture(event.pointerId);
    if (changed) {
      this.serialize();
      this.clearError();
      this.saveViewState();
    }
    this.scheduleDraw();
  }

  onWheel(event) {
    event.preventDefault();
    event.stopPropagation();
    const duration = this.sequenceFrameCount();
    const total = Math.max(1, this.viewEnd - this.viewStart);
    if (event.shiftKey) {
      const shift = total * Math.sign(event.deltaY) * 0.12;
      this.viewStart = clamp(this.viewStart + shift, 0, Math.max(0, duration - total));
      this.viewEnd = Math.min(duration, this.viewStart + total);
    } else {
      const center = this.frameAtEvent(event);
      const factor = event.deltaY > 0 ? 1.18 : 0.84;
      const minimum = Math.min(duration, Math.max(1, Math.round(this.fps() / 2)));
      const nextRange = clamp(total * factor, minimum, duration);
      const ratio = (center - this.viewStart) / total;
      this.viewStart = clamp(center - nextRange * ratio, 0, Math.max(0, duration - nextRange));
      this.viewEnd = Math.min(duration, this.viewStart + nextRange);
      this.viewStart = Math.max(0, this.viewEnd - nextRange);
    }
    this.saveViewState();
    this.scheduleDraw();
  }

  zoom(factor) {
    const duration = this.sequenceFrameCount();
    const total = Math.max(1, this.viewEnd - this.viewStart);
    const center = (this.viewStart + this.viewEnd) / 2;
    const minimum = Math.min(duration, Math.max(1, Math.round(this.fps() / 2)));
    const nextRange = clamp(total * factor, minimum, duration);
    this.viewStart = clamp(center - nextRange / 2, 0, Math.max(0, duration - nextRange));
    this.viewEnd = Math.min(duration, this.viewStart + nextRange);
    this.saveViewState();
    this.scheduleDraw();
  }

  onKeyDown(event) {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLSelectElement) {
      return;
    }
    if ((event.key === "Delete" || event.key === "Backspace") && this.selectedClip()) {
      this.deleteClip();
      event.preventDefault();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "d") {
      this.duplicateClip();
      event.preventDefault();
      return;
    }
    if ((event.key === "ArrowLeft" || event.key === "ArrowRight") && this.selectedClip()) {
      const clip = this.selectedClip();
      const direction = event.key === "ArrowLeft" ? -1 : 1;
      const delta = direction * (event.shiftKey ? 10 : 1);
      const previous = this.clips[this.selectedIndex - 1];
      const next = this.clips[this.selectedIndex + 1];
      const maximum = this.maximumFrame();
      if ((!previous || clip.start + delta >= previous.end) &&
          (!next || clip.end + delta <= next.start) &&
          (!Number.isFinite(maximum) || clip.end + delta <= maximum) &&
          clip.start + delta >= 0) {
        clip.start += delta;
        clip.end += delta;
        this.serialize();
        this.syncInspector();
        this.scheduleDraw();
      }
      event.preventDefault();
    }
  }

  hitTest(x, y) {
    for (let index = this.clipRects.length - 1; index >= 0; index--) {
      const rect = this.clipRects[index];
      if (y < rect.y || y > rect.y + rect.height) continue;
      const selected = rect.index === this.selectedIndex;
      if (selected && y <= rect.y + 20) {
        if (Math.abs(x - rect.fadeInX) <= 10) return { index: rect.index, type: "fade-in" };
        if (Math.abs(x - rect.fadeOutX) <= 10) return { index: rect.index, type: "fade-out" };
      }
      if (Math.abs(x - rect.x) <= 14) return { index: rect.index, type: "start" };
      if (Math.abs(x - (rect.x + rect.width)) <= 14) return { index: rect.index, type: "end" };
      if (x >= rect.x && x <= rect.x + rect.width) return { index: rect.index, type: "move" };
    }
    return null;
  }

  zoomToFit(save = true) {
    this.viewStart = 0;
    this.viewEnd = this.sequenceFrameCount();
    if (save) this.saveViewState();
    this.scheduleDraw();
  }

  frameToX(frame, width) {
    const right = width - TIMELINE_RIGHT;
    return TIMELINE_LEFT +
      ((frame - this.viewStart) / Math.max(EPSILON, this.viewEnd - this.viewStart)) *
      (right - TIMELINE_LEFT);
  }

  scheduleDraw() {
    if (this.pendingFrame) return;
    this.pendingFrame = requestAnimationFrame(() => {
      this.pendingFrame = null;
      this.draw();
    });
  }

  drawWaveformLane(ctx, width, top, bottom) {
    const right = width - TIMELINE_RIGHT;
    const center = (top + bottom) / 2;
    const preview = this.beatData?.waveformPreview;

    ctx.fillStyle = "#121b20";
    ctx.fillRect(TIMELINE_LEFT, top, right - TIMELINE_LEFT, bottom - top);
    ctx.strokeStyle = "#26343b";
    ctx.strokeRect(TIMELINE_LEFT + 0.5, top + 0.5, right - TIMELINE_LEFT - 1, bottom - top - 1);
    ctx.fillStyle = "#71717a";
    ctx.font = "8px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText("WAVE", TIMELINE_LEFT - 5, center);

    const selected = this.selectedClip();
    if (selected) {
      const selectionStart = clamp(this.frameToX(selected.start, width), TIMELINE_LEFT, right);
      const selectionEnd = clamp(this.frameToX(selected.end, width), TIMELINE_LEFT, right);
      if (selectionEnd > selectionStart) {
        ctx.fillStyle = "rgba(34,211,238,.10)";
        ctx.fillRect(selectionStart, top + 1, selectionEnd - selectionStart, bottom - top - 2);
      }
    }

    ctx.strokeStyle = "#33444c";
    ctx.beginPath();
    ctx.moveTo(TIMELINE_LEFT, center + 0.5);
    ctx.lineTo(right, center + 0.5);
    ctx.stroke();

    if (!preview) {
      ctx.fillStyle = "#64748b";
      ctx.font = "9px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Run FL Audio BPM Analyzer once to load the waveform", (TIMELINE_LEFT + right) / 2, center);
      return;
    }

    const binCount = preview.peaks.length / 2;
    const waveformEndFrame = preview.duration * this.fps();
    const visibleStart = Math.max(this.viewStart, 0);
    const visibleEnd = Math.min(this.viewEnd, waveformEndFrame);
    if (!(visibleEnd > visibleStart)) return;

    const startX = Math.max(TIMELINE_LEFT, Math.floor(this.frameToX(visibleStart, width)));
    const endX = Math.min(right, Math.ceil(this.frameToX(visibleEnd, width)));
    const plotHeight = Math.max(1, (bottom - top) / 2 - 5);
    ctx.strokeStyle = "#38bdf8";
    ctx.globalAlpha = 0.82;
    ctx.beginPath();
    for (let x = startX; x <= endX; x++) {
      const firstFrame = this.frameAtX(x);
      const lastFrame = this.frameAtX(Math.min(endX, x + 1));
      const firstBin = clamp(
        Math.floor((firstFrame / this.fps() / preview.duration) * binCount),
        0,
        binCount - 1,
      );
      const lastBin = clamp(
        Math.ceil((lastFrame / this.fps() / preview.duration) * binCount),
        firstBin + 1,
        binCount,
      );
      let minimum = preview.scale;
      let maximum = -preview.scale;
      for (let bin = firstBin; bin < lastBin; bin++) {
        minimum = Math.min(minimum, preview.peaks[bin * 2]);
        maximum = Math.max(maximum, preview.peaks[bin * 2 + 1]);
      }
      ctx.moveTo(x + 0.5, center - (maximum / preview.scale) * plotHeight);
      ctx.lineTo(x + 0.5, center - (minimum / preview.scale) * plotHeight);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    if (this.hover?.y < top || this.hover?.y > bottom ||
        this.hover.x < TIMELINE_LEFT || this.hover.x > right) {
      return;
    }
    const frame = clamp(Math.round(this.frameAtX(this.hover.x)), 0, Math.round(waveformEndFrame));
    const seconds = frame / this.fps();
    const bin = clamp(Math.floor((seconds / preview.duration) * binCount), 0, binCount - 1);
    const amplitude = Math.max(
      Math.abs(preview.peaks[bin * 2]),
      Math.abs(preview.peaks[bin * 2 + 1]),
    ) / preview.scale;
    const x = this.frameToX(frame, width);
    ctx.strokeStyle = "#fbbf24";
    ctx.beginPath();
    ctx.moveTo(x + 0.5, top);
    ctx.lineTo(x + 0.5, bottom);
    ctx.stroke();

    const text = `frame ${frame} · ${formatClock(seconds)} · ${this.nearestBeatLabel(frame)} · ${Math.round(amplitude * 100)}%`;
    ctx.font = "9px Inter, sans-serif";
    const boxWidth = ctx.measureText(text).width + 12;
    const boxX = clamp(x - boxWidth / 2, TIMELINE_LEFT, right - boxWidth);
    ctx.fillStyle = "#082f49";
    ctx.fillRect(boxX, top + 4, boxWidth, 18);
    ctx.fillStyle = "#e0f2fe";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(text, boxX + 6, top + 13);
  }

  drawBeatLane(ctx, width, waveformBottom = null) {
    const right = width - TIMELINE_RIGHT;
    ctx.fillStyle = "#15151a";
    ctx.fillRect(TIMELINE_LEFT, 31, right - TIMELINE_LEFT, 26);
    ctx.fillStyle = "#71717a";
    ctx.font = "8px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText("BEATS", TIMELINE_LEFT - 5, 44);

    const frames = this.beatFrames();
    let hovered = null;
    for (let index = 0; index < frames.length; index++) {
      const frame = frames[index];
      if (frame < this.viewStart || frame > this.viewEnd) continue;
      const x = this.frameToX(frame, width);
      ctx.strokeStyle = "#22d3ee";
      ctx.globalAlpha = 0.72;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 42);
      ctx.lineTo(x + 0.5, 56);
      ctx.stroke();
      if (waveformBottom != null) {
        ctx.globalAlpha = 0.16;
        ctx.beginPath();
        ctx.moveTo(x + 0.5, 57);
        ctx.lineTo(x + 0.5, waveformBottom);
        ctx.stroke();
      }
      ctx.fillStyle = "#67e8f9";
      ctx.globalAlpha = 0.72;
      ctx.beginPath();
      ctx.arc(x, 40, 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      if (this.hover?.y >= 31 && this.hover?.y <= 58 && Math.abs(this.hover.x - x) <= 6) {
        hovered = { index, frame, x };
      }
    }

    if (hovered) {
      const text = `Beat ${hovered.index} · frame ${hovered.frame} · ${formatClock(hovered.frame / this.fps())}`;
      ctx.font = "9px Inter, sans-serif";
      const boxWidth = ctx.measureText(text).width + 12;
      const boxX = clamp(hovered.x - boxWidth / 2, TIMELINE_LEFT, right - boxWidth);
      ctx.fillStyle = "#083344";
      ctx.fillRect(boxX, 59, boxWidth, 18);
      ctx.fillStyle = "#cffafe";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(text, boxX + 6, 68);
    }
  }

  draw() {
    const cssWidth = Math.max(1, this.canvas.clientWidth);
    const cssHeight = Math.max(1, this.canvas.clientHeight);
    const dpr = window.devicePixelRatio || 1;
    if (this.canvas.width !== Math.round(cssWidth * dpr) || this.canvas.height !== Math.round(cssHeight * dpr)) {
      this.canvas.width = Math.round(cssWidth * dpr);
      this.canvas.height = Math.round(cssHeight * dpr);
    }
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);
    ctx.fillStyle = "#101013";
    ctx.fillRect(0, 0, cssWidth, cssHeight);

    const right = cssWidth - TIMELINE_RIGHT;
    const waveformTop = 62;
    const waveformBottom = 132;
    const trackTop = this.waveformVisible ? 142 : 80;
    const trackBottom = cssHeight - 12;
    const trackHeight = Math.max(80, trackBottom - trackTop);

    ctx.fillStyle = "#18181c";
    ctx.fillRect(TIMELINE_LEFT, trackTop, right - TIMELINE_LEFT, trackHeight);
    ctx.strokeStyle = "#303036";
    ctx.strokeRect(TIMELINE_LEFT + 0.5, trackTop + 0.5, right - TIMELINE_LEFT - 1, trackHeight - 1);

    if (this.waveformVisible) {
      this.drawWaveformLane(ctx, cssWidth, waveformTop, waveformBottom);
    }

    const range = Math.max(1, this.viewEnd - this.viewStart);
    const step = niceFrameStep(range, right - TIMELINE_LEFT, this.fps());
    const minor = step % 4 === 0 ? step / 4 : step % 2 === 0 ? step / 2 : step;
    const firstMinor = Math.ceil(this.viewStart / minor) * minor;
    for (let frame = firstMinor; frame <= this.viewEnd + EPSILON; frame += minor) {
      const x = this.frameToX(frame, cssWidth);
      const major = frame % step === 0;
      ctx.strokeStyle = major ? "#34343a" : "#27272c";
      ctx.beginPath();
      ctx.moveTo(x + 0.5, trackTop);
      ctx.lineTo(x + 0.5, trackBottom);
      ctx.stroke();
    }

    const firstTick = Math.ceil(this.viewStart / step) * step;
    ctx.font = "9px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let frame = firstTick; frame <= this.viewEnd + EPSILON; frame += step) {
      const x = this.frameToX(frame, cssWidth);
      ctx.fillStyle = "#a1a1aa";
      ctx.fillText(String(Math.round(frame)), x, 7);
    }
    ctx.fillStyle = "#71717a";
    ctx.textAlign = "right";
    ctx.fillText("FRAMES", TIMELINE_LEFT - 5, 7);

    this.drawBeatLane(ctx, cssWidth, this.waveformVisible ? waveformBottom : null);

    this.clipRects = [];
    for (let index = 0; index < this.clips.length; index++) {
      const clip = this.clips[index];
      if (clip.end < this.viewStart || clip.start > this.viewEnd) continue;
      const startX = this.frameToX(clip.start, cssWidth);
      const endX = this.frameToX(clip.end, cssWidth);
      const fadeInX = this.frameToX(clip.start + clip.fadeIn, cssWidth);
      const fadeOutX = this.frameToX(clip.end - clip.fadeOut, cssWidth);
      const x = clamp(startX, TIMELINE_LEFT, right);
      const clippedEnd = clamp(endX, TIMELINE_LEFT, right);
      const width = Math.max(2, clippedEnd - x);
      const y = trackTop + 14;
      const height = trackHeight - 28;
      const selected = index === this.selectedIndex;

      ctx.fillStyle = selected ? "#155e75" : "#31313a";
      ctx.strokeStyle = selected ? "#67e8f9" : "#52525b";
      ctx.lineWidth = selected ? 2 : 1;
      ctx.beginPath();
      ctx.roundRect(x, y, width, height, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = selected ? "rgba(103,232,249,.17)" : "rgba(161,161,170,.10)";
      if (fadeInX > startX) {
        ctx.beginPath();
        ctx.moveTo(clamp(startX, TIMELINE_LEFT, right), y + height);
        ctx.lineTo(clamp(fadeInX, TIMELINE_LEFT, right), y);
        ctx.lineTo(clamp(fadeInX, TIMELINE_LEFT, right), y + height);
        ctx.closePath();
        ctx.fill();
      }
      if (fadeOutX < endX) {
        ctx.beginPath();
        ctx.moveTo(clamp(fadeOutX, TIMELINE_LEFT, right), y);
        ctx.lineTo(clamp(endX, TIMELINE_LEFT, right), y + height);
        ctx.lineTo(clamp(fadeOutX, TIMELINE_LEFT, right), y + height);
        ctx.closePath();
        ctx.fill();
      }

      if (selected) {
        for (const handleX of [fadeInX, fadeOutX]) {
          if (handleX < TIMELINE_LEFT || handleX > right) continue;
          ctx.fillStyle = "#a5f3fc";
          ctx.beginPath();
          ctx.moveTo(handleX, y + 2);
          ctx.lineTo(handleX - 5, y + 11);
          ctx.lineTo(handleX + 5, y + 11);
          ctx.closePath();
          ctx.fill();
        }
        ctx.fillStyle = "#a5f3fc";
        ctx.fillRect(x, y + 20, Math.min(3, width), Math.max(10, height - 40));
        ctx.fillRect(Math.max(x, x + width - 3), y + 20, Math.min(3, width), Math.max(10, height - 40));
      }

      ctx.save();
      ctx.beginPath();
      ctx.rect(x + 8, y + 8, Math.max(0, width - 16), height - 16);
      ctx.clip();
      ctx.fillStyle = "#fafafa";
      ctx.font = "600 10px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(clip.prompt.replace(/\s+/g, " "), x + 9, y + 10);
      ctx.fillStyle = "#a1a1aa";
      ctx.font = "8px Inter, sans-serif";
      ctx.fillText(`${clip.start}–${clip.end} frames · ${(clip.start / this.fps()).toFixed(2)}–${(clip.end / this.fps()).toFixed(2)}s`, x + 9, y + 28);
      ctx.restore();

      this.clipRects.push({ index, x, y, width, height, fadeInX, fadeOutX });
    }

    if (this.snapGuideFrame != null && this.snapGuideFrame >= this.viewStart && this.snapGuideFrame <= this.viewEnd) {
      const x = this.frameToX(this.snapGuideFrame, cssWidth);
      ctx.strokeStyle = "#f0abfc";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 31);
      ctx.lineTo(x + 0.5, trackBottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (this.playheadFrame != null && this.playheadFrame >= this.viewStart && this.playheadFrame <= this.viewEnd) {
      const x = this.frameToX(this.playheadFrame, cssWidth);
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 4);
      ctx.lineTo(x + 0.5, trackBottom);
      ctx.stroke();
    }

    if (this.migrationPending) {
      this.emptyEl.textContent = "Run once to convert this legacy beat schedule into frames.";
    } else {
      this.emptyEl.textContent = this.clips.length ? "" : "Open Raw to repair the schedule, or add a prompt clip.";
    }
    this.updateSummary();
  }

  updateSummary() {
    const frames = this.sequenceFrameCount();
    const beatCount = this.beatData?.beatTimes?.length || 0;
    const waveformBins = (this.beatData?.waveformPreview?.peaks?.length || 0) / 2;
    this.summaryEl.textContent =
      `${this.clips.length} prompts · ${frames} frames · ${(frames / this.fps()).toFixed(3)}s · ` +
      `${beatCount} beats · ${waveformBins ? `${waveformBins} waveform bins` : "waveform unavailable"}`;
  }

  dispose() {
    if (this.resizeObserver) this.resizeObserver.disconnect();
    if (this.pendingFrame) cancelAnimationFrame(this.pendingFrame);
    for (const restore of this.callbackRestorers) restore();
    this.callbackRestorers = [];
    this.root.remove();
  }
}

app.registerExtension({
  name: "ComfyUI.FL_Audio_Beat_Prompt_Schedule",

  nodeCreated(node) {
    const comfyClass = node.constructor?.comfyClass || "";
    if (comfyClass !== "FL_Audio_Beat_Prompt_Schedule") return;

    const widgets = {
      timeline: findWidget(node, "timeline"),
      defaultFadeIn: findWidget(node, "default_fade_in"),
      defaultFadeOut: findWidget(node, "default_fade_out"),
      curve: findWidget(node, "curve"),
      timeUnit: findWidget(node, "time_unit"),
      fps: findWidget(node, "fps"),
      sequenceDuration: findWidget(node, "sequence_duration"),
    };
    const hiddenWidgets = [widgets.timeline, widgets.timeUnit];
    for (const widget of hiddenWidgets) hideWidget(widget);

    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.height = "100%";
    container.style.minHeight = "650px";

    const domWidget = node.addDOMWidget(
      "beat_prompt_sequencer",
      "fl-beat-prompt-sequencer",
      container,
      {
        getMinHeight: () => 680,
        hideOnZoom: false,
        serialize: false,
      },
    );

    enforceMinimumNodeSize(node);
    requestAnimationFrame(() => enforceMinimumNodeSize(node));

    setTimeout(() => {
      const editor = new BeatPromptSequencer({ node, container, widgets });
      INSTANCES.set(node.id, editor);
    }, 50);

    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      originalOnExecuted?.apply(this, arguments);
      INSTANCES.get(node.id)?.updateFromExecution(message);
    };

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const result = originalOnConfigure?.apply(this, args);
      for (const widget of hiddenWidgets) hideWidget(widget);
      requestAnimationFrame(() => enforceMinimumNodeSize(this));
      const editor = INSTANCES.get(node.id);
      if (editor) {
        editor.loadTimeline();
        editor.refreshBeatStatus();
        editor.scheduleDraw();
      }
      return result;
    };

    const originalOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (type, slot) {
      const result = originalOnConnectionsChange?.apply(this, arguments);
      if (type === 1 && this.inputs?.[slot]?.name === "beat_positions") {
        INSTANCES.get(node.id)?.markBeatDataCached();
      }
      return result;
    };

    domWidget.onRemove = () => {
      const editor = INSTANCES.get(node.id);
      if (editor) {
        editor.dispose();
        INSTANCES.delete(node.id);
      }
    };
  },
});
