import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const DEFAULT_SETTINGS = {
  version: 1,
  start_time: 0,
  end_time: 0,
  sample_mode: "source",
  target_fps: 24,
  select_every_nth: 1,
  frame_load_cap: 0,
  resize_mode: "original",
  width: 0,
  height: 0,
  include_audio: true,
};

const MIN_NODE_WIDTH = 420;
const MIN_NODE_HEIGHT = 440;
const MIN_PANEL_HEIGHT = 340;
const VIDEO_EXTENSIONS = new Set(["avi", "gif", "m4v", "mkv", "mov", "mp4", "webm"]);

const STYLES = `
  .flvl-panel {
    --flvl-accent: #8b5cf6;
    --flvl-border: var(--border-color, #343741);
    --flvl-control: var(--comfy-input-bg, #24262d);
    --flvl-muted: var(--descrip-text, #979cab);
    --flvl-surface: #111218;
    --flvl-surface-raised: #191a20;
    background: var(--comfy-menu-bg, #17181d);
    border: 1px solid var(--flvl-border);
    border-radius: 9px;
    box-sizing: border-box;
    color: var(--input-text, #f4f4f5);
    display: grid;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 11px;
    gap: 6px;
    grid-template-rows: 29px minmax(0, 1fr) 40px 78px;
    height: 100%;
    min-height: 0;
    overflow: hidden;
    padding: 6px;
    position: relative;
    width: 100%;
  }
  .flvl-panel * { box-sizing: border-box; }
  .flvl-source-bar {
    align-items: center;
    display: flex;
    gap: 5px;
    min-width: 0;
    padding: 0 1px;
  }
  .flvl-filename {
    color: #d8dbe2;
    flex: 1;
    font-size: 10.5px;
    font-weight: 650;
    min-width: 0;
    overflow: hidden;
    padding: 0 4px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .flvl-button,
  .flvl-icon-button,
  .flvl-menu-close,
  .flvl-reset-button {
    background: var(--flvl-control);
    border: 1px solid var(--flvl-border);
    color: inherit;
    cursor: pointer;
    font: inherit;
    transition: background-color 120ms ease, border-color 120ms ease, color 120ms ease;
  }
  .flvl-button {
    border-radius: 5px;
    height: 26px;
    padding: 0 9px;
  }
  .flvl-more-button {
    border-radius: 5px;
    font-size: 16px;
    height: 26px;
    line-height: 18px;
    padding: 0;
    width: 28px;
  }
  .flvl-button:hover,
  .flvl-icon-button:hover,
  .flvl-menu-close:hover,
  .flvl-reset-button:hover {
    background: color-mix(in srgb, var(--flvl-control) 82%, var(--flvl-accent));
    border-color: var(--flvl-accent);
  }
  .flvl-button:focus-visible,
  .flvl-icon-button:focus-visible,
  .flvl-menu-close:focus-visible,
  .flvl-reset-button:focus-visible,
  .flvl-drop-zone:focus-visible {
    outline: 2px solid var(--flvl-accent);
    outline-offset: 1px;
  }
  .flvl-preview {
    background: #050507;
    border: 1px solid var(--flvl-border);
    border-radius: 7px;
    height: 100%;
    min-height: 0;
    min-width: 0;
    overflow: hidden;
    position: relative;
    width: 100%;
  }
  .flvl-preview video {
    display: block;
    height: 100%;
    object-fit: contain;
    width: 100%;
  }
  .flvl-drop-zone {
    align-items: center;
    background:
      radial-gradient(circle at 50% 44%, rgba(139, 92, 246, .13), transparent 45%),
      #08090d;
    border: 1px dashed #565b68;
    color: #a3a8b4;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    gap: 7px;
    inset: 7px;
    justify-content: center;
    padding: 18px;
    position: absolute;
    text-align: center;
    z-index: 4;
  }
  .flvl-drop-zone[hidden] { display: none; }
  .flvl-drop-icon {
    color: #c4b5fd;
    font-size: 24px;
    line-height: 1;
  }
  .flvl-drop-title {
    color: #e6e7eb;
    font-size: 12px;
    font-weight: 700;
  }
  .flvl-drop-help {
    color: var(--flvl-muted);
    font-size: 9px;
  }
  .flvl-panel[data-dragging="true"] .flvl-drop-zone {
    background: rgba(76, 29, 149, .93);
    border-color: #c4b5fd;
    display: flex;
    inset: 4px;
  }
  .flvl-panel[data-dragging="true"] .flvl-drop-title {
    color: white;
  }
  .flvl-preview-info,
  .flvl-status {
    backdrop-filter: blur(5px);
    background: rgba(20, 21, 26, .78);
    border: 1px solid rgba(255, 255, 255, .07);
    border-radius: 999px;
    max-width: calc(70% - 8px);
    overflow: hidden;
    padding: 3px 7px;
    position: absolute;
    text-overflow: ellipsis;
    top: 7px;
    white-space: nowrap;
    z-index: 3;
  }
  .flvl-preview-info {
    color: #c7cad1;
    font-size: 9px;
    left: 7px;
  }
  .flvl-status {
    align-items: center;
    color: #c8ccd5;
    display: flex;
    font-size: 9px;
    font-weight: 700;
    gap: 5px;
    right: 7px;
    text-transform: uppercase;
  }
  .flvl-status::before {
    background: currentColor;
    border-radius: 50%;
    content: "";
    flex: 0 0 5px;
    height: 5px;
    width: 5px;
  }
  .flvl-status[data-state="ready"] { color: #86efac; }
  .flvl-status[data-state="stale"] { color: #fde68a; }
  .flvl-status[data-state="busy"] { color: #c4b5fd; }
  .flvl-status[data-state="error"] { color: #fda4af; }
  .flvl-preview-controls {
    align-items: center;
    background: linear-gradient(to bottom, rgba(9, 10, 13, 0), rgba(9, 10, 13, .94) 44%);
    bottom: 0;
    display: flex;
    gap: 6px;
    left: 0;
    min-height: 39px;
    padding: 10px 7px 5px;
    position: absolute;
    right: 0;
    z-index: 3;
  }
  .flvl-icon-button {
    background: rgba(30, 32, 38, .84);
    border-radius: 5px;
    flex: 0 0 25px;
    height: 25px;
    line-height: 20px;
    padding: 0;
  }
  .flvl-time {
    color: #d0d3da;
    flex: 0 0 auto;
    font-variant-numeric: tabular-nums;
    min-width: 69px;
  }
  .flvl-preview-volume {
    accent-color: var(--flvl-accent);
    cursor: pointer;
    height: 15px;
    margin: 0;
    min-width: 0;
  }
  .flvl-control-spacer { flex: 1 1 auto; }
  .flvl-preview-volume { flex: 0 1 58px; max-width: 58px; }
  .flvl-preview-volume-value {
    color: var(--flvl-muted);
    flex: 0 0 27px;
    font-size: 9px;
    opacity: .82;
    text-align: right;
  }
  .flvl-browser-error {
    align-items: center;
    background: rgba(5, 5, 7, .92);
    color: #b8bdc8;
    display: flex;
    inset: 32px 0 39px;
    justify-content: center;
    padding: 20px;
    position: absolute;
    text-align: center;
    z-index: 2;
  }
  .flvl-browser-error[hidden] { display: none; }
  .flvl-trim-timeline {
    background: var(--flvl-surface);
    border: 1px solid var(--flvl-border);
    border-radius: 6px;
    display: grid;
    gap: 1px;
    grid-template-rows: 13px minmax(0, 1fr);
    min-width: 0;
    overflow: hidden;
    padding: 3px 6px 4px;
  }
  .flvl-trim-labels {
    align-items: center;
    color: var(--flvl-muted);
    display: flex;
    font-size: 8px;
    font-weight: 700;
    gap: 6px;
    letter-spacing: .03em;
    min-width: 0;
    text-transform: uppercase;
  }
  .flvl-trim-labels strong {
    color: #ddd6fe;
    font-size: 9px;
    font-variant-numeric: tabular-nums;
  }
  .flvl-trim-help {
    flex: 1;
    font-size: 8px;
    font-weight: 500;
    overflow: hidden;
    text-align: center;
    text-overflow: ellipsis;
    text-transform: none;
    white-space: nowrap;
  }
  .flvl-trim-help strong {
    color: #c4b5fd;
  }
  .flvl-trim-timeline canvas {
    cursor: pointer;
    display: block;
    height: 19px;
    touch-action: none;
    width: 100%;
  }
  .flvl-trim-timeline[data-disabled="true"] {
    opacity: .48;
  }
  .flvl-controls {
    display: grid;
    gap: 4px;
    grid-template-rows: 37px 37px;
    min-width: 0;
    overflow: hidden;
  }
  .flvl-control-row,
  .flvl-control-group {
    align-items: end;
    display: grid;
    gap: 4px;
    min-width: 0;
  }
  .flvl-range-row {
    grid-template-columns: 42px repeat(3, minmax(0, 1fr));
  }
  .flvl-processing-row {
    display: grid;
    gap: 6px;
    grid-template-columns: minmax(145px, .8fr) minmax(0, 1.25fr);
    min-width: 0;
  }
  .flvl-sampling-group {
    grid-template-columns: 36px minmax(0, 1fr) minmax(44px, .72fr);
  }
  .flvl-output-group {
    grid-template-columns:
      38px
      minmax(0, 1.1fr)
      minmax(0, .7fr)
      minmax(0, .7fr)
      minmax(44px, .76fr);
  }
  .flvl-group-label {
    align-items: center;
    align-self: end;
    border-right: 1px solid var(--flvl-border);
    color: #b8bcc7;
    display: flex;
    font-size: 7.5px;
    font-weight: 750;
    height: 26px;
    letter-spacing: .06em;
    line-height: 1;
    overflow: hidden;
    padding-right: 5px;
    text-transform: uppercase;
    white-space: nowrap;
  }
  .flvl-param,
  .flvl-menu-field {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .flvl-param-label,
  .flvl-menu-field > span {
    color: var(--flvl-muted);
    font-size: 8px;
    font-weight: 700;
    letter-spacing: .05em;
    line-height: 9px;
    overflow: hidden;
    text-overflow: ellipsis;
    text-transform: uppercase;
    white-space: nowrap;
  }
  .flvl-param input,
  .flvl-param select,
  .flvl-menu-field input,
  .flvl-menu-field select {
    appearance: none;
    background: var(--flvl-control);
    border: 1px solid var(--flvl-border);
    border-radius: 5px;
    color: inherit;
    font: inherit;
    height: 26px;
    min-width: 0;
    padding: 2px 5px;
    width: 100%;
  }
  .flvl-param input[type="number"],
  .flvl-menu-field input[type="number"] { appearance: textfield; }
  .flvl-param input[type="number"]::-webkit-inner-spin-button,
  .flvl-menu-field input[type="number"]::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  .flvl-param input:focus,
  .flvl-param select:focus,
  .flvl-menu-field input:focus,
  .flvl-menu-field select:focus {
    border-color: var(--flvl-accent);
    outline: none;
  }
  .flvl-param input:disabled,
  .flvl-param select:disabled {
    color: var(--flvl-muted);
    cursor: default;
    opacity: .65;
  }
  .flvl-audio-toggle {
    align-items: center;
    background: var(--flvl-control);
    border: 1px solid var(--flvl-border);
    border-radius: 5px;
    color: var(--flvl-muted);
    cursor: pointer;
    display: flex;
    font-size: 9px;
    font-weight: 700;
    gap: 4px;
    height: 26px;
    justify-content: center;
    min-width: 0;
    padding: 0 4px;
  }
  .flvl-audio-toggle input {
    appearance: none;
    background: #30323b;
    border: 1px solid #4a4d59;
    border-radius: 999px;
    cursor: pointer;
    height: 12px;
    margin: 0;
    padding: 0;
    position: relative;
    transition: background-color 120ms ease, border-color 120ms ease;
    width: 20px;
  }
  .flvl-audio-toggle input::after {
    background: #a7abb5;
    border-radius: 50%;
    content: "";
    height: 6px;
    left: 2px;
    position: absolute;
    top: 2px;
    transition: background-color 120ms ease, transform 120ms ease;
    width: 6px;
  }
  .flvl-audio-toggle input:checked {
    background: var(--flvl-accent);
    border-color: #a78bfa;
  }
  .flvl-audio-toggle input:checked::after {
    background: white;
    transform: translateX(8px);
  }
  .flvl-audio-toggle[data-enabled="true"] { color: #86efac; }
  .flvl-menu {
    background: var(--comfy-menu-bg, #1b1c22);
    border: 1px solid var(--flvl-border);
    border-radius: 7px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, .45);
    display: grid;
    gap: 8px;
    max-height: calc(100% - 43px);
    overflow: auto;
    padding: 9px;
    position: absolute;
    right: 6px;
    top: 39px;
    width: min(310px, calc(100% - 12px));
    z-index: 8;
  }
  .flvl-menu[hidden] { display: none; }
  .flvl-menu-header {
    align-items: center;
    display: flex;
    justify-content: space-between;
  }
  .flvl-menu-title {
    font-size: 11px;
    font-weight: 750;
  }
  .flvl-menu-close {
    border-radius: 4px;
    height: 24px;
    width: 25px;
  }
  .flvl-menu-help,
  .flvl-memory {
    color: var(--flvl-muted);
    font-size: 9px;
    line-height: 1.35;
  }
  .flvl-memory[data-warning="true"] { color: #fbbf24; }
  .flvl-menu-divider {
    border-top: 1px solid var(--flvl-border);
  }
  .flvl-reset-button {
    border-radius: 4px;
    height: 26px;
  }
  .flvl-error {
    background: #3f1d25;
    border: 1px solid #7f1d2d;
    border-radius: 5px;
    color: #fecdd3;
    font-size: 9px;
    left: 12px;
    padding: 6px;
    position: absolute;
    right: 12px;
    top: 42px;
    z-index: 10;
  }
  .flvl-error[hidden] { display: none; }
  .flvl-upload-progress {
    background: #292b32;
    border-radius: 999px;
    height: 4px;
    margin-top: 2px;
    overflow: hidden;
    width: min(180px, 75%);
  }
  .flvl-upload-progress > span {
    animation: flvl-upload 1s ease-in-out infinite alternate;
    background: var(--flvl-accent);
    display: block;
    height: 100%;
    width: 45%;
  }
  @keyframes flvl-upload {
    from { transform: translateX(-20%); }
    to { transform: translateX(140%); }
  }
`;

function injectStyles() {
  if (document.getElementById("flvl-styles")) return;
  const style = document.createElement("style");
  style.id = "flvl-styles";
  style.textContent = STYLES;
  document.head.appendChild(style);
}

function hideWidget(widget) {
  if (!widget) return;
  widget.hidden = true;
  widget.computeSize = () => [0, 0];
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
  if (width !== node.size[0] || height !== node.size[1]) {
    node.setSize([width, height]);
  }
}

function filenameFromPath(path) {
  return String(path || "").replace(/ \[(input|output|temp)\]$/, "").replace(/\\/g, "/").split("/").pop() || "";
}

function previewReference(path) {
  const normalized = String(path || "").replace(/ \[(input|output|temp)\]$/, "").replace(/\\/g, "/");
  const parts = normalized.split("/");
  return {
    filename: parts.pop() || "",
    subfolder: parts.join("/"),
  };
}

function formatTime(value, precise = false) {
  if (!Number.isFinite(value)) return precise ? "00:00.00" : "00:00";
  const minutes = Math.floor(value / 60);
  const seconds = precise ? (value % 60).toFixed(2).padStart(5, "0") : String(Math.floor(value % 60)).padStart(2, "0");
  return `${String(minutes).padStart(2, "0")}:${seconds}`;
}

function formatBytes(value) {
  if (!Number.isFinite(value) || value <= 0) return "0 MB";
  const gb = value / (1024 ** 3);
  if (gb >= 1) return `${gb.toFixed(gb >= 10 ? 1 : 2)} GB`;
  return `${Math.max(1, Math.round(value / (1024 ** 2)))} MB`;
}

function supportedVideoFile(file) {
  const extension = file?.name?.split(".").pop()?.toLowerCase();
  return Boolean(file && (file.type?.startsWith("video/") || file.type === "image/gif" || VIDEO_EXTENSIONS.has(extension)));
}

class LoadVideoPanel {
  constructor(node, videoWidget, settingsWidget, container) {
    this.node = node;
    this.videoWidget = videoWidget;
    this.settingsWidget = settingsWidget;
    this.container = container;
    this.settings = { ...DEFAULT_SETTINGS };
    this.sourceInfo = null;
    this.executionInfo = null;
    this.configError = "";
    this.objectUrl = null;
    this.probeId = 0;
    this.dragDepth = 0;
    this.trimDrag = null;
    this.disposed = false;

    this.node.properties ||= {};
    if (!Number.isFinite(this.node.properties.previewVolume)) this.node.properties.previewVolume = 0.8;
    if (typeof this.node.properties.previewMuted !== "boolean") this.node.properties.previewMuted = true;

    this.readSettings();
    this.build();
    this.bind();
    this.syncControls();
    this.restoreSource();
  }

  readSettings() {
    try {
      const parsed = JSON.parse(this.settingsWidget.value);
      if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
        throw new Error("Load settings must be a JSON object.");
      }
      if (parsed.version !== undefined && parsed.version !== 1) {
        throw new Error(`Settings version ${parsed.version} is unsupported.`);
      }
      this.settings = { ...DEFAULT_SETTINGS, ...parsed, version: 1 };
      this.configError = "";
    } catch (error) {
      this.settings = { ...DEFAULT_SETTINGS };
      this.configError = error.message;
    }
  }

  build() {
    injectStyles();
    this.container.innerHTML = `
      <div class="flvl-panel" data-dragging="false">
        <div class="flvl-source-bar">
          <div class="flvl-filename" data-role="filename">No video selected</div>
          <button class="flvl-button" data-action="replace" data-role="source-action" type="button">Choose</button>
          <button class="flvl-button flvl-more-button" data-action="more" type="button" title="Source and memory details" aria-label="Source and memory details" aria-expanded="false">⋯</button>
        </div>

        <div class="flvl-error" data-role="error" role="alert" hidden></div>

        <div class="flvl-preview" data-role="preview">
          <video data-role="video" playsinline preload="metadata"></video>
          <div class="flvl-preview-info" data-role="preview-info">Choose a video</div>
          <div class="flvl-status" data-role="status" data-state="idle">empty</div>
          <div class="flvl-browser-error" data-role="browser-error" hidden>
            Browser preview unavailable. This file can still be loaded.
          </div>
          <div class="flvl-drop-zone" data-role="drop-zone" role="button" tabindex="0" aria-label="Choose or drop a video">
            <div class="flvl-drop-icon">＋</div>
            <div class="flvl-drop-title" data-role="drop-title">Drop a video here</div>
            <div class="flvl-drop-help" data-role="drop-help">or click to browse</div>
            <div class="flvl-upload-progress" data-role="upload-progress" hidden><span></span></div>
          </div>
          <div class="flvl-preview-controls">
            <button class="flvl-icon-button" data-action="play" type="button" title="Play or pause preview">▶</button>
            <span class="flvl-time" data-role="time">00:00 / 00:00</span>
            <span class="flvl-control-spacer"></span>
            <button class="flvl-icon-button" data-action="mute" type="button" title="Mute preview">🔇</button>
            <input class="flvl-preview-volume" data-role="volume" aria-label="Preview volume" type="range" min="0" max="100" step="1">
            <span class="flvl-preview-volume-value" data-role="volume-value">80%</span>
          </div>
        </div>

        <div class="flvl-trim-timeline" data-role="trim-timeline" data-disabled="true">
          <div class="flvl-trim-labels">
            <span>In <strong data-role="trim-in-label">00:00.00</strong></span>
            <span class="flvl-trim-help"><strong data-role="trim-frame-label">0</strong> selected frames</span>
            <span>Out <strong data-role="trim-out-label">00:00.00</strong></span>
          </div>
          <canvas data-role="trim-canvas" tabindex="0" aria-label="Video trim timeline"></canvas>
        </div>

        <div class="flvl-controls">
          <div class="flvl-control-row flvl-range-row" role="group" aria-label="Video range parameters">
            <span class="flvl-group-label">Range</span>
            <label class="flvl-param">
              <span class="flvl-param-label">In</span>
              <input data-setting="start_time" type="number" min="0" step="0.01">
            </label>
            <label class="flvl-param">
              <span class="flvl-param-label">Out · 0 = end</span>
              <input data-setting="end_time" type="number" min="0" step="0.01">
            </label>
            <label class="flvl-param">
              <span class="flvl-param-label">Max frames</span>
              <input data-setting="frame_load_cap" type="number" min="0" step="1" title="Sets the Out point from the current In point. Use 0 for no cap.">
            </label>
          </div>
          <div class="flvl-processing-row">
            <div class="flvl-control-group flvl-sampling-group" role="group" aria-label="Video sampling parameters">
              <span class="flvl-group-label">Sample</span>
              <label class="flvl-param">
                <span class="flvl-param-label">Mode</span>
                <select data-setting="sample_mode">
                  <option value="source">Source FPS</option>
                  <option value="target_fps">Target FPS</option>
                  <option value="every_nth">Every Nth</option>
                </select>
              </label>
              <label class="flvl-param">
                <span class="flvl-param-label" data-role="sample-value-label">FPS</span>
                <input data-role="sample-value" type="number" min="1" max="120" step="0.01" disabled>
              </label>
            </div>
            <div class="flvl-control-group flvl-output-group" role="group" aria-label="Video output parameters">
              <span class="flvl-group-label">Output</span>
              <label class="flvl-param">
                <span class="flvl-param-label">Resize</span>
                <select data-setting="resize_mode">
                  <option value="original">Original</option>
                  <option value="fit">Fit</option>
                  <option value="crop">Fill / crop</option>
                </select>
              </label>
              <label class="flvl-param">
                <span class="flvl-param-label">Width</span>
                <input data-setting="width" type="number" min="0" max="16384" step="1">
              </label>
              <label class="flvl-param">
                <span class="flvl-param-label">Height</span>
                <input data-setting="height" type="number" min="0" max="16384" step="1">
              </label>
              <label class="flvl-param">
                <span class="flvl-param-label">Audio</span>
                <span class="flvl-audio-toggle" data-role="audio-toggle">
                  <input data-setting="include_audio" type="checkbox">
                  <span data-role="audio-toggle-value">On</span>
                </span>
              </label>
            </div>
          </div>
        </div>

        <div class="flvl-menu" data-role="settings-menu" hidden>
          <div class="flvl-menu-header">
            <span class="flvl-menu-title">Source and memory</span>
            <button class="flvl-menu-close" data-action="menu-close" type="button" aria-label="Close source and memory details">×</button>
          </div>
          <label class="flvl-menu-field">
            <span>Comfy input video</span>
            <select data-role="input-video"></select>
          </label>
          <div class="flvl-menu-help">Dropped files are copied into the ComfyUI input directory.</div>
          <div class="flvl-menu-divider"></div>
          <div class="flvl-memory" data-role="memory">Decoded-memory estimate appears after probing.</div>
          <div class="flvl-menu-divider"></div>
          <button class="flvl-button" data-action="remove" type="button">Remove selected video</button>
          <button class="flvl-reset-button" data-action="reset" type="button">Reset processing settings</button>
        </div>
      </div>
    `;

    this.panel = this.container.querySelector(".flvl-panel");
    this.filename = this.container.querySelector('[data-role="filename"]');
    this.preview = this.container.querySelector('[data-role="preview"]');
    this.video = this.container.querySelector('[data-role="video"]');
    this.previewInfo = this.container.querySelector('[data-role="preview-info"]');
    this.status = this.container.querySelector('[data-role="status"]');
    this.browserError = this.container.querySelector('[data-role="browser-error"]');
    this.dropZone = this.container.querySelector('[data-role="drop-zone"]');
    this.dropTitle = this.container.querySelector('[data-role="drop-title"]');
    this.dropHelp = this.container.querySelector('[data-role="drop-help"]');
    this.uploadProgress = this.container.querySelector('[data-role="upload-progress"]');
    this.error = this.container.querySelector('[data-role="error"]');
    this.time = this.container.querySelector('[data-role="time"]');
    this.volume = this.container.querySelector('[data-role="volume"]');
    this.volumeValue = this.container.querySelector('[data-role="volume-value"]');
    this.trimTimeline = this.container.querySelector('[data-role="trim-timeline"]');
    this.trimCanvas = this.container.querySelector('[data-role="trim-canvas"]');
    this.trimInLabel = this.container.querySelector('[data-role="trim-in-label"]');
    this.trimOutLabel = this.container.querySelector('[data-role="trim-out-label"]');
    this.trimFrameLabel = this.container.querySelector('[data-role="trim-frame-label"]');
    this.playButton = this.container.querySelector('[data-action="play"]');
    this.muteButton = this.container.querySelector('[data-action="mute"]');
    this.sourceAction = this.container.querySelector('[data-role="source-action"]');
    this.moreButton = this.container.querySelector('[data-action="more"]');
    this.settingsMenu = this.container.querySelector('[data-role="settings-menu"]');
    this.inputVideoSelect = this.container.querySelector('[data-role="input-video"]');
    this.memory = this.container.querySelector('[data-role="memory"]');
    this.sampleValue = this.container.querySelector('[data-role="sample-value"]');
    this.sampleValueLabel = this.container.querySelector('[data-role="sample-value-label"]');
    this.audioToggle = this.container.querySelector('[data-role="audio-toggle"]');
    this.audioToggleValue = this.container.querySelector('[data-role="audio-toggle-value"]');
    this.settingControls = [...this.container.querySelectorAll("[data-setting]")];

    this.fileInput = document.createElement("input");
    this.fileInput.type = "file";
    this.fileInput.accept = "video/*,image/gif,.avi,.gif,.m4v,.mkv,.mov,.mp4,.webm";
    this.fileInput.hidden = true;
    this.container.appendChild(this.fileInput);
  }

  bind() {
    this.container.querySelector('[data-action="replace"]').addEventListener("click", () => this.chooseFile());
    this.dropZone.addEventListener("click", () => this.chooseFile());
    this.dropZone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        this.chooseFile();
      }
    });
    this.fileInput.addEventListener("change", () => {
      const file = this.fileInput.files?.[0];
      if (file) this.uploadFile(file);
      this.fileInput.value = "";
    });

    for (const control of this.settingControls) {
      const eventName = control.type === "number" ? "input" : "change";
      control.addEventListener(eventName, () => {
        let value;
        if (control.type === "checkbox") {
          value = control.checked;
        } else if (control.type === "number") {
          value = Number(control.value);
          if (!Number.isFinite(value)) return;
          if (["select_every_nth", "frame_load_cap", "width", "height"].includes(control.dataset.setting)) {
            value = Math.trunc(value);
          }
        } else {
          value = control.value;
        }
        this.updateSetting(control.dataset.setting, value);
      });
    }
    this.sampleValue.addEventListener("input", () => {
      if (this.settings.sample_mode === "source") return;
      let value = Number(this.sampleValue.value);
      if (!Number.isFinite(value)) return;
      const name = this.settings.sample_mode === "target_fps" ? "target_fps" : "select_every_nth";
      if (name === "select_every_nth") value = Math.trunc(value);
      this.updateSetting(name, value);
    });

    this.moreButton.addEventListener("click", () => this.setMenuOpen(this.settingsMenu.hidden));
    this.container.querySelector('[data-action="menu-close"]').addEventListener("click", () => this.setMenuOpen(false));
    this.container.querySelector('[data-action="reset"]').addEventListener("click", () => this.resetSettings());
    this.container.querySelector('[data-action="remove"]').addEventListener("click", () => this.removeSource());
    this.inputVideoSelect.addEventListener("change", () => {
      if (this.inputVideoSelect.value) this.selectSource(this.inputVideoSelect.value);
      else this.removeSource();
    });

    this.handleDocumentPointerDown = (event) => {
      if (!this.settingsMenu.hidden && !this.settingsMenu.contains(event.target) && !this.moreButton.contains(event.target)) {
        this.setMenuOpen(false);
      }
    };
    this.handleDocumentKeyDown = (event) => {
      if (event.key === "Escape" && !this.settingsMenu.hidden) {
        this.setMenuOpen(false);
        this.moreButton.focus();
      }
    };
    document.addEventListener("pointerdown", this.handleDocumentPointerDown);
    document.addEventListener("keydown", this.handleDocumentKeyDown);

    this.panel.addEventListener("dragenter", (event) => {
      if (!event.dataTransfer?.types?.includes("Files")) return;
      event.preventDefault();
      this.dragDepth += 1;
      this.panel.dataset.dragging = "true";
    });
    this.panel.addEventListener("dragover", (event) => {
      if (!event.dataTransfer?.types?.includes("Files")) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
    });
    this.panel.addEventListener("dragleave", (event) => {
      if (!event.dataTransfer?.types?.includes("Files")) return;
      this.dragDepth = Math.max(0, this.dragDepth - 1);
      if (this.dragDepth === 0) this.panel.dataset.dragging = "false";
    });
    this.panel.addEventListener("drop", (event) => {
      event.preventDefault();
      this.dragDepth = 0;
      this.panel.dataset.dragging = "false";
      const files = [...(event.dataTransfer?.files || [])];
      const file = files.find(supportedVideoFile);
      if (!file) {
        this.showError("Drop a supported video file.");
        return;
      }
      this.uploadFile(file);
    });

    this.playButton.addEventListener("click", () => {
      if (!this.video.src) return;
      if (this.video.paused) this.video.play().catch(() => {});
      else this.video.pause();
    });
    this.video.addEventListener("play", () => {
      this.playButton.textContent = "❚❚";
    });
    this.video.addEventListener("pause", () => {
      this.playButton.textContent = "▶";
    });
    this.video.addEventListener("loadedmetadata", () => {
      this.browserError.hidden = true;
      this.applyTrimWindow();
      this.updateTime();
      this.renderTrimTimeline();
    });
    this.video.addEventListener("timeupdate", () => {
      const end = this.activePreviewEnd();
      if (!this.video.paused && end && this.video.currentTime >= end) {
        this.video.currentTime = Math.min(this.settings.start_time, this.video.duration || 0);
        this.video.play().catch(() => {});
      }
      this.updateTime();
      this.renderTrimTimeline();
    });
    this.video.addEventListener("error", () => {
      if (this.video.src) this.browserError.hidden = false;
    });
    this.trimCanvas.addEventListener("pointerdown", (event) => this.beginTrimPointer(event));
    this.trimCanvas.addEventListener("pointermove", (event) => this.moveTrimPointer(event));
    this.trimCanvas.addEventListener("pointerup", (event) => this.endTrimPointer(event));
    this.trimCanvas.addEventListener("pointercancel", (event) => this.endTrimPointer(event));
    this.trimCanvas.addEventListener("keydown", (event) => this.handleTrimKey(event));
    this.muteButton.addEventListener("click", () => {
      this.node.properties.previewMuted = !this.node.properties.previewMuted;
      this.applyPreviewAudio();
    });
    this.volume.addEventListener("input", () => {
      this.node.properties.previewVolume = Number(this.volume.value) / 100;
      this.applyPreviewAudio();
    });

    this.trimResizeObserver = new ResizeObserver(() => this.renderTrimTimeline());
    this.trimResizeObserver.observe(this.trimCanvas);
  }

  chooseFile() {
    this.fileInput.click();
  }

  updateSourceAction(hasSource) {
    this.sourceAction.textContent = hasSource ? "Replace" : "Choose";
    this.sourceAction.title = hasSource ? "Replace the selected video" : "Choose a video";
  }

  async uploadFile(file) {
    if (!supportedVideoFile(file)) {
      this.showError("Choose a supported video file.");
      return;
    }

    const previousSource = this.videoWidget.value;
    this.clearError();
    this.setStatus("busy", "uploading");
    this.dropZone.hidden = false;
    this.dropTitle.textContent = `Uploading ${file.name}`;
    this.dropHelp.textContent = "Copying into ComfyUI input…";
    this.uploadProgress.hidden = false;
    this.setObjectPreview(file);

    try {
      const body = new FormData();
      body.append("image", file);
      body.append("type", "input");
      const response = await api.fetchApi("/upload/image", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `Upload failed (${response.status}).`);
      const path = [payload.subfolder, payload.name].filter(Boolean).join("/").replace(/\\/g, "/");
      this.addVideoOption(path);
      await this.selectSource(path);
    } catch (error) {
      const message = error.message || "Video upload failed.";
      if (previousSource) await this.selectSource(previousSource);
      else this.removeSource(false);
      this.showError(message);
    } finally {
      this.uploadProgress.hidden = true;
    }
  }

  setObjectPreview(file) {
    this.revokeObjectUrl();
    this.sourceInfo = null;
    this.objectUrl = URL.createObjectURL(file);
    this.video.src = this.objectUrl;
    this.video.load();
    this.filename.textContent = file.name;
    this.updateSourceAction(true);
  }

  addVideoOption(path) {
    const values = this.videoWidget.options?.values;
    if (Array.isArray(values) && !values.includes(path)) values.push(path);
  }

  async selectSource(path, markGraph = true) {
    if (!path) {
      this.removeSource(markGraph);
      return;
    }
    this.addVideoOption(path);
    this.videoWidget.value = path;
    this.videoWidget.callback?.(path);
    if (markGraph) this.node.graph?.change?.();
    this.sourceInfo = null;
    this.executionInfo = null;
    this.filename.textContent = filenameFromPath(path);
    this.updateSourceAction(true);
    this.syncInputVideoOptions();
    this.loadServerPreview(path);
    await this.probeSource(path);
  }

  removeSource(markGraph = true) {
    this.probeId += 1;
    this.revokeObjectUrl();
    this.video.pause();
    this.video.removeAttribute("src");
    this.video.load();
    this.videoWidget.value = "";
    this.videoWidget.callback?.("");
    if (markGraph) this.node.graph?.change?.();
    this.sourceInfo = null;
    this.executionInfo = null;
    this.filename.textContent = "No video selected";
    this.updateSourceAction(false);
    this.previewInfo.textContent = "Choose a video";
    this.browserError.hidden = true;
    this.dropTitle.textContent = "Drop a video here";
    this.dropHelp.textContent = "or click to browse";
    this.dropZone.hidden = false;
    this.setStatus("idle", "empty");
    this.syncInputVideoOptions();
    this.syncSampleValue();
    this.updateMemoryEstimate();
    this.renderTrimTimeline();
  }

  loadServerPreview(path) {
    this.revokeObjectUrl();
    const reference = previewReference(path);
    const params = new URLSearchParams({
      filename: reference.filename,
      subfolder: reference.subfolder,
      type: "input",
      timestamp: Date.now(),
    });
    this.video.src = api.apiURL(`/view?${params.toString()}`);
    this.video.load();
    this.browserError.hidden = true;
    this.dropZone.hidden = true;
    this.setStatus("busy", "probing");
  }

  async probeSource(path) {
    const probeId = ++this.probeId;
    try {
      const params = new URLSearchParams({ filename: path });
      const response = await api.fetchApi(`/fl/load-video/info?${params.toString()}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `Video probe failed (${response.status}).`);
      if (this.disposed || probeId !== this.probeId || this.videoWidget.value !== path) return;
      this.sourceInfo = payload;
      this.node.properties.lastSourceInfo = { ...payload, filename: path };
      if (this.settings.frame_load_cap > 0 && this.syncFrameRange("frame_load_cap")) {
        this.settingsWidget.value = JSON.stringify(this.settings);
        this.syncControls();
      }
      this.syncSampleValue();
      this.updateSourceSummary();
      this.setStatus("ready", "ready");
      this.applyTrimWindow();
      this.updateMemoryEstimate();
      this.clearError();
    } catch (error) {
      if (this.disposed || probeId !== this.probeId) return;
      this.sourceInfo = null;
      this.showError(error.message || "Could not inspect video.");
    }
  }

  restoreSource() {
    this.syncInputVideoOptions();
    const path = this.videoWidget.value;
    if (!path) {
      this.removeSource(false);
      if (this.configError) this.showError(this.configError);
      return;
    }
    const cached = this.node.properties.lastSourceInfo;
    if (cached?.filename === path) {
      this.sourceInfo = { ...cached };
      delete this.sourceInfo.filename;
    }
    this.filename.textContent = filenameFromPath(path);
    this.updateSourceAction(true);
    this.loadServerPreview(path);
    this.updateSourceSummary();
    this.probeSource(path);
    if (this.configError) this.showError(this.configError);
  }

  syncInputVideoOptions() {
    const current = this.videoWidget.value || "";
    const values = Array.isArray(this.videoWidget.options?.values)
      ? this.videoWidget.options.values.filter(Boolean)
      : [];
    if (current && !values.includes(current)) values.push(current);
    values.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
    this.inputVideoSelect.replaceChildren();
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = "Choose a video…";
    this.inputVideoSelect.appendChild(empty);
    for (const value of values) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      this.inputVideoSelect.appendChild(option);
    }
    this.inputVideoSelect.value = current;
  }

  updateSetting(name, value) {
    this.settings[name] = value;
    const linkedRangeChanged = this.syncFrameRange(name);
    this.settingsWidget.value = JSON.stringify(this.settings);
    this.configError = "";
    this.syncControls(linkedRangeChanged ? null : name);
    this.applyTrimWindow();
    this.updateMemoryEstimate();
    if (this.executionInfo) this.setStatus("stale", "settings changed");
    this.node.graph?.change?.();
    this.node.setDirtyCanvas(true, true);
  }

  resetSettings() {
    this.settings = { ...DEFAULT_SETTINGS };
    this.settingsWidget.value = JSON.stringify(this.settings);
    this.configError = "";
    this.syncControls();
    this.applyTrimWindow();
    this.updateMemoryEstimate();
    if (this.executionInfo) this.setStatus("stale", "settings changed");
    this.node.graph?.change?.();
    this.node.setDirtyCanvas(true, true);
  }

  syncControls(changedName = null) {
    for (const control of this.settingControls) {
      const name = control.dataset.setting;
      if (changedName && name !== changedName) continue;
      if (control.type === "checkbox") control.checked = Boolean(this.settings[name]);
      else control.value = this.settings[name];
    }

    for (const control of this.settingControls.filter((item) => ["width", "height"].includes(item.dataset.setting))) {
      control.disabled = this.settings.resize_mode === "original";
    }
    this.syncSampleValue();
    const audioEnabled = Boolean(this.settings.include_audio);
    this.audioToggle.dataset.enabled = String(audioEnabled);
    this.audioToggleValue.textContent = audioEnabled ? "On" : "Off";
    this.applyPreviewAudio();
  }

  syncSampleValue() {
    if (!this.sampleValue) return;
    if (this.settings.sample_mode === "target_fps") {
      this.sampleValueLabel.textContent = "FPS";
      this.sampleValue.disabled = false;
      this.sampleValue.min = "1";
      this.sampleValue.max = "120";
      this.sampleValue.step = "0.01";
      this.sampleValue.value = String(this.settings.target_fps);
    } else if (this.settings.sample_mode === "every_nth") {
      this.sampleValueLabel.textContent = "Every N";
      this.sampleValue.disabled = false;
      this.sampleValue.min = "1";
      this.sampleValue.removeAttribute("max");
      this.sampleValue.step = "1";
      this.sampleValue.value = String(this.settings.select_every_nth);
    } else {
      this.sampleValueLabel.textContent = "FPS";
      this.sampleValue.disabled = true;
      this.sampleValue.min = "1";
      this.sampleValue.max = "120";
      this.sampleValue.step = "0.01";
      this.sampleValue.value = this.sourceInfo?.frame_rate
        ? String(Number(this.sourceInfo.frame_rate).toFixed(2))
        : "";
    }
  }

  timelineDuration() {
    if (Number.isFinite(this.sourceInfo?.duration) && this.sourceInfo.duration > 0) return this.sourceInfo.duration;
    if (Number.isFinite(this.video.duration) && this.video.duration > 0) return this.video.duration;
    return 0;
  }

  effectiveFrameRate() {
    let fps = Number(this.sourceInfo?.frame_rate);
    if (!Number.isFinite(fps) || fps <= 0) return 0;
    if (this.settings.sample_mode === "target_fps") {
      fps = Number(this.settings.target_fps);
    } else if (this.settings.sample_mode === "every_nth") {
      fps /= Math.max(1, Math.trunc(Number(this.settings.select_every_nth) || 1));
    }
    return Number.isFinite(fps) && fps > 0 ? fps : 0;
  }

  trimBounds() {
    const duration = this.timelineDuration();
    const start = Math.min(duration, Math.max(0, Number(this.settings.start_time) || 0));
    const configuredEnd = Number(this.settings.end_time) || 0;
    const end = configuredEnd ? Math.min(duration, Math.max(start, configuredEnd)) : duration;
    return { duration, start, end };
  }

  selectedFrameCount(bounds = this.trimBounds()) {
    const fps = this.effectiveFrameRate();
    if (!fps || !bounds.duration || bounds.end <= bounds.start) return 0;
    return Math.max(1, Math.ceil((bounds.end - bounds.start) * fps - 1e-7));
  }

  syncFrameRange(changedName) {
    const linkedSettings = new Set([
      "start_time",
      "end_time",
      "sample_mode",
      "target_fps",
      "select_every_nth",
      "frame_load_cap",
    ]);
    if (!linkedSettings.has(changedName)) return false;

    const bounds = this.trimBounds();
    const fps = this.effectiveFrameRate();
    if (!bounds.duration || !fps) return false;

    if (changedName !== "frame_load_cap") {
      const frameCount = this.selectedFrameCount(bounds);
      if (!frameCount || this.settings.frame_load_cap === frameCount) return false;
      this.settings.frame_load_cap = frameCount;
      return true;
    }

    const requestedFrames = Math.max(0, Math.trunc(Number(this.settings.frame_load_cap) || 0));
    this.settings.frame_load_cap = requestedFrames;
    if (!requestedFrames) return false;

    const availableFrames = Math.max(1, Math.ceil((bounds.duration - bounds.start) * fps - 1e-7));
    const frameCount = Math.min(requestedFrames, availableFrames);
    const end = Math.min(bounds.duration, bounds.start + frameCount / fps);
    this.settings.frame_load_cap = frameCount;
    this.settings.end_time = end >= bounds.duration - 1e-7 ? 0 : Number(end.toFixed(6));
    return frameCount !== requestedFrames || this.settings.end_time !== Number(bounds.end.toFixed(6));
  }

  trimGeometry() {
    const rect = this.trimCanvas.getBoundingClientRect();
    const left = 9;
    const right = Math.max(left + 1, rect.width - 9);
    return { rect, left, right, width: right - left };
  }

  trimTimeAtPointer(event) {
    const { duration } = this.trimBounds();
    const { rect, left, width } = this.trimGeometry();
    const x = Math.max(left, Math.min(left + width, event.clientX - rect.left));
    return duration * ((x - left) / width);
  }

  trimXForTime(time, geometry, duration) {
    if (!duration) return geometry.left;
    return geometry.left + (Math.max(0, Math.min(duration, time)) / duration) * geometry.width;
  }

  trimPointerMode(event) {
    const bounds = this.trimBounds();
    if (!bounds.duration) return null;
    const geometry = this.trimGeometry();
    const x = event.clientX - geometry.rect.left;
    const inX = this.trimXForTime(bounds.start, geometry, bounds.duration);
    const outX = this.trimXForTime(bounds.end, geometry, bounds.duration);
    const inDistance = Math.abs(x - inX);
    const outDistance = Math.abs(x - outX);
    if (Math.min(inDistance, outDistance) <= 10) {
      return inDistance <= outDistance ? "in" : "out";
    }
    return "scrub";
  }

  beginTrimPointer(event) {
    const mode = this.trimPointerMode(event);
    if (!mode) return;
    event.preventDefault();
    event.stopPropagation();
    this.trimDrag = mode;
    this.trimCanvas.setPointerCapture(event.pointerId);
    this.moveTrimPointer(event);
  }

  moveTrimPointer(event) {
    if (!this.trimDrag) {
      const mode = this.trimPointerMode(event);
      this.trimCanvas.style.cursor = mode === "in" || mode === "out" ? "ew-resize" : "pointer";
      return;
    }
    event.preventDefault();
    const bounds = this.trimBounds();
    const time = this.trimTimeAtPointer(event);
    if (this.trimDrag === "scrub") {
      this.video.currentTime = time;
      this.updateTime();
      this.renderTrimTimeline();
      return;
    }

    const frameStep = Math.max(0.01, 1 / Math.max(1, Number(this.sourceInfo?.frame_rate) || 100));
    if (this.trimDrag === "in") {
      const value = Math.min(Math.max(0, time), Math.max(0, bounds.end - frameStep));
      this.updateSetting("start_time", Number(value.toFixed(3)));
    } else {
      const value = Math.max(bounds.start + frameStep, Math.min(bounds.duration, time));
      this.updateSetting(
        "end_time",
        value >= bounds.duration - frameStep / 2 ? 0 : Number(value.toFixed(3)),
      );
    }
  }

  endTrimPointer(event) {
    if (!this.trimDrag) return;
    if (this.trimCanvas.hasPointerCapture(event.pointerId)) {
      this.trimCanvas.releasePointerCapture(event.pointerId);
    }
    this.trimDrag = null;
    this.trimCanvas.style.cursor = this.trimPointerMode(event) === "scrub" ? "pointer" : "ew-resize";
  }

  handleTrimKey(event) {
    const bounds = this.trimBounds();
    if (!bounds.duration || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const frameStep = 1 / Math.max(1, Number(this.sourceInfo?.frame_rate) || 30);
    const direction = event.key === "ArrowLeft" ? -1 : 1;
    if (event.shiftKey && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
      const value = Math.max(0, Math.min(bounds.end - frameStep, bounds.start + direction * frameStep));
      this.updateSetting("start_time", Number(value.toFixed(3)));
      return;
    }
    if (event.altKey && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
      const value = Math.max(bounds.start + frameStep, Math.min(bounds.duration, bounds.end + direction * frameStep));
      this.updateSetting("end_time", value >= bounds.duration - frameStep / 2 ? 0 : Number(value.toFixed(3)));
      return;
    }
    if (event.key === "Home") this.video.currentTime = 0;
    else if (event.key === "End") this.video.currentTime = bounds.duration;
    else this.video.currentTime = Math.max(0, Math.min(bounds.duration, this.video.currentTime + direction * frameStep));
    this.updateTime();
    this.renderTrimTimeline();
  }

  renderTrimTimeline() {
    if (!this.trimCanvas) return;
    const bounds = this.trimBounds();
    const geometry = this.trimGeometry();
    const cssWidth = Math.max(1, geometry.rect.width);
    const cssHeight = Math.max(1, geometry.rect.height);
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const pixelWidth = Math.round(cssWidth * ratio);
    const pixelHeight = Math.round(cssHeight * ratio);
    if (this.trimCanvas.width !== pixelWidth || this.trimCanvas.height !== pixelHeight) {
      this.trimCanvas.width = pixelWidth;
      this.trimCanvas.height = pixelHeight;
    }

    const context = this.trimCanvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, cssWidth, cssHeight);
    const trackY = Math.round(cssHeight / 2);
    const trackHeight = 6;
    const inX = this.trimXForTime(bounds.start, geometry, bounds.duration);
    const outX = this.trimXForTime(bounds.end, geometry, bounds.duration);

    context.fillStyle = "#30333d";
    context.fillRect(geometry.left, trackY - trackHeight / 2, geometry.width, trackHeight);
    if (bounds.duration) {
      context.fillStyle = "#8b5cf6";
      context.fillRect(inX, trackY - trackHeight / 2, Math.max(1, outX - inX), trackHeight);
      context.fillStyle = "rgba(0, 0, 0, .42)";
      context.fillRect(geometry.left, trackY - trackHeight / 2, Math.max(0, inX - geometry.left), trackHeight);
      context.fillRect(outX, trackY - trackHeight / 2, Math.max(0, geometry.left + geometry.width - outX), trackHeight);

      context.fillStyle = "#5d6270";
      for (let index = 0; index <= 4; index += 1) {
        const x = geometry.left + geometry.width * (index / 4);
        context.fillRect(Math.round(x), trackY - 5, 1, 10);
      }

      const drawHandle = (x) => {
        context.fillStyle = "#c4b5fd";
        context.fillRect(Math.round(x) - 2, trackY - 8, 4, 16);
        context.beginPath();
        context.arc(x, trackY, 3.5, 0, Math.PI * 2);
        context.fill();
      };
      drawHandle(inX);
      drawHandle(outX);

      const playheadX = this.trimXForTime(Number(this.video.currentTime) || 0, geometry, bounds.duration);
      context.fillStyle = "#ffffff";
      context.fillRect(Math.round(playheadX), trackY - 8, 1, 16);
      context.beginPath();
      context.arc(playheadX + 0.5, trackY - 7, 2.5, 0, Math.PI * 2);
      context.fill();
    }

    this.trimInLabel.textContent = formatTime(bounds.start, true);
    this.trimOutLabel.textContent = formatTime(bounds.end, true);
    this.trimFrameLabel.textContent = String(this.selectedFrameCount(bounds));
    this.trimTimeline.dataset.disabled = String(!bounds.duration);
  }

  applyTrimWindow() {
    if (!Number.isFinite(this.video.duration)) return;
    const duration = this.video.duration;
    const start = Math.min(Math.max(0, Number(this.settings.start_time) || 0), duration);
    const end = this.activePreviewEnd();
    if (this.video.currentTime < start || (end && this.video.currentTime > end)) {
      this.video.currentTime = start;
    }
    this.updateTime();
    this.renderTrimTimeline();
  }

  activePreviewEnd() {
    const duration = Number.isFinite(this.video.duration) ? this.video.duration : 0;
    const configured = Number(this.settings.end_time) || 0;
    return configured ? Math.min(configured, duration || configured) : duration;
  }

  applyPreviewAudio() {
    const volume = Math.max(0, Math.min(1, Number(this.node.properties.previewVolume)));
    this.volume.value = String(Math.round(volume * 100));
    this.volumeValue.textContent = `${Math.round(volume * 100)}%`;
    this.video.volume = volume;
    this.video.muted = Boolean(this.node.properties.previewMuted);
    this.muteButton.textContent = this.video.muted ? "🔇" : "🔊";
    this.muteButton.title = this.video.muted ? "Unmute preview" : "Mute preview";
  }

  updateTime() {
    const current = Number(this.video.currentTime) || 0;
    const end = this.activePreviewEnd() || Number(this.video.duration) || 0;
    this.time.textContent = `${formatTime(current)} / ${formatTime(end)}`;
  }

  updateSourceSummary() {
    const info = this.executionInfo || this.sourceInfo;
    if (!info) return;
    if (this.executionInfo) {
      const audio = info.has_audio ? "audio" : "silent";
      this.previewInfo.textContent = `${info.loaded_width}×${info.loaded_height} · ${Number(info.loaded_fps).toFixed(2)} fps · ${info.loaded_frame_count} frames · ${audio}`;
      return;
    }
    const audio = info.has_audio ? "audio" : "silent";
    const frameLabel = info.frame_count_estimated ? `~${info.frame_count}` : info.frame_count;
    this.previewInfo.textContent = `${info.width}×${info.height} · ${Number(info.frame_rate).toFixed(2)} fps · ${frameLabel} frames · ${audio}`;
  }

  updateMemoryEstimate() {
    if (!this.sourceInfo) {
      this.memory.textContent = "Decoded-memory estimate appears after probing.";
      this.memory.dataset.warning = "false";
      return;
    }
    const info = this.sourceInfo;
    const start = Math.min(this.settings.start_time, info.duration);
    const end = Math.min(this.settings.end_time || info.duration, info.duration);
    const duration = Math.max(0, end - start);
    let fps = info.frame_rate;
    if (this.settings.sample_mode === "target_fps") fps = this.settings.target_fps;
    if (this.settings.sample_mode === "every_nth") fps /= Math.max(1, this.settings.select_every_nth);
    let outputFrames = Math.max(1, Math.ceil(duration * fps));
    if (this.settings.frame_load_cap) outputFrames = Math.min(outputFrames, this.settings.frame_load_cap);
    const decodeDuration = this.settings.frame_load_cap
      ? Math.min(duration, this.settings.frame_load_cap / fps)
      : duration;
    const sourceFrames = Math.max(1, Math.ceil(decodeDuration * info.frame_rate));

    let width = info.width;
    let height = info.height;
    if (this.settings.resize_mode === "crop") {
      width = this.settings.width || width;
      height = this.settings.height || height;
    } else if (this.settings.resize_mode === "fit" && (this.settings.width || this.settings.height)) {
      const scale = this.settings.width && this.settings.height
        ? Math.min(this.settings.width / width, this.settings.height / height)
        : this.settings.width ? this.settings.width / width : this.settings.height / height;
      width = Math.max(1, Math.round(width * scale));
      height = Math.max(1, Math.round(height * scale));
    }

    const sourceBytes = sourceFrames * info.width * info.height * 3 * 4;
    const outputBytes = outputFrames * width * height * 3 * 4;
    const estimate = sourceBytes + outputBytes + Math.min(sourceBytes, outputBytes);
    this.memory.textContent = `Estimated peak decoded memory: ${formatBytes(estimate)}.`;
    this.memory.dataset.warning = String(estimate >= 1024 ** 3);
  }

  updateFromExecution(message) {
    const info = message?.fl_load_video?.[0];
    if (!info) return;
    this.executionInfo = { ...info };
    this.node.properties.lastExecutionInfo = { ...info };
    this.node.properties.lastLoadSettings = this.settingsWidget.value;
    this.updateSourceSummary();
    this.setStatus("ready", "loaded");
  }

  setStatus(state, label) {
    this.status.dataset.state = state;
    this.status.textContent = label;
  }

  setMenuOpen(open) {
    this.settingsMenu.hidden = !open;
    this.moreButton.setAttribute("aria-expanded", String(open));
    if (open) this.syncInputVideoOptions();
  }

  showError(message) {
    this.error.textContent = message;
    this.error.hidden = false;
    this.setStatus("error", "error");
  }

  clearError() {
    this.error.textContent = "";
    this.error.hidden = true;
  }

  configure() {
    hideWidget(this.videoWidget);
    hideWidget(this.settingsWidget);
    this.readSettings();
    this.syncControls();
    this.setMenuOpen(false);
    this.executionInfo = this.node.properties.lastExecutionInfo || null;
    this.restoreSource();
    if (this.executionInfo && this.node.properties.lastLoadSettings !== this.settingsWidget.value) {
      this.setStatus("stale", "settings changed");
    }
  }

  revokeObjectUrl() {
    if (!this.objectUrl) return;
    URL.revokeObjectURL(this.objectUrl);
    this.objectUrl = null;
  }

  dispose() {
    this.disposed = true;
    this.probeId += 1;
    document.removeEventListener("pointerdown", this.handleDocumentPointerDown);
    document.removeEventListener("keydown", this.handleDocumentKeyDown);
    this.trimResizeObserver?.disconnect();
    this.revokeObjectUrl();
    this.video.pause();
    this.video.removeAttribute("src");
    this.video.load();
    this.container.replaceChildren();
  }
}

app.registerExtension({
  name: "ComfyUI.FL_LoadVideo",
  nodeCreated(node) {
    if (node.comfyClass !== "FL_LoadVideo") return;

    const videoWidget = node.widgets?.find((widget) => widget.name === "video");
    const settingsWidget = node.widgets?.find((widget) => widget.name === "load_settings");
    if (!videoWidget || !settingsWidget) return;
    hideWidget(videoWidget);
    hideWidget(settingsWidget);

    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.height = "100%";
    container.style.minHeight = `${MIN_PANEL_HEIGHT}px`;
    container.style.overflow = "hidden";

    const domWidget = node.addDOMWidget("fl_load_video_panel", "fl-load-video", container, {
      getMinHeight: () => MIN_PANEL_HEIGHT,
      hideOnZoom: false,
      serialize: false,
    });
    enforceMinimumNodeSize(node);
    requestAnimationFrame(() => enforceMinimumNodeSize(node));

    const panel = new LoadVideoPanel(node, videoWidget, settingsWidget, container);

    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      originalOnExecuted?.apply(this, arguments);
      panel.updateFromExecution(message);
    };

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const result = originalOnConfigure?.apply(this, args);
      panel.configure();
      requestAnimationFrame(() => enforceMinimumNodeSize(this));
      return result;
    };

    domWidget.onRemove = () => panel.dispose();
  },
});
