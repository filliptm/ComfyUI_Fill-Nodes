import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const STYLE_ID = "fl-beat-prompt-sequencer-styles";
const INSTANCES = new Map();
const HEADER_RE = /^\s*\[\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)(?:\s*\|\s*(.*?))?\s*\]\s*$/;
const EPSILON = 1e-6;
const FORMAT_VERSION = 11;
const COMPATIBLE_FORMAT_VERSIONS = new Set([6, 7, 8, 9, 10, FORMAT_VERSION]);
const COMPACT_NODE_WIDTH = 380;
const MEDIA_FILE_RE = /\.(?:aac|aiff?|flac|m4a|mka|mkv|mov|mp3|mp4|oga|ogg|opus|wav|webm|wma)$/i;
const TIMELINE_LEFT = 16;
const TIMELINE_RIGHT = 12;
const DEFAULT_CLIP_GRID_INTERVALS = 4;
let activeModal = null;
const GRID_DENSITY_LABELS = {
  every_2_beats: "Every 2 beats",
  every_beat: "Every beat",
  half_beat: "Half-beat",
};
const STYLES = `
  .flbps-root {
    height: 100%;
    min-height: 0;
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
  .flbps-root:focus { outline: none; }
  .flbps-root:focus-visible { outline: 1px solid #525762; outline-offset: -1px; }
  .flbps-toolbar, .flbps-actions, .flbps-footer {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 7px 9px;
    border-bottom: 1px solid #2b2b31;
    background: #1c1c20;
  }
  .flbps-status {
    max-width: 390px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    padding: 4px 8px;
    border-radius: 10px;
    color: #a1a1aa;
    background: #27272a;
    font-size: 9px;
  }
  .flbps-status.fresh { color: #d1fae5; background: #065f46; }
  .flbps-status.cached { color: #fef3c7; background: #713f12; }
  .flbps-status.error { color: #fee2e2; background: #7f1d1d; }
  .flbps-toolbar {
    flex-wrap: wrap;
    gap: 8px;
    padding-top: 6px;
    padding-bottom: 6px;
    background: #17191e;
  }
  .flbps-control-group { display: flex; align-items: center; gap: 7px; }
  .flbps-toolbar-divider { width: 1px; height: 20px; background: #343740; }
  .flbps-transport {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 6px 9px;
    border-bottom: 1px solid #2b2b31;
    background: #18181c;
  }
  .flbps-transport-time {
    min-width: 105px;
    color: #fbbf24;
    font: 10px "Cascadia Mono", Consolas, monospace;
  }
  .flbps-source-label {
    min-width: 0;
    overflow: hidden;
    color: #a1a1aa;
    font-size: 9px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .flbps-auto {
    display: flex;
    align-items: center;
    gap: 4px;
    color: #a1a1aa;
    font-size: 9px;
  }
  .flbps-control {
    display: flex;
    align-items: center;
    gap: 4px;
    color: #a1a1aa;
    font-size: 9px;
  }
  .flbps-control select, .flbps-control input[type="number"], .flbps-inspector input,
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
  .flbps-control input[type="range"] {
    width: 110px;
    accent-color: #22d3ee;
  }
  .flbps-control input[type="number"] {
    width: 62px;
    height: 23px;
    padding: 2px 4px;
    font-size: 9px;
    text-align: right;
  }
  .flbps-offset-frames {
    min-width: 66px;
    color: #67e8f9;
    font: 9px "Cascadia Mono", Consolas, monospace;
  }
  .flbps-control select:focus, .flbps-control input[type="number"]:focus, .flbps-inspector input:focus,
  .flbps-inspector textarea:focus, .flbps-raw textarea:focus { border-color: #22d3ee; }
  .flbps-canvas-wrap {
    position: relative;
    height: clamp(300px, 45vh, 420px);
    flex: 0 1 420px;
    min-height: 280px;
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
    transition: color .1s ease, background .1s ease, border-color .1s ease, opacity .1s ease;
  }
  .flbps-button:hover { color: #fff; border-color: #52525b; background: #303036; }
  .flbps-button.primary { color: #ecfeff; border-color: #0e7490; background: #155e75; }
  .flbps-button.active { color: #cffafe; border-color: #0891b2; background: #164e63; }
  .flbps-button.danger:hover { border-color: #b91c1c; background: #7f1d1d; }
  .flbps-button:disabled { opacity: .4; cursor: default; }
  .flbps-spacer { flex: 1; }
  .flbps-inspector {
    flex: 1 1 150px;
    min-height: 128px;
    display: flex;
    flex-direction: column;
    padding: 8px 9px;
    background: #19191d;
    border-bottom: 1px solid #2b2b31;
  }
  .flbps-inspector.disabled { opacity: 0.45; pointer-events: none; }
  .flbps-inspector-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
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
    color: #c4b5fd;
    text-transform: none;
    letter-spacing: 0;
  }
  .flbps-inspector textarea {
    width: 100%;
    min-height: 68px;
    flex: 1 1 auto;
    resize: vertical;
    padding: 7px;
    font-size: 10px;
    line-height: 1.4;
  }
  .flbps-raw { display: none; flex: 0 0 auto; padding: 8px 9px; background: #17171a; border-bottom: 1px solid #2b2b31; }
  .flbps-raw.open { display: block; }
  .flbps-raw-label { margin-bottom: 5px; color: #a1a1aa; font-size: 9px; }
  .flbps-raw textarea { width: 100%; height: 130px; resize: vertical; padding: 7px; font-family: "Cascadia Mono", Consolas, monospace; font-size: 9px; line-height: 1.35; }
  .flbps-raw-actions { display: flex; gap: 6px; margin-top: 6px; justify-content: flex-end; }
  .flbps-footer {
    justify-content: flex-end;
    border-bottom: 0;
    color: #71717a;
    font-size: 8px;
  }
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
  .flbps-context-menu {
    position: fixed;
    z-index: 10020;
    min-width: 220px;
    padding: 5px;
    color: #e4e4e7;
    background: #202127;
    border: 1px solid #555b68;
    border-radius: 7px;
    box-shadow: 0 14px 38px rgba(0, 0, 0, .58);
    font: 10px Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  .flbps-context-title {
    padding: 5px 7px 6px;
    color: #8ddde8;
    border-bottom: 1px solid #363a43;
    font: 9px "Cascadia Mono", Consolas, monospace;
  }
  .flbps-context-menu button {
    width: 100%;
    display: block;
    padding: 7px;
    color: #e4e4e7;
    background: transparent;
    border: 0;
    border-radius: 4px;
    font: inherit;
    text-align: left;
    cursor: pointer;
  }
  .flbps-context-menu button:hover { color: #fff; background: #343844; }
  .flbps-context-menu button:disabled { color: #646975; cursor: default; }
  .flbps-context-menu button:disabled:hover { background: transparent; }
  .flbps-modal-overlay {
    position: fixed;
    inset: 0;
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2.5vh 2.5vw;
    background: rgba(0, 0, 0, .84);
    backdrop-filter: blur(4px);
    animation: flbps-fade-in .15s ease-out;
  }
  .flbps-modal-shell {
    width: 95vw;
    height: 94vh;
    max-width: 1900px;
    max-height: 1400px;
    min-width: 760px;
    min-height: 600px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    color: #e4e4e7;
    background: #111114;
    border: 1px solid #3f3f46;
    border-radius: 12px;
    box-shadow: 0 24px 80px rgba(0, 0, 0, .72);
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    animation: flbps-modal-in .18s ease-out;
  }
  .flbps-modal-header {
    flex: 0 0 auto;
    min-height: 52px;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px 9px 16px;
    background: #1b1b20;
    border-bottom: 1px solid #303036;
  }
  .flbps-modal-heading { min-width: 0; display: flex; flex-direction: column; gap: 2px; }
  .flbps-modal-title { color: #fafafa; font-size: 14px; font-weight: 700; }
  .flbps-modal-subtitle {
    max-width: 62vw;
    overflow: hidden;
    color: #a1a1aa;
    font-size: 10px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .flbps-modal-main { flex: 1 1 auto; min-height: 0; display: flex; }
  .flbps-library {
    width: 310px;
    flex: 0 0 310px;
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: 9px;
    padding: 11px;
    overflow: hidden;
    background: #17171b;
    border-right: 1px solid #303036;
    transition: width .16s ease, flex-basis .16s ease, padding .16s ease,
      opacity .12s ease, border-color .12s ease;
  }
  .flbps-modal-shell.library-collapsed .flbps-library {
    width: 0;
    flex-basis: 0;
    padding-left: 0;
    padding-right: 0;
    opacity: 0;
    border-right-color: transparent;
    pointer-events: none;
  }
  .flbps-library-section { flex: 0 0 auto; display: flex; flex-direction: column; gap: 6px; }
  .flbps-library-label {
    color: #8b8b95;
    font-size: 8px;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
  }
  .flbps-drop-zone {
    min-height: 70px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 10px;
    color: #a1a1aa;
    background: #202027;
    border: 1px dashed #52525b;
    border-radius: 7px;
    font-size: 10px;
    line-height: 1.4;
    text-align: center;
    cursor: pointer;
  }
  .flbps-drop-zone.dragging { color: #cffafe; background: #164e63; border-color: #22d3ee; }
  .flbps-library-actions, .flbps-library-tabs { display: flex; gap: 6px; }
  .flbps-library-actions .flbps-button, .flbps-library-tabs .flbps-button { flex: 1; }
  .flbps-library-search, .flbps-library-folder, .flbps-setting input, .flbps-setting select {
    width: 100%;
    height: 28px;
    padding: 4px 7px;
    color: #f4f4f5;
    background: #252529;
    border: 1px solid #3f3f46;
    border-radius: 5px;
    outline: none;
    font: inherit;
    font-size: 10px;
  }
  .flbps-library-search:focus, .flbps-library-folder:focus,
  .flbps-setting input:focus, .flbps-setting select:focus { border-color: #22d3ee; }
  .flbps-library-results {
    flex: 1 1 180px;
    min-height: 120px;
    overflow: auto;
    background: #121216;
    border: 1px solid #2f2f35;
    border-radius: 6px;
  }
  .flbps-file-row {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 7px 8px;
    color: #d4d4d8;
    background: transparent;
    border: 0;
    border-bottom: 1px solid #25252a;
    font: inherit;
    text-align: left;
    cursor: pointer;
  }
  .flbps-file-row:hover { background: #27272e; }
  .flbps-file-row.selected { color: #cffafe; background: #164e63; }
  .flbps-file-name { overflow: hidden; font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
  .flbps-file-folder { overflow: hidden; color: #71717a; font-size: 8px; text-overflow: ellipsis; white-space: nowrap; }
  .flbps-library-message { color: #8b8b95; font-size: 9px; line-height: 1.35; }
  .flbps-settings {
    flex: 0 0 auto;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 7px;
  }
  .flbps-setting { min-width: 0; display: flex; flex-direction: column; gap: 3px; }
  .flbps-setting label { color: #8b8b95; font-size: 8px; }
  .flbps-setting.checkbox { flex-direction: row; align-items: center; padding-top: 15px; }
  .flbps-setting.checkbox input { width: auto; height: auto; }
  .flbps-editor-host { flex: 1 1 auto; min-width: 0; min-height: 0; padding: 8px; }
  .flbps-sidebar-toggle { min-width: 82px; }
  .flbps-modal-close { min-width: 66px; }
  @keyframes flbps-fade-in { from { opacity: 0; } to { opacity: 1; } }
  @keyframes flbps-modal-in {
    from { opacity: 0; transform: scale(.975) translateY(-8px); }
    to { opacity: 1; transform: scale(1) translateY(0); }
  }
  @media (max-width: 980px) {
    .flbps-modal-overlay { padding: 0; }
    .flbps-modal-shell { width: 100vw; height: 100vh; min-width: 0; min-height: 0; border-radius: 0; }
    .flbps-library { width: 250px; flex-basis: 250px; }
    .flbps-status { display: none; }
    .flbps-toolbar-divider { display: none; }
  }
  @media (max-width: 1250px) and (min-width: 981px) {
    .flbps-status { max-width: 220px; }
    .flbps-source-label { max-width: 130px; }
  }
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

function formatRulerTime(seconds) {
  const value = Math.max(0, finiteNumber(seconds));
  if (value < 60) return `${value.toFixed(value < 10 ? 1 : 0)}s`;
  const minutes = Math.floor(value / 60);
  return `${minutes}:${String(Math.floor(value % 60)).padStart(2, "0")}`;
}

function canvasTextLines(ctx, text, maxWidth, maximumLines = 2) {
  const words = String(text || "").replace(/\s+/g, " ").trim().split(" ").filter(Boolean);
  const lines = [];
  let line = "";
  while (words.length && lines.length < maximumLines) {
    const word = words.shift();
    const candidate = line ? `${line} ${word}` : word;
    if (!line || ctx.measureText(candidate).width <= maxWidth) {
      line = candidate;
      continue;
    }
    lines.push(line);
    line = word;
  }
  if (line && lines.length < maximumLines) lines.push(line);
  if (words.length && lines.length) {
    let last = `${lines.pop()}…`;
    while (last.length > 1 && ctx.measureText(last).width > maxWidth) {
      last = `${last.slice(0, -2)}…`;
    }
    lines.push(last);
  }
  return lines;
}

function filenameFromPath(value) {
  const parts = String(value || "").replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || "";
}

function executionPayload(message) {
  const values = message?.fl_prompt_sequencer ?? message?.ui?.fl_prompt_sequencer;
  return Array.isArray(values) ? values[0] : values;
}

function isSupportedMediaFile(file) {
  return Boolean(
    file &&
    ((file.type || "").startsWith("audio/") ||
      (file.type || "").startsWith("video/") ||
      MEDIA_FILE_RE.test(file.name || "")),
  );
}

function setWidgetValue(widget, value) {
  if (!widget) return;
  widget.value = value;
  widget.callback?.call(widget, value);
}

function parseOptions(raw, defaultFadeIn, defaultFadeOut, lineNumber) {
  const values = {
    fadeIn: finiteNumber(defaultFadeIn),
    fadeOut: finiteNumber(defaultFadeOut),
    crossfade: 0,
  };
  if (!raw) return values;
  for (const part of raw.split("|")) {
    const separator = part.indexOf("=");
    if (separator < 0) {
      throw new Error(
        `Line ${lineNumber}: options must use fade_in=value, fade_out=value, or crossfade=value.`,
      );
    }
    const name = part.slice(0, separator).trim();
    const value = Number(part.slice(separator + 1).trim());
    if (name !== "fade_in" && name !== "fade_out" && name !== "crossfade") {
      throw new Error(`Line ${lineNumber}: unknown option '${name}'.`);
    }
    if (!Number.isFinite(value) || value < 0) {
      throw new Error(`Line ${lineNumber}: ${name} must be zero or greater.`);
    }
    if (name === "fade_in") values.fadeIn = value;
    else if (name === "fade_out") values.fadeOut = value;
    else values.crossfade = value;
  }
  return values;
}

function validateCrossfades(clips) {
  for (let index = 0; index < clips.length; index++) {
    const clip = clips[index];
    const previous = clips[index - 1];
    if (!(clip.crossfade > EPSILON)) continue;
    if (!previous) throw new Error(`Line ${clip.line}: the first section cannot crossfade.`);
    if (Math.abs(previous.end - clip.start) > EPSILON) {
      throw new Error(`Line ${clip.line}: crossfade requires a touching previous section.`);
    }
    const maximum = Math.min(previous.end - previous.start, clip.end - clip.start);
    if (clip.crossfade > maximum + EPSILON) {
      throw new Error(`Line ${clip.line}: crossfade exceeds the shorter adjacent section.`);
    }
    previous.fadeOut = 0;
    clip.fadeIn = 0;
  }
  return clips;
}

function normalizeCrossfades(clips) {
  for (let index = 0; index < clips.length; index++) {
    const clip = clips[index];
    clip.crossfade = Math.max(0, Math.round(finiteNumber(clip.crossfade)));
    const previous = clips[index - 1];
    if (!previous || previous.end !== clip.start) {
      clip.crossfade = 0;
      continue;
    }
    clip.crossfade = Math.min(
      clip.crossfade,
      previous.end - previous.start,
      clip.end - clip.start,
    );
    if (clip.crossfade > 0) {
      previous.fadeOut = 0;
      clip.fadeIn = 0;
    }
  }
  return clips;
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
        crossfade: options.crossfade,
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
  return validateCrossfades(clips);
}

function validateFrameClips(clips) {
  for (const clip of clips) {
    for (const [name, value] of Object.entries({
      start: clip.start,
      end: clip.end,
      fade_in: clip.fadeIn,
      fade_out: clip.fadeOut,
      crossfade: clip.crossfade,
    })) {
      if (Math.abs(value - Math.round(value)) > EPSILON) {
        throw new Error(`Line ${clip.line}: ${name} must be a whole frame.`);
      }
    }
    clip.start = Math.round(clip.start);
    clip.end = Math.round(clip.end);
    clip.fadeIn = Math.round(clip.fadeIn);
    clip.fadeOut = Math.round(clip.fadeOut);
    clip.crossfade = Math.round(clip.crossfade);
  }
  return validateCrossfades(clips);
}

function serializeTimeline(clips) {
  return clips.map((clip) => (
    `[${Math.round(clip.start)} - ${Math.round(clip.end)} | ` +
    `fade_in=${Math.round(clip.fadeIn)} | fade_out=${Math.round(clip.fadeOut)} | ` +
    `crossfade=${Math.round(clip.crossfade)}]\n` +
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

function waveformPreviewFromBuffer(buffer) {
  const bucketCount = Math.min(8192, Math.max(1, Math.ceil(buffer.duration * 60)));
  const peaks = new Array(bucketCount * 2);
  for (let bucket = 0; bucket < bucketCount; bucket++) {
    const start = Math.floor(bucket / bucketCount * buffer.length);
    const end = Math.max(start + 1, Math.floor((bucket + 1) / bucketCount * buffer.length));
    let minimum = 1;
    let maximum = -1;
    for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
      const samples = buffer.getChannelData(channel);
      for (let index = start; index < end; index++) {
        minimum = Math.min(minimum, samples[index]);
        maximum = Math.max(maximum, samples[index]);
      }
    }
    peaks[bucket * 2] = Math.round(clamp(minimum, -1, 1) * 32767);
    peaks[bucket * 2 + 1] = Math.round(clamp(maximum, -1, 1) * 32767);
  }
  return { version: 1, duration: buffer.duration, scale: 32767, peaks };
}

function cropWaveformPreview(preview, startSeconds, duration) {
  if (!preview || !(duration > 0)) return null;
  const bucketCount = preview.peaks.length / 2;
  const startBucket = clamp(Math.floor(startSeconds / preview.duration * bucketCount), 0, bucketCount - 1);
  const endBucket = clamp(
    Math.ceil((startSeconds + duration) / preview.duration * bucketCount),
    startBucket + 1,
    bucketCount,
  );
  return {
    version: 1,
    duration,
    scale: preview.scale,
    peaks: preview.peaks.slice(startBucket * 2, endBucket * 2),
  };
}

function audioViewURL(value) {
  const match = String(value || "").match(/^(.*?)(?:\s+\[(input|output|temp)\])?$/);
  const relative = (match?.[1] || "").replace(/\\/g, "/");
  const slash = relative.lastIndexOf("/");
  const filename = slash >= 0 ? relative.slice(slash + 1) : relative;
  const subfolder = slash >= 0 ? relative.slice(0, slash) : "";
  const params = new URLSearchParams({
    filename,
    subfolder,
    type: match?.[2] || "input",
  });
  return api.apiURL(`/view?${params.toString()}`);
}

class BeatPromptSequencer {
  constructor({ node, container, widgets, onStateChange = null }) {
    this.node = node;
    this.container = container;
    this.widgets = widgets;
    this.onStateChange = onStateChange;
    this.clips = [];
    this.selectedIndex = -1;
    this.playheadFrame = null;
    this.snapGuideFrame = null;
    this.drag = null;
    this.resnapPending = false;
    this.clipRects = [];
    this.crossfadeRects = [];
    this.pendingFrame = null;
    this.resizeObserver = null;
    this.callbackRestorers = [];
    this.rawInvalid = false;
    this.migrationPending = false;
    this.hover = null;
    this.sourceWaveformPreview = null;
    this.sourceAudioDuration = 0;
    this.audioElement = null;
    this.audioURL = "";
    this.playbackFrameRequest = null;
    this.analysisTimer = null;
    this.analysisRequest = 0;
    this.loadingAudio = false;
    this.separationJobId = node._flAudioSeparationJobId || null;
    this.separationTimer = null;
    this.contextMenu = null;
    this.documentPointerHandler = (event) => {
      if (this.contextMenu && !this.contextMenu.contains(event.target)) {
        this.closeContextMenu();
      }
    };

    const saved = node.properties?.flBeatPromptSequencer || {};
    const savedCompatible = COMPATIBLE_FORMAT_VERSIONS.has(finiteNumber(saved.formatVersion));
    this.beatData = savedCompatible ? saved.beatData || null : null;
    if (this.beatData) {
      this.beatData.waveformPreview = normalizeWaveformPreview(this.beatData.waveformPreview);
    }
    this.dataFresh = false;
    this.viewStart = savedCompatible ? finiteNumber(saved.viewStart, 0) : 0;
    this.viewEnd = savedCompatible ? finiteNumber(saved.viewEnd, 0) : 0;
    this.waveformVisible = saved.waveformVisible !== false;
    this.autoAnalyze = saved.autoAnalyze !== false;

    injectStyles();
    this.build();
    this.bindWidgetCallbacks();
    if (this.separationJobId) {
      this.root.querySelector('[data-action="separate"]').textContent = "Cancel separation";
      this.pollSeparation();
    }
    this.applyBeatOffset();
    this.loadTimeline();
    this.resnapClipsToGrid();
    this.refreshBeatStatus();
    if (!(this.viewEnd > this.viewStart)) this.zoomToFit(false);
    if (this.widgets.audioFile?.value) this.loadAudioSource();
    this.scheduleDraw();
  }

  fps() {
    return Math.max(1, finiteNumber(this.widgets.fps?.value, 24));
  }

  beatOffsetMs() {
    return clamp(Math.round(finiteNumber(this.widgets.beatOffset?.value, 0)), -1000, 1000);
  }

  beatGridDensity() {
    const value = this.widgets.beatGridDensity?.value;
    return value in GRID_DENSITY_LABELS ? value : "every_beat";
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
      <div class="flbps-transport">
        <button class="flbps-button" data-action="play" title="Play or pause the selected audio crop; playback loops continuously">Play</button>
        <button class="flbps-button" data-action="stop" title="Stop and return to the crop start">Stop</button>
        <span class="flbps-transport-time" data-role="transport-time">00:00.000 / 00:00.000</span>
        <span class="flbps-source-label" data-role="source-label">No audio selected</span>
        <span class="flbps-spacer"></span>
        <span class="flbps-status" data-role="status">Choose audio to load the timeline</span>
        <label class="flbps-auto" title="Refresh beat, onset, and drum markers after audio or trim changes">
          <input data-role="auto-analyze" type="checkbox"> Auto analyze
        </label>
        <button class="flbps-button" data-action="analyze" title="Analyze beats, onsets, and drums without queueing the workflow">Analyze</button>
        <button class="flbps-button" data-action="separate" title="Explicitly separate and cache stems for analysis">Separate stems</button>
      </div>
      <div class="flbps-toolbar">
        <div class="flbps-control-group">
          <label class="flbps-control" title="Choose the spacing of the cyan grid used for display, snapping, and prompt timing.">
            Grid
            <select data-role="beat-grid-density">
              <option value="every_2_beats">Every 2 beats</option>
              <option value="every_beat">Every beat</option>
              <option value="half_beat">Half-beat</option>
            </select>
          </label>
        </div>
        <span class="flbps-toolbar-divider"></span>
        <div class="flbps-control-group">
          <label class="flbps-control" title="Shift the cyan beat grid over the stationary waveform and detected reference ticks.">
            Beat offset
            <input data-role="beat-offset" type="range" min="-1000" max="1000" step="1">
            <input data-role="beat-offset-number" type="number" min="-1000" max="1000" step="1" aria-label="Beat offset in milliseconds">
            <span class="flbps-offset-frames" data-role="beat-offset-frames"></span>
          </label>
          <button class="flbps-button" data-action="reset-offset" title="Reset the beat offset to zero">Zero</button>
        </div>
        <span class="flbps-toolbar-divider"></span>
        <div class="flbps-control-group">
          <button class="flbps-button" data-action="zoom-out" title="Show more frames">Zoom -</button>
          <button class="flbps-button" data-action="zoom-in" title="Show fewer frames">Zoom +</button>
          <button class="flbps-button" data-action="fit" title="Show the complete frame range">Fit</button>
          <button class="flbps-button" data-action="waveform" title="Show or hide the aligned audio waveform">Waveform</button>
        </div>
        <span class="flbps-spacer"></span>
        <span class="flbps-control">Frames</span>
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
          <div class="flbps-field"><label>Crossfade frames</label><input data-field="crossfade" type="number" min="0" step="1" title="Blend from the touching previous prompt into this prompt"></div>
        </div>
        <div class="flbps-prompt-label">
          <span>Prompt</span><span class="flbps-prompt-meta" data-role="prompt-meta"></span>
        </div>
        <textarea data-field="prompt" placeholder="Describe what should happen during this frame range."></textarea>
      </div>
      <div class="flbps-raw" data-role="raw-panel">
        <div class="flbps-raw-label">Advanced frame schedule. All positions, fades, and crossfades must be integer frames.</div>
        <textarea data-role="raw-text"></textarea>
        <div class="flbps-raw-actions">
          <button class="flbps-button" data-action="raw-cancel">Close</button>
          <button class="flbps-button primary" data-action="raw-apply">Apply frames</button>
        </div>
      </div>
      <div class="flbps-footer">
        <span>drag a shared cut to resize both prompts · double-click it for crossfade · right-click a beat for audio In/Out · Space play/pause</span>
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
    this.promptMetaEl = this.root.querySelector('[data-role="prompt-meta"]');
    this.transportTimeEl = this.root.querySelector('[data-role="transport-time"]');
    this.sourceLabelEl = this.root.querySelector('[data-role="source-label"]');
    this.controls = {
      beatGridDensity: this.root.querySelector('[data-role="beat-grid-density"]'),
      autoAnalyze: this.root.querySelector('[data-role="auto-analyze"]'),
      beatOffset: this.root.querySelector('[data-role="beat-offset"]'),
      beatOffsetNumber: this.root.querySelector('[data-role="beat-offset-number"]'),
      beatOffsetFrames: this.root.querySelector('[data-role="beat-offset-frames"]'),
    };
    this.fields = {
      start: this.root.querySelector('[data-field="start"]'),
      end: this.root.querySelector('[data-field="end"]'),
      duration: this.root.querySelector('[data-field="clip-duration"]'),
      fadeIn: this.root.querySelector('[data-field="fade-in"]'),
      fadeOut: this.root.querySelector('[data-field="fade-out"]'),
      crossfade: this.root.querySelector('[data-field="crossfade"]'),
      prompt: this.root.querySelector('[data-field="prompt"]'),
    };
    this.editButtons = [
      ...this.root.querySelectorAll('[data-action="add"], [data-action="split"], [data-action="duplicate"], [data-action="delete"]'),
    ];

    this.controls.beatGridDensity.value = this.beatGridDensity();
    this.controls.autoAnalyze.checked = this.autoAnalyze;
    this.syncBeatOffsetControls();
    this.waveformButton = this.root.querySelector('[data-action="waveform"]');
    this.waveformButton.classList.toggle("active", this.waveformVisible);
    this.controls.beatGridDensity.addEventListener("change", () => {
      this.setBeatGridDensity(this.controls.beatGridDensity.value);
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
    this.controls.autoAnalyze.addEventListener("change", () => {
      this.autoAnalyze = this.controls.autoAnalyze.checked;
      this.saveViewState();
      if (this.autoAnalyze) this.requestAnalysis();
    });
    this.controls.beatOffset.addEventListener("input", () => {
      this.setBeatOffset(this.controls.beatOffset.value);
    });
    this.controls.beatOffset.addEventListener("change", () => {
      this.setBeatOffset(this.controls.beatOffset.value, true, true);
    });
    this.controls.beatOffsetNumber.addEventListener("input", () => {
      if (this.controls.beatOffsetNumber.value !== "") {
        this.setBeatOffset(this.controls.beatOffsetNumber.value);
      }
    });
    this.controls.beatOffsetNumber.addEventListener("change", () => {
      this.setBeatOffset(this.controls.beatOffsetNumber.value, true, true);
    });
    this.root.querySelector('[data-action="reset-offset"]').addEventListener("click", () => {
      this.setBeatOffset(0, true, true);
    });
    this.root.querySelector('[data-action="play"]').addEventListener("click", () => this.togglePlayback());
    this.root.querySelector('[data-action="stop"]').addEventListener("click", () => this.stopPlayback());
    this.root.querySelector('[data-action="analyze"]').addEventListener("click", () => this.requestAnalysis(true));
    this.root.querySelector('[data-action="separate"]').addEventListener("click", () => this.startSeparation());
    this.root.querySelector('[data-action="add"]').addEventListener("click", () => this.addClip());
    this.root.querySelector('[data-action="split"]').addEventListener("click", () => this.splitClip());
    this.root.querySelector('[data-action="duplicate"]').addEventListener("click", () => this.duplicateClip());
    this.root.querySelector('[data-action="delete"]').addEventListener("click", () => this.deleteClip());
    this.root.querySelector('[data-action="raw"]').addEventListener("click", () => this.toggleRaw());
    this.root.querySelector('[data-action="raw-cancel"]').addEventListener("click", () => this.toggleRaw(false));
    this.root.querySelector('[data-action="raw-apply"]').addEventListener("click", () => this.applyRaw());

    for (const name of ["start", "end", "fadeIn", "fadeOut", "crossfade"]) {
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
    this.canvas.addEventListener("dblclick", (event) => this.onDoubleClick(event));
    this.canvas.addEventListener("contextmenu", (event) => this.onContextMenu(event));
    this.canvas.addEventListener("auxclick", (event) => {
      if (event.button === 1) event.preventDefault();
    });
    this.canvas.addEventListener("wheel", (event) => this.onWheel(event), { passive: false });
    this.root.addEventListener("keydown", (event) => this.onKeyDown(event));
    document.addEventListener("pointerdown", this.documentPointerHandler, true);

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

    bind(this.widgets.timeUnit, () => {
      this.loadTimeline();
      this.resnapClipsToGrid();
    });
    bind(this.widgets.fps, () => {
      this.syncInspector();
      this.syncBeatOffsetControls();
      this.refreshBrowserCrop();
      this.resnapClipsToGrid();
      this.zoomToFit();
      this.markDirty();
      this.scheduleAnalysis();
    });
    bind(this.widgets.sequenceDuration, () => {
      this.refreshBrowserCrop();
      this.resnapClipsToGrid();
      this.zoomToFit();
      this.markDirty();
      this.scheduleAnalysis();
    });
    bind(this.widgets.audioFile, () => this.loadAudioSource());
    bind(this.widgets.trimStartFrame, () => {
      this.refreshBrowserCrop();
      this.scheduleAnalysis();
    });
    bind(this.widgets.bpmMethod, () => this.scheduleAnalysis());
    bind(this.widgets.halfTime, () => this.scheduleAnalysis());
    bind(this.widgets.beatOffset, (value) => this.setBeatOffset(value, false, true));
    bind(this.widgets.analysisSource, () => this.scheduleAnalysis());
    bind(this.widgets.beatGridDensity, (value) => this.setBeatGridDensity(value, false));
    bind(this.widgets.defaultFadeIn, () => this.markDirty());
    bind(this.widgets.defaultFadeOut, () => this.markDirty());
    bind(this.widgets.curve, () => this.markDirty());
  }

  markDirty() {
    this.node.graph?.change?.();
    this.onStateChange?.();
  }

  syncBeatOffsetControls() {
    if (!this.controls?.beatOffset) return;
    const offset = this.beatOffsetMs();
    const sign = offset > 0 ? "+" : "";
    const frames = offset / 1000 * this.fps();
    const frameSign = frames > 0 ? "+" : "";
    this.controls.beatOffset.value = String(offset);
    this.controls.beatOffsetNumber.value = String(offset);
    this.controls.beatOffsetFrames.textContent =
      `${sign}${offset} ms · ${frameSign}${frames.toFixed(2)} fr`;
  }

  baseGridIntervalSeconds() {
    const values = this.beatData?.baseBeatTimes || [];
    const configured = finiteNumber(this.beatData?.baseGridIntervalSeconds);
    if (configured > 0) return configured;
    if (values.length > 1) {
      const intervals = values
        .slice(1)
        .map((value, index) => finiteNumber(value) - finiteNumber(values[index]))
        .filter((value) => value > EPSILON)
        .sort((left, right) => left - right);
      if (intervals.length) {
        const middle = Math.floor(intervals.length / 2);
        return intervals.length % 2
          ? intervals[middle]
          : (intervals[middle - 1] + intervals[middle]) / 2;
      }
    }
    const bpm = finiteNumber(this.beatData?.bpm);
    return bpm > 0 ? 60 / bpm : 0;
  }

  gridIntervalSeconds() {
    const interval = this.baseGridIntervalSeconds();
    if (this.beatGridDensity() === "every_2_beats") return interval * 2;
    if (this.beatGridDensity() === "half_beat") return interval / 2;
    return interval;
  }

  baseGridTimes() {
    const values = this.beatData?.baseBeatTimes || [];
    if (this.beatGridDensity() === "every_2_beats") {
      return values.filter((_, index) => index % 2 === 0);
    }
    if (this.beatGridDensity() !== "half_beat") return values;
    const result = [];
    for (let index = 0; index < values.length; index++) {
      result.push(values[index]);
      if (index + 1 < values.length) {
        result.push((values[index] + values[index + 1]) / 2);
      }
    }
    return result;
  }

  gridBeatTimes(offsetMs = this.beatOffsetMs()) {
    const values = this.baseGridTimes();
    const duration = Math.max(0, finiteNumber(this.beatData?.audioDuration));
    const interval = this.gridIntervalSeconds();
    if (!values.length || !(duration > 0) || !(interval > 0)) return [];
    const offset = finiteNumber(offsetMs) / 1000;
    const shifted = values.map((value) => finiteNumber(value) + offset);
    const result = shifted.filter((value) => value >= 0 && value < duration);
    if (offset > 0) {
      for (let beatTime = shifted[0] - interval; beatTime >= 0; beatTime -= interval) {
        if (beatTime < duration) result.unshift(beatTime);
      }
    } else if (offset < 0) {
      for (let beatTime = shifted[shifted.length - 1] + interval;
        beatTime < duration;
        beatTime += interval) {
        if (beatTime >= 0) result.push(beatTime);
      }
    }
    return result;
  }

  applyBeatOffset() {
    this.syncBeatOffsetControls();
    const density = this.beatGridDensity();
    if (this.widgets.beatGridDensity?.value !== density) {
      this.widgets.beatGridDensity.value = density;
    }
    if (this.controls?.beatGridDensity) {
      this.controls.beatGridDensity.value = density;
    }
    if (!this.beatData) {
      this.scheduleDraw();
      return;
    }
    const interval = this.gridIntervalSeconds();
    this.beatData.beatTimes = this.gridBeatTimes();
    this.beatData.detectedBeatTimes = [...(this.beatData.baseDetectedBeatTimes || [])];
    this.beatData.beatOffsetMs = this.beatOffsetMs();
    this.beatData.beatGridDensity = density;
    this.beatData.baseGridIntervalSeconds = this.baseGridIntervalSeconds();
    this.beatData.gridIntervalSeconds = interval;
    this.beatData.gridBpm = interval > 0 ? 60 / interval : 0;
    this.refreshBeatStatus();
    this.scheduleDraw();
  }

  setBeatOffset(value, updateWidget = true, resnap = false) {
    const offset = clamp(Math.round(finiteNumber(value, 0)), -1000, 1000);
    if (this.widgets.beatOffset && (updateWidget || this.widgets.beatOffset.value !== offset)) {
      this.widgets.beatOffset.value = offset;
    }
    this.applyBeatOffset();
    if (resnap) this.resnapClipsToGrid();
    this.markDirty();
  }

  setBeatGridDensity(value, updateWidget = true) {
    const density = value in GRID_DENSITY_LABELS ? value : "every_beat";
    if (this.widgets.beatGridDensity &&
        (updateWidget || this.widgets.beatGridDensity.value !== density)) {
      this.widgets.beatGridDensity.value = density;
    }
    this.applyBeatOffset();
    this.resnapClipsToGrid();
    this.markDirty();
  }

  saveViewState() {
    this.node.properties = this.node.properties || {};
    const savedBeatData = this.beatData ? { ...this.beatData, waveformPreview: null } : null;
    const previous = { ...(this.node.properties.flBeatPromptSequencer || {}) };
    delete previous.magnetMode;
    delete previous.snapMode;
    this.node.properties.flBeatPromptSequencer = {
      ...previous,
      formatVersion: FORMAT_VERSION,
      beatData: savedBeatData,
      viewStart: this.viewStart,
      viewEnd: this.viewEnd,
      waveformVisible: this.waveformVisible,
      autoAnalyze: this.autoAnalyze,
    };
    this.markDirty();
  }

  trimStartFrame() {
    return Math.max(0, Math.round(finiteNumber(this.widgets.trimStartFrame?.value, 0)));
  }

  cropStartSeconds() {
    return this.trimStartFrame() / this.fps();
  }

  cropDurationSeconds() {
    const configured = this.configuredFrameCount();
    if (configured > 0) return configured / this.fps();
    return Math.max(0, this.sourceAudioDuration - this.cropStartSeconds());
  }

  setStatus(text, state = "") {
    this.statusEl.className = `flbps-status${state ? ` ${state}` : ""}`;
    this.statusEl.textContent = text;
  }

  invalidateAnalysis() {
    if (!this.beatData) return;
    this.beatData.baseBeatTimes = [];
    this.beatData.baseDetectedBeatTimes = [];
    this.beatData.beatTimes = [];
    this.beatData.detectedBeatTimes = [];
    this.beatData.onsetTimes = [];
    this.beatData.drumTimes = {};
    this.dataFresh = false;
    this.setStatus("Audio changed · analysis pending", "cached");
    this.scheduleDraw();
  }

  refreshBrowserCrop() {
    if (!this.sourceWaveformPreview || !(this.sourceAudioDuration > 0)) {
      this.updateTransportTime();
      return;
    }
    const start = this.cropStartSeconds();
    const duration = Math.min(this.cropDurationSeconds(), Math.max(0, this.sourceAudioDuration - start));
    const cropPreview = cropWaveformPreview(this.sourceWaveformPreview, start, duration);
    this.beatData = {
      ...(this.beatData || {}),
      bpm: finiteNumber(this.beatData?.bpm),
      beatTimes: this.beatData?.beatTimes || [],
      detectedBeatTimes: this.beatData?.detectedBeatTimes || [],
      onsetTimes: this.beatData?.onsetTimes || [],
      drumTimes: this.beatData?.drumTimes || {},
      audioDuration: duration,
      sourceDuration: this.sourceAudioDuration,
      sourceStart: start,
      waveformPreview: cropPreview,
    };
    this.updateTransportTime();
    this.scheduleDraw();
  }

  async loadAudioSource() {
    const filename = String(this.widgets.audioFile?.value || "");
    const request = ++this.analysisRequest;
    this.stopPlayback();
    this.sourceWaveformPreview = null;
    this.sourceAudioDuration = 0;
    if (!filename) {
      this.beatData = null;
      this.audioElement = null;
      this.sourceLabelEl.textContent = "No audio selected";
      this.setStatus("Choose audio or connect beat positions");
      this.scheduleDraw();
      return;
    }

    this.loadingAudio = true;
    this.sourceLabelEl.textContent = filename;
    this.setStatus("Decoding waveform…");
    const url = audioViewURL(filename);
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Audio preview request failed (${response.status}).`);
      const bytes = await response.arrayBuffer();
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) throw new Error("This browser does not support audio waveform decoding.");
      const context = new AudioContextClass();
      let buffer;
      try {
        buffer = await context.decodeAudioData(bytes);
      } finally {
        await context.close();
      }
      if (request !== this.analysisRequest) return;

      this.sourceAudioDuration = buffer.duration;
      this.sourceWaveformPreview = waveformPreviewFromBuffer(buffer);
      const availableFrames = Math.max(1, Math.floor(buffer.duration * this.fps()) - this.trimStartFrame());
      let settingsChanged = false;
      if (this.trimStartFrame() >= Math.floor(buffer.duration * this.fps())) {
        this.widgets.trimStartFrame.value = 0;
        settingsChanged = true;
      }
      if (!this.configuredFrameCount() || this.configuredFrameCount() > availableFrames) {
        this.widgets.sequenceDuration.value = Math.max(
          1,
          Math.floor((buffer.duration - this.cropStartSeconds()) * this.fps()),
        );
        settingsChanged = true;
      }
      if (settingsChanged) this.markDirty();
      this.audioURL = url;
      this.audioElement = new Audio(url);
      this.audioElement.preload = "auto";
      this.audioElement.addEventListener("timeupdate", () => this.updatePlaybackPosition());
      this.audioElement.addEventListener("pause", () => {
        this.stopPlaybackLoop();
        if (!this.audioElement.ended) this.updatePlaybackPosition();
        this.updatePlayButton();
      });
      this.audioElement.addEventListener("play", () => {
        this.updatePlayButton();
        this.startPlaybackLoop();
      });
      this.audioElement.addEventListener("ended", () => this.loopPlayback(true));
      this.refreshBrowserCrop();
      this.invalidateAnalysis();
      this.zoomToFit(false);
      this.scheduleAnalysis(0);
    } catch (error) {
      if (request !== this.analysisRequest) return;
      this.showError(`${error.message} Server analysis will still be used.`);
      this.scheduleAnalysis(0);
    } finally {
      if (request === this.analysisRequest) this.loadingAudio = false;
    }
  }

  scheduleAnalysis(delay = 250) {
    if (!this.autoAnalyze || !this.widgets.audioFile?.value) return;
    clearTimeout(this.analysisTimer);
    this.analysisTimer = setTimeout(() => this.requestAnalysis(), delay);
  }

  async requestAnalysis(force = false) {
    if (!force && !this.autoAnalyze) return;
    const audioFile = String(this.widgets.audioFile?.value || "");
    if (!audioFile) {
      this.showError("Choose an audio file before analyzing.");
      return;
    }
    clearTimeout(this.analysisTimer);
    const request = ++this.analysisRequest;
    this.setStatus("Analyzing beats, onsets, and drums…");
    try {
      const response = await api.fetchApi("/fl/audio-prompt-timeline/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio_file: audioFile,
          fps: this.fps(),
          trim_start_frame: this.trimStartFrame(),
          length_frames: this.configuredFrameCount(),
          bpm_method: this.widgets.bpmMethod?.value || "beat_intervals",
          half_time: Boolean(this.widgets.halfTime?.value),
          beat_offset_ms: 0,
          analysis_source: this.widgets.analysisSource?.value || "mix",
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `Analysis failed (${response.status}).`);
      if (request !== this.analysisRequest) return;
      this.applyAnalysis(payload, true);
      if (this.migrationPending) this.loadTimeline();
      this.resnapClipsToGrid();
      this.clearError();
      this.saveViewState();
    } catch (error) {
      if (request !== this.analysisRequest) return;
      this.showError(error.message);
    }
  }

  applyAnalysis(payload, fresh) {
    const payloadOffset = finiteNumber(payload.beat_offset_ms, 0) / 1000;
    const payloadBeatTimes = (payload.beat_times || []).map((value) => finiteNumber(value));
    const payloadDetectedBeatTimes = (payload.detected_beat_times || []).map(
      (value) => finiteNumber(value),
    );
    this.beatData = {
      bpm: finiteNumber(payload.bpm),
      gridBpm: finiteNumber(payload.grid_bpm, payload.bpm),
      baseGridIntervalSeconds: finiteNumber(payload.base_grid_interval_seconds),
      gridIntervalSeconds: finiteNumber(payload.grid_interval_seconds),
      beatGridDensity: payload.beat_grid_density || this.beatGridDensity(),
      baseBeatTimes: (payload.base_beat_times || payloadBeatTimes.map(
        (value) => value - payloadOffset,
      )).map((value) => finiteNumber(value)),
      baseDetectedBeatTimes: (
        payload.base_detected_beat_times ||
        payloadDetectedBeatTimes
      ).map((value) => finiteNumber(value)),
      beatTimes: [],
      detectedBeatTimes: [],
      onsetTimes: (payload.onset_times || []).map((value) => finiteNumber(value)),
      drumTimes: payload.drum_times || {},
      audioDuration: finiteNumber(payload.audio_duration),
      sourceDuration: finiteNumber(payload.source_duration, this.sourceAudioDuration),
      sourceStart: finiteNumber(payload.source_start, this.cropStartSeconds()),
      fps: finiteNumber(payload.fps, this.fps()),
      waveformPreview: normalizeWaveformPreview(payload.waveform_preview) ||
        cropWaveformPreview(this.sourceWaveformPreview, this.cropStartSeconds(), this.cropDurationSeconds()),
      cacheKey: payload.cache_key || "",
    };
    this.dataFresh = fresh;
    this.applyBeatOffset();
  }

  updatePlayButton() {
    const button = this.root.querySelector('[data-action="play"]');
    button.textContent = this.audioElement && !this.audioElement.paused ? "Pause" : "Play";
  }

  updateTransportTime() {
    const current = this.playheadFrame == null ? 0 : this.playheadFrame / this.fps();
    this.transportTimeEl.textContent =
      `${formatClock(current)} / ${formatClock(this.cropDurationSeconds())}`;
  }

  updatePlaybackPosition() {
    if (!this.audioElement) return;
    const relative = this.audioElement.currentTime - this.cropStartSeconds();
    const duration = this.cropDurationSeconds();
    if (relative >= duration - 0.005) {
      this.loopPlayback();
      return;
    }
    this.playheadFrame = clamp(relative * this.fps(), 0, this.sequenceFrameCount());
    this.updateTransportTime();
    this.scheduleDraw();
  }

  async loopPlayback(resume = false) {
    if (!this.audioElement) return;
    this.audioElement.currentTime = this.cropStartSeconds();
    this.playheadFrame = 0;
    this.updateTransportTime();
    this.scheduleDraw();
    if (!resume || !this.audioElement.paused) return;
    try {
      await this.audioElement.play();
    } catch (error) {
      this.showError(`Audio playback failed: ${error.message}`);
    }
  }

  startPlaybackLoop() {
    this.stopPlaybackLoop();
    const tick = () => {
      this.playbackFrameRequest = null;
      if (!this.audioElement || this.audioElement.paused) return;
      this.updatePlaybackPosition();
      this.playbackFrameRequest = requestAnimationFrame(tick);
    };
    this.playbackFrameRequest = requestAnimationFrame(tick);
  }

  stopPlaybackLoop() {
    if (this.playbackFrameRequest != null) {
      cancelAnimationFrame(this.playbackFrameRequest);
      this.playbackFrameRequest = null;
    }
  }

  async togglePlayback() {
    if (!this.audioElement) {
      this.showError("Choose an audio file before playing.");
      return;
    }
    if (!this.audioElement.paused) {
      this.audioElement.pause();
      return;
    }
    const frame = clamp(this.playheadFrame || 0, 0, this.sequenceFrameCount() - 1);
    this.audioElement.currentTime = this.cropStartSeconds() + frame / this.fps();
    try {
      await this.audioElement.play();
    } catch (error) {
      this.showError(`Audio playback failed: ${error.message}`);
    }
  }

  stopPlayback() {
    this.stopPlaybackLoop();
    if (this.audioElement) {
      this.audioElement.pause();
      this.audioElement.currentTime = this.cropStartSeconds();
    }
    this.playheadFrame = 0;
    this.updatePlayButton();
    if (this.transportTimeEl) this.updateTransportTime();
    this.scheduleDraw();
  }

  async startSeparation() {
    if (this.separationJobId) {
      const response = await api.fetchApi(
        `/fl/audio-prompt-timeline/separate/${encodeURIComponent(this.separationJobId)}/cancel`,
        { method: "POST" },
      );
      const payload = await response.json();
      if (!response.ok) this.showError(payload.error || "Could not cancel stem separation.");
      else this.setStatus(payload.message || "Cancelling stem separation…");
      return;
    }
    const audioFile = String(this.widgets.audioFile?.value || "");
    if (!audioFile) {
      this.showError("Choose an audio file before separating stems.");
      return;
    }
    this.setStatus("Starting explicit stem separation…");
    try {
      const response = await api.fetchApi("/fl/audio-prompt-timeline/separate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio_file: audioFile }),
      });
      const payload = await response.json();
      if (payload.status === "completed") {
        this.finishSeparation(payload);
        return;
      }
      const job = payload.job || payload;
      if (!response.ok && !job.job_id) {
        throw new Error(payload.error || `Stem separation failed (${response.status}).`);
      }
      this.separationJobId = job.job_id;
      this.node._flAudioSeparationJobId = this.separationJobId;
      this.root.querySelector('[data-action="separate"]').textContent = "Cancel separation";
      this.setStatus(job.message || "Stem separation running…");
      this.pollSeparation();
    } catch (error) {
      this.showError(error.message);
    }
  }

  async pollSeparation() {
    if (!this.separationJobId) return;
    try {
      const response = await api.fetchApi(
        `/fl/audio-prompt-timeline/separate/${encodeURIComponent(this.separationJobId)}`,
      );
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not read stem separation status.");
      const percent = Math.round(finiteNumber(payload.progress) * 100);
      this.setStatus(`${payload.message || payload.status} · ${percent}%`);
      if (payload.status === "completed") {
        this.finishSeparation(payload);
        return;
      }
      if (payload.status === "error" || payload.status === "cancelled") {
        this.separationJobId = null;
        this.node._flAudioSeparationJobId = null;
        this.root.querySelector('[data-action="separate"]').textContent = "Separate stems";
        if (payload.status === "error") this.showError(payload.message || "Stem separation failed.");
        else this.setStatus("Stem separation cancelled", "cached");
        return;
      }
      this.separationTimer = setTimeout(() => this.pollSeparation(), 750);
    } catch (error) {
      this.separationJobId = null;
      this.node._flAudioSeparationJobId = null;
      this.root.querySelector('[data-action="separate"]').textContent = "Separate stems";
      this.showError(error.message);
    }
  }

  finishSeparation(payload) {
    clearTimeout(this.separationTimer);
    this.separationJobId = null;
    this.node._flAudioSeparationJobId = null;
    this.root.querySelector('[data-action="separate"]').textContent = "Separate stems";
    this.setStatus(payload.message || "Stem separation complete", "fresh");
    if (this.widgets.analysisSource) this.widgets.analysisSource.value = "drums";
    this.requestAnalysis(true);
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
      const crossfadeStartSeconds = clip.crossfade > 0
        ? this.sourceToSeconds(clip.start - clip.crossfade * 0.5, unit)
        : startSeconds;
      const crossfadeEndSeconds = clip.crossfade > 0
        ? this.sourceToSeconds(clip.start + clip.crossfade * 0.5, unit)
        : startSeconds;
      if ([
        startSeconds,
        endSeconds,
        fadeInSeconds,
        fadeOutSeconds,
        crossfadeStartSeconds,
        crossfadeEndSeconds,
      ].some((value) => value == null)) {
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
        crossfade: Math.max(0, Math.round((crossfadeEndSeconds - crossfadeStartSeconds) * fps)),
      };
    });
  }

  finishMigration(clips) {
    if (clips.some((clip) => !clip)) {
      throw new Error("The legacy schedule could not be converted to frames.");
    }
    validateFrameClips(normalizeCrossfades(clips));
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
      crossfade: Math.round(finiteNumber(section.crossfade_frames)),
      prompt: String(section.prompt || ""),
    }));
  }

  updateFromExecution(message) {
    const payload = executionPayload(message);
    if (!payload || !Array.isArray(payload.beat_times)) return;
    this.applyAnalysis(payload, true);

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
    this.resnapClipsToGrid();
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
      this.statusEl.textContent = "Choose audio or connect beat positions";
      return;
    }
    const count = this.beatData.beatTimes?.length || 0;
    const detected = this.beatData.detectedBeatTimes?.length || 0;
    const onsets = this.beatData.onsetTimes?.length || 0;
    const offset = this.beatOffsetMs();
    const offsetText = offset ? ` · offset ${offset > 0 ? "+" : ""}${offset} ms` : "";
    const density = GRID_DENSITY_LABELS[this.beatGridDensity()];
    const text = `${finiteNumber(this.beatData.gridBpm, this.beatData.bpm).toFixed(2)} grid BPM · ${density} · ${count} grid · ` +
      `${detected} detected · ${onsets} onsets · ${finiteNumber(this.beatData.audioDuration).toFixed(2)} sec` +
      offsetText;
    if (this.dataFresh) {
      this.statusEl.classList.add("fresh");
      this.statusEl.textContent = text;
    } else {
      this.statusEl.classList.add("cached");
      this.statusEl.textContent = `${text} · cached`;
    }
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
    this.fields.crossfade.value = String(clip.crossfade);
    this.fields.prompt.value = clip.prompt;
    const previous = this.clips[this.selectedIndex - 1];
    this.fields.crossfade.disabled = !previous || previous.end !== clip.start;
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
    let fadeIn = Math.max(0, Math.round(finiteNumber(this.fields.fadeIn.value, clip.fadeIn)));
    let fadeOut = Math.max(0, Math.round(finiteNumber(this.fields.fadeOut.value, clip.fadeOut)));
    const crossfade = Math.max(
      0,
      Math.round(finiteNumber(this.fields.crossfade.value, clip.crossfade)),
    );
    if (!(end > start)) {
      this.showError("The selected prompt must end after it starts.");
      this.syncInspector();
      return;
    }
    if (crossfade > 0) fadeIn = 0;
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
    if (crossfade > 0) {
      if (!previous || previous.end !== start) {
        this.showError("Crossfade requires this prompt to touch the previous prompt.");
        this.syncInspector();
        return;
      }
      if (crossfade > Math.min(previous.end - previous.start, end - start)) {
        this.showError("Crossfade exceeds the shorter adjacent prompt.");
        this.syncInspector();
        return;
      }
      previous.fadeOut = 0;
      fadeIn = 0;
    }
    Object.assign(clip, { start, end, fadeIn, fadeOut, crossfade });
    normalizeCrossfades(this.clips);
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
  }

  beatFrames() {
    return (this.beatData?.beatTimes || []).map((seconds) => Math.round(seconds * this.fps()));
  }

  detectedBeatFrames() {
    return (this.beatData?.detectedBeatTimes || []).map((seconds) => Math.round(seconds * this.fps()));
  }

  onsetFrames() {
    return (this.beatData?.onsetTimes || []).map((seconds) => Math.round(seconds * this.fps()));
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

  nearestBeatIndex(value, markers = this.beatFrames()) {
    if (!markers.length) return -1;
    let nearest = 0;
    for (let index = 1; index < markers.length; index++) {
      if (Math.abs(markers[index] - value) < Math.abs(markers[nearest] - value)) nearest = index;
    }
    return nearest;
  }

  editingSnapFrames(minimum = 0, maximum = this.sequenceFrameCount()) {
    const markers = this.beatFrames()
      .filter((marker) => marker >= minimum && marker <= maximum);
    if (!markers.length) return [];
    markers.push(Math.round(minimum));
    if (Number.isFinite(maximum)) markers.push(Math.round(maximum));
    return [...new Set(markers)].sort((left, right) => left - right);
  }

  snapFrame(value, minimum = 0, maximum = Infinity) {
    const frame = clamp(Math.round(value), minimum, maximum);
    const markers = this.editingSnapFrames(minimum, maximum);
    if (!markers.length) return frame;
    let nearest = markers[0];
    for (let index = 1; index < markers.length; index++) {
      if (Math.abs(markers[index] - frame) < Math.abs(nearest - frame)) nearest = markers[index];
    }
    return nearest;
  }

  orderedSnapTargets(values, markers) {
    if (!values.length) return [];
    if (markers.length < values.length) return null;

    const previousMarkers = Array.from(
      { length: values.length },
      () => Array(markers.length).fill(-1),
    );
    let costs = markers.map((marker, index) => (
      index <= markers.length - values.length
        ? Math.abs(marker - values[0])
        : Infinity
    ));

    for (let valueIndex = 1; valueIndex < values.length; valueIndex++) {
      const nextCosts = Array(markers.length).fill(Infinity);
      const maximumMarker = markers.length - values.length + valueIndex;
      let bestCost = Infinity;
      let bestMarker = -1;
      for (let markerIndex = 0; markerIndex < markers.length; markerIndex++) {
        const previousIndex = markerIndex - 1;
        if (previousIndex >= 0 && costs[previousIndex] < bestCost) {
          bestCost = costs[previousIndex];
          bestMarker = previousIndex;
        }
        if (markerIndex < valueIndex || markerIndex > maximumMarker || bestMarker < 0) continue;
        nextCosts[markerIndex] = bestCost + Math.abs(markers[markerIndex] - values[valueIndex]);
        previousMarkers[valueIndex][markerIndex] = bestMarker;
      }
      costs = nextCosts;
    }

    let markerIndex = -1;
    let bestCost = Infinity;
    for (let index = values.length - 1; index < markers.length; index++) {
      if (costs[index] < bestCost) {
        bestCost = costs[index];
        markerIndex = index;
      }
    }
    if (markerIndex < 0) return null;

    const targets = Array(values.length);
    for (let valueIndex = values.length - 1; valueIndex >= 0; valueIndex--) {
      targets[valueIndex] = markers[markerIndex];
      markerIndex = previousMarkers[valueIndex][markerIndex];
    }
    return targets;
  }

  resnapClipsToGrid(markers = null) {
    if (this.drag) {
      this.resnapPending = true;
      return false;
    }
    this.resnapPending = false;
    if (this.rawInvalid || this.migrationPending || !this.clips.length) return false;

    const snapMarkers = markers || this.editingSnapFrames(0, this.sequenceFrameCount());
    if (!snapMarkers.length) return false;

    const values = [];
    const startBoundaries = [];
    const endBoundaries = [];
    for (let index = 0; index < this.clips.length; index++) {
      const clip = this.clips[index];
      const previous = this.clips[index - 1];
      if (previous && previous.end === clip.start) {
        startBoundaries.push(endBoundaries[index - 1]);
      } else {
        startBoundaries.push(values.length);
        values.push(clip.start);
      }
      endBoundaries.push(values.length);
      values.push(clip.end);
    }

    const targets = this.orderedSnapTargets(values, snapMarkers);
    if (!targets) return false;
    const before = this.clips.map((clip) => (
      [clip.start, clip.end, clip.fadeIn, clip.fadeOut, clip.crossfade]
    ));

    for (let index = 0; index < this.clips.length; index++) {
      const clip = this.clips[index];
      clip.start = targets[startBoundaries[index]];
      clip.end = targets[endBoundaries[index]];
      const duration = clip.end - clip.start;
      clip.fadeIn = clamp(Math.round(finiteNumber(clip.fadeIn)), 0, duration);
      clip.fadeOut = clamp(
        Math.round(finiteNumber(clip.fadeOut)),
        0,
        Math.max(0, duration - clip.fadeIn),
      );
    }
    normalizeCrossfades(this.clips);

    const changed = this.clips.some((clip, index) => (
      clip.start !== before[index][0] ||
      clip.end !== before[index][1] ||
      clip.fadeIn !== before[index][2] ||
      clip.fadeOut !== before[index][3] ||
      clip.crossfade !== before[index][4]
    ));
    if (!changed) return false;

    this.serialize();
    this.syncInspector();
    this.scheduleDraw();
    return true;
  }

  defaultClipLength() {
    return Math.max(1, Math.round(this.fps() * 2));
  }

  crossfadeBounds(index) {
    const clip = this.clips[index];
    if (!clip) return null;
    const frames = Math.max(0, Math.round(finiteNumber(clip.crossfade)));
    const before = Math.floor(frames / 2);
    return {
      start: clip.start - before,
      end: clip.start + frames - before,
      boundary: clip.start,
      frames,
    };
  }

  maximumCrossfade(index) {
    const clip = this.clips[index];
    const previous = this.clips[index - 1];
    if (!clip || !previous || previous.end !== clip.start) return 0;
    return Math.max(
      0,
      Math.min(previous.end - previous.start, clip.end - clip.start),
    );
  }

  defaultCrossfade(index) {
    const maximum = this.maximumCrossfade(index);
    if (!maximum) return 0;
    const beatFrames = Math.round(this.baseGridIntervalSeconds() * this.fps());
    return Math.min(maximum, Math.max(1, beatFrames || Math.round(this.fps() * 0.5)));
  }

  toggleCrossfade(index) {
    const clip = this.clips[index];
    const previous = this.clips[index - 1];
    if (!clip || !previous || previous.end !== clip.start) return;
    clip.crossfade = clip.crossfade > 0 ? 0 : this.defaultCrossfade(index);
    if (clip.crossfade > 0) {
      previous.fadeOut = 0;
      clip.fadeIn = 0;
    }
    this.select(index);
    this.clearError();
    this.serialize();
    this.syncInspector();
    this.scheduleDraw();
  }

  gridClipRangeAt(frame) {
    const markers = this.editingSnapFrames(0, this.sequenceFrameCount());
    const intervalFrames = this.gridIntervalSeconds() * this.fps();
    if (!markers.length || !(intervalFrames > EPSILON)) return undefined;
    const maximum = this.maximumFrame();
    const candidates = [];
    for (let index = 0; index < markers.length; index++) {
      const start = markers[index];
      const end = index + DEFAULT_CLIP_GRID_INTERVALS < markers.length
        ? markers[index + DEFAULT_CLIP_GRID_INTERVALS]
        : Math.round(start + intervalFrames * DEFAULT_CLIP_GRID_INTERVALS);
      if (end > start && (!Number.isFinite(maximum) || end <= maximum)) {
        candidates.push({ start, end });
      }
    }
    if (!candidates.length) return null;
    let nearest = candidates[0];
    for (let index = 1; index < candidates.length; index++) {
      if (Math.abs(candidates[index].start - frame) < Math.abs(nearest.start - frame)) {
        nearest = candidates[index];
      }
    }
    return nearest;
  }

  addClip(startOverride = null, endOverride = null) {
    if (this.migrationPending) return;
    const previousEnd = this.clips.length ? this.clips[this.clips.length - 1].end : 0;
    const exactRange = endOverride != null;
    const start = startOverride == null
      ? previousEnd
      : exactRange
      ? Math.max(0, Math.round(startOverride))
      : this.snapFrame(startOverride);
    let end = exactRange ? Math.round(endOverride) : start + this.defaultClipLength();
    if (!(end > start)) {
      this.showError("The new prompt must end after it starts.");
      return;
    }
    const maximum = this.maximumFrame();
    if (Number.isFinite(maximum)) {
      if (start >= maximum) {
        this.showError("There is no room for another prompt inside the configured frame length.");
        return;
      }
      if (exactRange && end > maximum) {
        this.showError("There is not enough room for a four-grid prompt inside the configured frame length.");
        return;
      }
      if (!exactRange) end = Math.min(end, maximum);
    }
    const duration = end - start;
    const fadeIn = Math.min(Math.round(this.defaultFadeIn()), duration);
    const fadeOut = Math.min(Math.round(this.defaultFadeOut()), duration - fadeIn);
    const clip = {
      start,
      end,
      fadeIn,
      fadeOut,
      crossfade: 0,
      prompt: "Describe this prompt section.",
    };
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
    const { y } = this.eventPosition(event);
    const layout = this.timelineLayout();
    if (y < layout.trackTop || y > layout.trackBottom) return;
    const frame = this.frameAtEvent(event);
    const range = this.gridClipRangeAt(frame);
    if (range === null) {
      this.showError("There is not enough room for a four-grid prompt inside the configured frame length.");
      return;
    }
    if (range) this.addClip(range.start, range.end);
    else this.addClip(frame);
  }

  onDoubleClick(event) {
    const { x, y } = this.eventPosition(event);
    const transition = this.crossfadeRects.find((rect) => (
      y >= rect.y &&
      y <= rect.y + rect.height &&
      Math.abs(x - rect.boundaryX) <= 10
    ));
    if (transition) {
      this.toggleCrossfade(transition.index);
      event.preventDefault();
      return;
    }
    const hit = this.hitTest(x, y);
    if (hit?.type.startsWith("crossfade")) {
      this.toggleCrossfade(hit.index);
      event.preventDefault();
      return;
    }
    this.addClipAtPointer(event);
  }

  closeContextMenu() {
    this.contextMenu?.remove();
    this.contextMenu = null;
  }

  cropPromptRange(start, end) {
    const cropped = [];
    let selectedIndex = -1;
    for (let index = 0; index < this.clips.length; index++) {
      const clip = this.clips[index];
      const visibleStart = Math.max(start, clip.start);
      const visibleEnd = Math.min(end, clip.end);
      if (visibleEnd <= visibleStart) continue;
      const fadeInEnd = clamp(clip.start + clip.fadeIn, visibleStart, visibleEnd);
      const fadeOutStart = clamp(clip.end - clip.fadeOut, visibleStart, visibleEnd);
      if (index === this.selectedIndex) selectedIndex = cropped.length;
      cropped.push({
        ...clip,
        start: visibleStart - start,
        end: visibleEnd - start,
        fadeIn: fadeInEnd - visibleStart,
        fadeOut: visibleEnd - fadeOutStart,
      });
    }
    this.clips = normalizeCrossfades(cropped);
    this.selectedIndex = selectedIndex;
  }

  applyAudioCrop(start, end) {
    const duration = this.sequenceFrameCount();
    const cropStart = clamp(Math.round(start), 0, duration - 1);
    const cropEnd = clamp(Math.round(end), cropStart + 1, duration);
    if (cropStart === 0 && cropEnd === duration) return;
    const croppedMarkers = this.editingSnapFrames(cropStart, cropEnd)
      .map((marker) => marker - cropStart);

    this.stopPlayback();
    this.cropPromptRange(cropStart, cropEnd);
    this.widgets.trimStartFrame.value = this.trimStartFrame() + cropStart;
    this.widgets.sequenceDuration.value = cropEnd - cropStart;
    this.resnapClipsToGrid(croppedMarkers);
    this.playheadFrame = 0;
    this.refreshBrowserCrop();
    this.invalidateAnalysis();
    this.serialize();
    this.syncInspector();
    this.zoomToFit(false);
    this.markDirty();
    this.scheduleAnalysis(0);
  }

  setAudioIn(frame) {
    const duration = this.sequenceFrameCount();
    if (frame <= 0 || frame >= duration) return;
    this.applyAudioCrop(frame, duration);
  }

  setAudioOut(frame) {
    const duration = this.sequenceFrameCount();
    if (frame <= 0 || frame >= duration) return;
    this.applyAudioCrop(0, frame);
  }

  onContextMenu(event) {
    event.preventDefault();
    event.stopPropagation();
    if (this.migrationPending || this.rawInvalid || !this.widgets.audioFile?.value) return;
    const { x, y } = this.eventPosition(event);
    const layout = this.timelineLayout();
    if (y < layout.rulerTop || y > layout.trackBottom ||
        x < TIMELINE_LEFT || x > this.canvas.clientWidth - TIMELINE_RIGHT) {
      return;
    }

    const duration = this.sequenceFrameCount();
    const frame = this.snapFrame(this.frameAtX(x), 0, duration);
    this.playheadFrame = frame;
    if (this.audioElement) {
      this.audioElement.currentTime = this.cropStartSeconds() + frame / this.fps();
    }
    this.updateTransportTime();
    this.scheduleDraw();
    this.closeContextMenu();

    const menu = document.createElement("div");
    menu.className = "flbps-context-menu";
    menu.addEventListener("contextmenu", (menuEvent) => menuEvent.preventDefault());
    const sourceFrame = this.trimStartFrame() + frame;
    menu.innerHTML = `
      <div class="flbps-context-title">Beat position · F${frame} · ${formatClock(frame / this.fps())} · source F${sourceFrame}</div>
      <button data-action="set-in"${frame <= 0 || frame >= duration ? " disabled" : ""}>Set audio In here</button>
      <button data-action="set-out"${frame <= 0 || frame >= duration ? " disabled" : ""}>Set audio Out here</button>
    `;
    menu.querySelector('[data-action="set-in"]').addEventListener("click", () => {
      this.closeContextMenu();
      this.setAudioIn(frame);
    });
    menu.querySelector('[data-action="set-out"]').addEventListener("click", () => {
      this.closeContextMenu();
      this.setAudioOut(frame);
    });
    document.body.appendChild(menu);
    this.contextMenu = menu;
    const rect = menu.getBoundingClientRect();
    menu.style.left = `${clamp(event.clientX, 6, window.innerWidth - rect.width - 6)}px`;
    menu.style.top = `${clamp(event.clientY, 6, window.innerHeight - rect.height - 6)}px`;
  }

  deleteClip() {
    if (!this.selectedClip()) return;
    const next = this.clips[this.selectedIndex + 1];
    if (next) next.crossfade = 0;
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
    this.clips.splice(this.selectedIndex + 1, 0, {
      ...clip,
      start,
      end,
      crossfade: 0,
    });
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
      crossfade: 0,
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

  timelineLayout(height = this.canvas.clientHeight) {
    const sourceVisible = this.waveformVisible && Boolean(this.sourceWaveformPreview);
    const sourceTop = sourceVisible ? 4 : null;
    const sourceBottom = sourceVisible ? 38 : null;
    const rulerTop = sourceVisible ? 44 : 4;
    const rulerBottom = rulerTop + 30;
    const waveformTop = this.waveformVisible ? rulerBottom + 2 : null;
    const waveformBottom = this.waveformVisible ? waveformTop + 92 : null;
    const trackTop = this.waveformVisible ? waveformBottom + 7 : rulerBottom + 7;
    return {
      sourceTop,
      sourceBottom,
      rulerTop,
      rulerBottom,
      waveformTop,
      waveformBottom,
      trackTop,
      trackBottom: height - 8,
    };
  }

  sourceFrameAtX(x) {
    const right = Math.max(TIMELINE_LEFT + 1, this.canvas.clientWidth - TIMELINE_RIGHT);
    const ratio = (clamp(x, TIMELINE_LEFT, right) - TIMELINE_LEFT) / (right - TIMELINE_LEFT);
    return Math.round(ratio * this.sourceAudioDuration * this.fps());
  }

  trimHandlePositions(width = this.canvas.clientWidth) {
    const right = Math.max(TIMELINE_LEFT + 1, width - TIMELINE_RIGHT);
    const sourceFrames = Math.max(1, Math.round(this.sourceAudioDuration * this.fps()));
    const start = this.trimStartFrame();
    const end = Math.min(sourceFrames, start + this.configuredFrameCount());
    return {
      start,
      end,
      startX: TIMELINE_LEFT + start / sourceFrames * (right - TIMELINE_LEFT),
      endX: TIMELINE_LEFT + end / sourceFrames * (right - TIMELINE_LEFT),
      sourceFrames,
    };
  }

  hitTestTrim(x, y) {
    const layout = this.timelineLayout();
    if (!this.waveformVisible ||
        !this.sourceWaveformPreview ||
        y < layout.sourceTop ||
        y > layout.sourceBottom) {
      return null;
    }
    const handles = this.trimHandlePositions();
    if (Math.abs(x - handles.startX) <= 10) return { type: "trim-start", handles };
    if (Math.abs(x - handles.endX) <= 10) return { type: "trim-end", handles };
    if (x > handles.startX && x < handles.endX) return { type: "trim-move", handles };
    return null;
  }

  updateTrimDrag(x) {
    const frame = this.sourceFrameAtX(x);
    const original = this.drag.original;
    if (this.drag.type === "trim-start") {
      const start = clamp(frame, 0, original.end - 1);
      this.widgets.trimStartFrame.value = start;
      this.widgets.sequenceDuration.value = original.end - start;
    } else if (this.drag.type === "trim-end") {
      const end = clamp(frame, original.start + 1, original.sourceFrames);
      this.widgets.sequenceDuration.value = end - original.start;
    } else {
      const duration = original.end - original.start;
      const delta = frame - this.drag.pointerStartFrame;
      this.widgets.trimStartFrame.value = clamp(
        original.start + delta,
        0,
        original.sourceFrames - duration,
      );
    }
    this.refreshBrowserCrop();
    this.invalidateAnalysis();
    this.markDirty();
  }

  onPointerDown(event) {
    this.root.focus({ preventScroll: true });
    const { x, y } = this.eventPosition(event);
    if (event.button === 1) {
      const duration = this.sequenceFrameCount();
      const range = this.viewEnd - this.viewStart;
      event.preventDefault();
      if (range >= duration - EPSILON) return;
      this.drag = {
        type: "timeline-pan",
        pointerStartX: x,
        pointerStartY: y,
        originalViewStart: this.viewStart,
        originalViewEnd: this.viewEnd,
        active: false,
      };
      this.canvas.setPointerCapture(event.pointerId);
      this.canvas.style.cursor = "grabbing";
      return;
    }
    if (event.button !== 0) return;
    const trimHit = this.hitTestTrim(x, y);
    if (trimHit) {
      this.drag = {
        type: trimHit.type,
        pointerStartX: x,
        pointerStartY: y,
        original: {
          start: trimHit.handles.start,
          end: trimHit.handles.end,
          sourceFrames: trimHit.handles.sourceFrames,
        },
        beatFrames: this.beatFrames(),
        pointerStartFrame: this.sourceFrameAtX(x),
        active: false,
      };
      this.canvas.setPointerCapture(event.pointerId);
      event.preventDefault();
      return;
    }
    if (this.migrationPending || this.rawInvalid) return;
    const hit = this.hitTest(x, y);
    if (!hit) {
      this.playheadFrame = this.snapFrame(this.frameAtX(x), 0, this.sequenceFrameCount());
      if (this.audioElement) {
        this.audioElement.currentTime = this.cropStartSeconds() + this.playheadFrame / this.fps();
      }
      this.updateTransportTime();
      this.scheduleDraw();
      return;
    }

    this.select(hit.index);
    const clip = this.selectedClip();
    const markers = this.editingSnapFrames(0, this.sequenceFrameCount());
    const startIndex = this.nearestBeatIndex(clip.start, markers);
    const endIndex = this.nearestBeatIndex(clip.end, markers);
    this.drag = {
      type: hit.type,
      pointerStartRaw: Math.max(0, Math.round(this.frameAtX(x))),
      pointerStartX: x,
      pointerStartY: y,
      pointerX: x,
      pointerY: y,
      gridSpan: startIndex >= 0 && endIndex >= 0 ? Math.max(1, endIndex - startIndex) : null,
      original: { ...clip },
      originalPrevious: hit.type === "shared-boundary"
        ? { ...this.clips[hit.index - 1] }
        : null,
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

  updateTimelinePan(x) {
    const right = Math.max(TIMELINE_LEFT + 1, this.canvas.clientWidth - TIMELINE_RIGHT);
    const pixels = right - TIMELINE_LEFT;
    const range = this.drag.originalViewEnd - this.drag.originalViewStart;
    const frameDelta = (x - this.drag.pointerStartX) / pixels * range;
    const duration = this.sequenceFrameCount();
    this.viewStart = clamp(
      this.drag.originalViewStart - frameDelta,
      0,
      Math.max(0, duration - range),
    );
    this.viewEnd = Math.min(duration, this.viewStart + range);
    this.viewStart = Math.max(0, this.viewEnd - range);
    this.scheduleDraw();
  }

  updateDrag(x) {
    const clip = this.selectedClip();
    if (!this.drag || !clip) return;
    this.panDuringDrag(x);
    const currentRaw = Math.max(0, Math.round(this.frameAtX(x)));
    const delta = currentRaw - this.drag.pointerStartRaw;
    const original = this.drag.original;
    const previous = this.clips[this.selectedIndex - 1];
    const next = this.clips[this.selectedIndex + 1];
    const maximum = this.maximumFrame();
    let guideFrame = null;

    if (this.drag.type === "shared-boundary") {
      const originalPrevious = this.drag.originalPrevious;
      const boundary = this.snapFrame(
        original.start + delta,
        originalPrevious.start + 1,
        original.end - 1,
      );
      previous.end = boundary;
      clip.start = boundary;

      const previousDuration = previous.end - previous.start;
      previous.fadeIn = Math.min(originalPrevious.fadeIn, previousDuration);
      previous.fadeOut = Math.min(
        originalPrevious.fadeOut,
        Math.max(0, previousDuration - previous.fadeIn),
      );
      const clipDuration = clip.end - clip.start;
      clip.fadeOut = Math.min(original.fadeOut, clipDuration);
      clip.fadeIn = Math.min(
        original.fadeIn,
        Math.max(0, clipDuration - clip.fadeOut),
      );
      clip.crossfade = Math.min(
        original.crossfade,
        previousDuration,
        clipDuration,
      );
      guideFrame = boundary;
    } else if (this.drag.type === "move") {
      const markers = this.editingSnapFrames(0, this.sequenceFrameCount());
      const span = this.drag.gridSpan;
      if (span && markers.length > span) {
        const proposedStart = original.start + delta;
        let nearest = null;
        for (let index = 0; index + span < markers.length; index++) {
          const start = markers[index];
          const end = markers[index + span];
          if ((previous && start < previous.end) ||
              (next && end > next.start) ||
              (Number.isFinite(maximum) && end > maximum)) {
            continue;
          }
          const distance = Math.abs(start - proposedStart);
          if (!nearest || distance < nearest.distance) nearest = { start, end, distance };
        }
        if (nearest) {
          clip.start = nearest.start;
          clip.end = nearest.end;
          guideFrame = nearest.start;
        }
      } else {
        const duration = original.end - original.start;
        let start = this.snapFrame(original.start + delta);
        if (previous) start = Math.max(start, previous.end);
        if (next) start = Math.min(start, next.start - duration);
        if (Number.isFinite(maximum)) start = Math.min(start, maximum - duration);
        clip.start = Math.round(start);
        clip.end = clip.start + duration;
        guideFrame = clip.start;
      }
    } else if (this.drag.type === "start") {
      const start = this.snapFrame(
        original.start + delta,
        previous?.end || 0,
        original.end - 1,
      );
      clip.start = Math.round(start);
      clip.fadeIn = Math.min(original.fadeIn, clip.end - clip.start - clip.fadeOut);
      guideFrame = clip.start;
    } else if (this.drag.type === "end") {
      const end = this.snapFrame(
        original.end + delta,
        original.start + 1,
        Math.min(next?.start ?? Infinity, maximum),
      );
      clip.end = Math.round(end);
      clip.fadeOut = Math.min(original.fadeOut, clip.end - clip.start - clip.fadeIn);
      guideFrame = clip.end;
    } else if (this.drag.type === "fade-in") {
      const current = this.snapFrame(currentRaw, clip.start, clip.end - clip.fadeOut);
      clip.fadeIn = clamp(current - clip.start, 0, clip.end - clip.start - clip.fadeOut);
      guideFrame = current;
    } else if (this.drag.type === "fade-out") {
      const current = this.snapFrame(currentRaw, clip.start + clip.fadeIn, clip.end);
      clip.fadeOut = clamp(clip.end - current, 0, clip.end - clip.start - clip.fadeIn);
      guideFrame = current;
    } else if (this.drag.type === "crossfade-start" ||
        this.drag.type === "crossfade-end" ||
        this.drag.type === "crossfade-create") {
      const boundary = original.start;
      const edge = this.snapFrame(
        currentRaw,
        previous?.start || 0,
        original.end,
      );
      const maximumCrossfade = Math.min(
        previous?.end - previous?.start || 0,
        original.end - original.start,
      );
      clip.crossfade = clamp(2 * Math.abs(edge - boundary), 0, maximumCrossfade);
      if (clip.crossfade > 0) {
        previous.fadeOut = 0;
        clip.fadeIn = 0;
      }
      const bounds = this.crossfadeBounds(this.selectedIndex);
      guideFrame = this.drag.type === "crossfade-start" ? bounds.start : bounds.end;
    }

    clip.fadeIn = Math.max(0, Math.round(clip.fadeIn));
    clip.fadeOut = Math.max(0, Math.round(clip.fadeOut));
    normalizeCrossfades(this.clips);
    this.snapGuideFrame = guideFrame;
    this.syncInspector();
    this.scheduleDraw();
  }

  onPointerMove(event) {
    const { x, y } = this.eventPosition(event);
    if (this.drag?.type === "timeline-pan") {
      this.hover = null;
      if (!this.drag.active) {
        const distance = Math.hypot(x - this.drag.pointerStartX, y - this.drag.pointerStartY);
        if (distance < 2) return;
        this.drag.active = true;
      }
      this.canvas.style.cursor = "grabbing";
      this.updateTimelinePan(x);
      event.preventDefault();
      return;
    }
    this.hover = { x, y };
    if (this.drag?.type === "trim-start" ||
        this.drag?.type === "trim-end" ||
        this.drag?.type === "trim-move") {
      if (!this.drag.active) {
        const distance = Math.hypot(x - this.drag.pointerStartX, y - this.drag.pointerStartY);
        if (distance < 3) return;
        this.drag.active = true;
      }
      this.canvas.style.cursor = this.drag.type === "trim-move" ? "grabbing" : "ew-resize";
      this.updateTrimDrag(x);
      event.preventDefault();
      return;
    }
    if (!this.drag || !this.selectedClip()) {
      const trimHit = this.hitTestTrim(x, y);
      const hit = this.hitTest(x, y);
      this.canvas.style.cursor = trimHit
        ? trimHit.type === "trim-move" ? "grab" : "ew-resize"
        : hit
        ? hit.type === "move"
          ? "grab"
          : hit.type === "shared-boundary"
          ? "col-resize"
          : "ew-resize"
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
    this.canvas.style.cursor = this.drag.type === "move"
      ? "grabbing"
      : this.drag.type === "shared-boundary"
      ? "col-resize"
      : "ew-resize";
    this.updateDrag(x);
    event.preventDefault();
  }

  onPointerUp(event) {
    if (!this.drag) return;
    const finishedDrag = this.drag;
    const trimChanged = this.drag.type === "trim-start" ||
      this.drag.type === "trim-end" ||
      this.drag.type === "trim-move";
    const viewChanged = this.drag.type === "timeline-pan";
    const changed = this.drag.active;
    this.drag = null;
    this.snapGuideFrame = null;
    this.canvas.style.cursor = "default";
    if (this.canvas.hasPointerCapture(event.pointerId)) this.canvas.releasePointerCapture(event.pointerId);
    if (changed) {
      if (trimChanged) {
        const start = this.trimStartFrame();
        const duration = this.sequenceFrameCount();
        const markers = (finishedDrag.beatFrames || [])
          .map((marker) => finishedDrag.original.start + marker - start)
          .filter((marker) => marker >= 0 && marker <= duration);
        if (markers.length) {
          markers.push(0, duration);
          this.resnapClipsToGrid([...new Set(markers)].sort((left, right) => left - right));
        }
        this.zoomToFit(false);
        this.scheduleAnalysis(0);
      } else if (viewChanged) {
        this.saveViewState();
      } else {
        normalizeCrossfades(this.clips);
        this.serialize();
        this.clearError();
        this.saveViewState();
      }
    }
    if (this.resnapPending) this.resnapClipsToGrid();
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
      const previous = this.clips[this.selectedIndex - 1];
      const next = this.clips[this.selectedIndex + 1];
      const maximum = this.maximumFrame();
      const markers = this.editingSnapFrames(0, this.sequenceFrameCount());
      const startIndex = this.nearestBeatIndex(clip.start, markers);
      const endIndex = this.nearestBeatIndex(clip.end, markers);
      const markerStep = direction * (event.shiftKey ? DEFAULT_CLIP_GRID_INTERVALS : 1);
      const targetStart = startIndex + markerStep;
      const targetEnd = endIndex + markerStep;
      if (startIndex >= 0 &&
          endIndex > startIndex &&
          targetStart >= 0 &&
          targetEnd < markers.length &&
          (!previous || markers[targetStart] >= previous.end) &&
          (!next || markers[targetEnd] <= next.start) &&
          (!Number.isFinite(maximum) || markers[targetEnd] <= maximum)) {
        clip.start = markers[targetStart];
        clip.end = markers[targetEnd];
        this.serialize();
        this.syncInspector();
        this.scheduleDraw();
      }
      event.preventDefault();
    }
  }

  hitTest(x, y) {
    for (let index = this.crossfadeRects.length - 1; index >= 0; index--) {
      const rect = this.crossfadeRects[index];
      if (y < rect.y || y > rect.y + Math.min(24, rect.height)) continue;
      if (rect.frames > 0) {
        if (Math.abs(x - rect.startX) <= 10) {
          return { index: rect.index, type: "crossfade-start" };
        }
        if (Math.abs(x - rect.endX) <= 10) {
          return { index: rect.index, type: "crossfade-end" };
        }
      } else if (Math.abs(x - rect.boundaryX) <= 10) {
        return { index: rect.index, type: "crossfade-create" };
      }
    }
    for (let index = this.crossfadeRects.length - 1; index >= 0; index--) {
      const rect = this.crossfadeRects[index];
      if (y >= rect.y && y <= rect.y + rect.height &&
          Math.abs(x - rect.boundaryX) <= 10) {
        return { index: rect.index, type: "shared-boundary" };
      }
    }
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

  drawSourceOverview(ctx, width, top, bottom) {
    const right = width - TIMELINE_RIGHT;
    const center = (top + bottom) / 2;
    const preview = this.sourceWaveformPreview;
    ctx.fillStyle = "#12151a";
    ctx.fillRect(TIMELINE_LEFT, top, right - TIMELINE_LEFT, bottom - top);
    ctx.strokeStyle = "#2d323a";
    ctx.strokeRect(TIMELINE_LEFT + 0.5, top + 0.5, right - TIMELINE_LEFT - 1, bottom - top - 1);
    ctx.fillStyle = "#656b76";
    ctx.font = "7px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("SOURCE TRIM", TIMELINE_LEFT + 6, top + 4);
    if (!preview) return;

    const bins = preview.peaks.length / 2;
    const plotHeight = Math.max(1, (bottom - top) / 2 - 3);
    ctx.strokeStyle = "#66798a";
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    for (let x = TIMELINE_LEFT; x <= right; x++) {
      const ratio = (x - TIMELINE_LEFT) / Math.max(1, right - TIMELINE_LEFT);
      const bin = clamp(Math.floor(ratio * bins), 0, bins - 1);
      const minimum = preview.peaks[bin * 2] / preview.scale;
      const maximum = preview.peaks[bin * 2 + 1] / preview.scale;
      ctx.moveTo(x + 0.5, center - maximum * plotHeight);
      ctx.lineTo(x + 0.5, center - minimum * plotHeight);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    const handles = this.trimHandlePositions(width);
    ctx.fillStyle = "rgba(34,211,238,.12)";
    ctx.fillRect(handles.startX, top + 1, Math.max(1, handles.endX - handles.startX), bottom - top - 2);
    ctx.fillStyle = "rgba(0,0,0,.52)";
    ctx.fillRect(TIMELINE_LEFT, top + 1, Math.max(0, handles.startX - TIMELINE_LEFT), bottom - top - 2);
    ctx.fillRect(handles.endX, top + 1, Math.max(0, right - handles.endX), bottom - top - 2);
    for (const x of [handles.startX, handles.endX]) {
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, top);
      ctx.lineTo(x + 0.5, bottom);
      ctx.stroke();
      ctx.fillStyle = "#67e8f9";
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x - 5, top + 7);
      ctx.lineTo(x + 5, top + 7);
      ctx.closePath();
      ctx.fill();
    }
  }

  drawWaveformLane(ctx, width, top, bottom) {
    const right = width - TIMELINE_RIGHT;
    const center = (top + bottom) / 2;
    const preview = this.beatData?.waveformPreview;

    ctx.fillStyle = "#14191e";
    ctx.fillRect(TIMELINE_LEFT, top, right - TIMELINE_LEFT, bottom - top);
    ctx.strokeStyle = "#293039";
    ctx.strokeRect(TIMELINE_LEFT + 0.5, top + 0.5, right - TIMELINE_LEFT - 1, bottom - top - 1);

    const selected = this.selectedClip();
    if (selected) {
      const selectionStart = clamp(this.frameToX(selected.start, width), TIMELINE_LEFT, right);
      const selectionEnd = clamp(this.frameToX(selected.end, width), TIMELINE_LEFT, right);
      if (selectionEnd > selectionStart) {
        ctx.fillStyle = "rgba(167,139,250,.09)";
        ctx.fillRect(selectionStart, top + 1, selectionEnd - selectionStart, bottom - top - 2);
      }
    }

    ctx.strokeStyle = "#303944";
    ctx.beginPath();
    ctx.moveTo(TIMELINE_LEFT, center + 0.5);
    ctx.lineTo(right, center + 0.5);
    ctx.stroke();

    if (!preview) {
      ctx.fillStyle = "#64748b";
      ctx.font = "9px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Choose an audio file to load its waveform", (TIMELINE_LEFT + right) / 2, center);
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
    ctx.strokeStyle = "#6d9bad";
    ctx.globalAlpha = 0.92;
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

    if (!this.hover || this.hover.y < top || this.hover.y > bottom ||
        this.hover.x < TIMELINE_LEFT || this.hover.x > right) {
      return;
    }
    const frame = clamp(Math.round(this.frameAtX(this.hover.x)), 0, Math.round(waveformEndFrame));
    const seconds = frame / this.fps();
    const x = this.frameToX(frame, width);
    ctx.strokeStyle = "rgba(251,191,36,.42)";
    ctx.beginPath();
    ctx.moveTo(x + 0.5, top);
    ctx.lineTo(x + 0.5, bottom);
    ctx.stroke();

    const text = `F${frame} · ${formatClock(seconds)} · ${this.nearestBeatLabel(frame)}`;
    ctx.font = "9px Inter, sans-serif";
    const boxWidth = ctx.measureText(text).width + 12;
    const boxX = clamp(x - boxWidth / 2, TIMELINE_LEFT, right - boxWidth);
    ctx.fillStyle = "rgba(28,25,23,.94)";
    ctx.fillRect(boxX, top + 4, boxWidth, 18);
    ctx.fillStyle = "#fef3c7";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(text, boxX + 6, top + 13);
  }

  drawRuler(ctx, width, top, bottom) {
    const right = width - TIMELINE_RIGHT;
    const range = Math.max(1, this.viewEnd - this.viewStart);
    const step = niceFrameStep(range, right - TIMELINE_LEFT, this.fps());
    const minor = step % 4 === 0 ? step / 4 : step % 2 === 0 ? step / 2 : step;

    ctx.fillStyle = "#17191f";
    ctx.fillRect(TIMELINE_LEFT, top, right - TIMELINE_LEFT, bottom - top);
    ctx.strokeStyle = "#343842";
    ctx.beginPath();
    ctx.moveTo(TIMELINE_LEFT, bottom - 0.5);
    ctx.lineTo(right, bottom - 0.5);
    ctx.stroke();

    const firstMinor = Math.ceil(this.viewStart / minor) * minor;
    for (let frame = firstMinor; frame <= this.viewEnd + EPSILON; frame += minor) {
      const x = this.frameToX(frame, width);
      const major = Math.abs(frame % step) < EPSILON;
      ctx.strokeStyle = major ? "#59606c" : "#3a3f48";
      ctx.globalAlpha = major ? 0.72 : 0.5;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, bottom - (major ? 8 : 4));
      ctx.lineTo(x + 0.5, bottom);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    const firstTick = Math.ceil(this.viewStart / step) * step;
    const tickWidth = Math.abs(
      this.frameToX(firstTick + step, width) - this.frameToX(firstTick, width),
    );
    for (let frame = firstTick; frame <= this.viewEnd + EPSILON; frame += step) {
      const x = this.frameToX(frame, width);
      ctx.fillStyle = "#c0c4cc";
      ctx.font = "8px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(`${Math.round(frame)}f`, x, top + 3);
      if (tickWidth >= 54) {
        ctx.fillStyle = "#737984";
        ctx.font = "7px Inter, sans-serif";
        ctx.fillText(formatRulerTime(frame / this.fps()), x, top + 14);
      }
    }
  }

  drawAnalysisMarkers(ctx, width, top, bottom, tooltipTop) {
    const right = width - TIMELINE_RIGHT;
    const families = [
      {
        label: "Detected",
        frames: this.detectedBeatFrames(),
        color: "#e879f9",
        startY: bottom - 10,
      },
      {
        label: "Onset",
        frames: this.onsetFrames(),
        color: "#f59e0b",
        startY: bottom - 5,
      },
    ];
    let hovered = null;
    for (const family of families) {
      ctx.strokeStyle = family.color;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.48;
      for (const frame of family.frames) {
        if (frame < this.viewStart || frame > this.viewEnd) continue;
        const x = this.frameToX(frame, width);
        ctx.beginPath();
        ctx.moveTo(x + 0.5, family.startY);
        ctx.lineTo(x + 0.5, bottom - 1);
        ctx.stroke();
        if (this.hover?.y >= top && this.hover?.y <= bottom &&
            Math.abs(this.hover.x - x) <= 4 &&
            (!hovered || Math.abs(this.hover.x - x) < hovered.distance)) {
          hovered = {
            ...family,
            frame,
            x,
            distance: Math.abs(this.hover.x - x),
          };
        }
      }
    }
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;

    if (hovered) {
      const text = `${hovered.label} · F${hovered.frame} · ${formatClock(hovered.frame / this.fps())}`;
      ctx.font = "9px Inter, sans-serif";
      const boxWidth = ctx.measureText(text).width + 12;
      const boxX = clamp(hovered.x - boxWidth / 2, TIMELINE_LEFT, right - boxWidth);
      ctx.fillStyle = "rgba(24,24,27,.95)";
      ctx.fillRect(boxX, tooltipTop + 5, boxWidth, 18);
      ctx.fillStyle = hovered.color;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(text, boxX + 6, tooltipTop + 14);
    }
  }

  drawBeatGrid(ctx, width, rulerBottom, contentTop, contentBottom) {
    const frames = this.beatFrames();
    const visibleFrames = frames.filter((frame) => frame >= this.viewStart && frame <= this.viewEnd);
    const markerSpacing = visibleFrames.length > 1
      ? Math.abs(this.frameToX(visibleFrames[1], width) - this.frameToX(visibleFrames[0], width))
      : Infinity;
    const groupStride = this.beatGridDensity() === "half_beat"
      ? 8
      : this.beatGridDensity() === "every_2_beats"
      ? 2
      : 4;

    for (let index = 0; index < frames.length; index++) {
      const frame = frames[index];
      if (frame < this.viewStart || frame > this.viewEnd) continue;
      const x = this.frameToX(frame, width);
      const accent = index % groupStride === 0;
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = accent ? 1.25 : 1;
      ctx.globalAlpha = accent ? 0.28 : 0.14;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, contentTop);
      ctx.lineTo(x + 0.5, contentBottom);
      ctx.stroke();

      ctx.globalAlpha = accent ? 0.9 : 0.72;
      ctx.fillStyle = "#67e8f9";
      ctx.beginPath();
      ctx.arc(x, rulerBottom - 4, accent ? 3 : 2.25, 0, Math.PI * 2);
      ctx.fill();

      if (accent || markerSpacing >= 38) {
        ctx.fillStyle = "#8ddde8";
        ctx.font = "7px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(String(index + 1), x, rulerBottom - 9);
      }
    }
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
  }

  drawPromptClips(ctx, width, top, bottom) {
    const right = width - TIMELINE_RIGHT;
    const trackHeight = Math.max(80, bottom - top);
    const cardY = top + 7;
    const cardHeight = trackHeight - 14;
    const previousHover = this.hover ? this.hitTest(this.hover.x, this.hover.y) : null;
    const sharedBoundaryIndex = this.drag?.type === "shared-boundary"
      ? this.selectedIndex
      : previousHover?.type === "shared-boundary"
      ? previousHover.index
      : -1;

    ctx.fillStyle = "#15171c";
    ctx.fillRect(TIMELINE_LEFT, top, right - TIMELINE_LEFT, trackHeight);
    ctx.strokeStyle = "#2d3038";
    ctx.strokeRect(TIMELINE_LEFT + 0.5, top + 0.5, right - TIMELINE_LEFT - 1, trackHeight - 1);

    this.clipRects = [];
    this.crossfadeRects = [];
    for (let index = 0; index < this.clips.length; index++) {
      const clip = this.clips[index];
      if (clip.end < this.viewStart || clip.start > this.viewEnd) continue;
      const startX = this.frameToX(clip.start, width);
      const endX = this.frameToX(clip.end, width);
      const fadeInX = this.frameToX(clip.start + clip.fadeIn, width);
      const fadeOutX = this.frameToX(clip.end - clip.fadeOut, width);
      const x = clamp(startX, TIMELINE_LEFT, right);
      const clippedEnd = clamp(endX, TIMELINE_LEFT, right);
      const cardWidth = Math.max(2, clippedEnd - x);
      const drawX = x + 1;
      const drawWidth = Math.max(1, cardWidth - 2);
      const selected = index === this.selectedIndex;
      const shared = index === sharedBoundaryIndex || index === sharedBoundaryIndex - 1;
      const hovered = previousHover?.index === index || shared;

      ctx.save();
      if (selected || shared) {
        ctx.shadowColor = shared ? "rgba(34,211,238,.3)" : "rgba(167,139,250,.28)";
        ctx.shadowBlur = 8;
      }
      ctx.fillStyle = shared
        ? "#38505c"
        : selected
        ? "#4a4263"
        : hovered
        ? "#414654"
        : index % 2
        ? "#363b47"
        : "#333844";
      ctx.strokeStyle = shared
        ? "#67e8f9"
        : selected
        ? "#b8a5ff"
        : hovered
        ? "#7b8292"
        : "#555c6b";
      ctx.lineWidth = selected || shared ? 2 : 1;
      ctx.beginPath();
      ctx.roundRect(drawX, cardY, drawWidth, cardHeight, 6);
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      if (fadeInX > startX) {
        const fadeStart = clamp(startX, TIMELINE_LEFT, right);
        const fadeEnd = clamp(fadeInX, TIMELINE_LEFT, right);
        const gradient = ctx.createLinearGradient(fadeStart, 0, fadeEnd, 0);
        gradient.addColorStop(0, "rgba(12,14,20,.46)");
        gradient.addColorStop(1, "rgba(12,14,20,0)");
        ctx.fillStyle = gradient;
        ctx.fillRect(fadeStart, cardY + 2, Math.max(0, fadeEnd - fadeStart), cardHeight - 4);
      }
      if (fadeOutX < endX) {
        const fadeStart = clamp(fadeOutX, TIMELINE_LEFT, right);
        const fadeEnd = clamp(endX, TIMELINE_LEFT, right);
        const gradient = ctx.createLinearGradient(fadeStart, 0, fadeEnd, 0);
        gradient.addColorStop(0, "rgba(12,14,20,0)");
        gradient.addColorStop(1, "rgba(12,14,20,.46)");
        ctx.fillStyle = gradient;
        ctx.fillRect(fadeStart, cardY + 2, Math.max(0, fadeEnd - fadeStart), cardHeight - 4);
      }

      if (selected || hovered) {
        ctx.fillStyle = shared ? "#67e8f9" : selected ? "#c4b5fd" : "#7b8190";
        ctx.fillRect(drawX, cardY + 22, Math.min(3, drawWidth), Math.max(12, cardHeight - 44));
        ctx.fillRect(Math.max(drawX, drawX + drawWidth - 3), cardY + 22, Math.min(3, drawWidth), Math.max(12, cardHeight - 44));
      }
      if (selected) {
        for (const handleX of [fadeInX, fadeOutX]) {
          if (handleX < TIMELINE_LEFT || handleX > right) continue;
          ctx.fillStyle = "#ddd6fe";
          ctx.beginPath();
          ctx.moveTo(handleX, cardY + 3);
          ctx.lineTo(handleX - 5, cardY + 9);
          ctx.lineTo(handleX, cardY + 15);
          ctx.lineTo(handleX + 5, cardY + 9);
          ctx.closePath();
          ctx.fill();
        }
      }

      if (drawWidth > 24) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(drawX + 9, cardY + 8, Math.max(0, drawWidth - 18), cardHeight - 16);
        ctx.clip();
        ctx.fillStyle = "#f4f4f5";
        ctx.font = "600 10px Inter, sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        const lines = canvasTextLines(ctx, clip.prompt, Math.max(1, drawWidth - 18), 2);
        lines.forEach((line, lineIndex) => ctx.fillText(line, drawX + 9, cardY + 10 + lineIndex * 14));
        ctx.fillStyle = shared ? "#a5f3fc" : selected ? "#c4b5fd" : "#9ca3af";
        ctx.font = "8px Inter, sans-serif";
        ctx.fillText(
          `${clip.start}–${clip.end}f · ${(clip.start / this.fps()).toFixed(2)}–${(clip.end / this.fps()).toFixed(2)}s`,
          drawX + 9,
          cardY + cardHeight - 19,
        );
        ctx.restore();
      }

      this.clipRects.push({
        index,
        x,
        y: cardY,
        width: cardWidth,
        height: cardHeight,
        fadeInX,
        fadeOutX,
      });
    }

    for (let index = 1; index < this.clips.length; index++) {
      const clip = this.clips[index];
      const previous = this.clips[index - 1];
      if (previous.end !== clip.start || clip.start < this.viewStart || clip.start > this.viewEnd) {
        continue;
      }
      const bounds = this.crossfadeBounds(index);
      const boundaryX = this.frameToX(bounds.boundary, width);
      const startX = clamp(this.frameToX(bounds.start, width), TIMELINE_LEFT, right);
      const endX = clamp(this.frameToX(bounds.end, width), TIMELINE_LEFT, right);
      const related = this.selectedIndex === index || this.selectedIndex === index - 1;
      const hovered = previousHover?.index === index &&
        previousHover?.type.startsWith("crossfade");
      const shared = sharedBoundaryIndex === index;
      this.crossfadeRects.push({
        index,
        x: startX,
        y: cardY,
        width: Math.max(1, endX - startX),
        height: cardHeight,
        startX,
        endX,
        boundaryX,
        frames: bounds.frames,
      });

      if (bounds.frames > 0) {
        const gradient = ctx.createLinearGradient(startX, 0, endX, 0);
        gradient.addColorStop(0, "rgba(167,139,250,.5)");
        gradient.addColorStop(0.5, "rgba(125,211,252,.38)");
        gradient.addColorStop(1, "rgba(34,211,238,.5)");
        ctx.fillStyle = gradient;
        ctx.fillRect(startX, cardY + 2, Math.max(1, endX - startX), cardHeight - 4);

        ctx.strokeStyle = "rgba(224,231,255,.7)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(startX, cardY + 2);
        ctx.lineTo(endX, cardY + cardHeight - 2);
        ctx.moveTo(startX, cardY + cardHeight - 2);
        ctx.lineTo(endX, cardY + 2);
        ctx.stroke();

        ctx.strokeStyle = "#e0f2fe";
        ctx.globalAlpha = 0.78;
        ctx.beginPath();
        ctx.moveTo(boundaryX + 0.5, cardY + 2);
        ctx.lineTo(boundaryX + 0.5, cardY + cardHeight - 2);
        ctx.stroke();
        ctx.globalAlpha = 1;

        if (endX - startX >= 42) {
          const label = `${bounds.frames}f`;
          ctx.font = "700 8px Inter, sans-serif";
          const labelWidth = ctx.measureText(label).width + 10;
          ctx.fillStyle = "rgba(15,23,42,.9)";
          ctx.fillRect(
            clamp(boundaryX - labelWidth / 2, startX, endX - labelWidth),
            cardY + 5,
            labelWidth,
            16,
          );
          ctx.fillStyle = "#e0f2fe";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(label, boundaryX, cardY + 13);
        }
      } else if (!related && !hovered && !shared) {
        continue;
      }

      for (const handleX of bounds.frames > 0 ? [startX, endX] : [boundaryX]) {
        ctx.fillStyle = hovered ? "#ffffff" : "#bae6fd";
        ctx.beginPath();
        ctx.moveTo(handleX, cardY + 2);
        ctx.lineTo(handleX - 5, cardY + 8);
        ctx.lineTo(handleX, cardY + 14);
        ctx.lineTo(handleX + 5, cardY + 8);
        ctx.closePath();
        ctx.fill();
      }

      if (shared) {
        ctx.fillStyle = "#67e8f9";
        ctx.fillRect(
          boundaryX - 2,
          cardY + 18,
          4,
          Math.max(8, cardHeight - 36),
        );
        const centerY = cardY + cardHeight / 2;
        ctx.beginPath();
        ctx.moveTo(boundaryX - 8, centerY);
        ctx.lineTo(boundaryX - 3, centerY - 5);
        ctx.lineTo(boundaryX - 3, centerY + 5);
        ctx.closePath();
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(boundaryX + 8, centerY);
        ctx.lineTo(boundaryX + 3, centerY - 5);
        ctx.lineTo(boundaryX + 3, centerY + 5);
        ctx.closePath();
        ctx.fill();
      }
    }
  }

  drawGuidesAndPlayhead(ctx, width, layout) {
    if (this.snapGuideFrame != null &&
        this.snapGuideFrame >= this.viewStart &&
        this.snapGuideFrame <= this.viewEnd) {
      const x = this.frameToX(this.snapGuideFrame, width);
      ctx.strokeStyle = "#c4b5fd";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x + 0.5, layout.rulerBottom);
      ctx.lineTo(x + 0.5, layout.trackBottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (this.playheadFrame != null &&
        this.playheadFrame >= this.viewStart &&
        this.playheadFrame <= this.viewEnd) {
      const x = this.frameToX(this.playheadFrame, width);
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, layout.rulerTop);
      ctx.lineTo(x + 0.5, layout.trackBottom);
      ctx.stroke();
      ctx.fillStyle = "#fbbf24";
      ctx.beginPath();
      ctx.moveTo(x, layout.rulerTop + 9);
      ctx.lineTo(x - 6, layout.rulerTop + 1);
      ctx.lineTo(x + 6, layout.rulerTop + 1);
      ctx.closePath();
      ctx.fill();
      ctx.lineWidth = 1;
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

    const layout = this.timelineLayout(cssHeight);
    const {
      sourceTop,
      sourceBottom,
      rulerTop,
      rulerBottom,
      waveformTop,
      waveformBottom,
      trackTop,
      trackBottom,
    } = layout;

    if (this.waveformVisible) {
      if (this.sourceWaveformPreview) {
        this.drawSourceOverview(ctx, cssWidth, sourceTop, sourceBottom);
      }
      this.drawWaveformLane(ctx, cssWidth, waveformTop, waveformBottom);
    }
    this.drawPromptClips(ctx, cssWidth, trackTop, trackBottom);
    this.drawRuler(ctx, cssWidth, rulerTop, rulerBottom);
    const contentTop = this.waveformVisible ? waveformTop : trackTop;
    this.drawAnalysisMarkers(ctx, cssWidth, rulerTop, rulerBottom, contentTop);
    this.drawBeatGrid(ctx, cssWidth, rulerBottom, contentTop, trackBottom);
    this.drawGuidesAndPlayhead(ctx, cssWidth, layout);

    if (this.migrationPending) {
      this.emptyEl.textContent = "Run once to convert this legacy beat schedule into frames.";
    } else {
      this.emptyEl.textContent = this.clips.length ? "" : "Open Raw to repair the schedule, or add a prompt clip.";
    }
  }

  dispose() {
    this.closeContextMenu();
    document.removeEventListener("pointerdown", this.documentPointerHandler, true);
    if (this.resizeObserver) this.resizeObserver.disconnect();
    if (this.pendingFrame) cancelAnimationFrame(this.pendingFrame);
    this.stopPlaybackLoop();
    clearTimeout(this.analysisTimer);
    clearTimeout(this.separationTimer);
    if (this.audioElement) this.audioElement.pause();
    this.analysisRequest++;
    for (const restore of this.callbackRestorers) restore();
    this.callbackRestorers = [];
    this.root.remove();
  }
}

function compactStatusText(widgets, editor = null, payload = null) {
  const audio = filenameFromPath(widgets.audioFile?.value) || "No audio selected";
  const frames = Math.max(0, Math.round(finiteNumber(widgets.sequenceDuration?.value)));
  let promptCount = editor?.clips?.length;
  if (!Number.isFinite(promptCount)) {
    try {
      promptCount = parseTimeline(
        widgets.timeline?.value || "",
        finiteNumber(widgets.defaultFadeIn?.value),
        finiteNumber(widgets.defaultFadeOut?.value),
      ).length;
    } catch {
      promptCount = 0;
    }
  }
  const bpm = finiteNumber(
    editor?.beatData?.gridBpm,
    finiteNumber(payload?.grid_bpm, payload?.bpm),
  );
  return `${audio} · ${promptCount} prompt${promptCount === 1 ? "" : "s"} · ` +
    `${frames || "auto"} frames${bpm > 0 ? ` · ${bpm.toFixed(2)} BPM` : ""}`;
}

function updateCompactStatus(node, widgets, statusWidget, editor = null, payload = null) {
  statusWidget.value = compactStatusText(widgets, editor, payload);
  app.graph?.setDirtyCanvas?.(true, false);
}

function compactNode(node, force) {
  node.min_size = [320, 120];
  requestAnimationFrame(() => {
    const computed = node.computeSize();
    const width = force
      ? COMPACT_NODE_WIDTH
      : Math.max(320, Math.min(node.size[0], 520));
    const height = Math.max(120, computed[1]);
    if (force || node.size[1] > height + 40 || node.size[0] > 520) {
      node.setSize([width, height]);
    }
  });
}

class BeatPromptSequencerModal {
  constructor(node, widgets, statusWidget) {
    this.node = node;
    this.widgets = widgets;
    this.statusWidget = statusWidget;
    this.editor = null;
    this.libraryEntries = [];
    this.localEntries = [];
    this.libraryMode = "library";
    this.libraryCollapsed = Boolean(node.properties?.flBeatPromptSequencer?.libraryCollapsed);
    this.widgetRestorers = [];
    this.previousBodyOverflow = "";
    this.closed = false;
    this.build();
  }

  build() {
    this.overlay = document.createElement("div");
    this.overlay.className = "flbps-modal-overlay";
    this.overlay.setAttribute("role", "dialog");
    this.overlay.setAttribute("aria-modal", "true");
    this.overlay.innerHTML = `
      <div class="flbps-modal-shell">
        <div class="flbps-modal-header">
          <div class="flbps-modal-heading">
            <div class="flbps-modal-title">FL Audio Beat Prompt Sequencer</div>
            <div class="flbps-modal-subtitle" data-role="modal-subtitle"></div>
          </div>
          <span class="flbps-spacer"></span>
          <button class="flbps-button flbps-sidebar-toggle" data-action="toggle-library" title="Show or hide the audio library and sequence settings">Hide library</button>
          <button class="flbps-button primary flbps-modal-close" data-action="modal-close">Done</button>
        </div>
        <div class="flbps-modal-main">
          <aside class="flbps-library">
            <div class="flbps-library-section">
              <div class="flbps-library-label">Audio source</div>
              <div class="flbps-drop-zone" data-role="drop-zone">
                Drop an audio or video file here<br>or click to choose one
              </div>
              <div class="flbps-library-actions">
                <button class="flbps-button" data-action="choose-file" title="Upload one audio or video file into ComfyUI input">Choose file</button>
                <button class="flbps-button" data-action="choose-folder" title="Search a local folder; only the file you select is uploaded">Choose folder</button>
              </div>
              <div class="flbps-library-message" data-role="library-message"></div>
            </div>
            <div class="flbps-library-section">
              <div class="flbps-library-tabs">
                <button class="flbps-button active" data-source="library">Comfy input</button>
                <button class="flbps-button" data-source="local">Local folder</button>
              </div>
              <input class="flbps-library-search" data-role="library-search" type="search" placeholder="Search audio files or folders">
              <select class="flbps-library-folder" data-role="library-folder" aria-label="Filter audio folder"></select>
            </div>
            <div class="flbps-library-results" data-role="library-results"></div>
            <div class="flbps-library-section">
              <div class="flbps-library-actions">
                <button class="flbps-button" data-action="refresh-library">Refresh input</button>
              </div>
            </div>
            <div class="flbps-library-section">
              <div class="flbps-library-label">Sequence settings</div>
              <div class="flbps-settings">
                <div class="flbps-setting" title="Frames per second used by the timeline and rendered video"><label>FPS</label><input data-setting="fps" type="number" min="1" max="240" step="0.001"></div>
                <div class="flbps-setting" title="Maximum sequence length in frames; zero uses the remaining audio"><label>Length frames</label><input data-setting="length" type="number" min="0" max="864000" step="1"></div>
                <div class="flbps-setting" title="Default number of frames used to fade a prompt in"><label>Default fade in</label><input data-setting="fade-in" type="number" min="0" max="864000" step="1"></div>
                <div class="flbps-setting" title="Default number of frames used to fade a prompt out"><label>Default fade out</label><input data-setting="fade-out" type="number" min="0" max="864000" step="1"></div>
                <div class="flbps-setting" title="Shape used for prompt fade-ins and fade-outs"><label>Curve</label><select data-setting="curve"><option value="linear">Linear</option><option value="cosine">Cosine</option></select></div>
                <div class="flbps-setting" title="Choose how the audio tempo is estimated"><label>BPM method</label><select data-setting="bpm-method"><option value="beat_intervals">Beat intervals</option><option value="onset_strength">Onset strength</option></select></div>
                <div class="flbps-setting" title="Analyze the full mix or a previously separated stem"><label>Analysis source</label><select data-setting="analysis-source"><option value="mix">Mix</option><option value="drums">Drums</option><option value="vocals">Vocals</option><option value="bass">Bass</option><option value="other">Other</option></select></div>
                <div class="flbps-setting checkbox" title="Use every other detected beat and report half the detected BPM"><input data-setting="half-time" type="checkbox"><label>Half-time</label></div>
              </div>
            </div>
          </aside>
          <main class="flbps-editor-host" data-role="editor-host"></main>
        </div>
      </div>
    `;
    this.shell = this.overlay.querySelector(".flbps-modal-shell");
    this.subtitle = this.overlay.querySelector('[data-role="modal-subtitle"]');
    this.library = this.overlay.querySelector(".flbps-library");
    this.results = this.overlay.querySelector('[data-role="library-results"]');
    this.searchInput = this.overlay.querySelector('[data-role="library-search"]');
    this.folderSelect = this.overlay.querySelector('[data-role="library-folder"]');
    this.libraryMessage = this.overlay.querySelector('[data-role="library-message"]');
    this.dropZone = this.overlay.querySelector('[data-role="drop-zone"]');
    this.editorHost = this.overlay.querySelector('[data-role="editor-host"]');
    this.libraryToggle = this.overlay.querySelector('[data-action="toggle-library"]');
    this.syncLibraryVisibility();

    this.fileInput = document.createElement("input");
    this.fileInput.type = "file";
    this.fileInput.accept = "audio/*,video/*,.aac,.aiff,.flac,.m4a,.mka,.mkv,.mov,.mp3,.mp4,.oga,.ogg,.opus,.wav,.webm,.wma";
    this.fileInput.hidden = true;
    this.folderInput = document.createElement("input");
    this.folderInput.type = "file";
    this.folderInput.multiple = true;
    this.folderInput.webkitdirectory = true;
    this.folderInput.hidden = true;
    this.library.append(this.fileInput, this.folderInput);

    this.overlay.querySelector('[data-action="modal-close"]').addEventListener("click", () => this.close());
    this.libraryToggle.addEventListener("click", () => this.toggleLibrary());
    this.overlay.addEventListener("pointerdown", (event) => {
      if (event.target === this.overlay) this.close();
    });
    for (const type of ["pointerdown", "pointermove", "pointerup", "wheel"]) {
      this.shell.addEventListener(type, (event) => event.stopPropagation(), { passive: type === "wheel" });
    }
    this.keyHandler = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        this.close();
      } else if ((event.code === "Space" || event.key === " ") &&
          !(event.target instanceof HTMLInputElement) &&
          !(event.target instanceof HTMLTextAreaElement) &&
          !(event.target instanceof HTMLSelectElement) &&
          !(event.target instanceof HTMLButtonElement) &&
          !event.target?.isContentEditable) {
        event.preventDefault();
        if (!event.repeat) this.editor?.togglePlayback();
      }
    };
    this.overlay.addEventListener("keydown", this.keyHandler);

    this.dropZone.addEventListener("click", () => this.chooseFile());
    this.overlay.querySelector('[data-action="choose-file"]').addEventListener("click", () => this.chooseFile());
    this.overlay.querySelector('[data-action="choose-folder"]').addEventListener("click", () => this.chooseFolder());
    this.overlay.querySelector('[data-action="refresh-library"]').addEventListener("click", () => this.refreshLibrary());
    this.fileInput.addEventListener("change", () => {
      const file = this.fileInput.files?.[0];
      if (file) this.uploadFile(file);
    });
    this.folderInput.addEventListener("change", () => this.loadLocalFolder());
    this.searchInput.addEventListener("input", () => this.renderFiles());
    this.folderSelect.addEventListener("change", () => this.renderFiles(false));
    for (const button of this.overlay.querySelectorAll("[data-source]")) {
      button.addEventListener("click", () => this.setLibraryMode(button.dataset.source));
    }
    this.library.addEventListener("dragover", (event) => {
      if (!event.dataTransfer?.types?.includes("Files")) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
      this.dropZone.classList.add("dragging");
    });
    this.library.addEventListener("dragleave", (event) => {
      if (!this.library.contains(event.relatedTarget)) this.dropZone.classList.remove("dragging");
    });
    this.library.addEventListener("drop", (event) => {
      event.preventDefault();
      this.dropZone.classList.remove("dragging");
      const file = [...(event.dataTransfer?.files || [])].find(isSupportedMediaFile);
      if (file) this.uploadFile(file);
      else this.setLibraryMessage("Drop a supported audio file or a video containing audio.", true);
    });
  }

  show() {
    if (activeModal && activeModal !== this) activeModal.close();
    activeModal = this;
    this.previousBodyOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    document.body.appendChild(this.overlay);
    this.editor = new BeatPromptSequencer({
      node: this.node,
      container: this.editorHost,
      widgets: this.widgets,
      onStateChange: () => this.handleEditorState(),
    });
    INSTANCES.set(this.node.id, this.editor);
    this.bindSettings();
    this.syncSettings();
    const pending = this.node._flSequencerExecutionMessage;
    if (pending) {
      this.editor.updateFromExecution(pending);
      this.node._flSequencerExecutionMessage = null;
    }
    this.refreshLibrary();
    requestAnimationFrame(() => {
      this.shell.tabIndex = -1;
      this.shell.focus({ preventScroll: true });
      this.editor.scheduleDraw();
    });
  }

  bindSettings() {
    this.settingSpecs = {
      fps: { widget: this.widgets.fps, parse: (value) => clamp(finiteNumber(value, 24), 1, 240) },
      length: { widget: this.widgets.sequenceDuration, parse: (value) => clamp(Math.round(finiteNumber(value)), 0, 864000) },
      "fade-in": { widget: this.widgets.defaultFadeIn, parse: (value) => clamp(Math.round(finiteNumber(value)), 0, 864000) },
      "fade-out": { widget: this.widgets.defaultFadeOut, parse: (value) => clamp(Math.round(finiteNumber(value)), 0, 864000) },
      curve: { widget: this.widgets.curve, parse: String },
      "bpm-method": { widget: this.widgets.bpmMethod, parse: String },
      "analysis-source": { widget: this.widgets.analysisSource, parse: String },
      "half-time": { widget: this.widgets.halfTime, parse: Boolean },
    };
    for (const [name, spec] of Object.entries(this.settingSpecs)) {
      const control = this.overlay.querySelector(`[data-setting="${name}"]`);
      control.addEventListener("change", () => {
        const raw = control.type === "checkbox" ? control.checked : control.value;
        setWidgetValue(spec.widget, spec.parse(raw));
      });
    }
    const syncedWidgets = [
      ...Object.values(this.settingSpecs).map((spec) => spec.widget),
      this.widgets.audioFile,
    ].filter(Boolean);
    for (const widget of syncedWidgets) {
      const original = widget.callback;
      const wrapped = (value) => {
        const result = original?.call(widget, value);
        this.syncSettings();
        if (widget === this.widgets.audioFile) this.renderFiles();
        return result;
      };
      widget.callback = wrapped;
      this.widgetRestorers.push(() => {
        if (widget.callback === wrapped) widget.callback = original;
      });
    }
  }

  syncSettings() {
    for (const [name, spec] of Object.entries(this.settingSpecs || {})) {
      const control = this.overlay.querySelector(`[data-setting="${name}"]`);
      if (!control) continue;
      if (control.type === "checkbox") control.checked = Boolean(spec.widget?.value);
      else control.value = String(spec.widget?.value ?? "");
    }
    const audio = String(this.widgets.audioFile?.value || "");
    this.subtitle.textContent = audio
      ? `${audio} · edits save directly to the node`
      : "Choose audio from Comfy input, drag a file, or browse a local folder";
    updateCompactStatus(this.node, this.widgets, this.statusWidget, this.editor);
  }

  handleEditorState() {
    this.syncSettings();
  }

  syncLibraryVisibility() {
    this.shell.classList.toggle("library-collapsed", this.libraryCollapsed);
    this.libraryToggle.textContent = this.libraryCollapsed ? "Show library" : "Hide library";
  }

  toggleLibrary() {
    this.libraryCollapsed = !this.libraryCollapsed;
    this.node.properties = this.node.properties || {};
    this.node.properties.flBeatPromptSequencer = {
      ...(this.node.properties.flBeatPromptSequencer || {}),
      formatVersion: FORMAT_VERSION,
      libraryCollapsed: this.libraryCollapsed,
    };
    this.syncLibraryVisibility();
    this.node.graph?.change?.();
    setTimeout(() => this.editor?.scheduleDraw(), 180);
  }

  chooseFile() {
    this.fileInput.value = "";
    this.fileInput.click();
  }

  chooseFolder() {
    this.folderInput.value = "";
    this.folderInput.click();
  }

  loadLocalFolder() {
    this.localEntries = [...(this.folderInput.files || [])]
      .filter(isSupportedMediaFile)
      .map((file) => {
        const path = (file.webkitRelativePath || file.name).replace(/\\/g, "/");
        const slash = path.lastIndexOf("/");
        return {
          path,
          folder: slash >= 0 ? path.slice(0, slash) : "",
          size: file.size,
          file,
        };
      })
      .sort((left, right) => left.path.localeCompare(right.path, undefined, { sensitivity: "base" }));
    this.setLibraryMode("local");
    this.setLibraryMessage(
      this.localEntries.length
        ? `${this.localEntries.length} supported files found. Only the file you select will upload.`
        : "No supported audio or video files were found in that folder.",
      !this.localEntries.length,
    );
  }

  setLibraryMode(mode) {
    this.libraryMode = mode === "local" ? "local" : "library";
    for (const button of this.overlay.querySelectorAll("[data-source]")) {
      button.classList.toggle("active", button.dataset.source === this.libraryMode);
    }
    this.renderFiles(true);
  }

  setLibraryMessage(message, error = false) {
    this.libraryMessage.textContent = message;
    this.libraryMessage.style.color = error ? "#fca5a5" : "#8b8b95";
  }

  async refreshLibrary() {
    this.setLibraryMessage("Refreshing ComfyUI input audio…");
    try {
      const response = await api.fetchApi("/fl/audio-prompt-timeline/files");
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `Audio library refresh failed (${response.status}).`);
      this.libraryEntries = Array.isArray(payload.files) ? payload.files : [];
      const values = this.widgets.audioFile?.options?.values;
      if (Array.isArray(values)) {
        for (const entry of this.libraryEntries) {
          if (!values.includes(entry.path)) values.push(entry.path);
        }
      }
      this.setLibraryMessage(`${this.libraryEntries.length} files available in ComfyUI input.`);
      this.renderFiles(true);
    } catch (error) {
      this.setLibraryMessage(error.message, true);
    }
  }

  renderFiles(resetFolder = false) {
    const entries = this.libraryMode === "local" ? this.localEntries : this.libraryEntries;
    const folders = [...new Set(entries.map((entry) => entry.folder || ""))]
      .sort((left, right) => left.localeCompare(right, undefined, { sensitivity: "base" }));
    const previousFolder = resetFolder ? "" : this.folderSelect.value;
    this.folderSelect.replaceChildren();
    const allOption = document.createElement("option");
    allOption.value = "";
    allOption.textContent = "All folders";
    this.folderSelect.appendChild(allOption);
    for (const folder of folders) {
      const option = document.createElement("option");
      option.value = folder;
      option.textContent = folder || "Input root";
      this.folderSelect.appendChild(option);
    }
    if (folders.includes(previousFolder)) this.folderSelect.value = previousFolder;

    const search = this.searchInput.value.trim().toLocaleLowerCase();
    const folder = this.folderSelect.value;
    const filtered = entries.filter((entry) => {
      if (folder && entry.folder !== folder) return false;
      return !search || entry.path.toLocaleLowerCase().includes(search);
    });
    this.results.replaceChildren();
    const selected = String(this.widgets.audioFile?.value || "").replace(/\\/g, "/");
    for (const entry of filtered.slice(0, 500)) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "flbps-file-row";
      if (this.libraryMode === "library" && entry.path === selected) row.classList.add("selected");
      const name = document.createElement("span");
      name.className = "flbps-file-name";
      name.textContent = filenameFromPath(entry.path);
      const folderLabel = document.createElement("span");
      folderLabel.className = "flbps-file-folder";
      folderLabel.textContent = entry.folder || (this.libraryMode === "library" ? "ComfyUI/input" : "Selected folder");
      row.append(name, folderLabel);
      row.addEventListener("click", () => {
        if (this.libraryMode === "local") this.uploadFile(entry.file);
        else this.selectAudioPath(entry.path);
      });
      this.results.appendChild(row);
    }
    if (!filtered.length) {
      const empty = document.createElement("div");
      empty.className = "flbps-library-message";
      empty.style.padding = "10px";
      empty.textContent = this.libraryMode === "local"
        ? "Choose a folder, then search its audio files here."
        : "No ComfyUI input files match this search.";
      this.results.appendChild(empty);
    } else if (filtered.length > 500) {
      const more = document.createElement("div");
      more.className = "flbps-library-message";
      more.style.padding = "8px";
      more.textContent = `Showing the first 500 of ${filtered.length} matches. Refine the search to narrow the list.`;
      this.results.appendChild(more);
    }
  }

  selectAudioPath(path) {
    const values = this.widgets.audioFile?.options?.values;
    if (Array.isArray(values) && !values.includes(path)) values.push(path);
    setWidgetValue(this.widgets.audioFile, path);
    this.node.graph?.change?.();
    this.setLibraryMessage(`Loaded ${filenameFromPath(path)}.`);
    this.renderFiles();
  }

  async uploadFile(file) {
    if (!isSupportedMediaFile(file)) {
      this.setLibraryMessage("Choose a supported audio file or video containing audio.", true);
      return;
    }
    this.setLibraryMessage(`Uploading ${file.name}…`);
    try {
      const body = new FormData();
      body.append("image", file);
      body.append("type", "input");
      const response = await api.fetchApi("/upload/image", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `Upload failed (${response.status}).`);
      const path = [payload.subfolder, payload.name].filter(Boolean).join("/").replace(/\\/g, "/");
      this.selectAudioPath(path);
      await this.refreshLibrary();
    } catch (error) {
      this.setLibraryMessage(error.message, true);
    }
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    for (const restore of this.widgetRestorers.reverse()) restore();
    this.widgetRestorers = [];
    if (this.editor) {
      this.editor.saveViewState();
      this.editor.dispose();
      INSTANCES.delete(this.node.id);
      this.editor = null;
    }
    this.overlay.removeEventListener("keydown", this.keyHandler);
    this.overlay.remove();
    document.body.style.overflow = this.previousBodyOverflow;
    if (activeModal === this) activeModal = null;
    updateCompactStatus(this.node, this.widgets, this.statusWidget);
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
      audioFile: findWidget(node, "audio_file"),
      trimStartFrame: findWidget(node, "trim_start_frame"),
      bpmMethod: findWidget(node, "bpm_method"),
      halfTime: findWidget(node, "half_time"),
      beatOffset: findWidget(node, "beat_offset_ms"),
      analysisSource: findWidget(node, "analysis_source"),
      beatGridDensity: findWidget(node, "beat_grid_density"),
    };
    const hiddenWidgets = Object.values(widgets).filter(Boolean);
    for (const widget of hiddenWidgets) hideWidget(widget);

    const previousFormat = finiteNumber(node.properties?.flBeatPromptSequencer?.formatVersion);
    node.properties = node.properties || {};
    const savedSequencer = {
      ...(node.properties.flBeatPromptSequencer || {}),
      formatVersion: FORMAT_VERSION,
    };
    delete savedSequencer.magnetMode;
    delete savedSequencer.snapMode;
    if (!COMPATIBLE_FORMAT_VERSIONS.has(previousFormat)) {
      savedSequencer.beatData = null;
      savedSequencer.viewStart = 0;
      savedSequencer.viewEnd = 0;
    }
    node.properties.flBeatPromptSequencer = savedSequencer;
    const openWidget = node.addWidget("button", "Open Audio Prompt Sequencer", null, () => {
      const modal = new BeatPromptSequencerModal(node, widgets, statusWidget);
      modal.show();
    }, { serialize: false });
    openWidget.serialize = false;
    const statusWidget = node.addWidget("text", "Timeline status", "", null, { serialize: false });
    statusWidget.disabled = true;
    statusWidget.serialize = false;
    updateCompactStatus(node, widgets, statusWidget);
    compactNode(node, previousFormat !== FORMAT_VERSION);

    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      originalOnExecuted?.apply(this, arguments);
      const editor = INSTANCES.get(node.id);
      if (editor) editor.updateFromExecution(message);
      else this._flSequencerExecutionMessage = message;
      updateCompactStatus(this, widgets, statusWidget, editor, executionPayload(message));
    };

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const result = originalOnConfigure?.apply(this, args);
      for (const widget of hiddenWidgets) hideWidget(widget);
      compactNode(this, false);
      const editor = INSTANCES.get(node.id);
      if (editor) {
        editor.applyBeatOffset();
        editor.loadTimeline();
        editor.resnapClipsToGrid();
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

    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
      if (activeModal?.node === this) activeModal.close();
      return originalOnRemoved?.apply(this, arguments);
    };
  },
});
