import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const DEFAULT_SETTINGS = {
  version: 1,
  filename_prefix: "FillVideo",
  frame_rate: 24,
  format: "mp4",
  codec: "h264",
  crf: 19,
  bit_depth: 8,
  include_audio: true,
  trim_video_to_audio: false,
  audio_gain_db: 0,
  output_directory: "",
  save_output: true,
  save_metadata: true,
};

const MIN_NODE_WIDTH = 420;
const MIN_NODE_HEIGHT = 360;
const MIN_PANEL_HEIGHT = 280;

const STYLES = `
  .flvc-panel {
    --flvc-accent: #8b5cf6;
    --flvc-border: var(--border-color, #343741);
    --flvc-control: var(--comfy-input-bg, #24262d);
    --flvc-muted: var(--descrip-text, #979cab);
    background: var(--comfy-menu-bg, #18191e);
    border: 1px solid var(--flvc-border);
    border-radius: 8px;
    box-sizing: border-box;
    color: var(--input-text, #f4f4f5);
    display: grid;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 11px;
    gap: 6px;
    grid-template-rows: auto minmax(0, 1fr);
    height: 100%;
    min-height: 0;
    overflow: hidden;
    padding: 6px;
    position: relative;
    width: 100%;
  }
  .flvc-panel * { box-sizing: border-box; }
  .flvc-toolbar {
    align-items: end;
    display: grid;
    gap: 4px;
    grid-template-columns:
      minmax(84px, 1.6fr)
      minmax(43px, .55fr)
      minmax(43px, .55fr)
      minmax(54px, .7fr)
      minmax(46px, .62fr)
      minmax(58px, .75fr)
      28px;
    min-width: 0;
  }
  .flvc-param {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .flvc-param-label {
    color: var(--flvc-muted);
    font-size: 8px;
    font-weight: 700;
    letter-spacing: .05em;
    line-height: 9px;
    overflow: hidden;
    text-overflow: ellipsis;
    text-transform: uppercase;
    white-space: nowrap;
  }
  .flvc-param input[type="text"],
  .flvc-param input[type="number"],
  .flvc-param select {
    appearance: none;
    background: var(--flvc-control);
    border: 1px solid var(--flvc-border);
    border-radius: 4px;
    color: inherit;
    font: inherit;
    height: 25px;
    min-width: 0;
    padding: 2px 5px;
    width: 100%;
  }
  .flvc-param input[type="number"] { appearance: textfield; }
  .flvc-param input[type="number"]::-webkit-inner-spin-button,
  .flvc-param input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  .flvc-param input:focus,
  .flvc-param select:focus {
    border-color: var(--flvc-accent);
    outline: none;
  }
  .flvc-param input:disabled { opacity: .45; }
  .flvc-toggle {
    align-items: center;
    background: var(--flvc-control);
    border: 1px solid var(--flvc-border);
    border-radius: 4px;
    cursor: pointer;
    display: flex;
    gap: 4px;
    height: 25px;
    justify-content: center;
    min-width: 0;
    padding: 0 4px;
  }
  .flvc-toggle input {
    accent-color: var(--flvc-accent);
    height: 13px;
    margin: 0;
    width: 13px;
  }
  .flvc-toggle-value {
    color: var(--flvc-muted);
    font-size: 9px;
    font-weight: 700;
  }
  .flvc-toggle[data-enabled="true"] .flvc-toggle-value { color: #86efac; }
  .flvc-more-button,
  .flvc-icon-button,
  .flvc-menu-close,
  .flvc-reset-button {
    background: var(--flvc-control);
    border: 1px solid var(--flvc-border);
    color: inherit;
    cursor: pointer;
    font: inherit;
  }
  .flvc-more-button {
    border-radius: 4px;
    font-size: 16px;
    height: 25px;
    line-height: 18px;
    padding: 0;
  }
  .flvc-more-button:hover,
  .flvc-icon-button:hover,
  .flvc-menu-close:hover,
  .flvc-reset-button:hover {
    border-color: var(--flvc-accent);
  }
  .flvc-more-button[data-custom="true"] {
    border-color: var(--flvc-accent);
    color: #c4b5fd;
  }
  .flvc-more-button:focus-visible,
  .flvc-icon-button:focus-visible,
  .flvc-menu-close:focus-visible,
  .flvc-reset-button:focus-visible {
    outline: 2px solid var(--flvc-accent);
    outline-offset: 1px;
  }
  .flvc-preview {
    background: #050507;
    border: 1px solid var(--flvc-border);
    border-radius: 6px;
    grid-row: 2;
    height: 100%;
    min-height: 0;
    min-width: 0;
    overflow: hidden;
    position: relative;
    width: 100%;
  }
  .flvc-preview video {
    display: block;
    height: 100%;
    object-fit: contain;
    width: 100%;
  }
  .flvc-placeholder {
    align-items: center;
    color: #717784;
    display: flex;
    inset: 0;
    justify-content: center;
    padding: 18px 18px 48px;
    position: absolute;
    text-align: center;
  }
  .flvc-status,
  .flvc-summary {
    backdrop-filter: blur(5px);
    background: rgba(24, 25, 30, .84);
    border-radius: 999px;
    max-width: calc(65% - 8px);
    overflow: hidden;
    position: absolute;
    text-overflow: ellipsis;
    top: 7px;
    white-space: nowrap;
    z-index: 2;
  }
  .flvc-summary {
    color: #c7cad1;
    font-size: 9px;
    left: 7px;
    padding: 4px 8px;
  }
  .flvc-status {
    color: #c8ccd5;
    font-size: 9px;
    font-weight: 700;
    padding: 4px 8px;
    right: 7px;
    text-transform: uppercase;
  }
  .flvc-status[data-state="ready"] { background: rgba(18, 59, 43, .9); color: #86efac; }
  .flvc-status[data-state="stale"] { background: rgba(70, 55, 24, .9); color: #fde68a; }
  .flvc-status[data-state="error"] { background: rgba(76, 29, 36, .9); color: #fda4af; }
  .flvc-preview-controls {
    align-items: center;
    background: linear-gradient(transparent, rgba(0, 0, 0, .82) 55%);
    bottom: 0;
    display: flex;
    gap: 6px;
    left: 0;
    min-width: 0;
    padding: 22px 7px 7px;
    position: absolute;
    right: 0;
    z-index: 2;
  }
  .flvc-icon-button {
    border-radius: 4px;
    flex: 0 0 28px;
    height: 25px;
    padding: 0;
  }
  .flvc-time {
    color: #e4e4e7;
    flex: 0 0 82px;
    font-size: 10px;
    font-variant-numeric: tabular-nums;
  }
  .flvc-control-spacer { flex: 1 1 auto; min-width: 4px; }
  .flvc-preview-volume { flex: 0 1 92px; min-width: 50px; }
  .flvc-preview-value {
    color: #d4d4d8;
    flex: 0 0 31px;
    font-size: 9px;
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  .flvc-menu {
    background: var(--comfy-menu-bg, #1b1c21);
    border: 1px solid var(--flvc-border);
    border-radius: 7px;
    box-shadow: 0 10px 26px rgba(0, 0, 0, .38);
    display: grid;
    gap: 8px;
    max-height: calc(100% - 54px);
    max-width: calc(100% - 12px);
    overflow-y: auto;
    padding: 9px;
    position: absolute;
    right: 6px;
    top: 47px;
    width: 300px;
    z-index: 5;
  }
  .flvc-menu[hidden] { display: none; }
  .flvc-menu-header {
    align-items: center;
    display: flex;
    justify-content: space-between;
  }
  .flvc-menu-title {
    font-size: 11px;
    font-weight: 750;
  }
  .flvc-menu-close {
    border-radius: 4px;
    height: 23px;
    padding: 0;
    width: 23px;
  }
  .flvc-menu-check {
    align-items: center;
    display: flex;
    gap: 7px;
    min-height: 24px;
  }
  .flvc-menu-check input {
    accent-color: var(--flvc-accent);
    margin: 0;
  }
  .flvc-menu-check[data-disabled="true"] { opacity: .45; }
  .flvc-menu-field {
    display: grid;
    gap: 4px;
  }
  .flvc-menu-field > span {
    color: var(--flvc-muted);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: .04em;
    text-transform: uppercase;
  }
  .flvc-menu-field input {
    background: var(--flvc-control);
    border: 1px solid var(--flvc-border);
    border-radius: 4px;
    color: inherit;
    font: inherit;
    height: 27px;
    min-width: 0;
    padding: 3px 6px;
    width: 100%;
  }
  .flvc-menu-field input:focus {
    border-color: var(--flvc-accent);
    outline: none;
  }
  .flvc-menu-help {
    color: var(--flvc-muted);
    font-size: 9px;
    line-height: 1.35;
  }
  .flvc-menu-info {
    border-top: 1px solid var(--flvc-border);
    display: grid;
    gap: 5px;
    padding-top: 8px;
  }
  .flvc-menu-info-row {
    align-items: center;
    color: var(--flvc-muted);
    display: flex;
    justify-content: space-between;
  }
  .flvc-menu-info-row strong {
    color: inherit;
    font-size: 10px;
  }
  .flvc-reset-button {
    border-radius: 4px;
    height: 26px;
  }
  .flvc-error {
    background: #3f1d25;
    border: 1px solid #7f1d2d;
    border-radius: 5px;
    color: #fecdd3;
    font-size: 9px;
    left: 8px;
    padding: 6px;
    position: absolute;
    right: 8px;
    top: 48px;
    z-index: 6;
  }
  .flvc-error[hidden] { display: none; }
`;

function injectStyles() {
  if (document.getElementById("flvc-styles")) return;
  const style = document.createElement("style");
  style.id = "flvc-styles";
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

function formatTime(value) {
  if (!Number.isFinite(value)) return "00:00";
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

class VideoCombinePanel {
  constructor(node, settingsWidget, container) {
    this.node = node;
    this.settingsWidget = settingsWidget;
    this.container = container;
    this.preview = null;
    this.settings = { ...DEFAULT_SETTINGS };
    this.configError = "";
    this.handleDocumentPointerDown = null;
    this.handleDocumentKeyDown = null;

    this.node.properties ||= {};
    if (!Number.isFinite(this.node.properties.previewVolume)) {
      this.node.properties.previewVolume = 0.8;
    }
    if (typeof this.node.properties.previewMuted !== "boolean") {
      this.node.properties.previewMuted = true;
    }

    this.readSettings();
    this.build();
    this.bind();
    this.syncControls();

    if (this.node.properties.lastPreview) {
      this.loadPreview(this.node.properties.lastPreview);
    }
    if (this.configError) {
      this.showConfigError();
    }
  }

  readSettings() {
    try {
      const parsed = JSON.parse(this.settingsWidget.value);
      if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
        throw new Error("Render settings must be a JSON object.");
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
      <div class="flvc-panel">
        <div class="flvc-toolbar" role="group" aria-label="Video render settings">
          <label class="flvc-param">
            <span class="flvc-param-label">Prefix</span>
            <input data-setting="filename_prefix" aria-label="Filename prefix" type="text">
          </label>
          <label class="flvc-param">
            <span class="flvc-param-label">FPS</span>
            <input data-setting="frame_rate" aria-label="Frame rate" title="Frame rate" type="number" min="1" max="120" step="0.01">
          </label>
          <label class="flvc-param">
            <span class="flvc-param-label">CRF</span>
            <input data-setting="crf" aria-label="Video quality CRF" title="Lower CRF means higher quality" type="number" min="0" max="51" step="1">
          </label>
          <label class="flvc-param">
            <span class="flvc-param-label">Depth</span>
            <select data-setting="bit_depth" aria-label="Bit depth" title="Video bit depth">
              <option value="8">8-bit</option>
              <option value="10">10-bit</option>
            </select>
          </label>
          <label class="flvc-param">
            <span class="flvc-param-label">Audio</span>
            <span class="flvc-toggle" data-role="audio-toggle">
              <input data-setting="include_audio" aria-label="Include connected audio" type="checkbox">
              <span class="flvc-toggle-value" data-role="audio-toggle-value">On</span>
            </span>
          </label>
          <label class="flvc-param">
            <span class="flvc-param-label">Gain dB</span>
            <input data-setting="audio_gain_db" aria-label="Export audio gain" title="Export audio gain in decibels" type="number" min="-60" max="12" step="0.5">
          </label>
          <button class="flvc-more-button" data-role="more" type="button" title="Output settings" aria-label="Output settings" aria-expanded="false">⋯</button>
        </div>

        <div class="flvc-error" data-role="error" role="alert" hidden></div>

        <div class="flvc-preview">
          <video data-role="video" loop playsinline></video>
          <div class="flvc-placeholder" data-role="placeholder">Queue the workflow to render a preview.</div>
          <div class="flvc-summary" data-role="summary">MP4 · H.264</div>
          <div class="flvc-status" data-role="status" data-state="idle">not rendered</div>
          <div class="flvc-preview-controls">
            <button class="flvc-icon-button" data-role="play" type="button" title="Play or pause preview">▶</button>
            <span class="flvc-time" data-role="time">00:00 / 00:00</span>
            <span class="flvc-control-spacer"></span>
            <button class="flvc-icon-button" data-role="preview-mute" type="button" title="Mute preview">🔇</button>
            <input class="flvc-preview-volume" data-role="preview-volume" aria-label="Preview volume" type="range" min="0" max="100" step="1">
            <span class="flvc-preview-value" data-role="preview-volume-value">80%</span>
          </div>
        </div>

        <div class="flvc-menu" data-role="settings-menu" hidden>
          <div class="flvc-menu-header">
            <span class="flvc-menu-title">Output settings</span>
            <button class="flvc-menu-close" data-role="menu-close" type="button" aria-label="Close output settings">×</button>
          </div>
          <label class="flvc-menu-field">
            <span>Custom directory</span>
            <input data-setting="output_directory" aria-label="Custom output directory" title="Absolute directory for rendered videos" type="text" placeholder="D:/Video Exports">
          </label>
          <div class="flvc-menu-help">Use an absolute path. Leave blank to use the ComfyUI output or temporary directory.</div>
          <label class="flvc-menu-check" data-role="save-output-row">
            <input data-setting="save_output" type="checkbox">
            Save to output directory
          </label>
          <label class="flvc-menu-check">
            <input data-setting="save_metadata" type="checkbox">
            Embed workflow metadata
          </label>
          <label class="flvc-menu-check" data-role="trim-audio-row">
            <input data-setting="trim_video_to_audio" type="checkbox">
            Match video duration to connected audio
          </label>
          <div class="flvc-menu-info">
            <div class="flvc-menu-info-row">
              <span>Container</span>
              <strong>MP4</strong>
            </div>
            <div class="flvc-menu-info-row">
              <span>Codec</span>
              <strong>H.264</strong>
            </div>
          </div>
          <button class="flvc-reset-button" data-role="reset-settings" type="button">Reset render settings</button>
        </div>
      </div>
    `;

    this.video = this.container.querySelector('[data-role="video"]');
    this.placeholder = this.container.querySelector('[data-role="placeholder"]');
    this.status = this.container.querySelector('[data-role="status"]');
    this.error = this.container.querySelector('[data-role="error"]');
    this.summary = this.container.querySelector('[data-role="summary"]');
    this.playButton = this.container.querySelector('[data-role="play"]');
    this.previewMuteButton = this.container.querySelector('[data-role="preview-mute"]');
    this.previewVolume = this.container.querySelector('[data-role="preview-volume"]');
    this.previewVolumeValue = this.container.querySelector('[data-role="preview-volume-value"]');
    this.time = this.container.querySelector('[data-role="time"]');
    this.audioToggle = this.container.querySelector('[data-role="audio-toggle"]');
    this.audioToggleValue = this.container.querySelector('[data-role="audio-toggle-value"]');
    this.moreButton = this.container.querySelector('[data-role="more"]');
    this.settingsMenu = this.container.querySelector('[data-role="settings-menu"]');
    this.saveOutputRow = this.container.querySelector('[data-role="save-output-row"]');
    this.menuCloseButton = this.container.querySelector('[data-role="menu-close"]');
    this.resetSettingsButton = this.container.querySelector('[data-role="reset-settings"]');
    this.saveOutputControl = this.container.querySelector('[data-setting="save_output"]');
    this.trimAudioControl = this.container.querySelector('[data-setting="trim_video_to_audio"]');
    this.trimAudioRow = this.container.querySelector('[data-role="trim-audio-row"]');
    this.settingControls = [...this.container.querySelectorAll("[data-setting]")];
  }

  bind() {
    for (const control of this.settingControls) {
      const eventName = control.type === "text" || control.type === "number" ? "input" : "change";
      control.addEventListener(eventName, () => {
        let value;
        if (control.type === "checkbox") {
          value = control.checked;
        } else if (control.dataset.setting === "bit_depth" || control.dataset.setting === "crf") {
          value = Number.parseInt(control.value, 10);
        } else if (control.type === "number" || control.type === "range") {
          value = Number.parseFloat(control.value);
        } else {
          value = control.value;
        }
        if (typeof value === "number" && !Number.isFinite(value)) return;
        this.updateSetting(control.dataset.setting, value);
      });
    }

    this.moreButton.addEventListener("click", () => {
      this.setMenuOpen(this.settingsMenu.hidden);
    });
    this.menuCloseButton.addEventListener("click", () => this.setMenuOpen(false));
    this.resetSettingsButton.addEventListener("click", () => this.resetSettings());

    this.handleDocumentPointerDown = (event) => {
      if (
        !this.settingsMenu.hidden
        && !this.settingsMenu.contains(event.target)
        && !this.moreButton.contains(event.target)
      ) {
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

    this.playButton.addEventListener("click", () => {
      if (!this.video.src) return;
      if (this.video.paused) {
        this.video.play().catch(() => {});
      } else {
        this.video.pause();
      }
    });
    this.video.addEventListener("play", () => {
      this.playButton.textContent = "❚❚";
    });
    this.video.addEventListener("pause", () => {
      this.playButton.textContent = "▶";
    });
    this.video.addEventListener("timeupdate", () => this.updateTime());
    this.video.addEventListener("loadedmetadata", () => {
      this.placeholder.style.display = "none";
      this.updateTime();
      this.video.play().catch(() => {});
    });
    this.video.addEventListener("error", () => {
      if (!this.video.src) return;
      this.placeholder.textContent = "Preview unavailable. The rendered file may still be valid.";
      this.placeholder.style.display = "flex";
    });

    this.previewMuteButton.addEventListener("click", () => {
      this.node.properties.previewMuted = !this.node.properties.previewMuted;
      this.applyPreviewAudio();
    });
    this.previewVolume.addEventListener("input", () => {
      this.node.properties.previewVolume = Number(this.previewVolume.value) / 100;
      this.applyPreviewAudio();
    });
  }

  setMenuOpen(open) {
    this.settingsMenu.hidden = !open;
    this.moreButton.setAttribute("aria-expanded", String(open));
  }

  updateSetting(name, value) {
    this.settings[name] = value;
    this.settingsWidget.value = JSON.stringify(this.settings);
    this.configError = "";
    this.showConfigError();
    this.syncControls(name);
    if (this.preview) {
      this.setStatus("stale", "settings changed");
    }
    this.node.setDirtyCanvas(true, true);
  }

  resetSettings() {
    this.settings = { ...DEFAULT_SETTINGS };
    this.settingsWidget.value = JSON.stringify(this.settings);
    this.configError = "";
    this.syncControls();
    if (this.preview) {
      this.setStatus("stale", "settings changed");
    }
    this.node.setDirtyCanvas(true, true);
  }

  syncControls(changedName = null) {
    for (const control of this.settingControls) {
      const name = control.dataset.setting;
      if (changedName && name !== changedName) continue;
      if (control.type === "checkbox") {
        control.checked = Boolean(this.settings[name]);
      } else {
        control.value = this.settings[name];
      }
    }

    const audioEnabled = Boolean(this.settings.include_audio);
    this.audioToggle.dataset.enabled = String(audioEnabled);
    this.audioToggleValue.textContent = audioEnabled ? "On" : "Off";
    for (const control of this.settingControls.filter((item) => item.dataset.setting === "audio_gain_db")) {
      control.disabled = !audioEnabled;
    }
    this.trimAudioControl.disabled = !audioEnabled;
    this.trimAudioRow.dataset.disabled = String(!audioEnabled);

    const customDirectory = String(this.settings.output_directory || "").trim();
    const customOutput = Boolean(customDirectory);
    this.saveOutputControl.disabled = customOutput;
    this.saveOutputRow.dataset.disabled = String(customOutput);
    this.moreButton.dataset.custom = String(customOutput);
    this.moreButton.title = customOutput ? `Custom output: ${customDirectory}` : "Output settings";

    this.showConfigError();
    this.applyPreviewAudio();
  }

  showConfigError() {
    if (this.configError) {
      this.error.textContent = this.configError;
      this.error.hidden = false;
      this.setStatus("error", "invalid settings");
    } else {
      this.error.textContent = "";
      this.error.hidden = true;
    }
  }

  applyPreviewAudio() {
    const volume = Math.max(0, Math.min(1, Number(this.node.properties.previewVolume)));
    this.previewVolume.value = Math.round(volume * 100);
    this.previewVolumeValue.textContent = `${Math.round(volume * 100)}%`;
    this.video.volume = volume;
    this.video.muted = Boolean(this.node.properties.previewMuted);
    this.previewMuteButton.textContent = this.video.muted ? "🔇" : "🔊";
    this.previewMuteButton.title = this.video.muted ? "Unmute preview" : "Mute preview";
  }

  updateTime() {
    this.time.textContent = `${formatTime(this.video.currentTime)} / ${formatTime(this.video.duration)}`;
  }

  setStatus(state, label) {
    this.status.dataset.state = state;
    this.status.textContent = label;
  }

  loadPreview(preview) {
    if (!preview?.filename) return;
    this.preview = preview;
    if (preview.preview_url) {
      const separator = preview.preview_url.includes("?") ? "&" : "?";
      this.video.src = api.apiURL(`${preview.preview_url}${separator}timestamp=${Date.now()}`);
    } else {
      const params = new URLSearchParams({
        filename: preview.filename,
        subfolder: preview.subfolder || "",
        type: preview.type || "output",
        timestamp: Date.now(),
      });
      this.video.src = api.apiURL(`/view?${params.toString()}`);
    }
    this.video.load();
    this.placeholder.textContent = "Loading preview…";
    this.placeholder.style.display = "flex";
    this.setStatus("ready", "ready");

    const padded = preview.source_width !== preview.encoded_width || preview.source_height !== preview.encoded_height;
    const dimensions = padded
      ? `${preview.source_width}×${preview.source_height} → ${preview.encoded_width}×${preview.encoded_height}`
      : `${preview.encoded_width}×${preview.encoded_height}`;
    const audio = preview.has_audio ? "audio" : "silent";
    this.summary.textContent = `${preview.frame_count} frames · ${Number(preview.frame_rate).toFixed(2)} fps · ${Number(preview.duration).toFixed(2)} sec · ${dimensions} · ${preview.bit_depth}-bit · ${audio}`;
    this.applyPreviewAudio();
  }

  updateFromExecution(message) {
    const preview = message?.fl_video_combine?.[0];
    if (!preview) return;
    this.node.properties.lastPreview = { ...preview };
    this.node.properties.lastRenderSettings = this.settingsWidget.value;
    this.loadPreview(preview);
  }

  configure() {
    hideWidget(this.settingsWidget);
    this.readSettings();
    this.setMenuOpen(false);
    this.syncControls();
    if (this.node.properties.lastPreview) {
      this.loadPreview(this.node.properties.lastPreview);
      if (this.node.properties.lastRenderSettings !== this.settingsWidget.value) {
        this.setStatus("stale", "settings changed");
      }
    }
    if (this.configError) {
      this.showConfigError();
    }
  }

  dispose() {
    if (this.handleDocumentPointerDown) {
      document.removeEventListener("pointerdown", this.handleDocumentPointerDown);
    }
    if (this.handleDocumentKeyDown) {
      document.removeEventListener("keydown", this.handleDocumentKeyDown);
    }
    this.video.pause();
    this.video.removeAttribute("src");
    this.video.load();
    this.container.replaceChildren();
  }
}

app.registerExtension({
  name: "ComfyUI.FL_VideoCombine",
  nodeCreated(node) {
    if (node.comfyClass !== "FL_VideoCombine") return;

    const settingsWidget = node.widgets?.find((widget) => widget.name === "render_settings");
    if (!settingsWidget) return;
    hideWidget(settingsWidget);

    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.height = "100%";
    container.style.minHeight = `${MIN_PANEL_HEIGHT}px`;
    container.style.overflow = "hidden";

    const domWidget = node.addDOMWidget("fl_video_combine_panel", "fl-video-combine", container, {
      getMinHeight: () => MIN_PANEL_HEIGHT,
      hideOnZoom: false,
      serialize: false,
    });
    enforceMinimumNodeSize(node);
    requestAnimationFrame(() => enforceMinimumNodeSize(node));

    const panel = new VideoCombinePanel(node, settingsWidget, container);

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
