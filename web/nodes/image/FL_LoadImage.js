import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const DEFAULT_SETTINGS = {
  version: 1,
  resize_mode: "original",
  width: 0,
  height: 0,
};

const MIN_NODE_WIDTH = 240;
const MIN_NODE_HEIGHT = 230;
const MIN_PANEL_HEIGHT = 160;
const IMAGE_EXTENSIONS = new Set(["bmp", "gif", "jpeg", "jpg", "png", "webp"]);

const STYLES = `
  .flli-container {
    container-type: size;
    height: 100%;
    min-height: 0;
    min-width: 0;
    overflow: hidden;
    width: 100%;
  }
  .flli-panel {
    --flli-accent: #8b5cf6;
    --flli-border: var(--border-color, #343741);
    --flli-control: var(--comfy-input-bg, #24262d);
    --flli-muted: var(--descrip-text, #979cab);
    background: var(--comfy-menu-bg, #17181d);
    border: 1px solid var(--flli-border);
    border-radius: 9px;
    box-sizing: border-box;
    color: var(--input-text, #f4f4f5);
    display: grid;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: clamp(7px, 2.6cqw, 11px);
    gap: clamp(3px, 1.4cqw, 6px);
    grid-template-rows: clamp(21px, 9cqh, 29px) minmax(0, 1fr) clamp(36px, 16cqh, 48px);
    height: 100%;
    min-height: 0;
    overflow: hidden;
    padding: clamp(3px, 1.4cqw, 6px);
    position: relative;
    width: 100%;
  }
  .flli-panel * { box-sizing: border-box; }
  .flli-source-bar {
    align-items: center;
    display: flex;
    gap: 5px;
    min-width: 0;
    overflow: hidden;
    padding: 0 1px;
  }
  .flli-filename {
    color: #d8dbe2;
    flex: 1;
    font-size: clamp(7px, 2.5cqw, 10.5px);
    font-weight: 650;
    min-width: 0;
    overflow: hidden;
    padding: 0 4px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .flli-button,
  .flli-more-button,
  .flli-menu-close,
  .flli-reset-button {
    background: var(--flli-control);
    border: 1px solid var(--flli-border);
    color: inherit;
    cursor: pointer;
    font: inherit;
  }
  .flli-button {
    border-radius: 5px;
    height: clamp(20px, 8cqh, 26px);
    min-width: 0;
    padding: 0 clamp(3px, 2cqw, 9px);
    white-space: nowrap;
  }
  .flli-more-button {
    border-radius: 5px;
    font-size: 16px;
    flex: 0 0 clamp(20px, 7cqw, 28px);
    height: clamp(20px, 8cqh, 26px);
    line-height: 18px;
    padding: 0;
    width: clamp(20px, 7cqw, 28px);
  }
  .flli-button:hover,
  .flli-more-button:hover,
  .flli-menu-close:hover,
  .flli-reset-button:hover {
    background: color-mix(in srgb, var(--flli-control) 82%, var(--flli-accent));
    border-color: var(--flli-accent);
  }
  .flli-button:focus-visible,
  .flli-more-button:focus-visible,
  .flli-menu-close:focus-visible,
  .flli-reset-button:focus-visible,
  .flli-drop-zone:focus-visible,
  .flli-control-row select:focus-visible,
  .flli-control-row input:focus-visible {
    outline: 2px solid var(--flli-accent);
    outline-offset: 1px;
  }
  .flli-preview {
    background: #050507;
    border: 1px solid var(--flli-border);
    border-radius: 7px;
    display: grid;
    height: 100%;
    min-height: 0;
    min-width: 0;
    overflow: hidden;
    position: relative;
    place-items: center;
    width: 100%;
  }
  .flli-image-stage {
    display: grid;
    height: 100%;
    margin: auto;
    max-height: 100%;
    max-width: 100%;
    min-height: 0;
    min-width: 0;
    overflow: hidden;
    place-items: center;
    width: 100%;
  }
  .flli-image-stage img {
    display: block;
    height: 100%;
    margin: auto;
    max-height: 100%;
    max-width: 100%;
    min-height: 0;
    min-width: 0;
    object-fit: contain;
    object-position: center center;
    width: 100%;
  }
  .flli-image-stage[data-resize-mode="crop"] img { object-fit: cover; }
  .flli-drop-zone {
    align-items: center;
    background:
      radial-gradient(circle at 50% 44%, rgba(139, 92, 246, .13), transparent 45%),
      #08090d;
    border: 1px dashed #565b68;
    color: #a3a8b4;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    gap: clamp(2px, 1.5cqw, 7px);
    inset: clamp(2px, 1.5cqw, 7px);
    justify-content: center;
    padding: clamp(4px, 4cqw, 18px);
    position: absolute;
    text-align: center;
    z-index: 4;
  }
  .flli-drop-zone[hidden] { display: none; }
  .flli-drop-icon {
    color: #c4b5fd;
    font-size: clamp(14px, 6cqw, 24px);
    line-height: 1;
  }
  .flli-drop-title {
    color: #e6e7eb;
    font-size: clamp(8px, 2.8cqw, 12px);
    font-weight: 700;
  }
  .flli-drop-help {
    color: var(--flli-muted);
    font-size: clamp(7px, 2.2cqw, 9px);
  }
  .flli-panel[data-dragging="true"] .flli-drop-zone {
    background: rgba(76, 29, 149, .93);
    border-color: #c4b5fd;
    display: flex;
    inset: 4px;
  }
  .flli-panel[data-dragging="true"] .flli-drop-title { color: white; }
  .flli-preview-info,
  .flli-status {
    backdrop-filter: blur(5px);
    background: rgba(20, 21, 26, .78);
    border: 1px solid rgba(255, 255, 255, .07);
    border-radius: 999px;
    max-width: calc(75% - 8px);
    overflow: hidden;
    padding: clamp(1px, .7cqw, 3px) clamp(3px, 1.6cqw, 7px);
    position: absolute;
    text-overflow: ellipsis;
    top: 7px;
    white-space: nowrap;
    z-index: 3;
  }
  .flli-preview-info { left: 7px; }
  .flli-status {
    max-width: 80px;
    right: 7px;
  }
  .flli-status[data-state="ready"] { color: #86efac; }
  .flli-status[data-state="busy"] { color: #fde68a; }
  .flli-status[data-state="stale"] { color: #fdba74; }
  .flli-status[data-state="error"] { color: #fda4af; }
  .flli-error {
    background: rgba(127, 29, 29, .96);
    border: 1px solid #ef4444;
    border-radius: 6px;
    color: #fee2e2;
    left: 8px;
    padding: 7px 9px;
    position: absolute;
    right: 8px;
    top: 40px;
    z-index: 8;
  }
  .flli-error[hidden] { display: none; }
  .flli-browser-error {
    background: rgba(24, 24, 27, .92);
    bottom: 8px;
    color: #fda4af;
    left: 8px;
    padding: 5px 7px;
    position: absolute;
    right: 8px;
    text-align: center;
    z-index: 3;
  }
  .flli-browser-error[hidden] { display: none; }
  .flli-control-row {
    align-items: end;
    background: rgba(255, 255, 255, .025);
    border: 1px solid var(--flli-border);
    border-radius: 6px;
    display: grid;
    gap: clamp(2px, 1.2cqw, 5px);
    grid-template-columns: minmax(0, .55fr) minmax(0, 1.4fr) minmax(0, 1fr) minmax(0, 1fr);
    min-width: 0;
    overflow: hidden;
    padding: clamp(2px, 1.2cqw, 5px) clamp(2px, 1.4cqw, 6px);
  }
  .flli-group-label {
    align-self: center;
    color: #c4b5fd;
    font-size: clamp(6px, 1.9cqw, 8px);
    min-width: 0;
    overflow: hidden;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
  }
  .flli-param {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .flli-param-label {
    color: var(--flli-muted);
    font-size: clamp(6px, 1.9cqw, 8px);
    font-weight: 700;
  }
  .flli-param input,
  .flli-param select,
  .flli-menu-field select {
    background: var(--flli-control);
    border: 1px solid var(--flli-border);
    border-radius: 4px;
    color: inherit;
    font: inherit;
    appearance: textfield;
    height: clamp(20px, 8cqh, 24px);
    min-width: 0;
    padding: 0 5px;
    width: 100%;
  }
  .flli-param input::-webkit-inner-spin-button,
  .flli-param input::-webkit-outer-spin-button { appearance: none; margin: 0; }
  .flli-param input:disabled { color: #696d78; }
  .flli-param[data-overridden="true"] .flli-param-label { color: #c4b5fd; }
  .flli-menu {
    background: rgba(20, 21, 26, .985);
    border: 1px solid var(--flli-border);
    border-radius: 8px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, .45);
    display: grid;
    gap: 8px;
    left: 7px;
    padding: 10px;
    position: absolute;
    right: 7px;
    top: 38px;
    z-index: 10;
  }
  .flli-menu[hidden] { display: none; }
  .flli-menu-header {
    align-items: center;
    display: flex;
    justify-content: space-between;
  }
  .flli-menu-title { font-weight: 750; }
  .flli-menu-close {
    border-radius: 4px;
    height: 24px;
    padding: 0;
    width: 24px;
  }
  .flli-menu-field {
    display: grid;
    gap: 4px;
  }
  .flli-menu-field > span,
  .flli-menu-help {
    color: var(--flli-muted);
    font-size: 9px;
  }
  .flli-menu-divider { border-top: 1px solid var(--flli-border); }
  .flli-reset-button {
    border-radius: 5px;
    height: 27px;
  }
  .flli-upload-progress {
    background: rgba(255, 255, 255, .12);
    border-radius: 999px;
    height: 3px;
    overflow: hidden;
    width: 120px;
  }
  .flli-upload-progress[hidden] { display: none; }
  .flli-upload-progress span {
    animation: flli-upload 900ms ease-in-out infinite;
    background: #a78bfa;
    display: block;
    height: 100%;
    width: 45%;
  }
  @keyframes flli-upload {
    from { transform: translateX(-100%); }
    to { transform: translateX(220%); }
  }
  @container (max-height: 220px) {
    .flli-preview-info,
    .flli-status { top: 3px; }
    .flli-panel { border-radius: 6px; }
    .flli-control-row { border-radius: 4px; }
  }
`;

function injectStyles() {
  if (document.getElementById("flli-styles")) return;
  const style = document.createElement("style");
  style.id = "flli-styles";
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
  if (width !== node.size[0] || height !== node.size[1]) node.setSize([width, height]);
}

function filenameFromPath(path) {
  return String(path || "").replace(/ \[(input|output|temp)\]$/, "").replace(/\\/g, "/").split("/").pop() || "";
}

function previewReference(path) {
  const parts = String(path || "").replace(/ \[(input|output|temp)\]$/, "").replace(/\\/g, "/").split("/");
  return { filename: parts.pop() || "", subfolder: parts.join("/") };
}

function supportedImageFile(file) {
  const extension = file?.name?.split(".").pop()?.toLowerCase();
  return Boolean(file && IMAGE_EXTENSIONS.has(extension));
}

class LoadImagePanel {
  constructor(node, rootWidget, legacyWidget, imageWidget, settingsWidget, container) {
    this.node = node;
    this.rootWidget = rootWidget;
    this.legacyWidget = legacyWidget;
    this.imageWidget = imageWidget;
    this.settingsWidget = settingsWidget;
    this.container = container;
    this.settings = { ...DEFAULT_SETTINGS };
    this.sourceInfo = null;
    this.executionInfo = null;
    this.configError = "";
    this.objectUrl = null;
    this.dragDepth = 0;
    this.uploading = false;
    this.disposed = false;
    this.overrideValues = { width: null, height: null };
    this.overrideTimer = null;

    this.readSettings();
    this.build();
    this.bind();
    this.refreshOverrideValues(false);
    this.syncControls();
    this.restoreSource();
    this.scheduleOverrideTracking();
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
      <div class="flli-panel" data-dragging="false">
        <div class="flli-source-bar">
          <div class="flli-filename" data-role="filename">No image selected</div>
          <button class="flli-button" data-action="replace" type="button">Choose</button>
          <button class="flli-more-button" data-action="more" type="button" title="Source details" aria-label="Source details" aria-expanded="false">⋯</button>
        </div>

        <div class="flli-error" data-role="error" role="alert" hidden></div>

        <div class="flli-preview">
          <div class="flli-image-stage" data-role="image-stage" data-resize-mode="original">
            <img data-role="image" alt="Selected image preview">
          </div>
          <div class="flli-preview-info" data-role="preview-info">Choose an image</div>
          <div class="flli-status" data-role="status" data-state="idle">empty</div>
          <div class="flli-browser-error" data-role="browser-error" hidden>Browser preview unavailable. This file can still be loaded.</div>
          <div class="flli-drop-zone" data-role="drop-zone" role="button" tabindex="0" aria-label="Choose or drop an image">
            <div class="flli-drop-icon">＋</div>
            <div class="flli-drop-title" data-role="drop-title">Drop an image here</div>
            <div class="flli-drop-help" data-role="drop-help">or click to browse</div>
            <div class="flli-upload-progress" data-role="upload-progress" hidden><span></span></div>
          </div>
        </div>

        <div class="flli-control-row" role="group" aria-label="Image output parameters">
          <span class="flli-group-label">Output</span>
          <label class="flli-param">
            <span class="flli-param-label">Resize</span>
            <select data-setting="resize_mode">
              <option value="original">Original</option>
              <option value="fit">Fit</option>
              <option value="crop">Fill / crop</option>
            </select>
          </label>
          <label class="flli-param">
            <span class="flli-param-label" data-role="width-label">Width</span>
            <input data-setting="width" type="number" min="0" max="16384" step="1">
          </label>
          <label class="flli-param">
            <span class="flli-param-label" data-role="height-label">Height</span>
            <input data-setting="height" type="number" min="0" max="16384" step="1">
          </label>
        </div>

        <div class="flli-menu" data-role="settings-menu" hidden>
          <div class="flli-menu-header">
            <span class="flli-menu-title">Image source</span>
            <button class="flli-menu-close" data-action="menu-close" type="button" aria-label="Close image source details">×</button>
          </div>
          <label class="flli-menu-field">
            <span>Comfy input image</span>
            <select data-role="input-image"></select>
          </label>
          <div class="flli-menu-help">Dropped files are copied into the ComfyUI input directory.</div>
          <div class="flli-menu-divider"></div>
          <button class="flli-button" data-action="remove" type="button">Remove selected image</button>
          <button class="flli-reset-button" data-action="reset" type="button">Reset processing settings</button>
        </div>
      </div>
    `;

    this.panel = this.container.querySelector(".flli-panel");
    this.preview = this.container.querySelector(".flli-preview");
    this.imageStage = this.container.querySelector('[data-role="image-stage"]');
    this.filename = this.container.querySelector('[data-role="filename"]');
    this.image = this.container.querySelector('[data-role="image"]');
    this.previewInfo = this.container.querySelector('[data-role="preview-info"]');
    this.status = this.container.querySelector('[data-role="status"]');
    this.browserError = this.container.querySelector('[data-role="browser-error"]');
    this.dropZone = this.container.querySelector('[data-role="drop-zone"]');
    this.dropTitle = this.container.querySelector('[data-role="drop-title"]');
    this.dropHelp = this.container.querySelector('[data-role="drop-help"]');
    this.uploadProgress = this.container.querySelector('[data-role="upload-progress"]');
    this.error = this.container.querySelector('[data-role="error"]');
    this.moreButton = this.container.querySelector('[data-action="more"]');
    this.settingsMenu = this.container.querySelector('[data-role="settings-menu"]');
    this.inputImageSelect = this.container.querySelector('[data-role="input-image"]');
    this.settingControls = [...this.container.querySelectorAll("[data-setting]")];

    this.fileInput = document.createElement("input");
    this.fileInput.type = "file";
    this.fileInput.accept = "image/*,.bmp,.gif,.jpeg,.jpg,.png,.webp";
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
      control.addEventListener(control.type === "number" ? "input" : "change", () => {
        let value = control.value;
        if (control.type === "number") {
          value = Math.trunc(Number(value));
          if (!Number.isFinite(value)) return;
        }
        this.updateSetting(control.dataset.setting, value);
      });
    }

    this.moreButton.addEventListener("click", () => this.setMenuOpen(this.settingsMenu.hidden));
    this.container.querySelector('[data-action="menu-close"]').addEventListener("click", () => this.setMenuOpen(false));
    this.container.querySelector('[data-action="reset"]').addEventListener("click", () => this.resetSettings());
    this.container.querySelector('[data-action="remove"]').addEventListener("click", () => this.removeSource());
    this.inputImageSelect.addEventListener("change", () => {
      if (this.inputImageSelect.value) this.selectSource(this.inputImageSelect.value);
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
      const file = [...(event.dataTransfer?.files || [])].find(supportedImageFile);
      if (!file) {
        this.showError("Drop a supported image file.");
        return;
      }
      this.uploadFile(file);
    });

    this.image.addEventListener("load", () => this.handleImageLoad());
    this.image.addEventListener("error", () => {
      if (!this.image.src) return;
      this.browserError.hidden = false;
      this.setStatus("error", "preview error");
    });
    this.previewResizeObserver = new ResizeObserver(() => this.applyPreviewGeometry());
    this.previewResizeObserver.observe(this.preview);
  }

  chooseFile() {
    this.fileInput.click();
  }

  updateSourceAction(hasSource) {
    const button = this.container.querySelector('[data-action="replace"]');
    button.textContent = hasSource ? "Replace" : "Choose";
    button.title = hasSource ? "Replace the selected image" : "Choose an image";
  }

  async uploadFile(file) {
    if (!supportedImageFile(file)) {
      this.showError("Choose a supported image file.");
      return;
    }

    const previousSource = this.imageWidget.value;
    this.uploading = true;
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
      this.addImageOption(path);
      this.uploading = false;
      this.selectSource(path);
    } catch (error) {
      this.uploading = false;
      if (previousSource) this.selectSource(previousSource);
      else this.removeSource(false);
      this.showError(error.message || "Image upload failed.");
    } finally {
      this.uploadProgress.hidden = true;
    }
  }

  setObjectPreview(file) {
    this.revokeObjectUrl();
    this.sourceInfo = null;
    this.objectUrl = URL.createObjectURL(file);
    this.image.src = this.objectUrl;
    this.filename.textContent = file.name;
    this.updateSourceAction(true);
  }

  addImageOption(path) {
    const values = this.imageWidget.options?.values;
    if (Array.isArray(values) && !values.includes(path)) values.push(path);
  }

  selectSource(path, markGraph = true) {
    if (!path) {
      this.removeSource(markGraph);
      return;
    }
    this.addImageOption(path);
    this.imageWidget.value = path;
    this.imageWidget.callback?.(path);
    this.legacyWidget.value = "";
    this.legacyWidget.callback?.("");
    if (markGraph) this.node.graph?.change?.();
    this.executionInfo = null;
    this.sourceInfo = null;
    this.filename.textContent = filenameFromPath(path);
    this.updateSourceAction(true);
    this.syncInputImageOptions();
    this.loadServerPreview(path);
  }

  removeSource(markGraph = true) {
    this.uploading = false;
    this.revokeObjectUrl();
    this.image.removeAttribute("src");
    this.imageWidget.value = "";
    this.imageWidget.callback?.("");
    this.legacyWidget.value = "";
    this.legacyWidget.callback?.("");
    if (markGraph) this.node.graph?.change?.();
    this.sourceInfo = null;
    this.executionInfo = null;
    this.filename.textContent = "No image selected";
    this.updateSourceAction(false);
    this.previewInfo.textContent = "Choose an image";
    this.browserError.hidden = true;
    this.dropTitle.textContent = "Drop an image here";
    this.dropHelp.textContent = "or click to browse";
    this.dropZone.hidden = false;
    this.setStatus("idle", "empty");
    this.syncInputImageOptions();
    this.applyPreviewGeometry();
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
    this.image.src = api.apiURL(`/view?${params.toString()}`);
    this.browserError.hidden = true;
    this.dropZone.hidden = true;
    this.setStatus("busy", "loading");
  }

  showLegacySource(path) {
    this.revokeObjectUrl();
    this.image.removeAttribute("src");
    this.sourceInfo = null;
    this.filename.textContent = filenameFromPath(path);
    this.updateSourceAction(true);
    this.previewInfo.textContent = "Legacy external source";
    this.dropTitle.textContent = "Legacy external image";
    this.dropHelp.textContent = "Preview unavailable. Replace to copy it into ComfyUI input.";
    this.dropZone.hidden = false;
    this.setStatus("stale", "legacy");
  }

  handleImageLoad() {
    if (this.disposed) return;
    this.sourceInfo = {
      width: this.image.naturalWidth,
      height: this.image.naturalHeight,
    };
    const path = this.imageWidget.value;
    if (path) this.node.properties.lastSourceInfo = { ...this.sourceInfo, filename: path };
    this.browserError.hidden = true;
    this.updateSourceSummary();
    this.applyPreviewGeometry();
    if (this.uploading) return;
    this.dropZone.hidden = true;
    this.setStatus("ready", "ready");
    this.clearError();
  }

  migrateLegacySource() {
    if (this.imageWidget.value || !this.legacyWidget.value) return this.imageWidget.value;
    const legacy = String(this.legacyWidget.value).replace(/\\/g, "/").toLowerCase();
    const match = this.imageOptions().find((value) => {
      const normalized = value.toLowerCase();
      return legacy === `input/${normalized}` || legacy.endsWith(`/input/${normalized}`);
    });
    if (match) {
      this.imageWidget.value = match;
      this.imageWidget.callback?.(match);
    }
    return match || "";
  }

  restoreSource() {
    this.syncInputImageOptions();
    const path = this.migrateLegacySource() || this.imageWidget.value;
    if (!path) {
      if (this.legacyWidget.value) this.showLegacySource(this.legacyWidget.value);
      else this.removeSource(false);
      if (this.configError) this.showError(this.configError);
      return;
    }

    const cached = this.node.properties.lastSourceInfo;
    if (cached?.filename === path) {
      this.sourceInfo = { width: cached.width, height: cached.height };
    }
    this.filename.textContent = filenameFromPath(path);
    this.updateSourceAction(true);
    this.loadServerPreview(path);
    this.updateSourceSummary();
    if (this.configError) this.showError(this.configError);
  }

  imageOptions() {
    const values = this.imageWidget.options?.values;
    return Array.isArray(values) ? values.filter(Boolean) : [];
  }

  syncInputImageOptions() {
    const current = this.imageWidget.value || "";
    const values = this.imageOptions();
    if (current && !values.includes(current)) values.push(current);
    values.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
    this.inputImageSelect.replaceChildren();
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = "Choose an image…";
    this.inputImageSelect.appendChild(empty);
    for (const value of values) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      this.inputImageSelect.appendChild(option);
    }
    this.inputImageSelect.value = current;
  }

  updateSetting(name, value) {
    this.settings[name] = value;
    this.settingsWidget.value = JSON.stringify(this.settings);
    this.configError = "";
    if (this.executionInfo) {
      this.executionInfo = null;
      this.setStatus("stale", "settings changed");
    }
    this.syncControls(name);
    this.updateSourceSummary();
    this.applyPreviewGeometry();
    this.node.graph?.change?.();
    this.node.setDirtyCanvas(true, true);
  }

  resetSettings() {
    this.settings = { ...DEFAULT_SETTINGS };
    this.settingsWidget.value = JSON.stringify(this.settings);
    this.configError = "";
    if (this.executionInfo) {
      this.executionInfo = null;
      this.setStatus("stale", "settings changed");
    }
    this.syncControls();
    this.updateSourceSummary();
    this.applyPreviewGeometry();
    this.node.graph?.change?.();
    this.node.setDirtyCanvas(true, true);
  }

  syncControls(changedName = null) {
    for (const control of this.settingControls) {
      const name = control.dataset.setting;
      if (changedName && name !== changedName) continue;
      control.value = this.settings[name];
    }
    this.syncOverrideControls();
  }

  isOverrideConnected(name) {
    return this.node.inputs?.some((input) => input.name === `${name}_override` && input.link != null) || false;
  }

  connectedOverrideValue(name) {
    const input = this.node.inputs?.find((candidate) => candidate.name === `${name}_override`);
    if (input?.link == null) return null;
    return this.numberFromLink(input.link, new Set());
  }

  numberFromLink(linkId, visited) {
    if (visited.has(linkId)) return null;
    visited.add(linkId);
    const graph = this.node.graph || app.graph;
    const link = graph?.links?.[linkId];
    const source = link && graph?.getNodeById?.(link.origin_id);
    if (!source) return null;

    const output = source.outputs?.[link.origin_slot];
    const outputName = String(output?.name || "").toLowerCase();
    const widgets = source.widgets || [];
    let widget = widgets.find((candidate) => String(candidate.name || "").toLowerCase() === outputName);
    const sourceType = `${source.type || ""} ${source.comfyClass || ""}`.toLowerCase();
    if (!widget && sourceType.includes("primitive")) {
      widget = widgets.find((candidate) => Number.isFinite(Number(candidate.value)));
    }
    if (widget) {
      const value = Number(widget.value);
      if (Number.isFinite(value)) {
        const integer = Math.trunc(value);
        return integer >= 0 && integer <= 16384 ? integer : null;
      }
    }

    if (sourceType.includes("reroute") && source.inputs?.[0]?.link != null) {
      return this.numberFromLink(source.inputs[0].link, visited);
    }
    return null;
  }

  refreshOverrideValues(markStale = true) {
    const next = {
      width: this.connectedOverrideValue("width"),
      height: this.connectedOverrideValue("height"),
    };
    const changed = next.width !== this.overrideValues.width || next.height !== this.overrideValues.height;
    this.overrideValues = next;
    if (!changed) return;
    if (markStale && this.executionInfo) {
      this.executionInfo = null;
      this.setStatus("stale", "inputs changed");
    }
    this.syncOverrideControls();
    this.updateSourceSummary();
    this.applyPreviewGeometry();
    this.node.setDirtyCanvas(true, true);
  }

  scheduleOverrideTracking() {
    if (this.disposed || this.overrideTimer != null || !["width", "height"].some((name) => this.isOverrideConnected(name))) return;
    this.overrideTimer = window.setTimeout(() => {
      this.overrideTimer = null;
      this.refreshOverrideValues();
      this.scheduleOverrideTracking();
    }, 100);
  }

  overrideDisplayValue(name) {
    if (this.overrideValues[name] != null) return this.overrideValues[name];
    const executed = this.executionInfo?.[`requested_${name}`];
    return Number.isInteger(executed) ? executed : null;
  }

  syncOverrideControls() {
    const originalSize = this.settings.resize_mode === "original";
    for (const name of ["width", "height"]) {
      const control = this.container.querySelector(`[data-setting="${name}"]`);
      const label = this.container.querySelector(`[data-role="${name}-label"]`);
      const connected = this.isOverrideConnected(name);
      const value = connected ? this.overrideDisplayValue(name) : null;
      control.value = value ?? this.settings[name];
      control.disabled = originalSize || connected;
      control.closest(".flli-param").dataset.overridden = String(connected);
      label.textContent = connected ? `${name[0].toUpperCase()}${name.slice(1)} · input` : `${name[0].toUpperCase()}${name.slice(1)}`;
      control.title = connected
        ? originalSize
          ? `Connected override${value == null ? "" : ` (${value})`} is ignored while Resize is Original.`
          : value == null ? "Connected value will appear after its source or this node executes." : `Connected input value: ${value}`
        : "";
    }
  }

  handleConnectionsChanged() {
    if (this.executionInfo) {
      this.executionInfo = null;
      this.setStatus("stale", "inputs changed");
    }
    this.refreshOverrideValues(false);
    this.syncOverrideControls();
    this.updateSourceSummary();
    this.applyPreviewGeometry();
    this.scheduleOverrideTracking();
    this.node.setDirtyCanvas(true, true);
  }

  effectiveResizeSettings() {
    const settings = { ...this.settings };
    for (const name of ["width", "height"]) {
      if (!this.isOverrideConnected(name)) continue;
      const value = this.overrideDisplayValue(name);
      if (value == null) return null;
      settings[name] = value;
    }
    return settings;
  }

  targetDimensions(width, height) {
    if (!width || !height || this.settings.resize_mode === "original") return { width, height };
    const settings = this.effectiveResizeSettings();
    if (!settings) return null;
    if (settings.resize_mode === "crop") {
      if (!settings.width || !settings.height) return null;
      return { width: settings.width, height: settings.height };
    }
    let scale;
    if (!settings.width && !settings.height) return null;
    if (!settings.width) scale = settings.height / height;
    else if (!settings.height) scale = settings.width / width;
    else scale = Math.min(settings.width / width, settings.height / height);
    if (!Number.isFinite(scale) || scale <= 0) return { width, height };
    return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
  }

  applyPreviewGeometry() {
    if (!this.preview || !this.imageStage) return;
    const info = this.executionInfo || this.sourceInfo;
    const sourceWidth = info?.source_width || info?.width;
    const sourceHeight = info?.source_height || info?.height;
    const target = sourceWidth && sourceHeight ? this.targetDimensions(sourceWidth, sourceHeight) : null;
    const crop = this.settings.resize_mode === "crop" && target;
    const aspectWidth = crop ? target.width : sourceWidth;
    const aspectHeight = crop ? target.height : sourceHeight;
    this.imageStage.dataset.resizeMode = crop ? "crop" : this.settings.resize_mode;
    if (!aspectWidth || !aspectHeight || !this.preview.clientWidth || !this.preview.clientHeight) {
      this.imageStage.style.width = "100%";
      this.imageStage.style.height = "100%";
      return;
    }

    const targetRatio = aspectWidth / aspectHeight;
    const previewRatio = this.preview.clientWidth / this.preview.clientHeight;
    if (targetRatio >= previewRatio) {
      this.imageStage.style.width = "100%";
      this.imageStage.style.height = `${this.preview.clientWidth / targetRatio}px`;
    } else {
      this.imageStage.style.width = `${this.preview.clientHeight * targetRatio}px`;
      this.imageStage.style.height = "100%";
    }
  }

  updateSourceSummary() {
    const info = this.executionInfo || this.sourceInfo;
    if (!info) return;
    const sourceWidth = info.source_width || info.width;
    const sourceHeight = info.source_height || info.height;
    const loadedWidth = info.loaded_width;
    const loadedHeight = info.loaded_height;
    if (loadedWidth && loadedHeight) {
      this.previewInfo.textContent = sourceWidth === loadedWidth && sourceHeight === loadedHeight
        ? `${loadedWidth}×${loadedHeight}`
        : `${sourceWidth}×${sourceHeight} → ${loadedWidth}×${loadedHeight}`;
      return;
    }
    const target = this.targetDimensions(sourceWidth, sourceHeight);
    if (!target) {
      this.previewInfo.textContent = `${sourceWidth}×${sourceHeight} → connected size`;
      return;
    }
    this.previewInfo.textContent = sourceWidth === target.width && sourceHeight === target.height
      ? `${sourceWidth}×${sourceHeight}`
      : `${sourceWidth}×${sourceHeight} → ${target.width}×${target.height}`;
  }

  updateFromExecution(message) {
    const info = message?.fl_load_image?.[0];
    if (!info) return;
    this.executionInfo = { ...info };
    this.node.properties.lastExecutionInfo = { ...info };
    this.node.properties.lastLoadSettings = this.settingsWidget.value;
    this.syncOverrideControls();
    this.updateSourceSummary();
    this.applyPreviewGeometry();
    this.setStatus("ready", "loaded");
  }

  setStatus(state, label) {
    this.status.dataset.state = state;
    this.status.textContent = label;
  }

  setMenuOpen(open) {
    this.settingsMenu.hidden = !open;
    this.moreButton.setAttribute("aria-expanded", String(open));
    if (open) this.syncInputImageOptions();
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
    hideWidget(this.rootWidget);
    hideWidget(this.legacyWidget);
    hideWidget(this.imageWidget);
    hideWidget(this.settingsWidget);
    this.readSettings();
    this.executionInfo = this.node.properties.lastExecutionInfo || null;
    this.refreshOverrideValues(false);
    this.syncControls();
    this.setMenuOpen(false);
    this.restoreSource();
    this.scheduleOverrideTracking();
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
    if (this.overrideTimer != null) window.clearTimeout(this.overrideTimer);
    this.previewResizeObserver.disconnect();
    document.removeEventListener("pointerdown", this.handleDocumentPointerDown);
    document.removeEventListener("keydown", this.handleDocumentKeyDown);
    this.revokeObjectUrl();
    this.image.removeAttribute("src");
    this.container.replaceChildren();
  }
}

app.registerExtension({
  name: "ComfyUI.FL_LoadImage",
  nodeCreated(node) {
    if (node.comfyClass !== "FL_LoadImage") return;

    const rootWidget = node.widgets?.find((widget) => widget.name === "root_directory");
    const legacyWidget = node.widgets?.find((widget) => widget.name === "selected_file");
    const imageWidget = node.widgets?.find((widget) => widget.name === "image");
    const settingsWidget = node.widgets?.find((widget) => widget.name === "load_settings");
    if (!rootWidget || !legacyWidget || !imageWidget || !settingsWidget) return;
    hideWidget(rootWidget);
    hideWidget(legacyWidget);
    hideWidget(imageWidget);
    hideWidget(settingsWidget);

    const container = document.createElement("div");
    container.className = "flli-container";
    container.style.width = "100%";
    container.style.height = "100%";
    container.style.minHeight = `${MIN_PANEL_HEIGHT}px`;
    container.style.overflow = "hidden";

    const domWidget = node.addDOMWidget("fl_load_image_panel", "fl-load-image", container, {
      getMinHeight: () => MIN_PANEL_HEIGHT,
      hideOnZoom: false,
      serialize: false,
    });
    enforceMinimumNodeSize(node);
    requestAnimationFrame(() => enforceMinimumNodeSize(node));

    const panel = new LoadImagePanel(node, rootWidget, legacyWidget, imageWidget, settingsWidget, container);

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

    const originalOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
      const result = originalOnConnectionsChange?.apply(this, args);
      panel.handleConnectionsChanged();
      return result;
    };

    domWidget.onRemove = () => panel.dispose();
  },
});
