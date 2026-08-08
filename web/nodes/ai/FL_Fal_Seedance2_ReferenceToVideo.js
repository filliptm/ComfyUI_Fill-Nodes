import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const EVENT_NAME = "fl_fal_seedance2_progress";
const NODE_CLASS = "FL_Fal_Seedance2_ReferenceToVideo";
const INSTANCES = new Map();

const STYLES = `
  .fl-seedance2 {
    --accent: #a78bfa;
    --accent-strong: #7c3aed;
    --bg: #121318;
    --panel: #1a1c23;
    --border: #30333d;
    --muted: #9ca3af;
    --text: #f4f4f5;
    background:
      radial-gradient(circle at top right, rgba(124, 58, 237, .18), transparent 42%),
      var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    display: flex;
    flex-direction: column;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    gap: 9px;
    min-height: 288px;
    overflow: hidden;
    padding: 11px;
    width: 100%;
  }
  .fl-seedance2 * { box-sizing: border-box; }
  .fl-seedance2__header {
    align-items: center;
    display: flex;
    gap: 8px;
    justify-content: space-between;
  }
  .fl-seedance2__eyebrow {
    color: #c4b5fd;
    font-size: 9px;
    font-weight: 750;
    letter-spacing: .12em;
    line-height: 1;
    text-transform: uppercase;
  }
  .fl-seedance2__title {
    font-size: 12px;
    font-weight: 700;
    line-height: 1.2;
    margin-top: 3px;
  }
  .fl-seedance2__badge {
    background: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 999px;
    color: #d4d4d8;
    font-size: 9px;
    font-weight: 750;
    padding: 4px 8px;
    text-transform: uppercase;
    white-space: nowrap;
  }
  .fl-seedance2[data-phase="preparing"] .fl-seedance2__badge,
  .fl-seedance2[data-phase="uploading"] .fl-seedance2__badge,
  .fl-seedance2[data-phase="queued"] .fl-seedance2__badge,
  .fl-seedance2[data-phase="generating"] .fl-seedance2__badge,
  .fl-seedance2[data-phase="downloading"] .fl-seedance2__badge {
    background: rgba(124, 58, 237, .22);
    border-color: rgba(167, 139, 250, .55);
    color: #ddd6fe;
  }
  .fl-seedance2[data-phase="complete"] .fl-seedance2__badge {
    background: rgba(22, 163, 74, .18);
    border-color: rgba(74, 222, 128, .45);
    color: #86efac;
  }
  .fl-seedance2[data-phase="error"] .fl-seedance2__badge {
    background: rgba(220, 38, 38, .18);
    border-color: rgba(248, 113, 113, .5);
    color: #fca5a5;
  }
  .fl-seedance2[data-phase="cancelled"] .fl-seedance2__badge {
    background: rgba(217, 119, 6, .16);
    border-color: rgba(251, 191, 36, .45);
    color: #fcd34d;
  }
  .fl-seedance2__section {
    background: rgba(26, 28, 35, .8);
    border: 1px solid rgba(63, 63, 70, .72);
    border-radius: 8px;
    padding: 8px;
  }
  .fl-seedance2__label {
    color: var(--muted);
    font-size: 8px;
    font-weight: 750;
    letter-spacing: .1em;
    margin-bottom: 6px;
    text-transform: uppercase;
  }
  .fl-seedance2__chips {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    min-height: 20px;
  }
  .fl-seedance2__chip {
    background: #242630;
    border: 1px solid #3b3e49;
    border-radius: 5px;
    color: #ddd6fe;
    cursor: pointer;
    font: inherit;
    font-size: 9px;
    font-weight: 650;
    padding: 3px 6px;
  }
  .fl-seedance2__chip:hover {
    background: #312e48;
    border-color: #7c3aed;
  }
  .fl-seedance2__empty {
    color: #71717a;
    font-size: 9px;
    line-height: 20px;
  }
  .fl-seedance2__settings {
    color: #d4d4d8;
    font-size: 9px;
    line-height: 1.35;
  }
  .fl-seedance2__progress {
    background: #292b34;
    border-radius: 999px;
    height: 7px;
    overflow: hidden;
    position: relative;
  }
  .fl-seedance2__progress-fill {
    background: linear-gradient(90deg, var(--accent-strong), #c084fc);
    border-radius: inherit;
    height: 100%;
    transition: width 150ms ease;
    width: 0%;
  }
  .fl-seedance2__progress.is-indeterminate .fl-seedance2__progress-fill {
    animation: fl-seedance2-indeterminate 1.25s ease-in-out infinite;
    width: 38%;
  }
  @keyframes fl-seedance2-indeterminate {
    from { transform: translateX(-110%); }
    to { transform: translateX(300%); }
  }
  .fl-seedance2__meta {
    color: #a1a1aa;
    display: flex;
    font-size: 9px;
    gap: 8px;
    justify-content: space-between;
    margin-top: 6px;
  }
  .fl-seedance2__message {
    color: #d4d4d8;
    font-size: 9px;
    line-height: 1.35;
    margin-top: 6px;
    max-height: 38px;
    overflow: hidden;
    overflow-wrap: anywhere;
  }
  .fl-seedance2[data-phase="error"] .fl-seedance2__message { color: #fecaca; }
  .fl-seedance2__result[hidden] { display: none; }
  .fl-seedance2__video {
    background: #09090b;
    border-radius: 6px;
    display: block;
    max-height: 160px;
    width: 100%;
  }
  .fl-seedance2__result-meta {
    color: #a1a1aa;
    font-size: 9px;
    margin-top: 6px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .fl-seedance2__actions {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 7px;
  }
  .fl-seedance2__action {
    background: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 5px;
    color: #e4e4e7;
    cursor: pointer;
    font: inherit;
    font-size: 9px;
    font-weight: 650;
    padding: 4px 7px;
  }
  .fl-seedance2__action:hover {
    background: #3f3f46;
    border-color: #71717a;
  }
`;

function injectStyles() {
  const id = "fl-seedance2-reference-styles";
  if (document.getElementById(id)) return;
  const style = document.createElement("style");
  style.id = id;
  style.textContent = STYLES;
  document.head.appendChild(style);
}

function nodeKey(value) {
  return value == null ? "" : String(value);
}

class SeedanceDashboard {
  constructor(node, container) {
    injectStyles();
    this.node = node;
    this.container = container;
    this.phase = "idle";
    this.startedAt = null;
    this.videoUrl = "";
    this.requestId = "";

    this.element = document.createElement("div");
    this.element.className = "fl-seedance2";
    this.element.dataset.phase = "idle";
    this.element.innerHTML = `
      <div class="fl-seedance2__header">
        <div>
          <div class="fl-seedance2__eyebrow">Fal queue</div>
          <div class="fl-seedance2__title">Seedance 2.0 Reference Studio</div>
        </div>
        <span class="fl-seedance2__badge" data-role="badge">Idle</span>
      </div>
      <div class="fl-seedance2__section">
        <div class="fl-seedance2__label">Connected references · click a tag to insert</div>
        <div class="fl-seedance2__chips" data-role="chips"></div>
      </div>
      <div class="fl-seedance2__section">
        <div class="fl-seedance2__label">Generation setup</div>
        <div class="fl-seedance2__settings" data-role="settings"></div>
      </div>
      <div class="fl-seedance2__section">
        <div class="fl-seedance2__progress" data-role="progress">
          <div class="fl-seedance2__progress-fill" data-role="fill"></div>
        </div>
        <div class="fl-seedance2__meta">
          <span data-role="stage">Ready</span>
          <span data-role="elapsed">00:00</span>
        </div>
        <div class="fl-seedance2__message" data-role="message">
          Connect references or use a prompt by itself, then queue the workflow.
        </div>
      </div>
      <div class="fl-seedance2__section fl-seedance2__result" data-role="result" hidden>
        <div class="fl-seedance2__label">Generated video</div>
        <video class="fl-seedance2__video" data-role="video" controls preload="metadata"></video>
        <div class="fl-seedance2__result-meta" data-role="result-meta"></div>
        <div class="fl-seedance2__actions">
          <button class="fl-seedance2__action" type="button" data-action="open">Open video</button>
          <button class="fl-seedance2__action" type="button" data-action="copy-url">Copy URL</button>
          <button class="fl-seedance2__action" type="button" data-action="copy-request">Copy request ID</button>
        </div>
      </div>
    `;
    container.appendChild(this.element);

    this.badgeEl = this.element.querySelector('[data-role="badge"]');
    this.chipsEl = this.element.querySelector('[data-role="chips"]');
    this.settingsEl = this.element.querySelector('[data-role="settings"]');
    this.progressEl = this.element.querySelector('[data-role="progress"]');
    this.fillEl = this.element.querySelector('[data-role="fill"]');
    this.stageEl = this.element.querySelector('[data-role="stage"]');
    this.elapsedEl = this.element.querySelector('[data-role="elapsed"]');
    this.messageEl = this.element.querySelector('[data-role="message"]');
    this.resultEl = this.element.querySelector('[data-role="result"]');
    this.videoEl = this.element.querySelector('[data-role="video"]');
    this.resultMetaEl = this.element.querySelector('[data-role="result-meta"]');

    this.onClick = (event) => this.handleClick(event);
    this.element.addEventListener("click", this.onClick);
    this.timer = window.setInterval(() => this.refresh(), 500);
    this.refresh();
  }

  getWidgetValue(name, fallback = "—") {
    const widget = this.node.widgets?.find((item) => item.name === name);
    const value = widget?.value;
    return value === undefined || value === null || value === "" ? fallback : value;
  }

  connectedTags() {
    const tags = [];
    for (const input of this.node.inputs || []) {
      if (input.link == null) continue;
      const match = /(?:^|\.)(image|video|audio)_(\d+)$/i.exec(input.name || "");
      if (!match) continue;
      const kind = match[1][0].toUpperCase() + match[1].slice(1).toLowerCase();
      tags.push({
        kind,
        order: { Image: 0, Video: 1, Audio: 2 }[kind] * 100 + Number(match[2]),
      });
    }
    const counters = { Image: 0, Video: 0, Audio: 0 };
    return tags.sort((a, b) => a.order - b.order).map((item) => ({
      tag: `@${item.kind}${++counters[item.kind]}`,
      order: item.order,
    }));
  }

  refresh() {
    const tags = this.connectedTags();
    const signature = tags.map((item) => item.tag).join(",");
    if (signature !== this.tagSignature) {
      this.tagSignature = signature;
      this.renderTags(tags);
    }

    const resolution = this.getWidgetValue("resolution");
    const duration = this.getWidgetValue("duration");
    const aspectRatio = this.getWidgetValue("aspect_ratio");
    const bitrate = this.getWidgetValue("bitrate_mode");
    const audio = this.getWidgetValue("generate_audio", true) ? "audio on" : "audio off";
    this.settingsEl.textContent =
      `${resolution} · ${duration === "auto" ? "auto duration" : `${duration}s`} · ` +
      `${aspectRatio} · ${bitrate} bitrate · ${audio}`;

    if (this.startedAt && !["complete", "error", "cancelled"].includes(this.phase)) {
      this.elapsedEl.textContent = this.formatElapsed(Date.now() - this.startedAt);
    }
  }

  renderTags(tags) {
    this.chipsEl.replaceChildren();
    if (!tags.length) {
      const empty = document.createElement("span");
      empty.className = "fl-seedance2__empty";
      empty.textContent = "No references connected — prompt-only generation is supported.";
      this.chipsEl.appendChild(empty);
      return;
    }
    for (const { tag } of tags) {
      const button = document.createElement("button");
      button.className = "fl-seedance2__chip";
      button.type = "button";
      button.dataset.tag = tag;
      button.textContent = tag;
      button.title = `Insert ${tag} into the prompt`;
      this.chipsEl.appendChild(button);
    }
  }

  insertTag(tag) {
    const prompt = this.node.widgets?.find((item) => item.name === "prompt");
    if (!prompt) return;
    const current = String(prompt.value || "");
    const separator = current && !/\s$/.test(current) ? " " : "";
    prompt.value = `${current}${separator}${tag} `;
    prompt.callback?.(prompt.value);
    app.graph?.setDirtyCanvas(true, true);
  }

  handleClick(event) {
    const tag = event.target.closest("[data-tag]")?.dataset.tag;
    if (tag) {
      this.insertTag(tag);
      return;
    }

    const action = event.target.closest("[data-action]")?.dataset.action;
    if (action === "open" && this.videoUrl) {
      window.open(this.videoUrl, "_blank", "noopener,noreferrer");
    } else if (action === "copy-url" && this.videoUrl) {
      this.copy(this.videoUrl, "Video URL copied.");
    } else if (action === "copy-request" && this.requestId) {
      this.copy(this.requestId, "Request ID copied.");
    }
  }

  async copy(value, message) {
    try {
      await navigator.clipboard.writeText(value);
      this.messageEl.textContent = message;
    } catch {
      this.messageEl.textContent = "Clipboard access was unavailable.";
    }
  }

  formatElapsed(milliseconds) {
    const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }

  setPhase(phase) {
    this.phase = phase;
    this.element.dataset.phase = phase;
    this.badgeEl.textContent = phase === "idle" ? "Idle" : phase;
  }

  setProgress(value, max, indeterminate = false) {
    const hasRange = Number(max) > 0;
    const percent = hasRange ? Math.max(0, Math.min(100, Number(value || 0) / Number(max) * 100)) : 0;
    this.progressEl.classList.toggle("is-indeterminate", indeterminate);
    this.fillEl.style.width = indeterminate ? "38%" : `${percent}%`;
  }

  reset() {
    this.startedAt = Date.now();
    this.videoUrl = "";
    this.requestId = "";
    this.setPhase("preparing");
    this.setProgress(0, 1);
    this.stageEl.textContent = "Starting";
    this.elapsedEl.textContent = "00:00";
    this.messageEl.textContent = "Validating references and preparing media…";
    this.videoEl.removeAttribute("src");
    this.videoEl.load();
    this.resultEl.hidden = true;
  }

  update(detail) {
    const phase = detail.phase || "idle";
    if (!this.startedAt && phase !== "idle") this.startedAt = Date.now();
    if (detail.request_id) this.requestId = detail.request_id;
    this.setPhase(phase);

    if (phase === "preparing") {
      this.setProgress(detail.value, detail.max);
      this.stageEl.textContent = "Preparing references";
      this.messageEl.textContent = "Encoding and validating local reference media…";
    } else if (phase === "uploading") {
      this.setProgress(detail.value, detail.max);
      this.stageEl.textContent = "Uploading references";
      this.messageEl.textContent = `${detail.value || 0} of ${detail.max || 0} reference files uploaded.`;
    } else if (phase === "queued") {
      this.setProgress(0, 0, true);
      this.stageEl.textContent =
        detail.queue_position == null ? "Waiting in Fal queue" : `Queue position ${detail.queue_position}`;
      this.messageEl.textContent = this.requestId
        ? `Request ${this.requestId}`
        : "Request submitted to Fal.";
    } else if (phase === "generating") {
      this.setProgress(0, 0, true);
      this.stageEl.textContent = "Seedance is generating";
      this.messageEl.textContent = detail.log || "Fal is processing the video…";
    } else if (phase === "downloading") {
      this.setProgress(detail.value, detail.max, !Number(detail.max));
      this.stageEl.textContent = "Downloading result";
      this.messageEl.textContent = detail.max
        ? `${(Number(detail.value || 0) / 1048576).toFixed(1)} / ${(Number(detail.max) / 1048576).toFixed(1)} MiB`
        : "Streaming the generated video into ComfyUI…";
    } else if (phase === "complete") {
      this.setProgress(1, 1);
      this.stageEl.textContent = "Complete";
      this.videoUrl = detail.video_url || "";
      this.requestId = detail.request_id || this.requestId;
      this.messageEl.textContent = `Generation complete · seed ${detail.seed}`;
      this.elapsedEl.textContent = this.startedAt
        ? this.formatElapsed(Date.now() - this.startedAt)
        : "00:00";
      this.showResult(detail);
    } else if (phase === "error") {
      this.setProgress(0, 1);
      this.stageEl.textContent = "Generation failed";
      this.messageEl.textContent = detail.message || "Fal returned an error.";
      this.elapsedEl.textContent = this.startedAt
        ? this.formatElapsed(Date.now() - this.startedAt)
        : "00:00";
    } else if (phase === "cancelled") {
      this.setProgress(0, 1);
      this.stageEl.textContent = "Cancelled";
      this.messageEl.textContent = "The Fal request was cancelled after ComfyUI interrupted execution.";
      this.elapsedEl.textContent = this.startedAt
        ? this.formatElapsed(Date.now() - this.startedAt)
        : "00:00";
    }
  }

  showResult(detail) {
    if (!this.videoUrl) return;
    this.videoEl.src = this.videoUrl;
    this.resultMetaEl.textContent =
      `seed ${detail.seed} · request ${this.requestId || "—"}`;
    this.resultEl.hidden = false;
    const computedSize = this.node.computeSize();
    this.node.setSize([
      Math.max(this.node.size[0], 430),
      Math.max(this.node.size[1], computedSize[1], 950),
    ]);
    app.graph?.setDirtyCanvas(true, true);
  }

  dispose() {
    window.clearInterval(this.timer);
    this.element.removeEventListener("click", this.onClick);
    this.videoEl.removeAttribute("src");
    this.videoEl.load();
    this.element.remove();
  }
}

function removeInstance(node) {
  const key = nodeKey(node.id);
  const instance = INSTANCES.get(key);
  if (instance) instance.dispose();
  INSTANCES.delete(key);
}

app.registerExtension({
  name: "ComfyUI.FL_Fal_Seedance2_ReferenceToVideo",
  nodeCreated(node) {
    const comfyClass = node.constructor?.comfyClass || "";
    if (comfyClass !== NODE_CLASS) return;

    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.minHeight = "288px";

    const widget = node.addDOMWidget(
      "seedance2_dashboard",
      "fl-seedance2-dashboard",
      container,
      {
        getMinHeight: () => Math.max(310, container.scrollHeight),
        hideOnZoom: false,
        serialize: false,
      },
    );

    node.setSize([
      Math.max(node.size[0], 430),
      Math.max(node.size[1], 760),
    ]);

    window.setTimeout(() => {
      removeInstance(node);
      INSTANCES.set(nodeKey(node.id), new SeedanceDashboard(node, container));
    }, 0);

    widget.onRemove = () => removeInstance(node);
    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
      removeInstance(this);
      return originalOnRemoved?.apply(this, arguments);
    };
  },
});

api.addEventListener("executing", (event) => {
  const detail = event.detail;
  const activeNode = detail && typeof detail === "object" ? detail.node : detail;
  const instance = INSTANCES.get(nodeKey(activeNode));
  instance?.reset();
});

api.addEventListener(EVENT_NAME, (event) => {
  const detail = event.detail;
  if (!detail) return;
  INSTANCES.get(nodeKey(detail.node))?.update(detail);
});
