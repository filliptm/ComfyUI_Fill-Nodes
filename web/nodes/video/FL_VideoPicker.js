import { app } from "../../../../../scripts/app.js";
import { api } from "../../../../../scripts/api.js";


let activePicker = null;


function element(tag, className, text) {
    const value = document.createElement(tag);
    if (className) value.className = className;
    if (text !== undefined) value.textContent = text;
    return value;
}


function formatTime(seconds) {
    if (!Number.isFinite(seconds)) return "0:00.0";
    const minutes = Math.floor(seconds / 60);
    return `${minutes}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
}


function installStyles() {
    if (document.getElementById("fl-video-picker-styles")) return;
    const style = element("style");
    style.id = "fl-video-picker-styles";
    style.textContent = `
        .fl-video-picker-overlay {
            position: fixed; inset: 0; z-index: 10050; display: grid; place-items: center;
            padding: 22px; background: rgba(7, 9, 14, .92); backdrop-filter: blur(12px);
            color: #eef2f5; font: 13px Inter, system-ui, sans-serif;
        }
        .fl-video-picker-shell {
            width: min(1760px, 97vw); height: min(1080px, 95vh); min-height: 520px;
            display: grid; grid-template-rows: auto auto minmax(0, 1fr) auto;
            overflow: hidden; border: 1px solid #3d4654; border-radius: 18px;
            background: linear-gradient(150deg, #202630, #15191f 55%, #111419);
            box-shadow: 0 28px 90px rgba(0, 0, 0, .6);
        }
        .fl-video-picker-header, .fl-video-picker-toolbar, .fl-video-picker-footer {
            display: flex; align-items: center; gap: 12px; padding: 15px 18px;
            border-color: #353d48; background: rgba(255, 255, 255, .025);
        }
        .fl-video-picker-header { justify-content: space-between; border-bottom: 1px solid #353d48; }
        .fl-video-picker-title { font-size: 20px; font-weight: 720; letter-spacing: -.02em; }
        .fl-video-picker-subtitle { margin-top: 4px; color: #9da9b8; }
        .fl-video-picker-timer { color: #aeb8c5; font-variant-numeric: tabular-nums; }
        .fl-video-picker-toolbar { border-bottom: 1px solid #353d48; flex-wrap: wrap; }
        .fl-video-picker-button {
            min-height: 34px; padding: 7px 13px; border: 1px solid #475261; border-radius: 9px;
            background: #29313b; color: #edf4f6; cursor: pointer; font: inherit;
        }
        .fl-video-picker-button:hover { border-color: #68d8cb; background: #323d48; }
        .fl-video-picker-button.primary { border-color: #5dd5c6; background: #319f93; color: #071715; font-weight: 700; }
        .fl-video-picker-button.danger:hover { border-color: #ef7b82; }
        .fl-video-picker-spacer { flex: 1; }
        .fl-video-picker-scrub { min-width: 180px; flex: 1 1 420px; accent-color: #5dd5c6; }
        .fl-video-picker-clock { min-width: 92px; color: #aeb8c5; font-variant-numeric: tabular-nums; text-align: right; }
        .fl-video-picker-grid {
            display: grid; grid-template-columns: repeat(var(--fl-columns), minmax(250px, 1fr));
            gap: 14px; align-content: start; overflow: auto; padding: 16px;
        }
        .fl-video-picker-card {
            position: relative; overflow: hidden; border: 2px solid #353e49; border-radius: 13px;
            align-self: start; background: #0a0c0f; cursor: pointer;
            transition: border-color .15s, transform .15s, box-shadow .15s;
        }
        .fl-video-picker-card:hover { transform: translateY(-1px); border-color: #687687; }
        .fl-video-picker-card.selected { border-color: #5dd5c6; box-shadow: 0 0 0 2px rgba(93, 213, 198, .18); }
        .fl-video-picker-card video { display: block; width: 100%; height: min(28vh, 330px); object-fit: contain; background: #050608; }
        .fl-video-picker-cardbar { display: flex; align-items: center; gap: 9px; padding: 9px 11px; color: #acb7c5; }
        .fl-video-picker-card.selected .fl-video-picker-cardbar { color: #dffffa; background: rgba(73, 184, 171, .12); }
        .fl-video-picker-cardname { color: #f3f7f8; font-weight: 700; }
        .fl-video-picker-cardmeta { flex: 1; }
        .fl-video-picker-check {
            display: grid; place-items: center; width: 22px; height: 22px; border: 1px solid #596574;
            border-radius: 7px; color: transparent; background: #1b2128; font-weight: 800;
        }
        .selected .fl-video-picker-check { color: #06221e; background: #5dd5c6; border-color: #5dd5c6; }
        .fl-video-picker-expand { position: absolute; top: 9px; right: 9px; opacity: .75; }
        .fl-video-picker-footer { justify-content: space-between; border-top: 1px solid #353d48; }
        .fl-video-picker-note { color: #909cab; max-width: 780px; line-height: 1.4; }
        .fl-video-picker-preview {
            position: fixed; inset: 0; z-index: 10060; display: grid; place-items: center;
            padding: 4vw; background: rgba(0, 0, 0, .92);
        }
        .fl-video-picker-preview video { max-width: 92vw; max-height: 86vh; background: #000; box-shadow: 0 20px 70px #000; }
        @media (max-width: 900px) {
            .fl-video-picker-grid { grid-template-columns: 1fr; }
            .fl-video-picker-note { display: none; }
        }
    `;
    document.head.appendChild(style);
}


class VideoPickerModal {
    constructor(detail) {
        this.detail = detail;
        this.selected = new Set();
        this.videos = [];
        this.cards = [];
        this.remaining = detail.timeout_seconds || 300;
        this.playing = false;
        this.frame = 0;
        this.closed = false;
        this.previewOverlay = null;
        this.build();
        this.startTimers();
    }

    build() {
        installStyles();
        this.overlay = element("div", "fl-video-picker-overlay");
        const shell = element("section", "fl-video-picker-shell");
        shell.setAttribute("role", "dialog");
        shell.setAttribute("aria-modal", "true");

        const header = element("header", "fl-video-picker-header");
        const heading = element("div");
        heading.append(element("div", "fl-video-picker-title", "Choose video slices"));
        heading.append(element(
            "div",
            "fl-video-picker-subtitle",
            `${this.detail.x_slices}×${this.detail.y_slices} spatial grid • ${this.detail.candidates.length} synchronized candidates`,
        ));
        this.timer = element("div", "fl-video-picker-timer");
        header.append(heading, this.timer);

        const toolbar = element("div", "fl-video-picker-toolbar");
        this.playButton = element("button", "fl-video-picker-button", "Play all");
        this.playButton.addEventListener("click", () => this.togglePlayback());
        const selectAll = element("button", "fl-video-picker-button", "Select all");
        selectAll.addEventListener("click", () => this.setSelection(this.detail.candidates.map(({ index }) => index)));
        const clear = element("button", "fl-video-picker-button", "Clear");
        clear.addEventListener("click", () => this.setSelection([]));
        const invert = element("button", "fl-video-picker-button", "Invert");
        invert.addEventListener("click", () => this.setSelection(
            this.detail.candidates.map(({ index }) => index).filter((index) => !this.selected.has(index)),
        ));
        this.scrub = element("input", "fl-video-picker-scrub");
        this.scrub.type = "range";
        this.scrub.min = "0";
        this.scrub.max = "1000";
        this.scrub.value = "0";
        this.scrub.addEventListener("input", () => this.seek(Number(this.scrub.value) / 1000));
        this.clock = element("div", "fl-video-picker-clock", "0:00.0 / 0:00.0");
        toolbar.append(this.playButton, selectAll, clear, invert, element("div", "fl-video-picker-spacer"), this.scrub, this.clock);

        const grid = element("main", "fl-video-picker-grid");
        grid.style.setProperty("--fl-columns", String(Math.min(this.detail.x_slices, 4)));
        for (const candidate of this.detail.candidates) grid.append(this.createCard(candidate));

        const footer = element("footer", "fl-video-picker-footer");
        this.summary = element("div", "fl-video-picker-note");
        const actions = element("div");
        actions.style.display = "flex";
        actions.style.gap = "10px";
        const cancel = element("button", "fl-video-picker-button danger", "Cancel render");
        cancel.addEventListener("click", () => this.submit(true));
        this.confirm = element("button", "fl-video-picker-button primary");
        this.confirm.addEventListener("click", () => this.submit(false));
        actions.append(cancel, this.confirm);
        footer.append(this.summary, actions);

        shell.append(header, toolbar, grid, footer);
        this.overlay.append(shell);
        document.body.append(this.overlay);
        this.keyHandler = (event) => {
            if (event.key === "Escape" && this.previewOverlay) this.closePreview();
            else if (event.key === "Escape") this.submit(true);
            if (event.key === " " && !["BUTTON", "INPUT"].includes(event.target?.tagName)) {
                event.preventDefault();
                this.togglePlayback();
            }
        };
        document.addEventListener("keydown", this.keyHandler);
        this.updateSelection();
    }

    createCard(candidate) {
        const card = element("article", "fl-video-picker-card");
        card.dataset.index = String(candidate.index);
        card.tabIndex = 0;
        card.setAttribute("role", "button");
        card.setAttribute("aria-label", `Select slice ${candidate.index + 1}, row ${candidate.row + 1}, column ${candidate.column + 1}`);
        const video = element("video");
        video.src = api.apiURL(candidate.preview_url);
        video.loop = true;
        video.muted = true;
        video.playsInline = true;
        video.preload = "metadata";
        video.addEventListener("loadedmetadata", () => this.updateClock());
        video.addEventListener("error", () => card.classList.add("error"));

        const expand = element("button", "fl-video-picker-button fl-video-picker-expand", "Expand");
        expand.addEventListener("click", (event) => {
            event.stopPropagation();
            this.openPreview(candidate, video.currentTime);
        });
        const bar = element("div", "fl-video-picker-cardbar");
        bar.append(
            element("div", "fl-video-picker-check", "✓"),
            element("div", "fl-video-picker-cardname", `Slice ${candidate.index + 1}`),
            element("div", "fl-video-picker-cardmeta", `row ${candidate.row + 1}, col ${candidate.column + 1} • ${candidate.width}×${candidate.height} • ${candidate.frame_count}f`),
        );
        card.append(video, expand, bar);
        card.addEventListener("click", () => this.toggleSelection(candidate.index));
        card.addEventListener("keydown", (event) => {
            if (!["Enter", " "].includes(event.key)) return;
            event.preventDefault();
            event.stopPropagation();
            this.toggleSelection(candidate.index);
        });
        this.videos.push(video);
        this.cards[candidate.index] = card;
        return card;
    }

    toggleSelection(index) {
        if (this.selected.has(index)) this.selected.delete(index);
        else this.selected.add(index);
        this.updateSelection();
    }

    setSelection(indices) {
        this.selected = new Set(indices);
        this.updateSelection();
    }

    updateSelection() {
        for (const [index, card] of this.cards.entries()) {
            const selected = this.selected.has(index);
            card?.classList.toggle("selected", selected);
            card?.setAttribute("aria-pressed", String(selected));
        }
        const count = this.selected.size;
        this.summary.textContent = count
            ? `${count} clip${count === 1 ? "" : "s"} selected. The output is a video list; keep the decoded H3 audio wired around this node.`
            : "Nothing selected: confirming returns every slice. Audio is intentionally routed around this visual picker.";
        this.confirm.textContent = count ? `Keep ${count} selected` : `Keep all ${this.detail.candidates.length}`;
    }

    duration() {
        return this.videos.find((video) => Number.isFinite(video.duration))?.duration || 0;
    }

    seek(ratio) {
        const duration = this.duration();
        if (!duration) return;
        const time = Math.max(0, Math.min(duration, ratio * duration));
        for (const video of this.videos) video.currentTime = Math.min(time, video.duration || time);
        this.updateClock();
    }

    async togglePlayback() {
        if (this.playing) {
            this.pauseAll();
            return;
        }
        this.playing = true;
        this.playButton.textContent = "Pause all";
        await Promise.allSettled(this.videos.map((video) => video.play()));
        this.animate();
    }

    pauseAll() {
        this.playing = false;
        this.playButton.textContent = "Play all";
        cancelAnimationFrame(this.frame);
        for (const video of this.videos) video.pause();
    }

    animate() {
        if (!this.playing || this.closed) return;
        const leader = this.videos[0];
        if (leader) {
            for (const video of this.videos.slice(1)) {
                if (Math.abs(video.currentTime - leader.currentTime) > 0.08) video.currentTime = leader.currentTime;
            }
        }
        this.updateClock();
        this.frame = requestAnimationFrame(() => this.animate());
    }

    updateClock() {
        const duration = this.duration();
        const current = this.videos[0]?.currentTime || 0;
        this.scrub.value = duration ? String(Math.round(current / duration * 1000)) : "0";
        this.clock.textContent = `${formatTime(current)} / ${formatTime(duration)}`;
    }

    openPreview(candidate, currentTime) {
        this.closePreview();
        const overlay = element("div", "fl-video-picker-preview");
        const video = element("video");
        video.src = api.apiURL(candidate.preview_url);
        video.controls = true;
        video.autoplay = this.playing;
        video.loop = true;
        video.muted = true;
        video.playsInline = true;
        video.addEventListener("loadedmetadata", () => { video.currentTime = Math.min(currentTime, video.duration || currentTime); });
        overlay.append(video);
        const close = () => {
            video.pause();
            video.removeAttribute("src");
            video.load();
            overlay.remove();
            if (this.previewOverlay === overlay) this.previewOverlay = null;
        };
        overlay.addEventListener("click", (event) => { if (event.target === overlay) close(); });
        document.body.append(overlay);
        overlay.closePickerPreview = close;
        this.previewOverlay = overlay;
    }

    closePreview() {
        this.previewOverlay?.closePickerPreview();
    }

    startTimers() {
        const renderTimer = () => { this.timer.textContent = `Render paused • ${formatTime(this.remaining)} remaining`; };
        renderTimer();
        this.countdown = window.setInterval(() => {
            this.remaining -= 1;
            renderTimer();
            if (this.remaining <= 0) this.submit(true);
        }, 1000);
        this.statusPoll = window.setInterval(async () => {
            try {
                const response = await api.fetchApi(`/fl_video_picker/status/${encodeURIComponent(this.detail.session_id)}`);
                const status = await response.json();
                if (status.status !== "active") this.close();
            } catch (_) {
                // A transient frontend connection error should not discard the selection dialog.
            }
        }, 3000);
    }

    async submit(cancelled) {
        if (this.closed) return;
        const selection = [...this.selected].sort((a, b) => a - b);
        const sessionId = this.detail.session_id;
        this.close();
        try {
            await api.fetchApi("/fl_video_picker/select", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId, selection, cancelled }),
            });
        } catch (error) {
            console.error("[FL Video Picker] Could not submit selection", error);
        }
    }

    close() {
        if (this.closed) return;
        this.closed = true;
        this.pauseAll();
        clearInterval(this.countdown);
        clearInterval(this.statusPoll);
        document.removeEventListener("keydown", this.keyHandler);
        this.closePreview();
        for (const video of this.videos) {
            video.removeAttribute("src");
            video.load();
        }
        this.overlay.remove();
        if (activePicker === this) activePicker = null;
    }
}


api.addEventListener("fl_video_picker_show", (event) => {
    if (activePicker) void activePicker.submit(true);
    activePicker = new VideoPickerModal(event.detail);
});


app.registerExtension({
    name: "FillNodes.VideoPicker",
    async nodeCreated(node) {
        if (node.comfyClass !== "FL_VideoPicker") return;
        const width = Math.max(node.size?.[0] || 0, 360);
        node.setSize([width, node.computeSize([width, 0])[1]]);
    },
});
