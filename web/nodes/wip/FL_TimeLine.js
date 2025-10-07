import { app } from "../../../../scripts/app.js";
import { ComfyWidgets } from "../../../../scripts/widgets.js";

app.registerExtension({
    name: "Comfy.FL_Timeline",
    async nodeCreated(node) {
        if (node.comfyClass === "FL_TimeLine") {
            new FL_TimelineNode(node);
        }
    }
});

class FL_TimelineNode {
    constructor(node) {
        this.node = node;
        this.canvas = null;
        this.ctx = null;

        // Constants for UI layout
        this.UI_OFFSET_X = 0;
        this.UI_OFFSET_Y = 50;
        this.UI_SCALE = 1;
        this.PADDING = 10 * this.UI_SCALE;
        this.WIDGET_HEIGHT = 20 * this.UI_SCALE;
        this.TIMELINE_HEIGHT = 100 * this.UI_SCALE;
        this.MARGIN = 7 * this.UI_SCALE;
        this.ROW_HEIGHT = 30 * this.UI_SCALE;
        this.HANDLE_WIDTH = 10 * this.UI_SCALE;

        // Timeline data structure
        this.timelineData = [];
        this.selectedRow = null;
        this.isDragging = false;
        this.dragType = null; // 'start', 'end', or 'move'

        this.setupProperties();
        this.setupWidgets();
        this.createUI();
        this.setupEventListeners();

        this.node.serialize = this.serialize.bind(this);
        this.node.onConfigure = this.onConfigure.bind(this);
    }

    setupProperties() {
        this.node.properties = this.node.properties || {};
        this.node.properties.ipadapter_preset = this.node.properties.ipadapter_preset || "LIGHT - SD1.5 only (low strength)";
        this.node.properties.video_width = this.node.properties.video_width || 512;
        this.node.properties.video_height = this.node.properties.video_height || 512;
        this.node.properties.interpolation_mode = this.node.properties.interpolation_mode || "Linear";
        this.node.properties.number_animation_frames = this.node.properties.number_animation_frames || 96;
        this.node.properties.frames_per_second = this.node.properties.frames_per_second || 12;
        this.node.properties.time_format = this.node.properties.time_format || "Frames";
    }

    setupWidgets() {
        this.widgets = {};
        this.widgets.model = this.node.addWidget("model", "model", null, (v) => {}, { serialize: false });
        this.widgets.ipadapter_preset = this.node.addWidget("combo", "ipadapter_preset", this.node.properties.ipadapter_preset, (v) => { this.node.properties.ipadapter_preset = v; }, { values: ["LIGHT - SD1.5 only (low strength)", "STANDARD (medium strength)", "VIT-G (medium strength)", "PLUS (high strength)", "PLUS FACE (portraits)", "FULL FACE - SD1.5 only (portraits stronger)"] });
        this.widgets.video_width = this.node.addWidget("number", "video_width", this.node.properties.video_width, (v) => { this.node.properties.video_width = v; }, { min: 64, max: 2048, step: 8 });
        this.widgets.video_height = this.node.addWidget("number", "video_height", this.node.properties.video_height, (v) => { this.node.properties.video_height = v; }, { min: 64, max: 2048, step: 8 });
        this.widgets.interpolation_mode = this.node.addWidget("combo", "interpolation_mode", this.node.properties.interpolation_mode, (v) => { this.node.properties.interpolation_mode = v; }, { values: ["Linear", "Ease_in", "Ease_out", "Ease_in_out"] });
        this.widgets.number_animation_frames = this.node.addWidget("number", "number_animation_frames", this.node.properties.number_animation_frames, (v) => {
            this.node.properties.number_animation_frames = v;
            this.updateTimeRuler();
        }, { min: 1, max: 1000, step: 1 });
        this.widgets.frames_per_second = this.node.addWidget("number", "frames_per_second", this.node.properties.frames_per_second, (v) => {
            this.node.properties.frames_per_second = v;
            this.updateTimeRuler();
        }, { min: 1, max: 60, step: 1 });
        this.widgets.time_format = this.node.addWidget("combo", "time_format", this.node.properties.time_format, (v) => {
            this.node.properties.time_format = v;
            this.updateTimeRuler();
        }, { values: ["Frames", "Seconds"] });

        this.widgets.addRow = this.node.addWidget("button", "Add Row", null, () => this.addTimelineRow());
        this.widgets.removeRow = this.node.addWidget("button", "Remove Row", null, () => this.removeTimelineRow());
        this.widgets.uploadImage = this.node.addWidget("file", "Upload Image", "", (file) => this.handleImageUpload(file));
    }

    createUI() {
        this.canvas = document.createElement("canvas");
        this.ctx = this.canvas.getContext("2d");
        this.node.addCustomWidget({
            name: "timeline",
            type: "FL_Timeline",
            callback: () => {},
            draw: (ctx, node, width, height) => this.drawTimeline(ctx, width, height),
        });
    }

    setupEventListeners() {
        const orig_onMouseDown = this.node.onMouseDown;
        this.node.onMouseDown = (e) => {
            orig_onMouseDown?.call(this.node, e);
            this.onMouseDown(e);
        };

        const orig_onMouseMove = this.node.onMouseMove;
        this.node.onMouseMove = (e) => {
            orig_onMouseMove?.call(this.node, e);
            this.onMouseMove(e);
        };

        const orig_onMouseUp = this.node.onMouseUp;
        this.node.onMouseUp = (e) => {
            orig_onMouseUp?.call(this.node, e);
            this.onMouseUp(e);
        };
    }

    drawTimeline(ctx, width, height) {
        const timeRulerHeight = 20;
        this.drawTimeRuler(ctx, 0, 0, width, timeRulerHeight);
        this.drawTimelineRows(ctx, 0, timeRulerHeight, width, height - timeRulerHeight);
    }

    drawTimeRuler(ctx, x, y, width, height) {
        ctx.fillStyle = "#2a2a2a";
        ctx.fillRect(x, y, width, height);

        const totalFrames = this.node.properties.number_animation_frames;
        const framesPerSecond = this.node.properties.frames_per_second;
        const isSeconds = this.node.properties.time_format === "Seconds";

        const totalMarkers = isSeconds ? Math.ceil(totalFrames / framesPerSecond) * 10 : totalFrames;
        const majorTickInterval = isSeconds ? 10 : Math.floor(totalFrames / 10);

        for (let i = 0; i <= totalMarkers; i++) {
            const markerX = x + (i / totalMarkers) * width;
            const isMajorTick = i % majorTickInterval === 0;

            ctx.strokeStyle = isMajorTick ? "#fff" : "#666";
            ctx.beginPath();
            ctx.moveTo(markerX, y + height);
            ctx.lineTo(markerX, y + height - (isMajorTick ? height : height / 2));
            ctx.stroke();

            if (isMajorTick) {
                ctx.fillStyle = "#fff";
                ctx.textAlign = "center";
                ctx.textBaseline = "top";
                const label = isSeconds ? (i / 10).toFixed(1) + "s" : i.toString();
                ctx.fillText(label, markerX, y);
            }
        }
    }

    drawTimelineRows(ctx, x, y, width, height) {
        ctx.fillStyle = "#2a2a2a";
        ctx.fillRect(x, y, width, height);

        const totalFrames = this.node.properties.number_animation_frames;

        this.timelineData.forEach((row, index) => {
            const rowY = y + index * this.ROW_HEIGHT;
            const startX = x + (row.start / totalFrames) * width;
            const endX = x + (row.end / totalFrames) * width;

            // Draw row background
            ctx.fillStyle = index === this.selectedRow ? "#3a3a3a" : "#4a4a4a";
            ctx.fillRect(startX, rowY, endX - startX, this.ROW_HEIGHT);

            // Draw handles
            ctx.fillStyle = "#666";
            ctx.fillRect(startX, rowY, this.HANDLE_WIDTH, this.ROW_HEIGHT);
            ctx.fillRect(endX - this.HANDLE_WIDTH, rowY, this.HANDLE_WIDTH, this.ROW_HEIGHT);

            // Draw row label
            ctx.fillStyle = "white";
            ctx.font = "12px Arial";
            ctx.fillText(`Row ${index + 1}`, x + 5, rowY + this.ROW_HEIGHT / 2 + 4);

            // Draw frame numbers
            ctx.fillStyle = "#aaa";
            ctx.font = "10px Arial";
            ctx.fillText(`${row.start}`, startX + 2, rowY + this.ROW_HEIGHT - 2);
            ctx.fillText(`${row.end}`, endX - 30, rowY + this.ROW_HEIGHT - 2);

            // Draw image thumbnail if available
            if (row.image) {
                const img = new Image();
                img.src = row.image;
                ctx.drawImage(img, endX - this.ROW_HEIGHT, rowY, this.ROW_HEIGHT, this.ROW_HEIGHT);
            }
        });
    }

    updateTimeRuler() {
        this.node.setDirtyCanvas(true);
    }

    addTimelineRow() {
        const totalFrames = this.node.properties.number_animation_frames;
        this.timelineData.push({ start: 0, end: totalFrames, image: null });
        this.updateTimelineOutput();
    }

    removeTimelineRow() {
        if (this.selectedRow !== null && this.timelineData.length > 1) {
            this.timelineData.splice(this.selectedRow, 1);
            this.selectedRow = null;
            this.updateTimelineOutput();
        }
    }

    handleImageUpload(file) {
        if (this.selectedRow !== null && file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                this.timelineData[this.selectedRow].image = e.target.result;
                this.updateTimelineOutput();
            };
            reader.readAsDataURL(file);
        }
    }

    updateTimelineOutput() {
        // Update the hidden widget with the timeline data
        if (this.widgets.timelineData) {
            this.widgets.timelineData.value = JSON.stringify(this.timelineData);
        }
        this.node.setDirtyCanvas(true);
    }

    onMouseDown(e) {
        const { x, y } = this.getLocalMousePosition(e);
        const timeRulerHeight = 20;

        if (y >= timeRulerHeight && y <= this.TIMELINE_HEIGHT) {
            const rowIndex = Math.floor((y - timeRulerHeight) / this.ROW_HEIGHT);
            if (rowIndex < this.timelineData.length) {
                const row = this.timelineData[rowIndex];
                const totalFrames = this.node.properties.number_animation_frames;
                const startX = (row.start / totalFrames) * this.node.size[0];
                const endX = (row.end / totalFrames) * this.node.size[0];

                if (Math.abs(x - startX) <= this.HANDLE_WIDTH) {
                    this.isDragging = true;
                    this.dragType = 'start';
                    this.selectedRow = rowIndex;
                } else if (Math.abs(x - endX) <= this.HANDLE_WIDTH) {
                    this.isDragging = true;
                    this.dragType = 'end';
                    this.selectedRow = rowIndex;
                } else if (x >= startX && x <= endX) {
                    this.isDragging = true;
                    this.dragType = 'move';
                    this.selectedRow = rowIndex;
                } else {
                    this.selectedRow = null;
                }
                this.node.setDirtyCanvas(true);
            }
        }
    }

    onMouseMove(e) {
        if (this.isDragging && this.selectedRow !== null) {
            const { x } = this.getLocalMousePosition(e);
            const totalFrames = this.node.properties.number_animation_frames;
            const frameX = Math.max(0, Math.min(totalFrames, Math.round((x / this.node.size[0]) * totalFrames)));

            const row = this.timelineData[this.selectedRow];
            if (this.dragType === 'start') {
                row.start = Math.min(frameX, row.end - 1);
            } else if (this.dragType === 'end') {
                row.end = Math.max(frameX, row.start + 1);
            } else if (this.dragType === 'move') {
                const duration = row.end - row.start;
                row.start = Math.max(0, Math.min(frameX, totalFrames - duration));
                row.end = row.start + duration;
            }

            this.updateTimelineOutput();
        }
    }

    onMouseUp(e) {
        this.isDragging = false;
        this.dragType = null;
    }

    getLocalMousePosition(e) {
        const rect = this.node.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    serialize() {
        return {
            timelineData: this.timelineData,
            ...this.node.properties
        };
    }

    onConfigure(data) {
        if (data.timelineData) {
            this.timelineData = data.timelineData;
        }
        Object.assign(this.node.properties, data);
        this.updateTimelineOutput();
    }
}