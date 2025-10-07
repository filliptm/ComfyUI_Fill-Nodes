import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

function hideWidgetForGood(node, widget) {
    if (!widget) return;
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.origSerializeValue = widget.serializeValue;
    widget.computeSize = () => [0, -4];
    widget.type = "converted-widget";
}

app.registerExtension({
    name: "Comfy.FL_ImageAdjuster",
    async nodeCreated(node) {
        if (node.comfyClass === "FL_ImageAdjuster") {
            addImageAdjusterUI(node);
        }
    }
});

function addImageAdjusterUI(node) {
    const MIN_WIDTH = 200;
    const MIN_HEIGHT = 300;
    const SLIDER_HEIGHT = 25;
    const PADDING = 0;

    // Default values for sliders
    const DEFAULT_VALUES = {
        hue: 0,
        saturation: 0,
        brightness: 0,
        contrast: 0,
        sharpness: 0
    };

    // Find and hide the original widgets
    const originalWidgets = {};
    ["hue", "saturation", "brightness", "contrast", "sharpness"].forEach(name => {
        const widget = node.widgets.find(w => w.name === name);
        if (widget) {
            originalWidgets[name] = widget;
            hideWidgetForGood(node, widget);
        }
    });

    // Create custom sliders
    const sliders = [
        createSlider("Hue", -180, 180, originalWidgets.hue),
        createSlider("Saturation", -100, 100, originalWidgets.saturation),
        createSlider("Brightness", -100, 100, originalWidgets.brightness),
        createSlider("Contrast", -100, 100, originalWidgets.contrast),
        createSlider("Sharpness", 0, 100, originalWidgets.sharpness)
    ];

    function createSlider(name, min, max, originalWidget) {
        const slider = node.addWidget("slider", name, originalWidget.value, (v) => {
            originalWidget.value = v;
            node.setDirtyCanvas(true);
        }, { min: min, max: max, step: 1 });
        return slider;
    }

    // Add reset button
    const resetButton = node.addWidget("button", "Reset", null, () => {
        resetSliders();
    });

    function resetSliders() {
        sliders.forEach(slider => {
            const defaultValue = DEFAULT_VALUES[slider.name.toLowerCase()];
            slider.value = defaultValue;
            slider.callback(defaultValue);
        });

        node.setDirtyCanvas(true);
        node.triggerSlot(0);
    }

    // Add image preview
    const img = new Image();
    img.onload = () => node.setDirtyCanvas(true);

    node.onDrawBackground = function(ctx) {
        if (!this.flags.collapsed) {
            const [w, h] = this.size;

            // Calculate the Y position of the last widget
            const lastWidget = node.widgets[node.widgets.length - 1];
            const lastWidgetY = lastWidget.last_y || 0;

            // Set the image Y offset to be just below the last widget
            const IMAGE_Y_OFFSET = lastWidgetY + SLIDER_HEIGHT + PADDING;

            const imageArea = h - IMAGE_Y_OFFSET - PADDING;

            // Draw image
            if (img.src) {
                const aspectRatio = img.width / img.height;
                let drawWidth = w - 2 * PADDING;
                let drawHeight = imageArea;

                if (drawWidth / drawHeight > aspectRatio) {
                    drawWidth = drawHeight * aspectRatio;
                } else {
                    drawHeight = drawWidth / aspectRatio;
                }

                const x = PADDING + (w - 2 * PADDING - drawWidth) / 2;
                const y = IMAGE_Y_OFFSET;

                ctx.drawImage(img, x, y, drawWidth, drawHeight);
            }
        }
    };

    // Listen for the image from the backend
    api.addEventListener("fl_image_adjuster", (event) => {
        if (event.detail.image) {
            img.src = event.detail.image;
        }
    });

    function updateNodeSize() {
        node.size[0] = Math.max(MIN_WIDTH, node.size[0]);
        node.size[1] = Math.max(MIN_HEIGHT, node.size[1]);
    }

    node.onResize = updateNodeSize;
    updateNodeSize();
}