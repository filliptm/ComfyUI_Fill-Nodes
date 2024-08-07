import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.ColorPicker",
    async nodeCreated(node) {
        if (node.comfyClass === "FL_ColorPicker") {
            addColorPicker(node);
        }
    }
});

function addColorPicker(node) {
    const PADDING = 10;
    const WIDGET_HEIGHT = 20;
    const WIDGET_MARGIN = 5;
    let gradientWidth, gradientHeight;
    let selectedColor = "#FF0000";
    let pickerPos = { x: 180, y: 5 }; // Start at top-right corner (red)
    let isPickerActive = false;

    // Find the existing widget
    const colorWidget = node.widgets.find(w => w.name === "selected_color");
    if (colorWidget) {
        colorWidget.hidden = true;  // Hide this widget from the node interface
    }

    // Add reset button
    node.addWidget("button", "Reset Color", "reset", () => {
        selectedColor = "#FF0000";
        pickerPos = { x: gradientWidth - 5, y: 5 };
        updateColorOutput();
    });

    function updateColorOutput() {
        if (colorWidget) {
            colorWidget.value = selectedColor;
        }
        node.setDirtyCanvas(true);
    }

    function rgb_to_hex(r, g, b) {
        return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
    }

    node.onDrawBackground = function(ctx) {
        if (!this.flags.collapsed) {
            ctx.save();
            ctx.translate(PADDING, PADDING);

            const totalWidgetHeight = (WIDGET_HEIGHT + WIDGET_MARGIN) * 2.6; // Space for reset button and margin
            gradientWidth = this.size[0] - PADDING * 2;
            gradientHeight = this.size[1] - PADDING * 2 - totalWidgetHeight - WIDGET_HEIGHT; // Extra WIDGET_HEIGHT for color display at bottom

            // Draw widgets background
            ctx.fillStyle = "#2A2A2A";
            ctx.fillRect(0, 0, gradientWidth, totalWidgetHeight - WIDGET_MARGIN);

            // Draw color gradient
            ctx.translate(0, totalWidgetHeight);
            const gradient = ctx.createLinearGradient(0, 0, gradientWidth, 0);
            gradient.addColorStop(0, "rgb(255, 0, 0)");
            gradient.addColorStop(0.17, "rgb(255, 255, 0)");
            gradient.addColorStop(0.33, "rgb(0, 255, 0)");
            gradient.addColorStop(0.5, "rgb(0, 255, 255)");
            gradient.addColorStop(0.67, "rgb(0, 0, 255)");
            gradient.addColorStop(0.83, "rgb(255, 0, 255)");
            gradient.addColorStop(1, "rgb(255, 0, 0)");
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, gradientWidth, gradientHeight);

            // Draw black gradient overlay
            const blackGradient = ctx.createLinearGradient(0, 0, 0, gradientHeight);
            blackGradient.addColorStop(0, "rgba(0, 0, 0, 0)");
            blackGradient.addColorStop(1, "rgba(0, 0, 0, 1)");
            ctx.fillStyle = blackGradient;
            ctx.fillRect(0, 0, gradientWidth, gradientHeight);

            // Draw color picker ball
            ctx.beginPath();
            ctx.arc(pickerPos.x, pickerPos.y, 5, 0, Math.PI * 2);
            ctx.strokeStyle = isPickerActive ? "yellow" : "white";
            ctx.lineWidth = 2;
            ctx.stroke();

            // Draw selected color
            ctx.fillStyle = selectedColor;
            ctx.fillRect(0, gradientHeight + WIDGET_MARGIN, gradientWidth, WIDGET_HEIGHT);

            // Draw color value text
            ctx.fillStyle = "white";
            ctx.font = "12px Arial";
            ctx.fillText(selectedColor, 5, gradientHeight + WIDGET_MARGIN + WIDGET_HEIGHT - 5);

            ctx.restore();
        }
    };

    function updateSelectedColor(x, y) {
        // Ensure x and y are within the gradient bounds
        x = Math.max(0, Math.min(x, gradientWidth));
        y = Math.max(0, Math.min(y, gradientHeight));

        pickerPos = { x, y };

        // Create an offscreen canvas to sample the color
        const offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = gradientWidth;
        offscreenCanvas.height = gradientHeight;
        const offscreenCtx = offscreenCanvas.getContext('2d');

        // Recreate the gradient on the offscreen canvas
        const gradient = offscreenCtx.createLinearGradient(0, 0, gradientWidth, 0);
        gradient.addColorStop(0, "rgb(255, 0, 0)");
        gradient.addColorStop(0.17, "rgb(255, 255, 0)");
        gradient.addColorStop(0.33, "rgb(0, 255, 0)");
        gradient.addColorStop(0.5, "rgb(0, 255, 255)");
        gradient.addColorStop(0.67, "rgb(0, 0, 255)");
        gradient.addColorStop(0.83, "rgb(255, 0, 255)");
        gradient.addColorStop(1, "rgb(255, 0, 0)");
        offscreenCtx.fillStyle = gradient;
        offscreenCtx.fillRect(0, 0, gradientWidth, gradientHeight);

        const blackGradient = offscreenCtx.createLinearGradient(0, 0, 0, gradientHeight);
        blackGradient.addColorStop(0, "rgba(0, 0, 0, 0)");
        blackGradient.addColorStop(1, "rgba(0, 0, 0, 1)");
        offscreenCtx.fillStyle = blackGradient;
        offscreenCtx.fillRect(0, 0, gradientWidth, gradientHeight);

        const imageData = offscreenCtx.getImageData(x, y, 1, 1).data;
        selectedColor = rgb_to_hex(imageData[0], imageData[1], imageData[2]);
        updateColorOutput();
    }

    node.onMouseDown = function(event) {
        const totalWidgetHeight = (WIDGET_HEIGHT + WIDGET_MARGIN) * 2.6;
        const localX = event.canvasX - this.pos[0] - PADDING;
        const localY = event.canvasY - this.pos[1] - PADDING - totalWidgetHeight;

        if (localX >= 0 && localX <= gradientWidth && localY >= 0 && localY <= gradientHeight) {
            isPickerActive = !isPickerActive;
            if (isPickerActive) {
                updateSelectedColor(localX, localY);
            }
            this.setDirtyCanvas(true);
        }
    };

    node.onMouseMove = function(event) {
        if (isPickerActive) {
            const totalWidgetHeight = (WIDGET_HEIGHT + WIDGET_MARGIN) * 2.6;
            const localX = event.canvasX - this.pos[0] - PADDING;
            const localY = event.canvasY - this.pos[1] - PADDING - totalWidgetHeight;

            updateSelectedColor(localX, localY);
            this.setDirtyCanvas(true);
        }
    };

    // Handle node resizing
    node.onResize = function() {
        this.setDirtyCanvas(true);
    };

    node.setSize([250, 300]); // Set initial size
    node.setDirtyCanvas(true);
}