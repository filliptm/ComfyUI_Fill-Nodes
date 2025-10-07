import { app } from "../../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.FL_GradGenerator",
    async nodeCreated(node) {
        if (node.comfyClass === "FL_GradGenerator") {
            addGradientImageGenerator(node);
        }
    }
});

function addGradientImageGenerator(node) {
    // New constants for UI adjustment
    const UI_OFFSET_X = 0;  // Adjust this to move UI elements horizontally
    const UI_OFFSET_Y = 50;  // Adjust this to move UI elements vertically
    const UI_SCALE = 1;     // Adjust this to scale UI elements (1 is normal size)

    const PADDING = 10 * UI_SCALE;
    const WIDGET_HEIGHT = 20 * UI_SCALE;
    const GRADIENT_HEIGHT = 40 * UI_SCALE;
    const MARGIN = 7 * UI_SCALE;
    const COLOR_PREVIEW_SIZE = 30 * UI_SCALE;
    const SLIDER_HEIGHT = 20 * UI_SCALE;

    let gradientColors = [
        { pos: 0, color: [255, 0, 0] },
        { pos: 0.5, color: [0, 255, 0] },
        { pos: 1, color: [0, 0, 255] }
    ];

    let selectedStop = gradientColors[0];
    let isMoving = false;
    let movingType = null; // 'marker' or 'slider'
    let activeSliderIndex = -1;

    // Find existing widgets
    const widthWidget = node.widgets.find(w => w.name === "width");
    const heightWidget = node.widgets.find(w => w.name === "height");
    const colorModeWidget = node.widgets.find(w => w.name === "color_mode");
    const interpolationWidget = node.widgets.find(w => w.name === "interpolation");
    const gradientColorsWidget = node.widgets.find(w => w.name === "gradient_colors");

    // Initialize gradient_colors widget
    if (gradientColorsWidget) {
        gradientColorsWidget.value = JSON.stringify(gradientColors);
    }

    // Add custom widgets
    const addColorWidget = node.addWidget("button", "+", null, () => {
        const newPos = gradientColors.length > 0 ?
            (gradientColors[gradientColors.length - 1].pos + 1) / 2 : 0.5;
        gradientColors.push({ pos: newPos, color: [255, 255, 255] });
        updateColorOutput();
    });

    const removeColorWidget = node.addWidget("button", "-", null, () => {
        if (gradientColors.length > 2) {
            gradientColors = gradientColors.filter(stop => stop !== selectedStop);
            selectedStop = gradientColors[0];
            updateColorOutput();
        }
    });

    function updateColorOutput() {
        if (gradientColorsWidget) {
            gradientColorsWidget.value = JSON.stringify(gradientColors);
        }
        node.setDirtyCanvas(true);
    }

    function rgbToHex(r, g, b) {
        return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
    }

    function drawGradient(ctx, x, y, width, height) {
        const gradient = ctx.createLinearGradient(x, y, x + width, y);
        gradientColors.forEach(stop => {
            gradient.addColorStop(stop.pos, rgbToHex(...stop.color));
        });
        ctx.fillStyle = gradient;
        ctx.fillRect(x, y, width, height);
    }

    function drawColorEditor(ctx, x, y, width, height) {
        ctx.fillStyle = "rgba(30,30,30,0.8)";
        ctx.fillRect(x, y, width, height);

        // Draw color preview
        ctx.fillStyle = rgbToHex(...selectedStop.color);
        ctx.fillRect(x + 10, y + 10, COLOR_PREVIEW_SIZE, COLOR_PREVIEW_SIZE);

        // Draw RGB sliders
        const sliderWidth = width - COLOR_PREVIEW_SIZE - 30;
        const sliderSpacing = (height - COLOR_PREVIEW_SIZE - 20) / 3;
        ["Red", "Green", "Blue"].forEach((label, index) => {
            const sliderY = y + COLOR_PREVIEW_SIZE + 20 + index * sliderSpacing;
            ctx.fillStyle = "#5e5e5e";
            ctx.fillRect(x + COLOR_PREVIEW_SIZE + 20, sliderY, sliderWidth, SLIDER_HEIGHT);
            ctx.fillStyle = "#" + "FF0000,00FF00,0000FF".split(",")[index];
            ctx.fillRect(x + COLOR_PREVIEW_SIZE + 20, sliderY, (sliderWidth * selectedStop.color[index]) / 255, SLIDER_HEIGHT);
            ctx.fillStyle = "white";
            ctx.fillText(`${label}: ${selectedStop.color[index]}`, x + COLOR_PREVIEW_SIZE + 20, sliderY - 5);

            // Highlight active slider
            if (isMoving && movingType === 'slider' && activeSliderIndex === index) {
                ctx.strokeStyle = "yellow";
                ctx.lineWidth = 2;
                ctx.strokeRect(x + COLOR_PREVIEW_SIZE + 19, sliderY - 1, sliderWidth + 2, SLIDER_HEIGHT + 2);
            }
        });
    }

    node.onDrawBackground = function(ctx) {
        if (!this.flags.collapsed) {
            const width = (this.size[0] - PADDING * 2) * UI_SCALE;
            let y = (PADDING + (WIDGET_HEIGHT + MARGIN) * 6) * UI_SCALE + UI_OFFSET_Y; // Adjusted to draw under input fields

            // Apply UI_OFFSET_X and UI_SCALE to all drawing operations
            ctx.save();
            ctx.translate(PADDING * UI_SCALE + UI_OFFSET_X, y);
            ctx.scale(UI_SCALE, UI_SCALE);

            // Draw gradient preview
            drawGradient(ctx, 0, 0, width, GRADIENT_HEIGHT);
            ctx.strokeStyle = "#666";
            ctx.strokeRect(0, 0, width, GRADIENT_HEIGHT);

            // Draw color stops
            gradientColors.forEach((stop) => {
                const x = stop.pos * width;
                ctx.fillStyle = rgbToHex(...stop.color);
                ctx.beginPath();
                ctx.moveTo(x, GRADIENT_HEIGHT);
                ctx.lineTo(x - 5, GRADIENT_HEIGHT + 10);
                ctx.lineTo(x + 5, GRADIENT_HEIGHT + 10);
                ctx.closePath();
                ctx.fill();
                ctx.strokeStyle = stop === selectedStop ? "white" : "#666";
                ctx.lineWidth = (isMoving && movingType === 'marker' && stop === selectedStop) ? 3 : 1;
                ctx.stroke();
            });

            // Draw color editor
            const editorHeight = (this.size[1] - y - GRADIENT_HEIGHT - MARGIN * 3) / UI_SCALE;
            drawColorEditor(ctx, 0, GRADIENT_HEIGHT + MARGIN, width, editorHeight);

            ctx.restore();
        }
    };

    node.onMouseDown = function(event) {
        const width = (this.size[0] - PADDING * 2) * UI_SCALE;
        const gradientY = (PADDING + (WIDGET_HEIGHT + MARGIN) * 6) * UI_SCALE + UI_OFFSET_Y;
        const localX = (event.canvasX - this.pos[0] - PADDING * UI_SCALE - UI_OFFSET_X) / UI_SCALE;
        const localY = (event.canvasY - this.pos[1] - gradientY) / UI_SCALE;

        if (localY >= 0 && localY <= GRADIENT_HEIGHT + 10) {
            const clickedPos = localX / width;
            const clickedStop = gradientColors.find(stop =>
                Math.abs(stop.pos - clickedPos) < 0.05
            );

            if (clickedStop) {
                selectedStop = clickedStop;
                if (isMoving && movingType === 'marker') {
                    isMoving = false;
                    movingType = null;
                } else {
                    isMoving = true;
                    movingType = 'marker';
                }
            } else if (!isMoving) {
                // Add new color stop
                const newStop = { pos: clickedPos, color: [255, 255, 255] };
                gradientColors.push(newStop);
                gradientColors.sort((a, b) => a.pos - b.pos);
                selectedStop = newStop;
            }
            updateColorOutput();
        } else if (localY >= GRADIENT_HEIGHT + MARGIN) {
            const editorHeight = (this.size[1] - gradientY - GRADIENT_HEIGHT - MARGIN * 3) / UI_SCALE;
            const sliderSpacing = (editorHeight - COLOR_PREVIEW_SIZE - 20) / 3;
            const sliderY = localY - (GRADIENT_HEIGHT + MARGIN + COLOR_PREVIEW_SIZE + 20);
            if (sliderY >= 0 && sliderY <= 3 * sliderSpacing && localX >= COLOR_PREVIEW_SIZE + 20 && localX <= width - 10) {
                const clickedSliderIndex = Math.floor(sliderY / sliderSpacing);
                if (isMoving && movingType === 'slider' && activeSliderIndex === clickedSliderIndex) {
                    isMoving = false;
                    movingType = null;
                    activeSliderIndex = -1;
                } else {
                    isMoving = true;
                    movingType = 'slider';
                    activeSliderIndex = clickedSliderIndex;
                    updateSliderValue(localX);
                }
            }
        }

        this.setDirtyCanvas(true);
        event.stopPropagation();
    };

    node.onMouseMove = function(event) {
        if (isMoving) {
            const width = (this.size[0] - PADDING * 2) * UI_SCALE;
            const localX = (event.canvasX - this.pos[0] - PADDING * UI_SCALE - UI_OFFSET_X) / UI_SCALE;
            if (movingType === 'marker') {
                selectedStop.pos = Math.max(0, Math.min(1, localX / width));
                gradientColors.sort((a, b) => a.pos - b.pos);
            } else if (movingType === 'slider') {
                updateSliderValue(localX);
            }
            updateColorOutput();
        }
    };

    node.onMouseUp = function(event) {
        if (isMoving) {
            isMoving = false;
            movingType = null;
            activeSliderIndex = -1;
            updateColorOutput();
        }
    };

    function updateSliderValue(x) {
        const width = (node.size[0] - PADDING * 2) * UI_SCALE;
        const sliderWidth = width - COLOR_PREVIEW_SIZE - 30;
        const sliderValue = Math.max(0, Math.min(255, Math.floor(((x - COLOR_PREVIEW_SIZE - 20) / sliderWidth) * 255)));
        selectedStop.color[activeSliderIndex] = sliderValue;
        updateColorOutput();
    }

    // Handle node resizing
    node.onResize = function() {
        this.setDirtyCanvas(true);
    };

    node.setSize([280, 400]); // Adjusted initial size
}