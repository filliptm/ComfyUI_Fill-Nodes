import { app } from "../../../../scripts/app.js";
import { ComfyWidgets } from "../../../../scripts/widgets.js";
import { api } from "../../../../scripts/api.js";

// Helper function to convert RGB to hex
function rgb_to_hex(r, g, b) {
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1).toUpperCase();
}

// Helper function to convert hex to RGB
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

// Create an eyedropper color picker widget
function createEyedropperPicker(node, colorWidget, label) {
    const PADDING = 10;
    const WIDGET_HEIGHT = 40;
    const WIDGET_MARGIN = 5;

    let selectedColor = colorWidget.value || "#FF0000";

    function updateColorOutput() {
        if (colorWidget) colorWidget.value = selectedColor;
        if (hexInputWidget) hexInputWidget.value = selectedColor;
        let rgb = hexToRgb(selectedColor);
        if (rSlider) rSlider.value = rgb.r;
        if (gSlider) gSlider.value = rgb.g;
        if (bSlider) bSlider.value = rgb.b;
        node.setDirtyCanvas(true, true);
    }

    // Check if EyeDropper API is supported
    const supportsEyeDropper = typeof window.EyeDropper !== 'undefined';

    // Eyedropper button widget
    const eyedropperWidget = {
        name: `${label}_EYEDROPPER_DISPLAY`,
        type: "CANVAS_WIDGET",
        draw: function (ctx, node, widgetWidth, widgetY, height) {
            const drawY = widgetY + PADDING;

            ctx.save();
            ctx.translate(PADDING, drawY);

            // Draw label
            ctx.fillStyle = "white";
            ctx.font = "12px Arial";
            ctx.textAlign = "left";
            ctx.fillText(label, 0, -5);

            // Draw eyedropper button
            const buttonWidth = widgetWidth - PADDING * 2;
            const buttonHeight = WIDGET_HEIGHT;
            
            // Button background
            ctx.fillStyle = supportsEyeDropper ? "#4a4a4a" : "#666666";
            ctx.fillRect(0, 0, buttonWidth, buttonHeight);
            
            // Button border
            ctx.strokeStyle = "#888888";
            ctx.lineWidth = 1;
            ctx.strokeRect(0, 0, buttonWidth, buttonHeight);

            // Eyedropper icon (simplified)
            ctx.strokeStyle = "white";
            ctx.lineWidth = 2;
            const centerX = buttonWidth / 2;
            const centerY = buttonHeight / 2;
            
            if (supportsEyeDropper) {
                // Draw eyedropper icon
                ctx.beginPath();
                ctx.moveTo(centerX - 8, centerY + 8);
                ctx.lineTo(centerX - 4, centerY + 4);
                ctx.lineTo(centerX + 4, centerY - 4);
                ctx.lineTo(centerX + 8, centerY - 8);
                ctx.stroke();
                
                ctx.beginPath();
                ctx.arc(centerX + 6, centerY - 6, 3, 0, Math.PI * 2);
                ctx.stroke();
                
                // Button text
                ctx.fillStyle = "white";
                ctx.font = "11px Arial";
                ctx.textAlign = "center";
                ctx.fillText("Pick Color", centerX, centerY + 15);
            } else {
                // Not supported message
                ctx.fillStyle = "#999999";
                ctx.font = "10px Arial";
                ctx.textAlign = "center";
                ctx.fillText("Eyedropper not supported", centerX, centerY - 2);
                ctx.fillText("Use manual input below", centerX, centerY + 10);
            }

            // Color preview bar
            ctx.fillStyle = selectedColor;
            ctx.fillRect(0, buttonHeight + WIDGET_MARGIN, buttonWidth, 20);
            
            // Color text
            ctx.fillStyle = "white";
            ctx.font = "10px Arial";
            ctx.textAlign = "left";
            ctx.fillText(selectedColor, 5, buttonHeight + WIDGET_MARGIN + 15);

            ctx.restore();
        },
        mouse: function (event, pos, node) {
            if (!supportsEyeDropper) return false;
            
            // Simplified click detection - check if click is within widget bounds
            const widgetRect = this.computeArea();
            const relativeX = pos[0] - PADDING;
            const relativeY = pos[1] - (widgetRect.y + PADDING + 12); // Account for label offset
            const buttonWidth = node.size[0] - PADDING * 2;
            const buttonHeight = WIDGET_HEIGHT;

            if (event.type === "pointerdown") {
                // Check if click is within button area
                if (relativeX >= 0 && relativeX <= buttonWidth &&
                    relativeY >= 0 && relativeY <= buttonHeight) {
                    
                    console.log("Eyedropper button clicked"); // Debug log
                    
                    // Prevent event bubbling immediately
                    event.preventDefault();
                    event.stopPropagation();
                    
                    // Add small delay to ensure click is processed before launching eyedropper
                    setTimeout(() => {
                        try {
                            const eyeDropper = new EyeDropper();
                            console.log("Opening eyedropper...");
                            
                            eyeDropper.open()
                                .then(result => {
                                    console.log("Color picked successfully:", result.sRGBHex);
                                    selectedColor = result.sRGBHex.toUpperCase();
                                    updateColorOutput();
                                    node.setDirtyCanvas(true, true); // Force redraw
                                })
                                .catch(err => {
                                    if (err.name === 'AbortError') {
                                        console.log("Eyedropper selection was canceled by user");
                                    } else {
                                        console.error("Eyedropper failed:", err);
                                    }
                                });
                        } catch (error) {
                            console.error("Error creating EyeDropper:", error);
                            alert("EyeDropper API is not available in this browser. Please use the hex input or RGB sliders below.");
                        }
                    }, 100); // Small delay to prevent conflicts
                    
                    return true;
                }
            }
            return false;
        },
        computeSize: function (width) {
            const totalHeight = PADDING * 2 + 12 + WIDGET_HEIGHT + WIDGET_MARGIN + 20; // label + button + color bar
            return [width, totalHeight];
        },
        computeArea: function() {
            let y = this.y || 0;
            let totalHeight = this.computeSize(node.size[0])[1];
            return { x: 0, y: y, w: node.size[0], h: totalHeight };
        }
    };

    // Manual hex input widget
    const hexInputWidget = node.addWidget("text", `${label} Hex`, selectedColor, (value) => {
        if (/^#([0-9A-Fa-f]{6})$/.test(value)) {
            selectedColor = value.toUpperCase();
            updateColorOutput();
        }
    });

    // RGB Sliders for manual adjustment
    let rgb = hexToRgb(selectedColor);
    const rSlider = node.addWidget("slider", `${label} R`, rgb.r, (value) => {
        rgb.r = Math.round(value);
        selectedColor = rgb_to_hex(rgb.r, rgb.g, rgb.b);
        updateColorOutput();
    }, { min: 0, max: 255, step: 1 });

    const gSlider = node.addWidget("slider", `${label} G`, rgb.g, (value) => {
        rgb.g = Math.round(value);
        selectedColor = rgb_to_hex(rgb.r, rgb.g, rgb.b);
        updateColorOutput();
    }, { min: 0, max: 255, step: 1 });

    const bSlider = node.addWidget("slider", `${label} B`, rgb.b, (value) => {
        rgb.b = Math.round(value);
        selectedColor = rgb_to_hex(rgb.r, rgb.g, rgb.b);
        updateColorOutput();
    }, { min: 0, max: 255, step: 1 });

    // Hide original widget
    colorWidget.type = "HIDDEN";

    // Setup synchronization
    const originalCallback = colorWidget.callback;
    colorWidget.callback = (value) => {
        if (value !== selectedColor) {
            selectedColor = value;
            node.setDirtyCanvas(true, true);
        }
    };

    const originalSetValue = colorWidget.setValue;
    colorWidget.setValue = function(value) {
        originalSetValue?.call(this, value);
        selectedColor = value || "#FF0000";
        updateColorOutput();
    };

    // Initial setup
    selectedColor = colorWidget.value || "#FF0000";
    setTimeout(() => {
        selectedColor = colorWidget.value || "#FF0000";
        updateColorOutput();
    }, 0);

    return eyedropperWidget;
}

app.registerExtension({
    name: "Comfy.FL_ReplaceColor",
    async nodeCreated(node) {
        if (node.comfyClass === "FL_ReplaceColor") {
            // Find the color widgets
            const sourceColorWidget = node.widgets.find(w => w.name === "source_color");
            const targetColorWidget = node.widgets.find(w => w.name === "target_color");
            
            if (!sourceColorWidget || !targetColorWidget) {
                console.error("FL_ReplaceColor: Could not find color widgets!");
                return;
            }

            // Create eyedropper pickers
            const sourcePickerWidget = createEyedropperPicker(node, sourceColorWidget, "Source Color");
            const targetPickerWidget = createEyedropperPicker(node, targetColorWidget, "Target Color");
            
            node.addCustomWidget(sourcePickerWidget);
            node.addCustomWidget(targetPickerWidget);

            // Find expand/contract widgets and add descriptions if needed
            const expandWidget = node.widgets.find(w => w.name === "expand_pixels");
            const contractWidget = node.widgets.find(w => w.name === "contract_pixels");
            
            if (expandWidget) {
                expandWidget.options = expandWidget.options || {};
                expandWidget.options.tooltip = "Expand replacement area by N pixels (dilation)";
            }
            
            if (contractWidget) {
                contractWidget.options = contractWidget.options || {};
                contractWidget.options.tooltip = "Contract replacement area by N pixels (erosion)";
            }

            // Recompute node size after adding widgets
            node.setSize(node.computeSize());
        }
    }
});