import { app } from "../../../../scripts/app.js";
 import { ComfyWidgets } from "../../../../scripts/widgets.js";
 import { api } from "../../../../scripts/api.js"; // Import api
 
 // Helper function to convert hex to RGB
 function hexToRgb(hex) {
     const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
     return result ? {
         r: parseInt(result[1], 16),
         g: parseInt(result[2], 16),
         b: parseInt(result[3], 16)
     } : null;
 }
 
 // Helper function to convert RGB to HSV
 function rgbToHsv(r, g, b) {
     r /= 255, g /= 255, b /= 255;
     let max = Math.max(r, g, b), min = Math.min(r, g, b);
     let h, s, v = max;
     let d = max - min;
     s = max == 0 ? 0 : d / max;
     if (max == min) {
         h = 0; // achromatic
     } else {
         switch (max) {
             case r: h = (g - b) / d + (g < b ? 6 : 0); break;
             case g: h = (b - r) / d + 2; break;
             case b: h = (r - g) / d + 4; break;
         }
         h /= 6;
     }
     return { h, s, v };
 }
 
 
 app.registerExtension({
     name: "Comfy.ColorPicker",
     async nodeCreated(node) {
         if (node.comfyClass === "FL_ColorPicker") {
             const PADDING = 10;
             const WIDGET_HEIGHT = 20;
             const WIDGET_MARGIN = 5;
             const PICKER_AREA_HEIGHT = 150; // Fixed height for the picker gradient
 
             let gradientWidth, gradientHeight = PICKER_AREA_HEIGHT;
             let pickerPos = { x: 0, y: 0 }; // Will be calculated based on initial color
             let isPickerActive = false;
 
             // Find the original widget
             const colorWidget = node.widgets.find(w => w.name === "selected_color");
             if (!colorWidget) {
                 console.error("FL_ColorPicker: Could not find 'selected_color' widget!");
                 return;
             }
 
             // Store the original properties
             const originalCallback = colorWidget.callback;
             const originalType = colorWidget.type;
             const originalOptions = { ...colorWidget.options };
 
             // --- State ---
             let selectedColor = colorWidget.value || "#FF0000";
 
             // --- Helper Functions ---
             function rgb_to_hex(r, g, b) {
                 return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1).toUpperCase();
             }
 
             // Function to calculate picker position from hex color (approximation)
             function calculatePickerPosFromHex(hex) {
                 const rgb = hexToRgb(hex);
                 if (!rgb) return { x: gradientWidth - 5, y: 5 }; // Default if invalid hex
                 const hsv = rgbToHsv(rgb.r, rgb.g, rgb.b);
 
                 // x corresponds to hue (saturation is mixed in the gradient)
                 // y corresponds to value (brightness)
                 const x = hsv.h * (gradientWidth || 200); // Map hue to width
                 const y = (1 - hsv.v) * gradientHeight; // Map inverse value to height
 
                 // Clamp values
                 return {
                     x: Math.max(0, Math.min(x, gradientWidth || 200)),
                     y: Math.max(0, Math.min(y, gradientHeight))
                 };
             }
 
 
             function updateColorOutput() {
                 if (colorWidget) colorWidget.value = selectedColor;
                 if (hexInputWidget) hexInputWidget.value = selectedColor;
                 rgb = hexToRgb(selectedColor);
                 if (rSlider) rSlider.value = rgb.r;
                 if (gSlider) gSlider.value = rgb.g;
                 if (bSlider) bSlider.value = rgb.b;
                 node.setDirtyCanvas(true, true);
             }
 
             function updateSelectedColor(x, y) {
                 // Clamp x and y
                 x = Math.max(0, Math.min(x, gradientWidth || 0));
                 y = Math.max(0, Math.min(y, gradientHeight || 0));
                 pickerPos = { x, y };
 
                 if (gradientWidth > 0 && gradientHeight > 0) {
                     // Sample color (using offscreen canvas for accuracy)
                     const offscreenCanvas = document.createElement('canvas');
                     offscreenCanvas.width = gradientWidth;
                     offscreenCanvas.height = gradientHeight;
                     const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
 
                     // Draw gradient on offscreen canvas
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
 
                     const pixelData = offscreenCtx.getImageData(Math.round(x), Math.round(y), 1, 1).data;
                     selectedColor = rgb_to_hex(pixelData[0], pixelData[1], pixelData[2]);
                 }
                 updateColorOutput();
             }
 
             // --- Widgets ---
 
             // 1. Reset Button
             const resetButton = node.addWidget("button", "Reset Color", "reset", () => {
                 selectedColor = "#FF0000";
                 pickerPos = calculatePickerPosFromHex(selectedColor); // Reset picker pos
                 updateColorOutput();
                 if (originalCallback) { // Also trigger original callback if exists
                     originalCallback(selectedColor);
                 }
             });
 
             // 2. Custom Picker Widget
             const pickerWidget = {
                 name: "COLOR_PICKER_DISPLAY",
                 type: "CANVAS_WIDGET", // Custom type for identification
                 y: (resetButton.y || 0) + (resetButton.computeSize ? resetButton.computeSize(node.size[0])[1] : WIDGET_HEIGHT) + WIDGET_MARGIN,
                 draw: function (ctx, node, widgetWidth, widgetY, height) {
                     gradientWidth = widgetWidth - PADDING * 2; // Update width based on widget size
                     const drawY = widgetY + PADDING; // Start drawing below padding
 
                     ctx.save();
                     ctx.translate(PADDING, drawY); // Position drawing area
 
                     // Draw Gradient
                     if (gradientWidth > 0 && gradientHeight > 0) {
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
 
                         const blackGradient = ctx.createLinearGradient(0, 0, 0, gradientHeight);
                         blackGradient.addColorStop(0, "rgba(0, 0, 0, 0)");
                         blackGradient.addColorStop(1, "rgba(0, 0, 0, 1)");
                         ctx.fillStyle = blackGradient;
                         ctx.fillRect(0, 0, gradientWidth, gradientHeight);
 
                         // Draw Picker Circle
                         const pickerDrawX = Math.max(0, Math.min(pickerPos.x, gradientWidth));
                         const pickerDrawY = Math.max(0, Math.min(pickerPos.y, gradientHeight));
                         ctx.beginPath();
                         ctx.arc(pickerDrawX, pickerDrawY, 5, 0, Math.PI * 2);
                         ctx.strokeStyle = isPickerActive ? "yellow" : "white";
                         ctx.lineWidth = 2;
                         ctx.stroke();
                     }
 
                     // Draw Color Display Bar
                     ctx.fillStyle = selectedColor;
                     ctx.fillRect(0, gradientHeight + WIDGET_MARGIN, gradientWidth, WIDGET_HEIGHT);
 
                     // Draw Color Value Text
                     ctx.fillStyle = "white";
                     ctx.font = "12px Arial";
                     ctx.textAlign = "left";
                     ctx.fillText(selectedColor, 5, gradientHeight + WIDGET_MARGIN + WIDGET_HEIGHT - 5);
 
                     ctx.restore();
                 },
                 mouse: function (event, pos, node) {
                     // Calculate position relative to the gradient area within this widget
                     const widgetRect = this.computeArea(); // Get widget boundaries
                     const gradientAreaX = PADDING;
                     const gradientAreaY = widgetRect.y + PADDING; // Y relative to widget top + padding
 
                     const clickX = pos[0] - gradientAreaX;
                     const clickY = pos[1] - gradientAreaY;
 
                     if (event.type === "pointerdown") {
                         if (clickX >= 0 && clickX <= gradientWidth && clickY >= 0 && clickY <= gradientHeight) {
                             isPickerActive = true;
                             updateSelectedColor(clickX, clickY);
                             event.stopPropagation();
                             return true; // Handled
                         }
                     } else if (event.type === "pointermove" && isPickerActive) {
                         updateSelectedColor(clickX, clickY);
                         return true; // Handled
                     } else if (event.type === "pointerup" && isPickerActive) {
                         isPickerActive = false;
                         // Trigger original callback on mouse up to ensure final value is processed
                         if (originalCallback) {
                             originalCallback(selectedColor);
                         }
                         return true; // Handled
                     }
                     return false; // Not handled
                 },
                 computeSize: function (width) {
                     // Calculate required height: Picker Area + Display Bar + Padding
                     const totalHeight = PADDING * 2 + gradientHeight + WIDGET_MARGIN + WIDGET_HEIGHT;
                     return [width, totalHeight];
                 },
                 computeArea: function() { // Helper to get widget screen bounds
                     let y = this.y || 0;
                     let totalHeight = this.computeSize(node.size[0])[1];
                     return { x: 0, y: y, w: node.size[0], h: totalHeight };
                 }
             };
             node.addCustomWidget(pickerWidget); // Add the custom drawing widget
 
             // 3. Manual Hex Input Widget
             const hexInputWidget = node.addWidget("text", "Hex Color", selectedColor, (value) => {
                 // Validate hex format
                 if (/^#([0-9A-Fa-f]{6})$/.test(value)) {
                     selectedColor = value.toUpperCase();
                     pickerPos = calculatePickerPosFromHex(selectedColor);
                     updateColorOutput();
                 }
             });
 
             // --- RGB Sliders ---
             let rgb = hexToRgb(selectedColor);
             // R
             const rSlider = node.addWidget("slider", "R", rgb.r, (value) => {
                 rgb.r = Math.round(value);
                 selectedColor = rgb_to_hex(rgb.r, rgb.g, rgb.b);
                 pickerPos = calculatePickerPosFromHex(selectedColor);
                 updateColorOutput();
             }, { min: 0, max: 255, step: 1 });
             // G
             const gSlider = node.addWidget("slider", "G", rgb.g, (value) => {
                 rgb.g = Math.round(value);
                 selectedColor = rgb_to_hex(rgb.r, rgb.g, rgb.b);
                 pickerPos = calculatePickerPosFromHex(selectedColor);
                 updateColorOutput();
             }, { min: 0, max: 255, step: 1 });
             // B
             const bSlider = node.addWidget("slider", "B", rgb.b, (value) => {
                 rgb.b = Math.round(value);
                 selectedColor = rgb_to_hex(rgb.r, rgb.g, rgb.b);
                 pickerPos = calculatePickerPosFromHex(selectedColor);
                 updateColorOutput();
             }, { min: 0, max: 255, step: 1 });
 
             // --- Modify Original Widget ---
             // Prevent the original text input from showing
             colorWidget.type = "HIDDEN"; // Change type to hide default rendering
 
             // Override original callback to sync external changes
             colorWidget.callback = (value) => {
                 if (value !== selectedColor) {
                      selectedColor = value;
                      // Update picker position based on the new value and current dimensions
                      gradientWidth = node.size[0] - PADDING * 2; // Ensure width is current
                      pickerPos = calculatePickerPosFromHex(selectedColor);
                      node.setDirtyCanvas(true, true); // Redraw needed
                 }
                 // Avoid calling originalCallback here to prevent potential loops if it also modifies the value
             };
 
             // Listen for when the workflow is loaded and the widget value is set
             const originalSetValue = colorWidget.setValue;
             colorWidget.setValue = function(value) {
                 originalSetValue?.call(this, value);
                 selectedColor = value || "#FF0000";
                 gradientWidth = node.size[0] - PADDING * 2;
                 pickerPos = calculatePickerPosFromHex(selectedColor);
                 updateColorOutput();
             };
 
             // --- Initial State Setup ---
 
             // 1. Read the loaded value (or default)
             selectedColor = colorWidget.value || "#FF0000";
 
             // 2. Compute the final size of the node based on its content
             node.setSize(node.computeSize());
 
             // 3. Calculate gradient dimensions based on the *final* node size
             gradientWidth = node.size[0] - PADDING * 2;
             // gradientHeight = PICKER_AREA_HEIGHT; // Stays fixed
 
             // 4. Calculate the picker position based on the loaded color and final dimensions
             pickerPos = calculatePickerPosFromHex(selectedColor);
 
             // 5. Ensure the hidden widget's value reflects the potentially loaded state
             //    (This might be redundant if colorWidget.value was already correct, but safe)
             //    Crucially, call setDirtyCanvas to trigger redraw *after* all state is set.
             updateColorOutput(); // This sets colorWidget.value and calls setDirtyCanvas
 
             // --- End Initial State Setup ---
 
             setTimeout(() => {
                 // This ensures it runs after ComfyUI loads all widget values
                 selectedColor = colorWidget.value || "#FF0000";
                 gradientWidth = node.size[0] - PADDING * 2;
                 pickerPos = calculatePickerPosFromHex(selectedColor);
                 updateColorOutput();
             }, 0);
         }
     }
 });