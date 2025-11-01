import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

app.registerExtension({
    name: "FL.PasteOnCanvas",
    async nodeCreated(node) {
        if (node.comfyClass === "FL_PasteOnCanvas") {
            const MIN_WIDTH = 250;
            const MIN_HEIGHT_WITH_PREVIEW = 650;  // Space for 10 widgets + preview
            const MIN_HEIGHT_WITHOUT_PREVIEW = 320;  // Space for 10 widgets only
            const PADDING = 10;

            // Add image preview
            const img = new Image();
            img.onload = () => node.setDirtyCanvas(true);

            node.onDrawBackground = function(ctx) {
                if (!this.flags.collapsed) {
                    // Get the show_preview widget value
                    const showPreviewWidget = this.widgets?.find(w => w.name === "show_preview");
                    const showPreview = showPreviewWidget ? showPreviewWidget.value : false;

                    // Only draw if preview is enabled and image is loaded
                    if (!showPreview || !img.src) {
                        return;
                    }

                    const [w, h] = this.size;

                    // Calculate the Y position of the last widget
                    const lastWidget = node.widgets[node.widgets.length - 1];
                    const lastWidgetY = lastWidget.last_y || 0;

                    // Set the image Y offset to be just below the last widget
                    const IMAGE_Y_OFFSET = lastWidgetY + 30;

                    const imageArea = h - IMAGE_Y_OFFSET - PADDING;

                    // Draw image
                    if (img.src && imageArea > 50) {
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
            api.addEventListener("fl_paste_on_canvas", (event) => {
                if (event.detail.image) {
                    img.src = event.detail.image;
                }
            });

            function updateNodeSize() {
                // Check if preview is enabled
                const showPreviewWidget = node.widgets?.find(w => w.name === "show_preview");
                const showPreview = showPreviewWidget ? showPreviewWidget.value : false;

                // Use different minimum heights based on preview state
                const minHeight = showPreview ? MIN_HEIGHT_WITH_PREVIEW : MIN_HEIGHT_WITHOUT_PREVIEW;

                node.size[0] = Math.max(MIN_WIDTH, node.size[0]);
                node.size[1] = Math.max(minHeight, node.size[1]);
            }

            node.onResize = updateNodeSize;
            updateNodeSize();

            // Update size when preview toggle changes
            const showPreviewWidget = node.widgets?.find(w => w.name === "show_preview");
            if (showPreviewWidget) {
                const originalCallback = showPreviewWidget.callback;
                showPreviewWidget.callback = function(value) {
                    if (originalCallback) {
                        originalCallback.apply(this, arguments);
                    }

                    // If toggling preview off, shrink the node height to minimum
                    if (!value) {
                        node.size[1] = MIN_HEIGHT_WITHOUT_PREVIEW;
                    } else {
                        // If toggling preview on, grow the node height to minimum with preview
                        node.size[1] = MIN_HEIGHT_WITH_PREVIEW;
                    }

                    updateNodeSize();
                    node.setDirtyCanvas(true);
                };
            }
        }
    }
});
