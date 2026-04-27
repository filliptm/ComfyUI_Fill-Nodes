import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

// Preview-toggle pattern lifted from FL_PasteOnCanvas: backend sends a base64
// PNG when show_preview is enabled, frontend draws it below the widgets.

app.registerExtension({
  name: "FL.DepthBlur",
  async nodeCreated(node) {
    if (node.comfyClass !== "FL_DepthBlur") return;

    const MIN_WIDTH = 280;
    const MIN_HEIGHT_WITH_PREVIEW = 760;
    const MIN_HEIGHT_WITHOUT_PREVIEW = 460;
    const PADDING = 10;

    const img = new Image();
    img.onload = () => node.setDirtyCanvas(true);

    node.onDrawBackground = function (ctx) {
      if (this.flags.collapsed) return;

      const showPreviewWidget = this.widgets?.find((w) => w.name === "show_preview");
      const showPreview = showPreviewWidget ? showPreviewWidget.value : false;
      if (!showPreview || !img.src) return;

      const [w, h] = this.size;
      const lastWidget = node.widgets[node.widgets.length - 1];
      const lastWidgetY = lastWidget?.last_y || 0;
      const IMAGE_Y_OFFSET = lastWidgetY + 30;
      const imageArea = h - IMAGE_Y_OFFSET - PADDING;
      if (imageArea <= 50) return;

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
    };

    api.addEventListener("fl_depth_blur_preview", (event) => {
      if (event.detail?.image) {
        img.src = event.detail.image;
      }
    });

    function updateNodeSize() {
      const showPreviewWidget = node.widgets?.find((w) => w.name === "show_preview");
      const showPreview = showPreviewWidget ? showPreviewWidget.value : false;
      const minHeight = showPreview ? MIN_HEIGHT_WITH_PREVIEW : MIN_HEIGHT_WITHOUT_PREVIEW;
      node.size[0] = Math.max(MIN_WIDTH, node.size[0]);
      node.size[1] = Math.max(minHeight, node.size[1]);
    }

    node.onResize = updateNodeSize;
    updateNodeSize();

    const showPreviewWidget = node.widgets?.find((w) => w.name === "show_preview");
    if (showPreviewWidget) {
      const originalCallback = showPreviewWidget.callback;
      showPreviewWidget.callback = function (value) {
        if (originalCallback) originalCallback.apply(this, arguments);
        if (!value) {
          node.size[1] = MIN_HEIGHT_WITHOUT_PREVIEW;
        } else {
          node.size[1] = MIN_HEIGHT_WITH_PREVIEW;
        }
        updateNodeSize();
        node.setDirtyCanvas(true);
      };
    }
  },
});
