import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

// FL_KsamplerSEG_Regions: on-node previz showing the Voronoi tessellation
// overlay. Updated via PromptServer event "fl_seg_regions_preview".

const STYLES = `
  .flks-seg-regions-widget {
    background: #18181b;
    border-radius: 10px;
    border: 1px solid #27272a;
    overflow: hidden;
    position: relative;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #fafafa;
    box-sizing: border-box;
    height: 100%;
    min-height: 280px;
    display: flex;
    flex-direction: column;
  }
  .flks-seg-regions-widget * { box-sizing: border-box; }
  .flks-seg-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 10px;
    background: #1f1f23;
    border-bottom: 1px solid #27272a;
    flex-shrink: 0;
  }
  .flks-seg-title {
    font-size: 11px;
    font-weight: 600;
    color: #fafafa;
  }
  .flks-seg-badge {
    font-size: 10px;
    padding: 2px 7px;
    background: #06b6d4;
    color: white;
    border-radius: 9px;
    font-weight: 600;
  }
  .flks-seg-canvas-wrap {
    flex: 1;
    position: relative;
    background: #0f0f12;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 200px;
  }
  .flks-seg-canvas-wrap img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
  }
  .flks-seg-empty {
    color: #71717a;
    font-size: 11px;
    text-align: center;
    padding: 20px;
  }
  .flks-seg-footer {
    padding: 5px 10px;
    background: #1f1f23;
    border-top: 1px solid #27272a;
    font-size: 9px;
    color: #71717a;
    text-align: center;
    flex-shrink: 0;
  }
`;

class SegRegionsPreview {
  constructor({ node, container }) {
    this.node = node;
    this.container = container;
    this.element = document.createElement("div");
    this.element.className = "flks-seg-regions-widget";
    this.injectStyles();
    this.build();
    this.container.appendChild(this.element);
  }

  injectStyles() {
    const id = "flks-seg-regions-styles";
    if (!document.getElementById(id)) {
      const s = document.createElement("style");
      s.id = id;
      s.textContent = STYLES;
      document.head.appendChild(s);
    }
  }

  build() {
    this.element.innerHTML = `
      <div class="flks-seg-header">
        <span class="flks-seg-title">Region Tessellation</span>
        <span class="flks-seg-badge" data-role="count">—</span>
      </div>
      <div class="flks-seg-canvas-wrap" data-role="canvas-wrap">
        <div class="flks-seg-empty">Run the node to preview the tessellation here.</div>
      </div>
      <div class="flks-seg-footer">cyan = cell boundary · yellow = region index</div>
    `;
    this.countEl = this.element.querySelector('[data-role="count"]');
    this.canvasWrap = this.element.querySelector('[data-role="canvas-wrap"]');
  }

  setPreview(imageDataUri, count, sizeWH) {
    this.canvasWrap.innerHTML = "";
    if (!imageDataUri) return;
    const img = document.createElement("img");
    img.src = imageDataUri;
    img.alt = `tessellation, ${count} regions`;
    if (sizeWH && sizeWH.length === 2) {
      img.title = `source ${sizeWH[0]}x${sizeWH[1]} px`;
    }
    this.canvasWrap.appendChild(img);
    if (this.countEl) this.countEl.textContent = `${count} regions`;
  }

  dispose() {
    if (this.element && this.element.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }
  }
}

const INSTANCES = new Map();

app.registerExtension({
  name: "ComfyUI.FL_KsamplerSEG_Regions",
  nodeCreated(node) {
    const comfyClass = (node.constructor && node.constructor.comfyClass) || "";
    if (comfyClass !== "FL_KsamplerSEG_Regions") return;

    const container = document.createElement("div");
    container.id = `flks-seg-regions-container-${node.id}`;
    container.style.width = "100%";
    container.style.height = "100%";
    container.style.minHeight = "280px";

    const widget = node.addDOMWidget(
      "regions_preview",
      "flks-seg-regions-preview",
      container,
      {
        getMinHeight: () => 320,
        hideOnZoom: false,
        serialize: false,
      }
    );

    const [oldW, oldH] = node.size;
    node.setSize([Math.max(oldW, 380), Math.max(oldH, 560)]);

    setTimeout(() => {
      const inst = new SegRegionsPreview({ node, container });
      INSTANCES.set(node.id, inst);
    }, 50);

    widget.onRemove = () => {
      const inst = INSTANCES.get(node.id);
      if (inst) {
        inst.dispose();
        INSTANCES.delete(node.id);
      }
    };
  },
});

api.addEventListener("fl_seg_regions_preview", (event) => {
  const detail = event.detail;
  if (!detail) return;
  const nodeId = parseInt(detail.node, 10);
  const inst = INSTANCES.get(nodeId);
  if (!inst) return;
  inst.setPreview(detail.image, detail.count, detail.size);
});
