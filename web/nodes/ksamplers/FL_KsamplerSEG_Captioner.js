import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

// FL_KsamplerSEG_Captioner: on-node thumbnail grid that fills with regions +
// captions as they arrive (event "fl_seg_captioner_progress").

const STYLES = `
  .flks-seg-cap-widget {
    background: #18181b;
    border-radius: 10px;
    border: 1px solid #27272a;
    overflow: hidden;
    position: relative;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #fafafa;
    box-sizing: border-box;
    height: 100%;
    min-height: 320px;
    display: flex;
    flex-direction: column;
  }
  .flks-seg-cap-widget * { box-sizing: border-box; }
  .flks-seg-cap-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 10px;
    background: #1f1f23;
    border-bottom: 1px solid #27272a;
    flex-shrink: 0;
  }
  .flks-seg-cap-title {
    font-size: 11px;
    font-weight: 600;
  }
  .flks-seg-cap-progress {
    font-size: 10px;
    padding: 2px 7px;
    background: #06b6d4;
    color: white;
    border-radius: 9px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }
  .flks-seg-cap-grid {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 8px;
    align-content: start;
    background: #0f0f12;
  }
  .flks-seg-cap-tile {
    background: #1f1f23;
    border-radius: 6px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    border: 1px solid #27272a;
  }
  .flks-seg-cap-tile-img-wrap {
    position: relative;
    width: 100%;
    aspect-ratio: 1;
    background: #0f0f12;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .flks-seg-cap-tile-img-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  .flks-seg-cap-tile-idx {
    position: absolute;
    top: 4px;
    left: 4px;
    background: #fde047;
    color: #0f0f12;
    font-size: 10px;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 3px;
  }
  .flks-seg-cap-tile-caption {
    padding: 5px 7px;
    font-size: 10px;
    line-height: 1.3;
    color: #d4d4d8;
    min-height: 50px;
    word-break: break-word;
  }
  .flks-seg-cap-tile-caption.error { color: #ef4444; }
  .flks-seg-cap-empty {
    grid-column: 1 / -1;
    color: #71717a;
    font-size: 11px;
    text-align: center;
    padding: 30px 10px;
  }
`;

class SegCaptionerWidget {
  constructor({ node, container }) {
    this.node = node;
    this.container = container;
    this.element = document.createElement("div");
    this.element.className = "flks-seg-cap-widget";
    this.tiles = new Map(); // region_index -> tile element
    this.expectedTotal = 0;
    this.received = 0;
    this.injectStyles();
    this.build();
    this.container.appendChild(this.element);
  }

  injectStyles() {
    const id = "flks-seg-captioner-styles";
    if (!document.getElementById(id)) {
      const s = document.createElement("style");
      s.id = id;
      s.textContent = STYLES;
      document.head.appendChild(s);
    }
  }

  build() {
    this.element.innerHTML = `
      <div class="flks-seg-cap-header">
        <span class="flks-seg-cap-title">Region Captions</span>
        <span class="flks-seg-cap-progress" data-role="progress">—</span>
      </div>
      <div class="flks-seg-cap-grid" data-role="grid">
        <div class="flks-seg-cap-empty">Run the node to caption regions.</div>
      </div>
    `;
    this.progressEl = this.element.querySelector('[data-role="progress"]');
    this.gridEl = this.element.querySelector('[data-role="grid"]');
  }

  reset(total) {
    this.tiles.clear();
    this.expectedTotal = total;
    this.received = 0;
    this.gridEl.innerHTML = "";
    this.progressEl.textContent = `0 / ${total}`;
  }

  addOrUpdate({ region_index, caption, thumbnail, total }) {
    if (this.expectedTotal === 0 && total) {
      this.reset(total);
    } else if (total && total !== this.expectedTotal) {
      this.expectedTotal = total;
    }

    let tile = this.tiles.get(region_index);
    if (!tile) {
      tile = document.createElement("div");
      tile.className = "flks-seg-cap-tile";
      tile.innerHTML = `
        <div class="flks-seg-cap-tile-img-wrap">
          <span class="flks-seg-cap-tile-idx">${region_index}</span>
        </div>
        <div class="flks-seg-cap-tile-caption"></div>
      `;
      this.tiles.set(region_index, tile);
      // Insert in index order.
      const sortedIdx = [...this.tiles.keys()].sort((a, b) => a - b);
      const myPos = sortedIdx.indexOf(region_index);
      const after = sortedIdx[myPos + 1];
      if (after !== undefined && this.tiles.has(after)) {
        this.gridEl.insertBefore(tile, this.tiles.get(after));
      } else {
        this.gridEl.appendChild(tile);
      }
    }

    if (thumbnail) {
      const wrap = tile.querySelector(".flks-seg-cap-tile-img-wrap");
      let img = wrap.querySelector("img");
      if (!img) {
        img = document.createElement("img");
        wrap.insertBefore(img, wrap.firstChild);
      }
      img.src = thumbnail;
    }

    const cap = tile.querySelector(".flks-seg-cap-tile-caption");
    cap.textContent = caption || "";
    cap.classList.toggle("error", !!(caption && caption.startsWith("[caption failed")));

    this.received = this.tiles.size;
    if (this.expectedTotal > 0) {
      this.progressEl.textContent = `${this.received} / ${this.expectedTotal}`;
    }
  }

  dispose() {
    if (this.element && this.element.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }
  }
}

const INSTANCES = new Map();

app.registerExtension({
  name: "ComfyUI.FL_KsamplerSEG_Captioner",
  nodeCreated(node) {
    const comfyClass = (node.constructor && node.constructor.comfyClass) || "";
    if (comfyClass !== "FL_KsamplerSEG_Captioner") return;

    const container = document.createElement("div");
    container.id = `flks-seg-cap-container-${node.id}`;
    container.style.width = "100%";
    container.style.height = "100%";
    container.style.minHeight = "320px";

    const widget = node.addDOMWidget(
      "captioner_preview",
      "flks-seg-captioner-preview",
      container,
      {
        getMinHeight: () => 360,
        hideOnZoom: false,
        serialize: false,
      }
    );

    const [oldW, oldH] = node.size;
    node.setSize([Math.max(oldW, 420), Math.max(oldH, 640)]);

    setTimeout(() => {
      const inst = new SegCaptionerWidget({ node, container });
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

// On execution start, reset the grid for this node.
api.addEventListener("executing", (event) => {
  const detail = event.detail;
  if (!detail || !detail.node) return;
  const nodeId = parseInt(detail.node, 10);
  const inst = INSTANCES.get(nodeId);
  if (inst) inst.reset(0);
});

api.addEventListener("fl_seg_captioner_progress", (event) => {
  const detail = event.detail;
  if (!detail) return;
  const nodeId = parseInt(detail.node, 10);
  const inst = INSTANCES.get(nodeId);
  if (!inst) return;
  inst.addOrUpdate({
    region_index: detail.region_index,
    caption: detail.caption,
    thumbnail: detail.thumbnail,
    total: detail.total,
  });
});
