/**
 * File: FL_ShowText.js
 * Project: ComfyUI_Fill-Nodes
 *
 * Displays the incoming text on the node via a styled DOM widget and
 * passes the same string through to the output socket.
 */

import { app } from "../../../../scripts/app.js";

const STYLES = `
  .flshowtext-host {
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
  }
  .flshowtext-widget {
    background: #17181c;
    border: 1px solid #2a2d34;
    border-radius: 8px;
    color: #f4f4f5;
    display: flex;
    flex: 1 1 auto;
    flex-direction: column;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 12px;
    line-height: 1.4;
    min-height: 0;
    overflow: hidden;
    padding: 0;
    box-sizing: border-box;
    width: 100%;
  }
  .flshowtext-widget * { box-sizing: border-box; }
  .flshowtext-header {
    align-items: center;
    background: #101114;
    border-bottom: 1px solid #2a2d34;
    color: #cbd5e1;
    display: flex;
    flex: 0 0 auto;
    font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 10px;
    font-weight: 700;
    gap: 8px;
    justify-content: space-between;
    letter-spacing: 0.04em;
    padding: 6px 10px;
    text-transform: uppercase;
  }
  .flshowtext-badge {
    background: #27272a;
    border-radius: 999px;
    color: #a1a1aa;
    font-size: 9px;
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    letter-spacing: 0;
    padding: 2px 7px;
    text-transform: none;
  }
  .flshowtext-body {
    color: #e5e7eb;
    flex: 1 1 auto;
    min-height: 0;
    overflow-x: hidden;
    overflow-y: auto;
    padding: 10px;
    white-space: pre-wrap;
    word-break: break-word;
    scrollbar-width: thin;
    scrollbar-color: #3f3f46 #17181c;
  }
  .flshowtext-body::-webkit-scrollbar { width: 8px; }
  .flshowtext-body::-webkit-scrollbar-track { background: #17181c; }
  .flshowtext-body::-webkit-scrollbar-thumb {
    background: #3f3f46;
    border-radius: 999px;
  }
  .flshowtext-body::-webkit-scrollbar-thumb:hover { background: #52525b; }
  .flshowtext-body.empty {
    color: #71717a;
    font-style: italic;
  }
`;

const PLACEHOLDER = "Waiting for input. Run the workflow to display text.";

function injectStyles() {
  const id = "flshowtext-styles";
  if (document.getElementById(id)) return;
  const style = document.createElement("style");
  style.id = id;
  style.textContent = STYLES;
  document.head.appendChild(style);
}

function formatLength(text) {
  const chars = text.length;
  const lines = text ? text.split("\n").length : 0;
  return `${chars} chars · ${lines} line${lines === 1 ? "" : "s"}`;
}

function buildPanel() {
  const root = document.createElement("div");
  root.className = "flshowtext-widget";
  root.innerHTML = `
    <div class="flshowtext-header">
      <span>Show Text</span>
      <span class="flshowtext-badge" data-role="length">empty</span>
    </div>
    <div class="flshowtext-body empty" data-role="body"></div>
  `;
  const body = root.querySelector('[data-role="body"]');
  const badge = root.querySelector('[data-role="length"]');
  body.textContent = PLACEHOLDER;
  return { root, body, badge };
}

function attachWheelCapture(body) {
  body.addEventListener(
    "wheel",
    (event) => {
      const canScrollDown =
        event.deltaY > 0 && body.scrollTop + body.clientHeight < body.scrollHeight - 1;
      const canScrollUp = event.deltaY < 0 && body.scrollTop > 0;
      if (canScrollDown || canScrollUp) {
        body.scrollTop += event.deltaY;
        event.preventDefault();
        event.stopPropagation();
      }
    },
    { passive: false }
  );
}

app.registerExtension({
  name: "fl.node.FL_ShowText",
  nodeCreated(node) {
    const comfyClass = (node.constructor && node.constructor.comfyClass) || "";
    if (comfyClass !== "FL_ShowText") return;

    injectStyles();

    const host = document.createElement("div");
    host.className = "flshowtext-host";

    const panel = buildPanel();
    host.appendChild(panel.root);

    attachWheelCapture(panel.body);

    node.addDOMWidget("fl_showtext_display", "fl-showtext", host, {
      getMinHeight: () => 80,
      hideOnZoom: false,
      serialize: false,
    });

    const setText = (text) => {
      const value = text == null ? "" : String(text);
      if (value === "") {
        panel.body.classList.add("empty");
        panel.body.textContent = PLACEHOLDER;
        panel.badge.textContent = "empty";
      } else {
        panel.body.classList.remove("empty");
        panel.body.textContent = value;
        panel.badge.textContent = formatLength(value);
      }
    };

    const [oldW, oldH] = node.size;
    node.setSize([Math.max(oldW, 320), Math.max(oldH, 220)]);

    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      originalOnExecuted?.apply(this, arguments);
      const lines = message?.text;
      if (Array.isArray(lines) && lines.length > 0) {
        setText(lines.join("\n"));
      }
    };

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
      originalOnConfigure?.apply(this, arguments);
      setText("");
    };
  },
});
