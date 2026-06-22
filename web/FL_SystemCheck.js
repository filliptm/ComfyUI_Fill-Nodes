// custom_nodes/FL_SystemCheck.js
// Render system info in a DOM widget so it stays visible on the current
// ComfyUI frontend instead of relying on legacy canvas foreground drawing.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ICONS = {
    "Operating System": "🖥️",
    "CPU": "⚙️",
    "RAM": "🧠",
    "GPU": "🎮",
    "CUDA version": "🚀",
    "Python version": "🐍",
    "PyTorch": "🔥",
    "xformers": "⚡",
    "torchvision": "👁️",
    "numpy": "🔢",
    "Pillow": "🖼️",
    "OpenCV": "📷",
    "transformers": "🤖",
};

const COLORS = {
    "Operating System": "#4a90e2",
    "CPU": "#50c878",
    "RAM": "#9b59b6",
    "GPU": "#e74c3c",
    "CUDA version": "#f39c12",
    "Python version": "#3498db",
    "PyTorch": "#e67e22",
    "xformers": "#1abc9c",
    "torchvision": "#34495e",
    "numpy": "#2ecc71",
    "Pillow": "#e84393",
    "OpenCV": "#c5a01c",
    "transformers": "#6c5ce7",
};

const iconFor = (k) => ICONS[k] || "ℹ️";
const colorFor = (k) => COLORS[k] || "#95a5a6";

app.registerExtension({
    name: "FL.SystemCheck",
    async nodeCreated(node) {
        const comfyClass = (node.constructor && node.constructor.comfyClass) || node.comfyClass || "";
        if (comfyClass !== "FL_SystemCheck") return;

        node.color = "#2a363b";
        node.bgcolor = "#4F0074";

        const container = document.createElement("div");
        container.style.cssText =
            "display:flex;flex-direction:column;gap:6px;padding:6px;box-sizing:border-box;" +
            "width:100%;height:100%;overflow:auto;font-family:Arial,sans-serif;font-size:12px;color:#fff;";

        const placeholder = document.createElement("div");
        placeholder.textContent = 'Click "Run System Check".';
        placeholder.style.opacity = "0.6";
        container.appendChild(placeholder);

        node.addWidget("button", "Run System Check", null, () => runSystemCheck(node, container));

        node.addDOMWidget("fl_system_info", "div", container, {
            getMinHeight: () => 260,
            hideOnZoom: false,
            serialize: false,
        });

        const [oldWidth, oldHeight] = node.size;
        node.setSize([Math.max(oldWidth, 360), Math.max(oldHeight, 340)]);
    },
});

async function runSystemCheck(node, container) {
    container.innerHTML = "";
    const loading = document.createElement("div");
    loading.textContent = "Checking...";
    loading.style.opacity = "0.7";
    container.appendChild(loading);

    try {
        const response = await api.fetchApi("/fl_system_info");
        if (!response.ok) throw new Error("HTTP " + response.status);
        const info = await response.json();
        ["Env: PYTHONPATH", "Env: CUDA_HOME", "Env: LD_LIBRARY_PATH"].forEach((k) => delete info[k]);
        renderInfo(container, info);
        node.setDirtyCanvas(true, true);
    } catch (e) {
        container.innerHTML = "";
        const err = document.createElement("div");
        err.style.color = "#ff6b6b";
        err.textContent = "Error: " + e.message + " (check the console).";
        container.appendChild(err);
        console.error("[FL System Check]", e);
    }
}

function renderInfo(container, info) {
    container.innerHTML = "";
    for (const [key, value] of Object.entries(info)) {
        const row = document.createElement("div");
        row.style.cssText =
            "display:flex;align-items:center;gap:8px;border-radius:5px;padding:5px 8px;" +
            "background:" + colorFor(key) + ";";

        const icon = document.createElement("span");
        icon.textContent = iconFor(key);

        const label = document.createElement("span");
        label.textContent = key;
        label.style.fontWeight = "bold";

        const val = document.createElement("span");
        val.textContent = value;
        val.style.cssText = "margin-left:auto;text-align:right;word-break:break-word;";

        row.append(icon, label, val);
        container.appendChild(row);
    }
}
