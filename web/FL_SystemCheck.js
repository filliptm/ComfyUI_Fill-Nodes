// custom_nodes/FL_SystemCheck.js

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FL.SystemCheck",
    async nodeCreated(node) {
        if (node.comfyClass === "FL_SystemCheck") {
            node.color = "#2a363b";
            node.bgcolor = "#4F0074";
            node.systemInfo = null;
            node.animationOffset = 0;

            // Add button widget
            const widget = node.addWidget("button", "Run System Check", null, () => {
                runSystemCheck(node);
            });
            widget.serialize = false;

            // Override the default drawing behavior
            node.onDrawForeground = function(ctx) {
                if (this.systemInfo) {
                    drawSystemInfo(ctx, this);
                }
            };

            // Start animation loop
            animateNode(node);
        }
    },
});

function animateNode(node) {
    node.animationOffset += 0.005; // Adjust speed as needed
    if (node.animationOffset > 1) node.animationOffset -= 1;
    node.setDirtyCanvas(true);
    requestAnimationFrame(() => animateNode(node));
}

async function runSystemCheck(node) {
    try {
        const response = await fetch('/fl_system_info');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        node.systemInfo = await response.json();
        // Remove ENV variables
        delete node.systemInfo["Env: PYTHONPATH"];
        delete node.systemInfo["Env: CUDA_HOME"];
        delete node.systemInfo["Env: LD_LIBRARY_PATH"];
        node.setDirtyCanvas(true);
    } catch (error) {
        console.error("Error fetching system info:", error);
        node.systemInfo = { "Error": "Failed to fetch system information. Check console for details." };
        node.setDirtyCanvas(true);
    }
}

function drawSystemInfo(ctx, node) {
    const margin = 15;
    const componentHeight = 30;
    const componentSpacing = 10;
    let y = 60; // Start below the node title and button

    ctx.save();
    ctx.font = "12px Arial";

    const components = Object.entries(node.systemInfo).map(([key, value]) => ({
        key,
        value,
        icon: getIconForKey(key),
        color: getColorForKey(key)
    }));

    // Calculate max width
    let maxWidth = 0;
    components.forEach(comp => {
        ctx.font = "bold 12px Arial";
        const labelWidth = ctx.measureText(comp.key).width;
        ctx.font = "12px Arial";
        const valueWidth = ctx.measureText(comp.value).width;
        const componentWidth = labelWidth + valueWidth + 80; // 80 for padding and icon
        maxWidth = Math.max(maxWidth, componentWidth);
    });

    // Draw components
    components.forEach((comp, index) => {
        const staggeredOffset = (node.animationOffset + index * 0.1) % 1; // Stagger the animation
        drawComponent(ctx, comp.key, comp.value, y, maxWidth, componentHeight, comp.color, comp.icon, staggeredOffset);
        y += componentHeight + componentSpacing;
    });

    // Adjust node size
    node.size[0] = maxWidth + 2 * margin;
    node.size[1] = y + margin;

    ctx.restore();
}

function drawComponent(ctx, label, value, y, width, height, color, icon, animationOffset) {
    const radius = 5;
    const iconWidth = 30;
    const padding = 5;

    // Draw component background
    ctx.fillStyle = color;
    roundRect(ctx, 15, y, width, height, radius, true, false);

    // Draw icon background
    ctx.fillStyle = lightenColor(color, 20);
    roundRect(ctx, 15, y, iconWidth, height, radius, true, false);

    // Draw icon (centered)
    ctx.font = "16px Arial";
    ctx.fillStyle = "#ffffff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(icon, 15 + iconWidth / 2, y + height / 2);

    // Reset text alignment for label and value
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";

    // Draw label and value
    ctx.font = "bold 12px Arial";
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, 55, y + height / 2 + 5);

    ctx.font = "12px Arial";
    const valueWidth = ctx.measureText(value).width;
    ctx.fillText(value, width - valueWidth - padding, y + height / 2 + 5);

    // Draw animation
    drawFluidAnimation(ctx, 15, y, width, height, lightenColor(color, 10), animationOffset);
}

function drawFluidAnimation(ctx, x, y, width, height, color, offset) {
    ctx.save();
    ctx.globalCompositeOperation = 'overlay';
    ctx.globalAlpha = 0.3;

    const gradient = ctx.createLinearGradient(x, y, x + width, y);
    gradient.addColorStop(0, 'rgba(255,255,255,0)');
    gradient.addColorStop(0.5, color);
    gradient.addColorStop(1, 'rgba(255,255,255,0)');

    ctx.fillStyle = gradient;

    // Create a wavy path
    ctx.beginPath();
    ctx.moveTo(x, y);

    const waveHeight = height * 0.4;
    const waveCount = 3;
    for (let i = 0; i <= width; i++) {
        const dx = i / width;
        const offsetY = Math.sin((dx + offset) * Math.PI * 2 * waveCount) * waveHeight;
        ctx.lineTo(x + i, y + height / 2 + offsetY);
    }

    ctx.lineTo(x + width, y + height);
    ctx.lineTo(x, y + height);
    ctx.closePath();
    ctx.fill();

    ctx.restore();
}

function roundRect(ctx, x, y, width, height, radius, fill, stroke) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + width, y, x + width, y + height, radius);
    ctx.arcTo(x + width, y + height, x, y + height, radius);
    ctx.arcTo(x, y + height, x, y, radius);
    ctx.arcTo(x, y, x + width, y, radius);
    ctx.closePath();
    if (fill) {
        ctx.fill();
    }
    if (stroke) {
        ctx.stroke();
    }
}

function lightenColor(color, percent) {
    const num = parseInt(color.replace("#", ""), 16),
          amt = Math.round(2.55 * percent),
          R = (num >> 16) + amt,
          G = (num >> 8 & 0x00FF) + amt,
          B = (num & 0x0000FF) + amt;
    return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
        (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
        (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1);
}

function getIconForKey(key) {
    const iconMap = {
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
        "diffusers": "🌈"
    };
    return iconMap[key] || "ℹ️";
}

function getColorForKey(key) {
    const colorMap = {
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
        "diffusers": "#00cec9"
    };
    return colorMap[key] || "#95a5a6";
}