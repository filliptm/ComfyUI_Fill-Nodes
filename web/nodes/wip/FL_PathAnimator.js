/**
 * File: FL_PathAnimator.js
 * Project: ComfyUI_Fill-Nodes
 *
 * Interactive path animator with modal drawing editor
 */

import { app } from "../../../../../scripts/app.js";
import { api } from "../../../../../scripts/api.js";

function moveWidgetToTop(node, widget) {
    if (!widget) return;

    // Find the widget's current index
    const widgetIndex = node.widgets.indexOf(widget);
    if (widgetIndex > 0) {
        // Remove from current position
        node.widgets.splice(widgetIndex, 1);
        // Insert at the beginning
        node.widgets.unshift(widget);
    }
}

app.registerExtension({
    name: "FillNodes.PathAnimator",
    async nodeCreated(node) {
        if (node.comfyClass === "FL_PathAnimator") {
            // Find the paths_data widget immediately
            const pathsDataWidget = node.widgets.find(w => w.name === "paths_data");
            if (!pathsDataWidget) {
                console.error("FL_PathAnimator: 'paths_data' widget not found!");
                return;
            }

            // Move it to the top of the widget list so it renders first
            moveWidgetToTop(node, pathsDataWidget);

            // Initialize cached background image storage on the widget
            pathsDataWidget._cachedBackgroundImage = null;

            // Add "Edit Paths" button
            const editButton = node.addWidget("button", "Edit Paths", null, () => {
                openPathEditor(node, pathsDataWidget);
            });

            // Add path count display (readonly)
            const pathCountWidget = node.addWidget("text", "Path Count", "0 paths", null);
            pathCountWidget.disabled = true;

            // Update path count when paths change
            function updatePathCount() {
                try {
                    const data = JSON.parse(pathsDataWidget.value);
                    const count = data.paths ? data.paths.length : 0;
                    const staticCount = data.paths ? data.paths.filter(p => p.isSinglePoint || p.points.length === 1).length : 0;
                    const motionCount = count - staticCount;
                    pathCountWidget.value = `${count} path${count !== 1 ? 's' : ''} (${staticCount} static, ${motionCount} motion)`;
                } catch (e) {
                    pathCountWidget.value = "0 paths";
                }
            }

            // Initial update
            updatePathCount();

            // Store update function for later use
            node._updatePathCount = updatePathCount;
        }
    }
});

function openPathEditor(node, pathsDataWidget) {
    // Get frame dimensions from node widgets
    const frameWidthWidget = node.widgets.find(w => w.name === "frame_width");
    const frameHeightWidget = node.widgets.find(w => w.name === "frame_height");

    const frameWidth = frameWidthWidget ? frameWidthWidget.value : 512;
    const frameHeight = frameHeightWidget ? frameHeightWidget.value : 512;

    // Create and show modal
    const modal = new PathEditorModal(node, pathsDataWidget, frameWidth, frameHeight);
    modal.show();
}

class PathEditorModal {
    constructor(node, pathsDataWidget, frameWidth, frameHeight) {
        this.node = node;
        this.pathsDataWidget = pathsDataWidget;
        this.frameWidth = frameWidth;
        this.frameHeight = frameHeight;
        this.paths = [];
        this.currentPath = null;
        this.selectedPathIndex = -1;
        this.isDrawing = false;
        this.tool = 'pencil'; // pencil, eraser, select, point
        this.currentColor = this.getRandomColor();
        this.backgroundImage = null;
        this.canvasScale = 1.0;
        this.canvasOffsetX = 0;
        this.canvasOffsetY = 0;
        this.pathThickness = 3; // Visual thickness for displaying paths
        this.shiftPressed = false; // Track shift key state

        // Load existing paths
        this.loadPaths();

        // Load cached background image if it exists
        this.loadCachedBackgroundImage();

        // Create modal elements
        this.createModal();

        // Setup keyboard handlers
        this.setupKeyboardHandlers();
    }

    setupKeyboardHandlers() {
        this.keydownHandler = (e) => {
            // Track shift key
            if (e.key === 'Shift') {
                this.shiftPressed = true;
            }

            // Escape to save and close
            if (e.key === 'Escape') {
                this.savePaths();
                this.close();
            }
        };

        this.keyupHandler = (e) => {
            if (e.key === 'Shift') {
                this.shiftPressed = false;
            }
        };

        document.addEventListener('keydown', this.keydownHandler);
        document.addEventListener('keyup', this.keyupHandler);
    }

    loadCachedBackgroundImage() {
        // Check if there's a cached background image
        if (this.pathsDataWidget._cachedBackgroundImage) {
            const img = new Image();
            img.onload = () => {
                this.backgroundImage = img;
                if (this.canvas) {
                    this.canvas.width = img.width;
                    this.canvas.height = img.height;
                    this.render();
                }
            };
            img.src = this.pathsDataWidget._cachedBackgroundImage;
        }
    }

    loadPaths() {
        try {
            const data = JSON.parse(this.pathsDataWidget.value);
            this.paths = data.paths || [];
        } catch (e) {
            console.error("Error loading paths:", e);
            this.paths = [];
        }
    }

    savePaths() {
        const data = {
            paths: this.paths,
            canvas_size: {
                width: this.canvas.width,
                height: this.canvas.height
            }
        };
        this.pathsDataWidget.value = JSON.stringify(data);

        // Update path count display
        if (this.node._updatePathCount) {
            this.node._updatePathCount();
        }
    }

    getRandomColor() {
        const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2'];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    createModal() {
        // Create modal overlay
        this.overlay = document.createElement('div');
        this.overlay.className = 'fl-path-editor-overlay';
        this.overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(4px);
            z-index: 10000;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: fadeIn 0.2s ease-out;
        `;

        // Create modal container
        this.container = document.createElement('div');
        this.container.className = 'fl-path-editor-container';
        this.container.style.cssText = `
            background: linear-gradient(145deg, #2d2d2d, #252525);
            border-radius: 12px;
            border: 1px solid #3a3a3a;
            width: 92%;
            height: 92%;
            max-width: 1600px;
            max-height: 1000px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.05);
            animation: slideIn 0.3s ease-out;
        `;

        // Create header
        this.createHeader();

        // Create main content area
        this.createMainContent();

        // Create footer
        this.createFooter();

        this.overlay.appendChild(this.container);

        // Close on overlay click
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) {
                this.close();
            }
        });
    }

    createHeader() {
        const header = document.createElement('div');
        header.style.cssText = `
            padding: 20px 24px;
            border-bottom: 1px solid #404040;
            background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0) 100%);
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-radius: 12px 12px 0 0;
        `;

        const titleContainer = document.createElement('div');
        titleContainer.style.cssText = `
            display: flex;
            flex-direction: column;
            gap: 4px;
        `;

        const title = document.createElement('h2');
        title.textContent = '✏️ Path Animator Editor';
        title.style.cssText = `
            margin: 0;
            color: #fff;
            font-size: 20px;
            font-weight: 600;
            letter-spacing: -0.5px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        `;

        const subtitle = document.createElement('div');
        subtitle.textContent = 'Press ESC to save & close | Hold SHIFT for straight lines';
        subtitle.style.cssText = `
            color: #888;
            font-size: 12px;
            font-weight: 400;
        `;

        titleContainer.appendChild(title);
        titleContainer.appendChild(subtitle);

        const closeBtn = document.createElement('button');
        closeBtn.textContent = '✕';
        closeBtn.style.cssText = `
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            color: #fff;
            font-size: 24px;
            cursor: pointer;
            padding: 0;
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        `;
        closeBtn.onmouseover = () => {
            closeBtn.style.background = 'rgba(255, 77, 77, 0.8)';
            closeBtn.style.borderColor = 'rgba(255, 77, 77, 1)';
            closeBtn.style.transform = 'scale(1.05)';
        };
        closeBtn.onmouseout = () => {
            closeBtn.style.background = 'rgba(255, 255, 255, 0.05)';
            closeBtn.style.borderColor = 'rgba(255, 255, 255, 0.1)';
            closeBtn.style.transform = 'scale(1)';
        };
        closeBtn.onclick = () => this.close();

        header.appendChild(titleContainer);
        header.appendChild(closeBtn);
        this.container.appendChild(header);
    }

    createMainContent() {
        const content = document.createElement('div');
        content.style.cssText = `
            flex: 1;
            display: flex;
            overflow: hidden;
        `;

        // Create toolbar
        this.createToolbar(content);

        // Create canvas area
        this.createCanvasArea(content);

        // Create sidebar
        this.createSidebar(content);

        this.container.appendChild(content);
    }

    createToolbarButton(icon, title, isActive = false) {
        const btn = document.createElement('button');
        btn.textContent = icon;
        btn.title = title;
        btn.style.cssText = `
            width: 50px;
            height: 50px;
            border: 2px solid ${isActive ? '#4ECDC4' : 'rgba(255, 255, 255, 0.15)'};
            background: ${isActive ? 'rgba(78, 205, 196, 0.2)' : 'rgba(255, 255, 255, 0.05)'};
            color: #fff;
            cursor: pointer;
            border-radius: 8px;
            font-size: 20px;
            transition: all 0.2s ease;
            box-shadow: ${isActive ? '0 0 12px rgba(78, 205, 196, 0.3)' : 'none'};
        `;
        btn.onmouseover = () => {
            if (!isActive) {
                btn.style.background = 'rgba(255, 255, 255, 0.1)';
                btn.style.borderColor = 'rgba(255, 255, 255, 0.25)';
                btn.style.transform = 'scale(1.05)';
            }
        };
        btn.onmouseout = () => {
            if (!isActive) {
                btn.style.background = 'rgba(255, 255, 255, 0.05)';
                btn.style.borderColor = 'rgba(255, 255, 255, 0.15)';
                btn.style.transform = 'scale(1)';
            }
        };
        return btn;
    }

    createToolbar(parent) {
        const toolbar = document.createElement('div');
        toolbar.style.cssText = `
            width: 70px;
            background: linear-gradient(180deg, #1e1e1e 0%, #181818 100%);
            border-right: 1px solid #3a3a3a;
            padding: 12px 10px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            box-shadow: 2px 0 8px rgba(0, 0, 0, 0.3);
        `;

        // Add image upload button
        const uploadBtn = this.createToolbarButton('🖼️', 'Load Background Image');
        uploadBtn.onclick = () => this.loadImage();
        toolbar.appendChild(uploadBtn);

        // Add clear image button
        const clearImgBtn = this.createToolbarButton('🚫', 'Clear Background Image');
        clearImgBtn.onclick = () => this.clearImage();
        toolbar.appendChild(clearImgBtn);

        // Add separator
        const separator = document.createElement('div');
        separator.style.cssText = `
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            margin: 8px 0;
        `;
        toolbar.appendChild(separator);

        const tools = [
            { name: 'pencil', icon: '✏️', title: 'Draw Path (Motion)' },
            { name: 'point', icon: '📍', title: 'Add Static Point (Anchor)' },
            { name: 'eraser', icon: '🗑️', title: 'Erase Path' },
            { name: 'select', icon: '↖️', title: 'Select Path' },
        ];

        const toolButtons = [];
        tools.forEach(tool => {
            const btn = this.createToolbarButton(tool.icon, tool.title, this.tool === tool.name);
            btn.dataset.tool = tool.name;
            btn.onclick = () => {
                this.tool = tool.name;
                // Update canvas cursor
                if (this.tool === 'point') {
                    this.canvas.style.cursor = 'crosshair';
                } else if (this.tool === 'pencil') {
                    this.canvas.style.cursor = 'crosshair';
                } else if (this.tool === 'eraser') {
                    this.canvas.style.cursor = 'not-allowed';
                } else if (this.tool === 'select') {
                    this.canvas.style.cursor = 'pointer';
                }
                // Update all tool button states
                toolButtons.forEach(tb => {
                    const isActive = tb.dataset.tool === tool.name;
                    tb.style.border = `2px solid ${isActive ? '#4ECDC4' : 'rgba(255, 255, 255, 0.15)'}`;
                    tb.style.background = isActive ? 'rgba(78, 205, 196, 0.2)' : 'rgba(255, 255, 255, 0.05)';
                    tb.style.boxShadow = isActive ? '0 0 12px rgba(78, 205, 196, 0.3)' : 'none';
                });
            };
            toolButtons.push(btn);
            toolbar.appendChild(btn);
        });

        // Add another separator
        const separator2 = document.createElement('div');
        separator2.style.cssText = separator.style.cssText;
        toolbar.appendChild(separator2);

        // Add Lock Perimeter button
        const lockPerimeterBtn = this.createToolbarButton('🔒', 'Lock Perimeter - Add static shapes around border');
        lockPerimeterBtn.onclick = () => this.lockPerimeter();
        toolbar.appendChild(lockPerimeterBtn);

        // Add another separator
        const separator3 = document.createElement('div');
        separator3.style.cssText = separator.style.cssText;
        toolbar.appendChild(separator3);

        // Add thickness control
        const thicknessContainer = document.createElement('div');
        thicknessContainer.style.cssText = `
            padding: 8px 5px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        `;

        const thicknessLabel = document.createElement('label');
        thicknessLabel.textContent = 'Width';
        thicknessLabel.style.cssText = `
            color: #fff;
            font-size: 11px;
            text-align: center;
            opacity: 0.8;
        `;

        const thicknessSlider = document.createElement('input');
        thicknessSlider.type = 'range';
        thicknessSlider.min = '1';
        thicknessSlider.max = '10';
        thicknessSlider.value = this.pathThickness;
        thicknessSlider.style.cssText = `
            width: 100%;
            cursor: pointer;
            accent-color: #4ECDC4;
        `;
        thicknessSlider.oninput = (e) => {
            this.pathThickness = parseInt(e.target.value);
            thicknessValue.textContent = this.pathThickness;
            this.render();
        };

        const thicknessValue = document.createElement('div');
        thicknessValue.textContent = this.pathThickness;
        thicknessValue.style.cssText = `
            color: #4ECDC4;
            font-size: 14px;
            font-weight: bold;
            text-align: center;
        `;

        thicknessContainer.appendChild(thicknessLabel);
        thicknessContainer.appendChild(thicknessSlider);
        thicknessContainer.appendChild(thicknessValue);
        toolbar.appendChild(thicknessContainer);

        // Add clear all button
        const clearBtn = this.createToolbarButton('🗑️', 'Clear All Paths');
        clearBtn.style.marginTop = 'auto';
        clearBtn.onclick = () => {
            if (confirm('Clear all paths?')) {
                this.paths = [];
                this.selectedPathIndex = -1;
                this.updateSidebar();
                this.render();
            }
        };
        toolbar.appendChild(clearBtn);

        parent.appendChild(toolbar);
    }

    loadImage() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    const img = new Image();
                    img.onload = () => {
                        this.backgroundImage = img;
                        // Update canvas size to match image
                        this.canvas.width = img.width;
                        this.canvas.height = img.height;

                        // Cache the image data URL for later use
                        this.pathsDataWidget._cachedBackgroundImage = event.target.result;

                        this.render();
                    };
                    img.src = event.target.result;
                };
                reader.readAsDataURL(file);
            }
        };
        input.click();
    }

    clearImage() {
        if (confirm('Clear background image?')) {
            this.backgroundImage = null;
            this.pathsDataWidget._cachedBackgroundImage = null;
            // Reset canvas to frame dimensions
            this.canvas.width = this.frameWidth;
            this.canvas.height = this.frameHeight;
            this.render();
        }
    }

    lockPerimeter() {
        // Prompt user for number of points
        const numPoints = prompt('How many shapes around the perimeter?', '12');
        if (!numPoints || isNaN(numPoints) || numPoints < 1) return;

        const count = parseInt(numPoints);
        const w = this.canvas.width;
        const h = this.canvas.height;
        const perimeter = 2 * (w + h);
        const spacing = perimeter / count;

        // Create paths around the perimeter
        for (let i = 0; i < count; i++) {
            const d = i * spacing;
            let x, y;

            if (d < w) {
                // Top edge
                x = d;
                y = 0;
            } else if (d < w + h) {
                // Right edge
                x = w;
                y = d - w;
            } else if (d < 2 * w + h) {
                // Bottom edge
                x = w - (d - w - h);
                y = h;
            } else {
                // Left edge
                x = 0;
                y = h - (d - 2 * w - h);
            }

            // Create a static path (single point) at this position
            const path = {
                id: 'path_' + Date.now() + '_' + i,
                name: 'Perimeter ' + (i + 1),
                points: [{ x: Math.round(x), y: Math.round(y) }],
                color: this.getRandomColor(),
                isSinglePoint: true
            };
            this.paths.push(path);
        }

        this.updateSidebar();
        this.render();
    }

    createCanvasArea(parent) {
        const canvasContainer = document.createElement('div');
        canvasContainer.style.cssText = `
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: radial-gradient(circle at center, #1e1e1e 0%, #0a0a0a 100%);
            position: relative;
            overflow: hidden;
            padding: 20px;
        `;

        this.canvas = document.createElement('canvas');
        this.canvas.width = this.frameWidth;
        this.canvas.height = this.frameHeight;
        this.canvas.style.cssText = `
            border: 1px solid #4a4a4a;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
            cursor: crosshair;
            max-width: 100%;
            max-height: 100%;
            border-radius: 4px;
        `;

        this.ctx = this.canvas.getContext('2d');

        // Setup canvas event listeners
        this.setupCanvasEvents();

        canvasContainer.appendChild(this.canvas);
        parent.appendChild(canvasContainer);

        // Initial render
        this.render();
    }

    setupCanvasEvents() {
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.canvas.addEventListener('mouseleave', (e) => this.onMouseUp(e));
    }

    getCanvasCoords(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    onMouseDown(e) {
        const pos = this.getCanvasCoords(e);

        if (this.tool === 'pencil') {
            this.isDrawing = true;
            this.currentPath = {
                id: 'path_' + Date.now(),
                name: 'Path ' + (this.paths.length + 1),
                points: [pos],
                color: this.currentColor,
                closed: false,
                isSinglePoint: false
            };
        } else if (this.tool === 'point') {
            // Add single static point
            const path = {
                id: 'path_' + Date.now(),
                name: 'Static ' + (this.paths.filter(p => p.isSinglePoint).length + 1),
                points: [pos],
                color: this.currentColor,
                isSinglePoint: true
            };
            this.paths.push(path);
            this.selectedPathIndex = this.paths.length - 1;
            this.currentColor = this.getRandomColor();
            this.updateSidebar();
            this.render();
        } else if (this.tool === 'select') {
            // Find clicked path
            this.selectedPathIndex = this.findPathAtPoint(pos);
            this.updateSidebar();
            this.render();
        } else if (this.tool === 'eraser') {
            // Erase path at this point
            const pathIndex = this.findPathAtPoint(pos);
            if (pathIndex !== -1) {
                this.paths.splice(pathIndex, 1);
                this.selectedPathIndex = -1;
                this.updateSidebar();
                this.render();
            }
        }
    }

    onMouseMove(e) {
        if (this.isDrawing && this.tool === 'pencil') {
            const pos = this.getCanvasCoords(e);

            // If shift is pressed, draw straight line from last point
            if (this.shiftPressed && this.currentPath.points.length > 0) {
                // Replace the preview point if it exists, or add it
                const lastPoint = this.currentPath.points[this.currentPath.points.length - 1];

                // Determine if line should be horizontal, vertical, or diagonal
                const dx = Math.abs(pos.x - lastPoint.x);
                const dy = Math.abs(pos.y - lastPoint.y);

                let constrainedPos;
                if (dx > dy * 2) {
                    // Horizontal
                    constrainedPos = { x: pos.x, y: lastPoint.y };
                } else if (dy > dx * 2) {
                    // Vertical
                    constrainedPos = { x: lastPoint.x, y: pos.y };
                } else {
                    // Diagonal 45 degrees
                    const dist = Math.min(dx, dy);
                    constrainedPos = {
                        x: lastPoint.x + (pos.x > lastPoint.x ? dist : -dist),
                        y: lastPoint.y + (pos.y > lastPoint.y ? dist : -dist)
                    };
                }

                // Store original last point count to know if we're previewing
                if (!this.shiftPreviewPoint) {
                    this.shiftPreviewPoint = true;
                    this.currentPath.points.push(constrainedPos);
                } else {
                    this.currentPath.points[this.currentPath.points.length - 1] = constrainedPos;
                }
                this.render();
            } else {
                // Normal drawing - clear shift preview flag
                this.shiftPreviewPoint = false;

                // Only add point if it's far enough from the last point (smoothing)
                const lastPoint = this.currentPath.points[this.currentPath.points.length - 1];
                const dist = Math.sqrt(Math.pow(pos.x - lastPoint.x, 2) + Math.pow(pos.y - lastPoint.y, 2));

                if (dist > 3) { // Minimum distance between points
                    this.currentPath.points.push(pos);
                    this.render();
                }
            }
        }
    }

    onMouseUp(e) {
        if (this.isDrawing && this.currentPath) {
            // Clear shift preview flag
            this.shiftPreviewPoint = false;

            if (this.currentPath.points.length > 1) {
                this.paths.push(this.currentPath);
                this.selectedPathIndex = this.paths.length - 1;
                this.currentColor = this.getRandomColor();
                this.updateSidebar();
            } else if (this.currentPath.points.length === 1) {
                // Single click became a static point
                this.currentPath.isSinglePoint = true;
                this.currentPath.name = 'Static ' + (this.paths.filter(p => p.isSinglePoint).length + 1);
                this.paths.push(this.currentPath);
                this.selectedPathIndex = this.paths.length - 1;
                this.currentColor = this.getRandomColor();
                this.updateSidebar();
            }
            this.currentPath = null;
            this.isDrawing = false;
            this.render();
        }
    }

    findPathAtPoint(point, baseThreshold = 10) {
        // Scale threshold based on canvas resolution
        const scale = this.getRenderScale();
        const threshold = baseThreshold * scale;

        // Check single points first (easier to select)
        for (let i = this.paths.length - 1; i >= 0; i--) {
            const path = this.paths[i];
            if (path.isSinglePoint || path.points.length === 1) {
                const p = path.points[0];
                const dist = Math.sqrt(Math.pow(point.x - p.x, 2) + Math.pow(point.y - p.y, 2));
                if (dist < threshold) {
                    return i;
                }
            }
        }

        // Then check multi-point paths
        for (let i = this.paths.length - 1; i >= 0; i--) {
            const path = this.paths[i];
            if (!path.isSinglePoint && path.points.length > 1) {
                for (let j = 0; j < path.points.length - 1; j++) {
                    const p1 = path.points[j];
                    const p2 = path.points[j + 1];

                    // Check if point is near line segment
                    const dist = this.distanceToSegment(point, p1, p2);
                    if (dist < threshold) {
                        return i;
                    }
                }
            }
        }
        return -1;
    }

    distanceToSegment(point, p1, p2) {
        const A = point.x - p1.x;
        const B = point.y - p1.y;
        const C = p2.x - p1.x;
        const D = p2.y - p1.y;

        const dot = A * C + B * D;
        const lenSq = C * C + D * D;
        let param = -1;

        if (lenSq !== 0) param = dot / lenSq;

        let xx, yy;

        if (param < 0) {
            xx = p1.x;
            yy = p1.y;
        } else if (param > 1) {
            xx = p2.x;
            yy = p2.y;
        } else {
            xx = p1.x + param * C;
            yy = p1.y + param * D;
        }

        const dx = point.x - xx;
        const dy = point.y - yy;
        return Math.sqrt(dx * dx + dy * dy);
    }

    getRenderScale() {
        // Calculate scale factor based on canvas dimensions
        // Use 512 as the base resolution for consistent rendering
        const baseResolution = 512;
        const minDimension = Math.min(this.canvas.width, this.canvas.height);
        return minDimension / baseResolution;
    }

    render() {
        if (!this.ctx) return;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw background image
        if (this.backgroundImage && this.backgroundImage.complete) {
            this.ctx.drawImage(this.backgroundImage, 0, 0, this.canvas.width, this.canvas.height);
        } else {
            this.ctx.fillStyle = '#333';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        }

        // Draw saved paths
        this.paths.forEach((path, index) => {
            this.drawPath(path, index === this.selectedPathIndex);
        });

        // Draw current path being drawn
        if (this.currentPath) {
            this.drawPath(this.currentPath, true);
        }
    }

    drawPath(path, isSelected = false) {
        const isSinglePoint = path.isSinglePoint || path.points.length === 1;
        const scale = this.getRenderScale();

        if (isSinglePoint) {
            // Draw single point as a square (static anchor)
            const point = path.points[0];
            const baseSize = isSelected ? 12 : 8;
            const size = baseSize * scale;

            this.ctx.fillStyle = path.color;
            this.ctx.fillRect(point.x - size / 2, point.y - size / 2, size, size);

            // Draw border
            this.ctx.strokeStyle = isSelected ? '#4ECDC4' : '#fff';
            this.ctx.lineWidth = (isSelected ? 3 : 2) * scale;
            this.ctx.strokeRect(point.x - size / 2, point.y - size / 2, size, size);

            // Draw label for static points
            if (isSelected) {
                this.ctx.fillStyle = '#fff';
                this.ctx.font = `${12 * scale}px sans-serif`;
                this.ctx.fillText('📍 Static', point.x + 10 * scale, point.y - 10 * scale);
            }
        } else if (path.points.length >= 2) {
            // Draw multi-point path (motion path)
            this.ctx.beginPath();
            this.ctx.moveTo(path.points[0].x, path.points[0].y);

            for (let i = 1; i < path.points.length; i++) {
                this.ctx.lineTo(path.points[i].x, path.points[i].y);
            }

            this.ctx.strokeStyle = path.color;
            this.ctx.lineWidth = (isSelected ? this.pathThickness + 2 : this.pathThickness) * scale;
            this.ctx.lineCap = 'round';
            this.ctx.lineJoin = 'round';
            this.ctx.stroke();

            // Draw points
            if (isSelected) {
                path.points.forEach((point, idx) => {
                    this.ctx.beginPath();
                    this.ctx.arc(point.x, point.y, Math.max(4, this.pathThickness) * scale, 0, Math.PI * 2);
                    this.ctx.fillStyle = path.color;
                    this.ctx.fill();

                    // Draw point numbers
                    if (path.points.length < 20) { // Only if not too many points
                        this.ctx.fillStyle = '#fff';
                        this.ctx.font = `${10 * scale}px sans-serif`;
                        this.ctx.fillText(idx, point.x + 8 * scale, point.y - 8 * scale);
                    }
                });

                // Draw label for motion paths
                const midPoint = path.points[Math.floor(path.points.length / 2)];
                this.ctx.fillStyle = '#fff';
                this.ctx.font = `${12 * scale}px sans-serif`;
                this.ctx.fillText(`↗️ Motion (${path.points.length} pts)`, midPoint.x + 10 * scale, midPoint.y - 10 * scale);
            }
        }
    }

    createSidebar(parent) {
        this.sidebar = document.createElement('div');
        this.sidebar.style.cssText = `
            width: 220px;
            background: #1e1e1e;
            border-left: 1px solid #444;
            padding: 15px;
            overflow-y: auto;
        `;

        const title = document.createElement('h3');
        title.textContent = 'Paths';
        title.style.cssText = `
            margin: 0 0 15px 0;
            color: #fff;
            font-size: 14px;
            font-weight: 500;
        `;
        this.sidebar.appendChild(title);

        this.pathList = document.createElement('div');
        this.pathList.style.cssText = `
            display: flex;
            flex-direction: column;
            gap: 8px;
        `;
        this.sidebar.appendChild(this.pathList);

        parent.appendChild(this.sidebar);

        this.updateSidebar();
    }

    updateSidebar() {
        if (!this.pathList) return;

        this.pathList.innerHTML = '';

        this.paths.forEach((path, index) => {
            const isSinglePoint = path.isSinglePoint || path.points.length === 1;

            const item = document.createElement('div');
            item.style.cssText = `
                padding: 10px;
                background: ${index === this.selectedPathIndex ? '#2d5a5a' : '#2b2b2b'};
                border: 1px solid ${index === this.selectedPathIndex ? '#4ECDC4' : '#444'};
                border-radius: 4px;
                cursor: pointer;
                color: #fff;
                font-size: 12px;
                display: flex;
                flex-direction: column;
                gap: 6px;
                transition: all 0.2s ease;
            `;

            const topRow = document.createElement('div');
            topRow.style.cssText = `
                display: flex;
                justify-content: space-between;
                align-items: center;
            `;

            const info = document.createElement('div');
            info.style.cssText = `
                display: flex;
                align-items: center;
                gap: 8px;
                flex: 1;
            `;

            const colorBox = document.createElement('div');
            colorBox.style.cssText = `
                width: 16px;
                height: 16px;
                background: ${path.color};
                border-radius: ${isSinglePoint ? '2px' : '50%'};
                border: 2px solid ${isSinglePoint ? '#fff' : 'transparent'};
            `;

            const nameContainer = document.createElement('div');
            nameContainer.style.cssText = `
                display: flex;
                flex-direction: column;
                gap: 2px;
            `;

            const name = document.createElement('span');
            name.textContent = path.name || `Path ${index + 1}`;
            name.style.fontWeight = '500';

            const typeLabel = document.createElement('span');
            typeLabel.textContent = isSinglePoint
                ? '📍 Static (1 pt)'
                : `↗️ Motion (${path.points.length} pts)`;
            typeLabel.style.cssText = `
                font-size: 10px;
                color: ${isSinglePoint ? '#F7DC6F' : '#4ECDC4'};
            `;

            nameContainer.appendChild(name);
            nameContainer.appendChild(typeLabel);

            info.appendChild(colorBox);
            info.appendChild(nameContainer);

            const deleteBtn = document.createElement('button');
            deleteBtn.textContent = '✕';
            deleteBtn.style.cssText = `
                background: rgba(255, 77, 77, 0.2);
                border: 1px solid rgba(255, 77, 77, 0.4);
                border-radius: 4px;
                color: #ff4d4d;
                cursor: pointer;
                font-size: 14px;
                padding: 4px 8px;
                transition: all 0.2s ease;
            `;
            deleteBtn.onmouseover = () => {
                deleteBtn.style.background = 'rgba(255, 77, 77, 0.4)';
                deleteBtn.style.borderColor = 'rgba(255, 77, 77, 0.8)';
            };
            deleteBtn.onmouseout = () => {
                deleteBtn.style.background = 'rgba(255, 77, 77, 0.2)';
                deleteBtn.style.borderColor = 'rgba(255, 77, 77, 0.4)';
            };
            deleteBtn.onclick = (e) => {
                e.stopPropagation();
                this.paths.splice(index, 1);
                this.selectedPathIndex = -1;
                this.updateSidebar();
                this.render();
            };

            topRow.appendChild(info);
            topRow.appendChild(deleteBtn);
            item.appendChild(topRow);

            item.onclick = () => {
                this.selectedPathIndex = index;
                this.updateSidebar();
                this.render();
            };

            this.pathList.appendChild(item);
        });
    }

    createFooter() {
        const footer = document.createElement('div');
        footer.style.cssText = `
            padding: 15px 20px;
            border-top: 1px solid #444;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
        `;

        const statsContainer = document.createElement('div');
        statsContainer.style.cssText = `
            color: #888;
            font-size: 12px;
        `;
        const staticCount = this.paths.filter(p => p.isSinglePoint || p.points.length === 1).length;
        const motionCount = this.paths.length - staticCount;
        statsContainer.textContent = `Total: ${this.paths.length} paths (${staticCount} static, ${motionCount} motion)`;

        const buttonContainer = document.createElement('div');
        buttonContainer.style.cssText = `
            display: flex;
            gap: 10px;
        `;

        const cancelBtn = document.createElement('button');
        cancelBtn.textContent = 'Cancel';
        cancelBtn.style.cssText = `
            padding: 8px 20px;
            background: #444;
            border: none;
            border-radius: 4px;
            color: #fff;
            cursor: pointer;
            font-size: 14px;
        `;
        cancelBtn.onclick = () => this.close();

        const saveBtn = document.createElement('button');
        saveBtn.textContent = 'Save Paths';
        saveBtn.style.cssText = `
            padding: 8px 20px;
            background: #4ECDC4;
            border: none;
            border-radius: 4px;
            color: #fff;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
        `;
        saveBtn.onclick = () => {
            this.savePaths();
            this.close();
        };

        buttonContainer.appendChild(cancelBtn);
        buttonContainer.appendChild(saveBtn);

        footer.appendChild(statsContainer);
        footer.appendChild(buttonContainer);
        this.container.appendChild(footer);
    }

    show() {
        // Add CSS animations if not already added
        if (!document.getElementById('fl-path-animator-styles')) {
            const style = document.createElement('style');
            style.id = 'fl-path-animator-styles';
            style.textContent = `
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                @keyframes slideIn {
                    from {
                        opacity: 0;
                        transform: scale(0.95) translateY(-20px);
                    }
                    to {
                        opacity: 1;
                        transform: scale(1) translateY(0);
                    }
                }
            `;
            document.head.appendChild(style);
        }
        document.body.appendChild(this.overlay);
    }

    close() {
        // Remove keyboard handlers
        document.removeEventListener('keydown', this.keydownHandler);
        document.removeEventListener('keyup', this.keyupHandler);

        // Fade out animation
        this.overlay.style.animation = 'fadeIn 0.15s ease-in reverse';
        this.container.style.animation = 'slideIn 0.15s ease-in reverse';
        setTimeout(() => {
            if (this.overlay.parentNode) {
                document.body.removeChild(this.overlay);
            }
        }, 150);
    }
}
