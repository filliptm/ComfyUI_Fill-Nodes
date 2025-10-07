import { app } from "../../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.FL_SaveImages",
    async nodeCreated(node) {
        if (node.comfyClass === "FL_SaveImages") {
            addFolderStructureUI(node);
        }
    }
});

function addFolderStructureUI(node) {
    const DEBUG_CURSOR_OFFSET = 10;

    const folderStructureWidget = node.widgets.find(w => w.name === "folder_structure");
    const baseDirectoryWidget = node.widgets.find(w => w.name === "base_directory");
    if (folderStructureWidget) {
        folderStructureWidget.hidden = true;
    }

    const MIN_WIDTH = 350;
    const MIN_HEIGHT = 80;
    const FOLDER_HEIGHT = 20;
    const TOP_PADDING = 250;
    const BOTTOM_PADDING = 20;
    const INDENT_WIDTH = 20;

    let folderStructure = { name: "root", children: [] };
    let selectedFolder = folderStructure;

    function refreshFolderStructure() {
        try {
            const savedStructure = JSON.parse(folderStructureWidget.value);
            folderStructure = { name: "root", children: savedStructure };
            selectedFolder = folderStructure;
        } catch (e) {
            console.error("Error parsing saved folder structure:", e);
            folderStructure = { name: "root", children: [] };
            selectedFolder = folderStructure;
        }
        updateNodeSize();
        node.setDirtyCanvas(true);
    }

    function updateFolderStructure() {
        folderStructureWidget.value = JSON.stringify(folderStructure.children);
        updateNodeSize();
        node.setDirtyCanvas(true);
    }

    function updateNodeSize() {
        const totalFolders = countTotalFolders(folderStructure);
        const maxDepth = getMaxDepth(folderStructure);
        const height = Math.max(MIN_HEIGHT, TOP_PADDING + (totalFolders * FOLDER_HEIGHT) + BOTTOM_PADDING);
        const width = Math.max(MIN_WIDTH, 200 + (maxDepth * INDENT_WIDTH));
        node.size[0] = width;
        node.size[1] = height;
    }

    function countTotalFolders(folder) {
        return 1 + folder.children.reduce((sum, child) => sum + countTotalFolders(child), 0);
    }

    function getMaxDepth(folder, depth = 0) {
        if (folder.children.length === 0) return depth;
        return Math.max(...folder.children.map(child => getMaxDepth(child, depth + 1)));
    }

    // Add folder input for subfolder
    const folderInput = node.addWidget("text", "New Subfolder", "", (v) => {});

    // Add folder button
    const addFolderButton = node.addWidget("button", "Add Subfolder", null, () => {
        if (folderInput.value) {
            const newFolder = { name: folderInput.value, children: [] };
            selectedFolder.children.push(newFolder);
            folderInput.value = "";
            updateFolderStructure();
        }
    });

    // Add remove selected folder button
    const removeFolderButton = node.addWidget("button", "Remove Selected Folder", null, () => {
        if (selectedFolder !== folderStructure) {
            removeFolder(folderStructure, selectedFolder);
            selectedFolder = folderStructure;
            updateFolderStructure();
        }
    });

    // Add refresh button
    const refreshButton = node.addWidget("button", "Refresh Folder Structure", null, () => {
        refreshFolderStructure();
    });

    function removeFolder(parent, folderToRemove) {
        const index = parent.children.findIndex(child => child === folderToRemove);
        if (index !== -1) {
            parent.children.splice(index, 1);
        } else {
            parent.children.forEach(child => removeFolder(child, folderToRemove));
        }
    }

    // Override the default behavior of the buttons
    addFolderButton.callback = () => {
        if (folderInput.value) {
            const newFolder = { name: folderInput.value, children: [] };
            selectedFolder.children.push(newFolder);
            folderInput.value = "";
            updateFolderStructure();
        }
    };

    removeFolderButton.callback = () => {
        if (selectedFolder !== folderStructure) {
            removeFolder(folderStructure, selectedFolder);
            selectedFolder = folderStructure;
            updateFolderStructure();
        }
    };

    refreshButton.callback = refreshFolderStructure;

    function drawFolderStructure(ctx, x, y, structure, level = 0) {
        const folderIcon = "📁";

        ctx.font = "14px Arial";
        ctx.fillStyle = "#ffffff";

        if (level === 0) {
            ctx.fillText(`${folderIcon} ${baseDirectoryWidget.value || "(default)"}`, x, y);
            y += FOLDER_HEIGHT;
        }

        for (const folder of structure.children) {
            const xPos = x + INDENT_WIDTH * (level + 1);

            // Draw connecting lines
            ctx.strokeStyle = "#ffffff";
            ctx.beginPath();
            ctx.moveTo(xPos - INDENT_WIDTH + 5, y - FOLDER_HEIGHT / 2);
            ctx.lineTo(xPos - 5, y - FOLDER_HEIGHT / 2);
            ctx.lineTo(xPos - 5, y + 2);
            ctx.stroke();

            // Highlight selected folder
            if (folder === selectedFolder) {
                ctx.fillStyle = "#4a90e2";
                ctx.fillRect(xPos - 2, y - 14, ctx.measureText(folder.name).width + 24, 18);
                ctx.fillStyle = "#ffffff";
            }

            ctx.fillText(`${folderIcon} ${folder.name}`, xPos, y);

            y += FOLDER_HEIGHT;
            y = drawFolderStructure(ctx, x, y, folder, level + 1);
        }

        return y;
    }

    node.onDrawBackground = function(ctx) {
        if (!this.flags.collapsed) {
            const pos = TOP_PADDING;
            ctx.fillStyle = "#2A2A2A";
            ctx.fillRect(0, pos, this.size[0], this.size[1] - pos);

            drawFolderStructure(ctx, 10, pos + 20, folderStructure);
        }
    };

    // Handle clicks on the node
    node.onMouseDown = function(event) {
        const pos = TOP_PADDING + 20; // Adjust for the 20px offset in drawFolderStructure
        if (event.canvasY - this.pos[1] > pos) {
            const localY = event.canvasY - this.pos[1] - pos + DEBUG_CURSOR_OFFSET;
            const clickedFolder = findClickedFolder(folderStructure, localY, 0);
            if (clickedFolder) {
                selectedFolder = clickedFolder;
                this.setDirtyCanvas(true);
            }
            return true; // Prevent dragging when clicking on the folder structure area
        }
    };

    function findClickedFolder(folder, y, currentY) {
        if (y >= currentY && y < currentY + FOLDER_HEIGHT) {
            return folder;
        }
        currentY += FOLDER_HEIGHT;
        for (const child of folder.children) {
            const result = findClickedFolder(child, y, currentY);
            if (result) return result;
            currentY += FOLDER_HEIGHT * countTotalFolders(child);
        }
        return null;
    }

    node.onResize = function() {
        this.setDirtyCanvas(true);
    };

    // Initial refresh
    refreshFolderStructure();
}