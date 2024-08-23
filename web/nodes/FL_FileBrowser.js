import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.FL_FileBrowser",
    async nodeCreated(node) {
        if (node.comfyClass === "FL_FileBrowser") {
            addFileBrowserUI(node);
        }
    }
});

function addFileBrowserUI(node) {
    const rootDirectoryWidget = node.widgets.find(w => w.name === "root_directory");
    const selectedFileWidget = node.widgets.find(w => w.name === "selected_file");

    rootDirectoryWidget.hidden = false;
    selectedFileWidget.hidden = true;

    const MIN_WIDTH = 400;
    const MIN_HEIGHT = 450;
    const TOP_PADDING = 140;
    const BOTTOM_PADDING = 20;
    const FOLDER_HEIGHT = 20;
    const INDENT_WIDTH = 15;
    const TOP_BAR_HEIGHT = 30;
    const THUMBNAIL_SIZE = 80;
    const THUMBNAIL_PADDING = 5;

    let currentDirectory = rootDirectoryWidget.value;
    let selectedFile = selectedFileWidget.value;
    let directoryStructure = { name: "root", children: [] };
    let fileList = [];
    let thumbnails = {};
    let scrollOffsetLeft = 0;
    let scrollOffsetRight = 0;

    async function updateDirectoryStructure() {
        try {
            const response = await fetch('/fl_file_browser/get_directory_structure', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: currentDirectory })
            });
            const data = await response.json();
            directoryStructure = data.structure;
            fileList = data.files;
            loadThumbnails();
            node.setDirtyCanvas(true);
        } catch (error) {
            console.error("Error updating directory structure:", error);
        }
    }

    async function loadThumbnails() {
        thumbnails = {};
        for (const file of fileList) {
            const response = await fetch('/fl_file_browser/get_thumbnail', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: currentDirectory, file: file })
            });
            const blob = await response.blob();
            thumbnails[file] = await createImageBitmap(blob);
        }
        node.setDirtyCanvas(true);
    }

    function updateSelectedFile(file) {
        selectedFile = file;
        selectedFileWidget.value = currentDirectory + '/' + file;
        node.setDirtyCanvas(true);
    }

    function goUpDirectory() {
        const parentDir = currentDirectory.split(/(\\|\/)/g).slice(0, -2).join('');
        if (parentDir) {
            currentDirectory = parentDir;
            rootDirectoryWidget.value = currentDirectory;
            updateDirectoryStructure();
        }
    }

    const refreshButton = node.addWidget("button", "Refresh", null, () => {
        currentDirectory = rootDirectoryWidget.value;
        updateDirectoryStructure();
    });

    rootDirectoryWidget.callback = () => {
        currentDirectory = rootDirectoryWidget.value;
        updateDirectoryStructure();
    };

    node.onDrawBackground = function(ctx) {
        if (!this.flags.collapsed) {
            const pos = TOP_PADDING - TOP_BAR_HEIGHT;
            ctx.fillStyle = "#2A2A2A";
            ctx.fillRect(0, pos, this.size[0], this.size[1] - pos);

            // Draw top bar
            ctx.fillStyle = "#3A3A3A";
            ctx.fillRect(0, pos, this.size[0], TOP_BAR_HEIGHT);

            // Draw back button
            ctx.fillStyle = "#4A4A4A";
            ctx.fillRect(5, pos + 5, 60, TOP_BAR_HEIGHT - 10);
            ctx.fillStyle = "#FFFFFF";
            ctx.font = "12px Arial";
            ctx.fillText("← Back", 15, pos + 20);

            // Draw current directory
            ctx.fillStyle = "#FFFFFF";
            ctx.font = "12px Arial";
            ctx.fillText(currentDirectory, 75, pos + 20);

            const midX = this.size[0] / 2;

            // Set up clipping regions for scrolling
            ctx.save();
            ctx.beginPath();
            ctx.rect(0, TOP_PADDING, midX, this.size[1] - TOP_PADDING - BOTTOM_PADDING);
            ctx.clip();
            drawDirectoryStructure(ctx, 10, TOP_PADDING - scrollOffsetLeft, directoryStructure);
            ctx.restore();

            ctx.save();
            ctx.beginPath();
            ctx.rect(midX, TOP_PADDING, this.size[0] - midX, this.size[1] - TOP_PADDING - BOTTOM_PADDING);
            ctx.clip();
            drawThumbnails(ctx, midX, TOP_PADDING - scrollOffsetRight, this.size[0] / 2 - 10, this.size[1] - TOP_PADDING - BOTTOM_PADDING);
            ctx.restore();
        }
    };

    function drawDirectoryStructure(ctx, x, y, structure, level = 0) {
        const folderIcon = "📁";
        ctx.font = "14px Arial";
        ctx.fillStyle = "#ffffff";

        for (const item of structure.children) {
            const xPos = x + INDENT_WIDTH * level;
            ctx.fillText(`${folderIcon} ${item.name}`, xPos, y);
            y += FOLDER_HEIGHT;

            if (item.children) {
                y = drawDirectoryStructure(ctx, x, y, item, level + 1);
            }
        }

        return y;
    }

    function drawThumbnails(ctx, x, y, width, height) {
        ctx.fillStyle = "#3A3A3A";
        ctx.fillRect(x, y, width, height);

        const thumbnailsPerRow = Math.floor(width / (THUMBNAIL_SIZE + THUMBNAIL_PADDING));
        fileList.forEach((file, index) => {
            const row = Math.floor(index / thumbnailsPerRow);
            const col = index % thumbnailsPerRow;
            const xPos = x + col * (THUMBNAIL_SIZE + THUMBNAIL_PADDING) + THUMBNAIL_PADDING;
            const yPos = y + row * (THUMBNAIL_SIZE + THUMBNAIL_PADDING) + THUMBNAIL_PADDING;

            if (thumbnails[file]) {
                ctx.drawImage(thumbnails[file], xPos, yPos, THUMBNAIL_SIZE, THUMBNAIL_SIZE);
            } else {
                ctx.fillStyle = "#4A4A4A";
                ctx.fillRect(xPos, yPos, THUMBNAIL_SIZE, THUMBNAIL_SIZE);
            }

            if (file === selectedFile) {
                ctx.strokeStyle = "#4a90e2";
                ctx.lineWidth = 2;
                ctx.strokeRect(xPos, yPos, THUMBNAIL_SIZE, THUMBNAIL_SIZE);
            }

            ctx.fillStyle = "#FFFFFF";
            ctx.font = "10px Arial";
            ctx.fillText(file.substring(0, 10) + (file.length > 10 ? "..." : ""), xPos, yPos + THUMBNAIL_SIZE + 12);
        });
    }

    node.onMouseDown = function(event) {
        const pos = TOP_PADDING - TOP_BAR_HEIGHT;
        const localY = event.canvasY - this.pos[1] - pos;
        const localX = event.canvasX - this.pos[0];

        if (localY >= 0 && localY <= TOP_BAR_HEIGHT) {
            // Click on top bar
            if (localX >= 5 && localX <= 65 && localY >= 5 && localY <= TOP_BAR_HEIGHT - 5) {
                // Click on back button
                goUpDirectory();
                return true;
            }
        } else if (localY > TOP_BAR_HEIGHT) {
            const midX = this.size[0] / 2;
            if (localX < midX) {
                // Click on directory structure
                const clickedItem = findClickedItem(directoryStructure, localY - TOP_BAR_HEIGHT + scrollOffsetLeft);
                if (clickedItem) {
                    currentDirectory = clickedItem.path;
                    rootDirectoryWidget.value = currentDirectory;
                    updateDirectoryStructure();
                }
            } else {
                // Click on thumbnails
                const thumbnailsPerRow = Math.floor((this.size[0] / 2 - 10) / (THUMBNAIL_SIZE + THUMBNAIL_PADDING));
                const clickedRow = Math.floor((localY - TOP_BAR_HEIGHT + scrollOffsetRight) / (THUMBNAIL_SIZE + THUMBNAIL_PADDING));
                const clickedCol = Math.floor((localX - midX) / (THUMBNAIL_SIZE + THUMBNAIL_PADDING));
                const clickedIndex = clickedRow * thumbnailsPerRow + clickedCol;
                if (clickedIndex >= 0 && clickedIndex < fileList.length) {
                    updateSelectedFile(fileList[clickedIndex]);
                }
            }
        }
        return true;
    };

    node.onMouseMove = function(event) {
        if (event.dragging && event.button === 2) { // Right mouse button
            const midX = this.size[0] / 2;
            if (event.canvasX - this.pos[0] < midX) {
                scrollOffsetLeft = Math.max(0, scrollOffsetLeft - event.deltay);
            } else {
                scrollOffsetRight = Math.max(0, scrollOffsetRight - event.deltay);
            }
            this.setDirtyCanvas(true);
        }
    };

    function findClickedItem(structure, y, currentY = 0) {
        for (const item of structure.children) {
            if (y >= currentY && y < currentY + FOLDER_HEIGHT) {
                return item;
            }
            currentY += FOLDER_HEIGHT;
            if (item.children) {
                const found = findClickedItem(item, y, currentY);
                if (found) return found;
                currentY += FOLDER_HEIGHT * countItems(item);
            }
        }
        return null;
    }

    function countItems(item) {
        return item.children ? item.children.reduce((sum, child) => sum + countItems(child), 1) : 1;
    }

    function updateNodeSize() {
        const width = Math.max(MIN_WIDTH, node.size[0]);
        const height = Math.max(MIN_HEIGHT, node.size[1]);
        node.size[0] = width;
        node.size[1] = height;
    }

    node.onResize = function() {
        updateNodeSize();
        this.setDirtyCanvas(true);
    };

    updateDirectoryStructure();
    updateNodeSize();
}