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
    // Tweakable variables
    const DIRECTORY_Y_OFFSET = 20; // Adjust this to move directory items up or down
    const CLICK_Y_OFFSET = 8; // Adjust this to fine-tune click detection

    const rootDirectoryWidget = node.widgets.find(w => w.name === "root_directory");
    const selectedFileWidget = node.widgets.find(w => w.name === "selected_file");

    rootDirectoryWidget.hidden = false;
    selectedFileWidget.hidden = true;

    const MIN_WIDTH = 730;
    const MIN_HEIGHT = 850;
    const TOP_PADDING = 150;
    const BOTTOM_PADDING = 20;
    const FOLDER_HEIGHT = 25;
    const INDENT_WIDTH = 15;
    const TOP_BAR_HEIGHT = 40;
    const THUMBNAIL_SIZE = 80;
    const THUMBNAIL_PADDING = 5;
    const SCROLLBAR_WIDTH = 13;

    let currentDirectory = rootDirectoryWidget.value;
    let selectedFile = selectedFileWidget.value;
    let directoryStructure = { name: "root", children: [], expanded: true, path: currentDirectory };
    let fileList = [];
    let thumbnails = {};
    let scrollOffsetLeft = 0;
    let scrollOffsetRight = 0;
    let isDraggingLeft = false;
    let isDraggingRight = false;

    async function updateDirectoryStructure() {
        try {
            const response = await fetch('/fl_file_browser/get_directory_structure', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: currentDirectory })
            });
            const data = await response.json();
            mergeDirectoryStructure(directoryStructure, data.structure);
            fileList = data.files;
            loadThumbnails();
            node.setDirtyCanvas(true);
        } catch (error) {
            console.error("Error updating directory structure:", error);
        }
    }

    function mergeDirectoryStructure(existing, updated) {
        existing.name = updated.name;
        existing.path = updated.path;

        const existingChildren = new Map(existing.children.map(child => [child.name, child]));
        existing.children = updated.children.map(updatedChild => {
            const existingChild = existingChildren.get(updatedChild.name);
            if (existingChild) {
                mergeDirectoryStructure(existingChild, updatedChild);
                return existingChild;
            } else {
                return { ...updatedChild, expanded: false };
            }
        });
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
            ctx.fillText("← Back", 15, pos + 25);

            // Draw current directory
            ctx.fillStyle = "#FFFFFF";
            ctx.font = "14px Arial";
            ctx.fillText(currentDirectory, 75, pos + 27);

            const midX = this.size[0] / 2;

            // Set up clipping regions for scrolling
            ctx.save();
            ctx.beginPath();
            ctx.rect(0, TOP_PADDING, midX - SCROLLBAR_WIDTH, this.size[1] - TOP_PADDING - BOTTOM_PADDING);
            ctx.clip();
            drawDirectoryStructure(ctx, 10, TOP_PADDING - scrollOffsetLeft + DIRECTORY_Y_OFFSET, directoryStructure);
            ctx.restore();

            ctx.save();
            ctx.beginPath();
            ctx.rect(midX, TOP_PADDING, this.size[0] - midX - SCROLLBAR_WIDTH, this.size[1] - TOP_PADDING - BOTTOM_PADDING);
            ctx.clip();
            drawThumbnails(ctx, midX, TOP_PADDING - scrollOffsetRight, this.size[0] / 2 - SCROLLBAR_WIDTH - 10, this.size[1] - TOP_PADDING - BOTTOM_PADDING);
            ctx.restore();

            // Draw scrollbars
            drawScrollbar(ctx, midX - SCROLLBAR_WIDTH, TOP_PADDING, SCROLLBAR_WIDTH, this.size[1] - TOP_PADDING - BOTTOM_PADDING, scrollOffsetLeft, getTotalDirectoryHeight());
            drawScrollbar(ctx, this.size[0] - SCROLLBAR_WIDTH, TOP_PADDING, SCROLLBAR_WIDTH, this.size[1] - TOP_PADDING - BOTTOM_PADDING, scrollOffsetRight, getTotalThumbnailHeight());
        }
    };

    function drawScrollbar(ctx, x, y, width, height, offset, totalHeight) {
        ctx.fillStyle = "#555555";
        ctx.fillRect(x, y, width, height);

        const visibleHeight = height;
        const scrollHeight = Math.max(height * (visibleHeight / totalHeight), 20);
        const maxOffset = Math.max(0, totalHeight - visibleHeight);
        const scrollY = y + (offset / maxOffset) * (height - scrollHeight);

        ctx.fillStyle = "#888888";
        ctx.fillRect(x, scrollY, width, scrollHeight);
    }

    function getTotalDirectoryHeight() {
        return countItems(directoryStructure) * FOLDER_HEIGHT;
    }

    function getTotalThumbnailHeight() {
        const thumbnailsPerRow = Math.floor((node.size[0] / 2 - SCROLLBAR_WIDTH - 10) / (THUMBNAIL_SIZE + THUMBNAIL_PADDING));
        return Math.ceil(fileList.length / thumbnailsPerRow) * (THUMBNAIL_SIZE + THUMBNAIL_PADDING);
    }

    function drawDirectoryStructure(ctx, x, y, structure, level = 0) {
        const folderIcon = structure.expanded ? "📂" : "📁";
        ctx.font = "14px Arial";
        ctx.fillStyle = structure.path === currentDirectory ? "#4a90e2" : "#ffffff";

        const xPos = x + INDENT_WIDTH * level;
        ctx.fillText(`${folderIcon} ${structure.name}`, xPos, y);
        y += FOLDER_HEIGHT;

        if (structure.expanded && structure.children) {
            for (const child of structure.children) {
                y = drawDirectoryStructure(ctx, x, y, child, level + 1);
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
        const localY = event.canvasY - this.pos[1] - pos + CLICK_Y_OFFSET;
        const localX = event.canvasX - this.pos[0];

        if (localY < 0 || localY > this.size[1] || localX < 0 || localX > this.size[0]) {
            return false; // Allow default behavior for dragging the node
        }

        if (localY >= 0 && localY <= TOP_BAR_HEIGHT) {
            // Click on top bar
            if (localX >= 5 && localX <= 65 && localY >= 5 && localY <= TOP_BAR_HEIGHT - 5) {
                // Click on back button
                goUpDirectory();
                return true;
            }
        } else if (localY > TOP_BAR_HEIGHT) {
            const midX = this.size[0] / 2;
            if (localX < midX - SCROLLBAR_WIDTH) {
                // Click on directory structure
                const clickedItem = findClickedItem(directoryStructure, localY - TOP_BAR_HEIGHT + scrollOffsetLeft - DIRECTORY_Y_OFFSET);
                if (clickedItem) {
                    if (clickedItem.children) {
                        clickedItem.expanded = !clickedItem.expanded;
                        this.setDirtyCanvas(true);
                    }
                    if (clickedItem.path !== currentDirectory) {
                        currentDirectory = clickedItem.path;
                        rootDirectoryWidget.value = currentDirectory;
                        updateDirectoryStructure();
                    }
                }
                return true;
            } else if (localX >= midX && localX < this.size[0] - SCROLLBAR_WIDTH) {
                // Click on thumbnails
                const thumbnailsPerRow = Math.floor((this.size[0] / 2 - SCROLLBAR_WIDTH - 10) / (THUMBNAIL_SIZE + THUMBNAIL_PADDING));
                const clickedRow = Math.floor((localY - TOP_BAR_HEIGHT + scrollOffsetRight) / (THUMBNAIL_SIZE + THUMBNAIL_PADDING));
                const clickedCol = Math.floor((localX - midX) / (THUMBNAIL_SIZE + THUMBNAIL_PADDING));
                const clickedIndex = clickedRow * thumbnailsPerRow + clickedCol;
                if (clickedIndex >= 0 && clickedIndex < fileList.length) {
                    updateSelectedFile(fileList[clickedIndex]);
                }
                return true;
            } else if (localX >= midX - SCROLLBAR_WIDTH && localX < midX) {
                // Click on left scrollbar
                isDraggingLeft = true;
                return true;
            } else if (localX >= this.size[0] - SCROLLBAR_WIDTH) {
                // Click on right scrollbar
                isDraggingRight = true;
                return true;
            }
        }
        return false; // Allow default behavior for dragging the node
    };

    node.onMouseMove = function(event) {
        if (isDraggingLeft) {
            const totalHeight = getTotalDirectoryHeight();
            const visibleHeight = this.size[1] - TOP_PADDING - BOTTOM_PADDING;
            const maxOffset = Math.max(0, totalHeight - visibleHeight);
            scrollOffsetLeft = Math.max(0, Math.min(maxOffset, (event.canvasY - this.pos[1] - TOP_PADDING) / visibleHeight * totalHeight));
            this.setDirtyCanvas(true);
            return true;
        } else if (isDraggingRight) {
            const totalHeight = getTotalThumbnailHeight();
            const visibleHeight = this.size[1] - TOP_PADDING - BOTTOM_PADDING;
            const maxOffset = Math.max(0, totalHeight - visibleHeight);
            scrollOffsetRight = Math.max(0, Math.min(maxOffset, (event.canvasY - this.pos[1] - TOP_PADDING) / visibleHeight * totalHeight));
            this.setDirtyCanvas(true);
            return true;
        }
        return false;
    };

    node.onMouseUp = function(event) {
        isDraggingLeft = false;
        isDraggingRight = false;
        return false;
    };

    function findClickedItem(structure, y, currentY = 0) {
        if (y >= currentY && y < currentY + FOLDER_HEIGHT) {
            return structure;
        }
        currentY += FOLDER_HEIGHT;

        if (structure.expanded && structure.children) {
            for (const child of structure.children) {
                const found = findClickedItem(child, y, currentY);
                if (found) return found;
                currentY += FOLDER_HEIGHT * (child.expanded ? countItems(child) : 1);
            }
        }
        return null;
    }

    function countItems(item) {
        if (!item.expanded || !item.children) return 1;
        return 1 + item.children.reduce((sum, child) => sum + countItems(child), 0);
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