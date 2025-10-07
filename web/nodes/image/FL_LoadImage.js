import { app } from "../../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.FL_LoadImage",
    async nodeCreated(node) {
        if (node.comfyClass === "FL_LoadImage") {
            addFileBrowserUI(node);
        }
    }
});

function addFileBrowserUI(node) {
    // Tweakable variables
    const DIRECTORY_Y_OFFSET = 30;
    const CLICK_Y_OFFSET = 0;
    const CLICK_X_OFFSET = -2;
    const FAVORITES_Y_OFFSET = 160;
    const DIVIDER_WIDTH = 13;

    const rootDirectoryWidget = node.widgets.find(w => w.name === "root_directory");
    const selectedFileWidget = node.widgets.find(w => w.name === "selected_file");

    rootDirectoryWidget.hidden = false;
    selectedFileWidget.hidden = true;

    const MIN_WIDTH = 430;
    const MIN_HEIGHT = 550;
    const TOP_PADDING = 210;
    const BOTTOM_PADDING = 20;
    const FOLDER_HEIGHT = 30;
    const INDENT_WIDTH = 20;
    const TOP_BAR_HEIGHT = 50;
    const THUMBNAIL_SIZE = 100;
    const THUMBNAIL_PADDING = 10;
    const SCROLLBAR_WIDTH = 13;

    const COLORS = {
        background: "#1e1e1e",
        topBar: "#252526",
        folder: "#2d2d30",
        folderHover: "#3e3e42",
        folderSelected: "#0e639c",
        text: "#cccccc",
        scrollbar: "#3e3e42",
        scrollbarHover: "#505050",
        thumbnailBorder: "#007acc",
        thumbnailBackground: "#252526",
        favoritesTab: "#5e5e5e",
        favoriteButton: "#0e639c",
        favoriteButtonHover: "#1177bb",
        divider: "#4f0074",
        dividerHover: "#16727c"
    };

    let currentDirectory = rootDirectoryWidget.value;
    let selectedFile = selectedFileWidget.value;
    let directoryStructure = { name: "root", children: [], expanded: true, path: currentDirectory };
    let fileList = [];
    let thumbnails = {};
    let scrollOffsetLeft = 0;
    let scrollOffsetRight = 0;
    let isDraggingLeft = false;
    let isDraggingRight = false;
    let hoveredFolder = null;
    let hoveredFavorite = null;
    let scrollStartY = 0;
    let scrollStartOffset = 0;
    let showFavorites = false;
    let favorites = JSON.parse(localStorage.getItem('fileBrowserFavorites') || '[]');
    let leftColumnWidth = node.size[0] / 2;
    let isDraggingDivider = false;
    let isHoveringDivider = false;

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

    function toggleFavorites() {
        showFavorites = !showFavorites;
        node.setDirtyCanvas(true);
    }

    function addToFavorites() {
        if (!favorites.includes(currentDirectory)) {
            favorites.push(currentDirectory);
            localStorage.setItem('fileBrowserFavorites', JSON.stringify(favorites));
            node.setDirtyCanvas(true);
        }
    }

    function removeFromFavorites(directory) {
        favorites = favorites.filter(fav => fav !== directory);
        localStorage.setItem('fileBrowserFavorites', JSON.stringify(favorites));
        node.setDirtyCanvas(true);
    }

    const refreshButton = node.addWidget("button", "Refresh", null, () => {
        currentDirectory = rootDirectoryWidget.value;
        updateDirectoryStructure();
    });

    const favoritesButton = node.addWidget("button", "Favorites", null, toggleFavorites);
    const addFavoriteButton = node.addWidget("button", "Add to Favorites", null, addToFavorites);

    rootDirectoryWidget.callback = () => {
        currentDirectory = rootDirectoryWidget.value;
        updateDirectoryStructure();
    };

    node.onDrawBackground = function(ctx) {
        if (!this.flags.collapsed) {
            const pos = TOP_PADDING - TOP_BAR_HEIGHT;
            ctx.fillStyle = COLORS.background;
            ctx.fillRect(0, pos, this.size[0], this.size[1] - pos);

            // Draw top bar
            ctx.fillStyle = COLORS.topBar;
            ctx.fillRect(0, pos, this.size[0], TOP_BAR_HEIGHT);

            // Draw back button
            drawRoundedRect(ctx, 10, pos + 10, 80, TOP_BAR_HEIGHT - 20, 5, COLORS.folder);
            ctx.fillStyle = COLORS.text;
            ctx.font = "14px Arial";
            ctx.fillText("← Back", 30, pos + 32);

            // Draw current directory
            ctx.fillStyle = COLORS.text;
            ctx.font = "14px Arial";
            ctx.fillText(currentDirectory, 100, pos + 32);

            if (showFavorites) {
                drawFavoritesTab(ctx);
            } else {
                // Set up clipping regions for scrolling
                ctx.save();
                ctx.beginPath();
                ctx.rect(0, TOP_PADDING, leftColumnWidth - SCROLLBAR_WIDTH, this.size[1] - TOP_PADDING - BOTTOM_PADDING);
                ctx.clip();
                drawDirectoryStructure(ctx, 10, TOP_PADDING - scrollOffsetLeft + DIRECTORY_Y_OFFSET, directoryStructure);
                ctx.restore();

                ctx.save();
                ctx.beginPath();
                ctx.rect(leftColumnWidth + DIVIDER_WIDTH, TOP_PADDING, this.size[0] - leftColumnWidth - DIVIDER_WIDTH - SCROLLBAR_WIDTH, this.size[1] - TOP_PADDING - BOTTOM_PADDING);
                ctx.clip();
                drawThumbnails(ctx, leftColumnWidth + DIVIDER_WIDTH, TOP_PADDING - scrollOffsetRight, this.size[0] - leftColumnWidth - DIVIDER_WIDTH - SCROLLBAR_WIDTH - 10, this.size[1] - TOP_PADDING - BOTTOM_PADDING);
                ctx.restore();

                // Draw scrollbars
                drawScrollbar(ctx, leftColumnWidth - SCROLLBAR_WIDTH, TOP_PADDING, SCROLLBAR_WIDTH, this.size[1] - TOP_PADDING - BOTTOM_PADDING, scrollOffsetLeft, getTotalDirectoryHeight());
                drawScrollbar(ctx, this.size[0] - SCROLLBAR_WIDTH, TOP_PADDING, SCROLLBAR_WIDTH, this.size[1] - TOP_PADDING - BOTTOM_PADDING, scrollOffsetRight, getTotalThumbnailHeight());

                // Draw divider
                ctx.fillStyle = isHoveringDivider ? COLORS.dividerHover : COLORS.divider;
                ctx.fillRect(leftColumnWidth, TOP_PADDING, DIVIDER_WIDTH, this.size[1] - TOP_PADDING - BOTTOM_PADDING);
            }
        }
    };

    function drawFavoritesTab(ctx) {
        const tabX = 0;
        const tabY = TOP_PADDING;
        const tabWidth = node.size[0];
        const tabHeight = node.size[1] - TOP_PADDING - BOTTOM_PADDING;

        ctx.fillStyle = COLORS.favoritesTab;
        ctx.fillRect(tabX, tabY, tabWidth, tabHeight);

        ctx.fillStyle = COLORS.text;
        ctx.font = "16px Arial";
        ctx.fillText("Favorites", tabX + 10, tabY + 30);

        favorites.forEach((fav, index) => {
            const y = tabY + 60 + index * 40;
            const isHovered = hoveredFavorite === index;

            drawRoundedRect(ctx, tabX + 10, y, tabWidth - 20, 30, 5, isHovered ? COLORS.folderHover : COLORS.folder);
            ctx.fillStyle = COLORS.text;
            ctx.font = "12px Arial";
            ctx.fillText(fav, tabX + 15, y + 20);

            // Draw remove button
            drawRoundedRect(ctx, tabX + tabWidth - 40, y + 5, 20, 20, 3, COLORS.favoriteButton);
            ctx.fillStyle = COLORS.text;
            ctx.font = "16px Arial";
            ctx.fillText("×", tabX + tabWidth - 34, y + 21);
        });
    }

    function drawRoundedRect(ctx, x, y, width, height, radius, color) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
    }

    function drawScrollbar(ctx, x, y, width, height, offset, totalHeight) {
        drawRoundedRect(ctx, x, y, width, height, width / 2, COLORS.scrollbar);

        const visibleHeight = height;
        const scrollHeight = Math.max(height * (visibleHeight / totalHeight), 20);
        const maxOffset = Math.max(0, totalHeight - visibleHeight);
        const scrollY = y + (offset / maxOffset) * (height - scrollHeight);

        drawRoundedRect(ctx, x, scrollY, width, scrollHeight, width / 2, COLORS.scrollbarHover);
    }

    function getTotalDirectoryHeight() {
        return countItems(directoryStructure) * FOLDER_HEIGHT;
    }

    function getTotalThumbnailHeight() {
        const thumbnailsPerRow = Math.floor((node.size[0] - leftColumnWidth - DIVIDER_WIDTH - SCROLLBAR_WIDTH - 10) / (THUMBNAIL_SIZE + THUMBNAIL_PADDING));
        return Math.ceil(fileList.length / thumbnailsPerRow) * (THUMBNAIL_SIZE + THUMBNAIL_PADDING);
    }

    function drawDirectoryStructure(ctx, x, y, structure, level = 0) {
        const folderIcon = structure.expanded ? "📂" : "📁";
        const xPos = x + INDENT_WIDTH * level;
        const yPos = y + FOLDER_HEIGHT / 2;

        const isHovered = structure === hoveredFolder;
        const isSelected = structure.path === currentDirectory;

        if (isSelected || isHovered) {
            drawRoundedRect(ctx, xPos - 5, y, leftColumnWidth - xPos, FOLDER_HEIGHT, 5, isSelected ? COLORS.folderSelected : COLORS.folderHover);
        }

        ctx.font = "14px Arial";
        ctx.fillStyle = COLORS.text;
        ctx.fillText(`${folderIcon} ${structure.name}`, xPos, yPos);

        y += FOLDER_HEIGHT;

        if (structure.expanded && structure.children) {
            for (const child of structure.children) {
                y = drawDirectoryStructure(ctx, x, y, child, level + 1);
            }
        }

        return y;
    }

    function drawThumbnails(ctx, x, y, width, height) {
        ctx.fillStyle = COLORS.background;
        ctx.fillRect(x, y, width, height);

        const thumbnailsPerRow = Math.floor(width / (THUMBNAIL_SIZE + THUMBNAIL_PADDING));
        fileList.forEach((file, index) => {
            const row = Math.floor(index / thumbnailsPerRow);
            const col = index % thumbnailsPerRow;
            const xPos = x + col * (THUMBNAIL_SIZE + THUMBNAIL_PADDING) + THUMBNAIL_PADDING;
            const yPos = y + row * (THUMBNAIL_SIZE + THUMBNAIL_PADDING) + THUMBNAIL_PADDING;

            drawRoundedRect(ctx, xPos, yPos, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 5, COLORS.thumbnailBackground);

            if (thumbnails[file]) {
                ctx.drawImage(thumbnails[file], xPos, yPos, THUMBNAIL_SIZE, THUMBNAIL_SIZE);
            }

            if (file === selectedFile) {
                ctx.strokeStyle = COLORS.thumbnailBorder;
                ctx.lineWidth = 3;
                ctx.strokeRect(xPos, yPos, THUMBNAIL_SIZE, THUMBNAIL_SIZE);
            }
        });
    }

    node.onMouseDown = function(event) {
        const pos = TOP_PADDING - TOP_BAR_HEIGHT;
        const localY = event.canvasY - this.pos[1] - pos + CLICK_Y_OFFSET;
        const localX = event.canvasX - this.pos[0] + CLICK_X_OFFSET; // Apply X offset

        if (localY < 0 || localY > this.size[1] || localX < 0 || localX > this.size[0]) {
            return false; // Allow default behavior for dragging the node
        }

        if (localY >= 0 && localY <= TOP_BAR_HEIGHT) {
            // Click on top bar
            if (localX >= 10 && localX <= 90 && localY >= 10 && localY <= TOP_BAR_HEIGHT - 10) {
                // Click on back button
                goUpDirectory();
                return true;
            }
        } else if (localY > TOP_BAR_HEIGHT) {
            if (showFavorites) {
                // Handle clicks on favorites tab
                const tabY = TOP_PADDING;
                const favoritesLocalY = localY + FAVORITES_Y_OFFSET; // Apply favorites offset
                favorites.forEach((fav, index) => {
                    const y = tabY + 60 + index * 40;
                    if (favoritesLocalY >= y && favoritesLocalY <= y + 30) {
                        if (localX >= this.size[0] - 40 && localX <= this.size[0] - 20) {
                            // Click on remove button
                            removeFromFavorites(fav);
                        } else {
                            // Click on favorite directory
                            currentDirectory = fav;
                            rootDirectoryWidget.value = currentDirectory;
                            updateDirectoryStructure();
                            showFavorites = false;
                        }
                        return true;
                    }
                });
            } else {
                if (Math.abs(localX - leftColumnWidth) <= DIVIDER_WIDTH / 2) {
                    // Click on divider
                    isDraggingDivider = true;
                    return true;
                } else if (localX < leftColumnWidth - SCROLLBAR_WIDTH) {
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
                } else if (localX >= leftColumnWidth + DIVIDER_WIDTH && localX < this.size[0] - SCROLLBAR_WIDTH) {
                    // Click on thumbnails
                    const thumbnailsPerRow = Math.floor((this.size[0] - leftColumnWidth - DIVIDER_WIDTH - SCROLLBAR_WIDTH - 10) / (THUMBNAIL_SIZE + THUMBNAIL_PADDING));
                    const clickedRow = Math.floor((localY - TOP_BAR_HEIGHT + scrollOffsetRight) / (THUMBNAIL_SIZE + THUMBNAIL_PADDING));
                    const clickedCol = Math.floor((localX - leftColumnWidth - DIVIDER_WIDTH) / (THUMBNAIL_SIZE + THUMBNAIL_PADDING));
                    const clickedIndex = clickedRow * thumbnailsPerRow + clickedCol;
                    if (clickedIndex >= 0 && clickedIndex < fileList.length) {
                        updateSelectedFile(fileList[clickedIndex]);
                    }
                    return true;
                } else if (localX >= leftColumnWidth - SCROLLBAR_WIDTH && localX < leftColumnWidth) {
                    // Click on left scrollbar
                    isDraggingLeft = true;
                    scrollStartY = event.canvasY;
                    scrollStartOffset = scrollOffsetLeft;
                    return true;
                } else if (localX >= this.size[0] - SCROLLBAR_WIDTH) {
                    // Click on right scrollbar
                    isDraggingRight = true;
                    scrollStartY = event.canvasY;
                    scrollStartOffset = scrollOffsetRight;
                    return true;
                }
            }
        }

        return false; // Allow default behavior for dragging the node
    };

    node.onMouseMove = function(event) {
        const pos = TOP_PADDING - TOP_BAR_HEIGHT;
        const localY = event.canvasY - this.pos[1] - pos + CLICK_Y_OFFSET;
        const localX = event.canvasX - this.pos[0] + CLICK_X_OFFSET; // Apply X offset

        if (isDraggingDivider) {
            leftColumnWidth = Math.max(100, Math.min(this.size[0] - 100, localX));
            this.setDirtyCanvas(true);
            return true;
        }

        if (isDraggingLeft) {
            const totalHeight = getTotalDirectoryHeight();
            const visibleHeight = this.size[1] - TOP_PADDING - BOTTOM_PADDING;
            const maxOffset = Math.max(0, totalHeight - visibleHeight);
            const scrollbarHeight = visibleHeight * (visibleHeight / totalHeight);
            const scrollMove = (event.canvasY - scrollStartY) * (totalHeight / visibleHeight);
            scrollOffsetLeft = Math.max(0, Math.min(maxOffset, scrollStartOffset + scrollMove));
            this.setDirtyCanvas(true);
            return true;
        } else if (isDraggingRight) {
            const totalHeight = getTotalThumbnailHeight();
            const visibleHeight = this.size[1] - TOP_PADDING - BOTTOM_PADDING;
            const maxOffset = Math.max(0, totalHeight - visibleHeight);
            const scrollbarHeight = visibleHeight * (visibleHeight / totalHeight);
            const scrollMove = (event.canvasY - scrollStartY) * (totalHeight / visibleHeight);
            scrollOffsetRight = Math.max(0, Math.min(maxOffset, scrollStartOffset + scrollMove));
            this.setDirtyCanvas(true);
            return true;
        }

        if (showFavorites) {
            // Hover effect for favorites
            const tabY = TOP_PADDING;
            const favoritesLocalY = localY + FAVORITES_Y_OFFSET; // Apply favorites offset
            let newHoveredFavorite = null;
            favorites.forEach((fav, index) => {
                const y = tabY + 60 + index * 40;
                if (favoritesLocalY >= y && favoritesLocalY <= y + 30) {
                    newHoveredFavorite = index;
                }
            });
            if (newHoveredFavorite !== hoveredFavorite) {
                hoveredFavorite = newHoveredFavorite;
                this.setDirtyCanvas(true);
            }
        } else {
            // Hover effect for divider
            const newIsHoveringDivider = Math.abs(localX - leftColumnWidth) <= DIVIDER_WIDTH / 2;
            if (newIsHoveringDivider !== isHoveringDivider) {
                isHoveringDivider = newIsHoveringDivider;
                this.setDirtyCanvas(true);
            }

            // Hover effect for folders
            if (localX < leftColumnWidth - SCROLLBAR_WIDTH && localY > TOP_BAR_HEIGHT) {
                const newHoveredFolder = findClickedItem(directoryStructure, localY - TOP_BAR_HEIGHT + scrollOffsetLeft - DIRECTORY_Y_OFFSET);
                if (newHoveredFolder !== hoveredFolder) {
                    hoveredFolder = newHoveredFolder;
                    this.setDirtyCanvas(true);
                }
            } else if (hoveredFolder) {
                hoveredFolder = null;
                this.setDirtyCanvas(true);
            }
        }

        // Update cursor style
        if (isHoveringDivider) {
            document.body.style.cursor = 'ew-resize';
        } else {
            document.body.style.cursor = 'default';
        }

        return false;
    };

    node.onMouseUp = function(event) {
        isDraggingLeft = false;
        isDraggingRight = false;
        isDraggingDivider = false;
        document.body.style.cursor = 'default';
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