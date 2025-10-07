import { app } from "../../../../scripts/app.js";

app.registerExtension({
    name: "Fill-Nodes.PixelArtAnimatedDisplay",
    async nodeCreated(node) {
        const animatedNodeClasses = [
            "FL_PixelArtShader",
        ];

        if (animatedNodeClasses.includes(node.comfyClass)) {
            addPixelArtAnimatedDisplay(node);
        }
    }
});

function addPixelArtAnimatedDisplay(node) {
    const ANIMATION_WIDTH = 50;  // Fixed width for the animation
    const ANIMATION_HEIGHT = 50; // Fixed height for the animation
    const PIXEL_SIZE = 4;        // Size of each "pixel"
    const GRID_WIDTH = Math.floor(ANIMATION_WIDTH / PIXEL_SIZE);
    const GRID_HEIGHT = Math.floor(ANIMATION_HEIGHT / PIXEL_SIZE);

    let grid = createRandomGrid(GRID_WIDTH, GRID_HEIGHT);
    let xOffset = 0;
    let yOffset = 60;


    function createRandomGrid(width, height) {
        return Array.from({ length: height }, () =>
            Array.from({ length: width }, () => Math.random() > 0.7)
        );
    }

    function updateGrid() {
        const newGrid = createRandomGrid(GRID_WIDTH, GRID_HEIGHT);
        for (let y = 0; y < GRID_HEIGHT; y++) {
            for (let x = 0; x < GRID_WIDTH; x++) {
                if (Math.random() > 0.1) {
                    newGrid[y][x] = grid[y][x];
                }
            }
        }
        grid = newGrid;
    }

    node.onDrawBackground = function(ctx) {
        if (!this.flags.collapsed) {
            ctx.save();

            // Calculate position to center the animation above the node, including offsets
            const nodeWidth = this.size[0];
            const baseXOffset = (nodeWidth - ANIMATION_WIDTH) / 2;
            ctx.translate(baseXOffset + xOffset, -ANIMATION_HEIGHT + yOffset);

            // Set composite operation to ensure transparency
            ctx.globalCompositeOperation = 'source-over';

            // Draw pixel art
            for (let y = 0; y < GRID_HEIGHT; y++) {
                for (let x = 0; x < GRID_WIDTH; x++) {
                    if (grid[y][x]) {
                        ctx.fillStyle = `hsl(${(x / GRID_WIDTH) * 360}, 100%, 50%)`;
                        ctx.fillRect(x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
                    }
                }
            }

            ctx.restore();

            // Update grid occasionally
            if (Math.random() > 0) {
                updateGrid();
            }

            this.setDirtyCanvas(true);
            requestAnimationFrame(() => this.setDirtyCanvas(true));
        }
    };

    node.setDirtyCanvas(true);
}