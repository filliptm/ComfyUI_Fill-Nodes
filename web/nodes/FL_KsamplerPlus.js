import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Fill-Nodes.KsamplerPlus",
    async nodeCreated(node) {
        const animatedNodeClasses = [
            "FL_KsamplerPlus",
        ];

        if (animatedNodeClasses.includes(node.comfyClass)) {
            addDenoisingTextAnimatedDisplay(node);
        }
    }
});

function addDenoisingTextAnimatedDisplay(node) {
    const ANIMATION_WIDTH = 120;
    const ANIMATION_HEIGHT = 80;
    const PIXEL_SIZE = 1;
    const GRID_WIDTH = Math.floor(ANIMATION_WIDTH / PIXEL_SIZE);
    const GRID_HEIGHT = Math.floor(ANIMATION_HEIGHT / PIXEL_SIZE);

    let grid = createEmptyGrid(GRID_WIDTH, GRID_HEIGHT);
    let animationProgress = 0;
    let xOffset = 0;
    let yOffset = 90;

    function createEmptyGrid(width, height) {
        return Array.from({ length: height }, () =>
            Array.from({ length: width }, () => ({ value: 0, hue: 0 }))
        );
    }

    function drawText(ctx) {
        ctx.font = 'bold 20px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = 'white';
        ctx.fillText('CUT', ANIMATION_WIDTH / 2, ANIMATION_HEIGHT / 2 - 15);
        ctx.fillText('THE NOISE', ANIMATION_WIDTH / 2, ANIMATION_HEIGHT / 2 + 15);
    }

    function updateGrid() {
        animationProgress += 0.005; // Slightly faster animation
        if (animationProgress > 1) {
            animationProgress = 0;
            grid = createEmptyGrid(GRID_WIDTH, GRID_HEIGHT);
        }

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = ANIMATION_WIDTH;
        tempCanvas.height = ANIMATION_HEIGHT;
        const tempCtx = tempCanvas.getContext('2d');
        drawText(tempCtx);

        const imageData = tempCtx.getImageData(0, 0, ANIMATION_WIDTH, ANIMATION_HEIGHT);

        for (let y = 0; y < GRID_HEIGHT; y++) {
            for (let x = 0; x < GRID_WIDTH; x++) {
                const i = (y * ANIMATION_WIDTH + x) * 4;
                if (imageData.data[i] > 0) {
                    if (grid[y][x].value === 0) {
                        grid[y][x].hue = Math.random() * 360; // Assign a random hue
                    }
                    if (Math.random() < animationProgress * 0.2) {
                        grid[y][x].value = Math.min(grid[y][x].value + 0.2, 1);
                    }
                }
            }
        }
    }

    node.onDrawBackground = function(ctx) {
        if (!this.flags.collapsed) {
            ctx.save();

            const nodeWidth = this.size[0];
            const baseXOffset = (nodeWidth - ANIMATION_WIDTH) / 2;
            ctx.translate(baseXOffset + xOffset, -ANIMATION_HEIGHT + yOffset);

            ctx.clearRect(0, 0, ANIMATION_WIDTH, ANIMATION_HEIGHT);

            // Draw appearing pixels with color and pulse effect
            for (let y = 0; y < GRID_HEIGHT; y++) {
                for (let x = 0; x < GRID_WIDTH; x++) {
                    const { value, hue } = grid[y][x];
                    if (value > 0) {
                        const pulseEffect = 0.7 + 0.3 * Math.sin(animationProgress * 10 + x * 0.1 + y * 0.1);
                        const alpha = value * pulseEffect;
                        ctx.fillStyle = `hsla(${hue}, 100%, 50%, ${alpha})`;
                        ctx.fillRect(x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
                    }
                }
            }

            ctx.restore();

            updateGrid();

            this.setDirtyCanvas(true);
            requestAnimationFrame(() => this.setDirtyCanvas(true));
        }
    };

    node.setDirtyCanvas(true);
}