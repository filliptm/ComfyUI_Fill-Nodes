import { app } from "../../../../scripts/app.js";

// Animation parameters
const ANIMATION_WIDTH = 120;
const ANIMATION_HEIGHT = 50;
const GHOST_SIZE = 15;
const ANIMATION_X_OFFSET = -10;
const ANIMATION_Y_OFFSET = 10;

app.registerExtension({
    name: "Ghost-API-Animation",
    async nodeCreated(node) {
        const animatedNodeClasses = [
            "FL_Dalle3",
            // Add other API-related node classes here
        ];

        if (animatedNodeClasses.includes(node.comfyClass)) {
            addGhostAPIAnimation(node);
        }
    }
});

function addGhostAPIAnimation(node) {
    let ghosts = [];

    function createGhost() {
        return {
            x: ANIMATION_WIDTH,
            y: Math.random() * ANIMATION_HEIGHT,
            speed: 0.5 + Math.random() * 1,
            size: GHOST_SIZE + Math.random() * 5,
            opacity: 1,
            waveFreq: 0.05 + Math.random() * 0.05,
            waveAmp: 1 + Math.random() * 2
        };
    }

    function updateGhosts() {
        ghosts = ghosts.filter(ghost => ghost.x > -ghost.size && ghost.opacity > 0);
        ghosts.forEach(ghost => {
            ghost.x -= ghost.speed;
            ghost.opacity -= 0.01;
        });

        if (Math.random() > 0.97) {
            ghosts.push(createGhost());
        }
    }

    function drawGhost(ctx, ghost) {
        ctx.save();
        ctx.translate(ghost.x, ghost.y + Math.sin(ghost.x * ghost.waveFreq) * ghost.waveAmp);

        // Ghost body
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.bezierCurveTo(-ghost.size/2, -ghost.size/2, -ghost.size/2, -ghost.size, 0, -ghost.size);
        ctx.bezierCurveTo(ghost.size/2, -ghost.size, ghost.size/2, -ghost.size/2, 0, 0);

        // Ghost tail
        ctx.quadraticCurveTo(-ghost.size/4, ghost.size/2, -ghost.size/2, ghost.size);
        ctx.quadraticCurveTo(-ghost.size/8, ghost.size/2, 0, ghost.size);
        ctx.quadraticCurveTo(ghost.size/8, ghost.size/2, ghost.size/2, ghost.size);
        ctx.quadraticCurveTo(ghost.size/4, ghost.size/2, 0, 0);

        ctx.fillStyle = `rgba(255, 255, 255, ${ghost.opacity})`;
        ctx.fill();

        // Eyes
        ctx.fillStyle = `rgba(0, 0, 0, ${ghost.opacity})`;
        ctx.beginPath();
        ctx.arc(-ghost.size/4, -ghost.size/2, ghost.size/10, 0, Math.PI * 2);
        ctx.arc(ghost.size/4, -ghost.size/2, ghost.size/10, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }

    node.onDrawBackground = function(ctx) {
        if (!this.flags.collapsed) {
            ctx.save();

            const nodeWidth = this.size[0];
            const baseXOffset = (nodeWidth - ANIMATION_WIDTH) / 2;
            ctx.translate(baseXOffset + ANIMATION_X_OFFSET, ANIMATION_Y_OFFSET);

            // Draw ghosts
            ghosts.forEach(ghost => drawGhost(ctx, ghost));

            // Draw API text
            ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
            ctx.font = '12px Arial';
            ctx.fillText('', 5, ANIMATION_HEIGHT - 5);

            ctx.restore();

            updateGhosts();

            this.setDirtyCanvas(true);
            requestAnimationFrame(() => this.setDirtyCanvas(true));
        }
    };

    node.setDirtyCanvas(true);
}