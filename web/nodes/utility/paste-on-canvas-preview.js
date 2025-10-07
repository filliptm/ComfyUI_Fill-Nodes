import { app } from "../../../../scripts/app.js";

app.registerExtension({
    name: "Fill-Nodes.PasteOnCanvasAnimatedDisplay",
    async nodeCreated(node) {
        if (node.comfyClass === "FL_PasteOnCanvas") {
            addAnimatedDisplay(node);
        }
    }
});

function addAnimatedDisplay(node) {
    // Increase the node size
    node.size = [220, 270];

    // Override the onDrawBackground method to add our animated display
    node.onDrawBackground = function(ctx) {

        // Calculate the current time to animate the balls
        const time = Date.now() * 0.001; // Current time in seconds

        // Draw moving circles and their mirrored counterparts
        for (let i = 0; i < 5; i++) {
            const x = 60 + (this.size[0] - 20) * (0.2 + 0.02 * Math.sin(time + i));
            const y = -20 + (this.size[1] - 40) * (0.2 + 0.02 * Math.cos(time * 1.5 + i));
            const mirrorX = this.size[0] - x; // Mirror X coordinate

            // Original set of circles
            ctx.beginPath();
            ctx.arc(x, y, 7, 0, Math.PI * 2);
            ctx.fillStyle = `hsl(${(time * 100 + i * 50) % 360}, 100%, 75%)`;
            ctx.fill();

            // Mirrored set of circles
            ctx.beginPath();
            ctx.arc(mirrorX, y, 7, 0, Math.PI * 2);
            ctx.fillStyle = `hsl(${(time * 100 + i * 50) % 360}, 100%, 75%)`;
            ctx.fill();
        }

        // Request next frame for continuous animation
        node.setDirtyCanvas(true);
        requestAnimationFrame(() => node.setDirtyCanvas(true));
    };

    // Force the initial redraw to start the animation
    node.setDirtyCanvas(true);
}
