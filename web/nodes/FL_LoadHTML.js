import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.FL_LoadHTML",
    async nodeCreated(node) {
        if (node.comfyClass === "FL_LoadHTML") {
            addImageSequenceUI(node);
        }
    }
});

function addImageSequenceUI(node) {
    const MIN_WIDTH = 300;
    const MIN_HEIGHT = 350;
    const PADDING = 10;

    let images = [];
    let currentFrame = 0;
    let isPlaying = false;
    let animationInterval;
    let fps = 24;

    // Create canvas for image display
    const canvas = document.createElement("canvas");
    canvas.style.position = "absolute";
    canvas.style.left = PADDING + "px";
    canvas.style.top = "60px";

    node.addWidget("custom", "canvas", canvas, {
        serializeValue: () => ({}),
        getValue: () => canvas,
    });

    // Create hidden file input
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.multiple = true;
    fileInput.accept = "image/*";
    fileInput.style.display = "none";
    document.body.appendChild(fileInput);

    // Add controls
    const loadImagesBtn = node.addWidget("button", "Load Images", null, () => {
        fileInput.click();
    });

    const playPauseBtn = node.addWidget("button", "Play", null, () => {
        isPlaying = !isPlaying;
        playPauseBtn.name = isPlaying ? "Pause" : "Play";
        if (isPlaying) {
            startAnimation();
        } else {
            stopAnimation();
        }
    });

    const fpsWidget = node.addWidget("number", "FPS", fps, (value) => {
        fps = value;
        if (isPlaying) {
            stopAnimation();
            startAnimation();
        }
    });
    fpsWidget.min = 1;
    fpsWidget.max = 60;
    fpsWidget.step = 1;

    fileInput.onchange = async (e) => {
        const files = Array.from(e.target.files);
        images = [];
        for (const file of files) {
            const base64 = await fileToBase64(file);
            const img = new Image();
            img.src = base64;
            await new Promise(resolve => {
                img.onload = () => {
                    images.push(img);
                    resolve();
                };
            });
        }
        currentFrame = 0;
        drawFrame();
        node.setDirtyCanvas(true);
    };

    function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    function drawFrame() {
        if (images.length === 0) return;

        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const image = images[currentFrame];
        const scale = Math.min(canvas.width / image.width, canvas.height / image.height);
        const x = (canvas.width - image.width * scale) / 2;
        const y = (canvas.height - image.height * scale) / 2;

        ctx.drawImage(image, x, y, image.width * scale, image.height * scale);
    }

    function startAnimation() {
        stopAnimation();
        animationInterval = setInterval(() => {
            currentFrame = (currentFrame + 1) % images.length;
            drawFrame();
            node.setDirtyCanvas(true);
        }, 1000 / fps);
    }

    function stopAnimation() {
        clearInterval(animationInterval);
    }

    node.onResize = function() {
        const minSize = [MIN_WIDTH, MIN_HEIGHT];
        node.size[0] = Math.max(minSize[0], node.size[0]);
        node.size[1] = Math.max(minSize[1], node.size[1]);
        canvas.width = node.size[0] - 2 * PADDING;
        canvas.height = node.size[1] - 100;
        drawFrame();
        node.setDirtyCanvas(true);
    };

    node.onDrawBackground = function(ctx) {
        if (!this.flags.collapsed) {
            ctx.fillStyle = "#1e1e1e";
            ctx.fillRect(0, 0, node.size[0], node.size[1]);

            ctx.fillStyle = "#9b59b6";
            ctx.font = "16px Arial";
            ctx.fillText("Image Sequence Animation", PADDING, 30);
        }
    };

    // Initial size update
    node.onResize();
}