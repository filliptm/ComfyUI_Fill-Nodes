import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Fill-Nodes.KsamplerPlus",
    async nodeCreated(node) {
        const animatedNodeClasses = [
            "FL_KsamplerPlus",
        ];

        if (animatedNodeClasses.includes(node.comfyClass)) {
            addModernSlotMachineAnimation(node);
        }
    }
});

function addModernSlotMachineAnimation(node) {
    // Easily adjustable parameters
    const PARAMS = {
        // Animation dimensions
        ANIMATION_WIDTH: 120,
        ANIMATION_HEIGHT: 80,
        X_OFFSET: 0,
        Y_OFFSET: 100,

        // Reel settings
        REEL_WIDTH: 30,
        REEL_HEIGHT: 30,
        REEL_COUNT: 3,
        SYMBOL_SIZE: 24,
        SPIN_SPEED: 0.2,

        // Game logic
        WIN_PROBABILITY: 0.25,
        SYMBOLS: ['💎', '🔮', '🎭', '🎨', '🎬', '🏆'],

        // Button settings
        BUTTON_WIDTH: 50,
        BUTTON_HEIGHT: 20,
        BUTTON_Y_OFFSET: 25,

        // Animation timings
        SPIN_DURATION: 1000,
        WIN_ANIMATION_SPEED: 0.02,

        // Visual styles
        BACKGROUND_COLOR: '#4B0082',
        FRAME_COLOR: '#8A2BE2',
        REEL_COLOR: '#2E0854',
        BUTTON_COLORS: {
            NORMAL: ['#8A2BE2', '#6A0DAD'],
            SPINNING: ['#6A0DAD', '#4B0082']
        },
        TEXT_COLOR: '#FFFFFF',

        // Particle system
        MAX_PARTICLES: 100,
        PARTICLE_DECAY_RATE: 0.02
    };

    let reels = Array(PARAMS.REEL_COUNT).fill().map(() => ({
        symbols: [...PARAMS.SYMBOLS],
        position: 0,
        spinning: false,
        finalPosition: 0,
        offset: 0
    }));

    let spinning = false;
    let winningSymbol = null;
    let winAnimation = null;
    let winAnimationProgress = 0;
    let particles = [];

    function drawModernSlotMachine(ctx) {
        // Background
        ctx.fillStyle = PARAMS.BACKGROUND_COLOR;
        ctx.fillRect(0, 0, PARAMS.ANIMATION_WIDTH, PARAMS.ANIMATION_HEIGHT);

        // Frame
        ctx.strokeStyle = PARAMS.FRAME_COLOR;
        ctx.lineWidth = 2;
        ctx.strokeRect(2, 2, PARAMS.ANIMATION_WIDTH - 4, PARAMS.ANIMATION_HEIGHT - 4);

        // Reels
        const reelStartX = (PARAMS.ANIMATION_WIDTH - (PARAMS.REEL_COUNT * PARAMS.REEL_WIDTH + (PARAMS.REEL_COUNT - 1) * 5)) / 2;
        reels.forEach((reel, i) => {
            ctx.fillStyle = PARAMS.REEL_COLOR;
            ctx.fillRect(reelStartX + i * (PARAMS.REEL_WIDTH + 5), 15, PARAMS.REEL_WIDTH, PARAMS.REEL_HEIGHT);

            ctx.save();
            ctx.beginPath();
            ctx.rect(reelStartX + i * (PARAMS.REEL_WIDTH + 5), 15, PARAMS.REEL_WIDTH, PARAMS.REEL_HEIGHT);
            ctx.clip();

            const symbolIndex = Math.floor(reel.position) % PARAMS.SYMBOLS.length;
            const y = 30 - (reel.position % 1) * PARAMS.SYMBOL_SIZE + reel.offset;
            ctx.font = `${PARAMS.SYMBOL_SIZE}px Arial`;
            ctx.fillStyle = PARAMS.TEXT_COLOR;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(reel.symbols[symbolIndex], reelStartX + PARAMS.REEL_WIDTH / 2 + i * (PARAMS.REEL_WIDTH + 5), y);

            ctx.restore();
        });

        // Spin button
        const buttonGradient = ctx.createLinearGradient(0, PARAMS.ANIMATION_HEIGHT - PARAMS.BUTTON_Y_OFFSET, 0, PARAMS.ANIMATION_HEIGHT - PARAMS.BUTTON_Y_OFFSET + PARAMS.BUTTON_HEIGHT);
        const buttonColors = spinning ? PARAMS.BUTTON_COLORS.SPINNING : PARAMS.BUTTON_COLORS.NORMAL;
        buttonGradient.addColorStop(0, buttonColors[0]);
        buttonGradient.addColorStop(1, buttonColors[1]);
        ctx.fillStyle = buttonGradient;
        ctx.beginPath();
        ctx.roundRect(PARAMS.ANIMATION_WIDTH / 2 - PARAMS.BUTTON_WIDTH / 2, PARAMS.ANIMATION_HEIGHT - PARAMS.BUTTON_Y_OFFSET, PARAMS.BUTTON_WIDTH, PARAMS.BUTTON_HEIGHT, 5);
        ctx.fill();
        ctx.fillStyle = PARAMS.TEXT_COLOR;
        ctx.font = 'bold 12px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('SPIN', PARAMS.ANIMATION_WIDTH / 2, PARAMS.ANIMATION_HEIGHT - PARAMS.BUTTON_Y_OFFSET + PARAMS.BUTTON_HEIGHT / 2);

        // Win animation
        if (winAnimation) {
            winAnimation(ctx, winAnimationProgress);
        }

        // Draw particles
        particles.forEach(p => {
            ctx.fillStyle = p.color;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    function spin() {
        if (spinning) return;
        spinning = true;
        winningSymbol = null;
        winAnimation = null;
        winAnimationProgress = 0;
        particles = [];

        reels.forEach(reel => {
            reel.spinning = true;
            reel.spinDuration = 50 + Math.random() * 50;
            reel.offset = Math.random() * PARAMS.SYMBOL_SIZE;
        });

        const isWin = Math.random() < PARAMS.WIN_PROBABILITY;
        if (isWin) {
            const winSymbolIndex = Math.floor(Math.random() * PARAMS.SYMBOLS.length);
            reels.forEach(reel => reel.finalPosition = winSymbolIndex);
            winningSymbol = PARAMS.SYMBOLS[winSymbolIndex];
        } else {
            reels.forEach(reel => reel.finalPosition = Math.floor(Math.random() * PARAMS.SYMBOLS.length));
        }

        setTimeout(() => {
            reels.forEach(reel => {
                reel.spinning = false;
                reel.position = reel.finalPosition;
                reel.offset = 0;
            });
            spinning = false;
            if (isWin) triggerWin();
        }, PARAMS.SPIN_DURATION);
    }

    function triggerWin() {
        switch (winningSymbol) {
            case '💎': winAnimation = diamondWinAnimation; break;
            case '🔮': winAnimation = crystalBallWinAnimation; break;
            case '🎭': winAnimation = masksWinAnimation; break;
            case '🎨': winAnimation = paletteWinAnimation; break;
            case '🎬': winAnimation = clapperboardWinAnimation; break;
            case '🏆': winAnimation = trophyWinAnimation; break;
        }
    }

    function diamondWinAnimation(ctx, progress) {
        const size = 40 * Math.sin(progress * Math.PI);
        ctx.font = `${size}px Arial`;
        ctx.fillStyle = `rgba(255, 255, 255, ${1 - progress})`;
        ctx.fillText('💎', PARAMS.ANIMATION_WIDTH / 2, PARAMS.ANIMATION_HEIGHT / 2);

        // Sparkling effect
        if (Math.random() < 0.3) {
            particles.push({
                x: Math.random() * PARAMS.ANIMATION_WIDTH,
                y: Math.random() * PARAMS.ANIMATION_HEIGHT,
                size: Math.random() * 3,
                color: `hsl(${Math.random() * 360}, 100%, 50%)`,
                life: 1
            });
        }
    }

    function crystalBallWinAnimation(ctx, progress) {
        ctx.fillStyle = `rgba(128, 0, 128, ${0.5 - progress * 0.5})`;
        ctx.beginPath();
        ctx.arc(PARAMS.ANIMATION_WIDTH / 2, PARAMS.ANIMATION_HEIGHT / 2, 30 * progress, 0, Math.PI * 2);
        ctx.fill();

        // Swirling mist effect
        for (let i = 0; i < 5; i++) {
            const angle = progress * Math.PI * 2 + i * Math.PI / 2.5;
            const x = PARAMS.ANIMATION_WIDTH / 2 + Math.cos(angle) * 20 * progress;
            const y = PARAMS.ANIMATION_HEIGHT / 2 + Math.sin(angle) * 20 * progress;
            ctx.font = '12px Arial';
            ctx.fillStyle = `rgba(255, 255, 255, ${1 - progress})`;
            ctx.fillText('✨', x, y);
        }

        ctx.font = '24px Arial';
        ctx.fillStyle = 'white';
        ctx.fillText('🔮', PARAMS.ANIMATION_WIDTH / 2, PARAMS.ANIMATION_HEIGHT / 2);
    }

    function masksWinAnimation(ctx, progress) {
        const angle = progress * Math.PI * 4;
        ctx.save();
        ctx.translate(PARAMS.ANIMATION_WIDTH / 2, PARAMS.ANIMATION_HEIGHT / 2);

        // Comedy mask
        ctx.rotate(angle);
        ctx.font = '24px Arial';
        ctx.fillStyle = `rgba(255, 255, 0, ${1 - progress})`;
        ctx.fillText('😂', 0, -20);

        // Tragedy mask
        ctx.rotate(Math.PI);
        ctx.fillStyle = `rgba(0, 0, 255, ${1 - progress})`;
        ctx.fillText('😭', 0, -20);

        ctx.restore();

        // Confetti effect
        if (Math.random() < 0.2) {
            particles.push({
                x: Math.random() * PARAMS.ANIMATION_WIDTH,
                y: -5,
                size: Math.random() * 5 + 2,
                color: Math.random() < 0.5 ? 'yellow' : 'blue',
                life: 1,
                vy: Math.random() * 2 + 1
            });
        }
    }

    function paletteWinAnimation(ctx, progress) {
        const colors = ['red', 'blue', 'green', 'yellow', 'purple'];
        colors.forEach((color, i) => {
            const angle = (i / colors.length + progress) * Math.PI * 4;
            const x = PARAMS.ANIMATION_WIDTH / 2 + Math.cos(angle) * 30 * Math.sin(progress * Math.PI);
            const y = PARAMS.ANIMATION_HEIGHT / 2 + Math.sin(angle) * 30 * Math.sin(progress * Math.PI);
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fill();

            // Paint splatter effect
            if (Math.random() < 0.1) {
                particles.push({
                    x: x,
                    y: y,
                    size: Math.random() * 3 + 1,
                    color: color,
                    life: 1,
                    vx: (Math.random() - 0.5) * 2,
                    vy: (Math.random() - 0.5) * 2
                });
            }
        });

        ctx.font = '24px Arial';
        ctx.fillStyle = 'white';
        ctx.fillText('🎨', PARAMS.ANIMATION_WIDTH / 2, PARAMS.ANIMATION_HEIGHT / 2);
    }

    function clapperboardWinAnimation(ctx, progress) {
        ctx.save();
        ctx.translate(PARAMS.ANIMATION_WIDTH / 2, PARAMS.ANIMATION_HEIGHT / 2);

        // Clapping effect
        const clapAngle = Math.sin(progress * Math.PI * 8) * 0.3;
        ctx.rotate(clapAngle);
        ctx.font = '24px Arial';
        ctx.fillStyle = 'white';
        ctx.fillText('🎬', 0, 0);

        // Starburst effect
        for (let i = 0; i < 8; i++) {
            const angle = i * Math.PI / 4 + progress * Math.PI * 2;
            const length = 20 + Math.sin(progress * Math.PI) * 10;
            ctx.strokeStyle = `rgba(255, 255, 0, ${1 - progress})`;
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(Math.cos(angle) * length, Math.sin(angle) * length);
            ctx.stroke();
        }

        ctx.restore();
    }

    function trophyWinAnimation(ctx, progress) {
        const y = PARAMS.ANIMATION_HEIGHT / 2 + Math.sin(progress * Math.PI * 2) * 10;
        ctx.font = '24px Arial';
        ctx.fillStyle = `rgba(255, 215, 0, ${1 - progress})`;
        ctx.fillText('🏆', PARAMS.ANIMATION_WIDTH / 2, y);

        // Fireworks effect
        if (Math.random() < 0.2) {
            const x = Math.random() * PARAMS.ANIMATION_WIDTH;
            const y = PARAMS.ANIMATION_HEIGHT;
            for (let i = 0; i < 20; i++) {
                const angle = Math.random() * Math.PI * 2;
                const speed = Math.random() * 2 + 1;
                particles.push({
                    x: x,
                    y: y,
                    size: Math.random() * 2 + 1,
                    color: `hsl(${Math.random() * 360}, 100%, 50%)`,
                    life: 1,
                    vx: Math.cos(angle) * speed,
                    vy: Math.sin(angle) * speed - 2
                });
            }
        }

        ctx.font = '16px Arial';
        ctx.fillStyle = `rgba(255, 255, 255, ${progress})`;
        ctx.fillText('JACKPOT!', PARAMS.ANIMATION_WIDTH / 2, PARAMS.ANIMATION_HEIGHT / 2 + 30);
    }

    function updateAnimation() {
        reels.forEach(reel => {
            if (reel.spinning) {
                reel.position += PARAMS.SPIN_SPEED;
                if (reel.offset > 0) {
                    reel.offset -= 1;
                }
            }
        });

        if (winAnimation) {
            winAnimationProgress += PARAMS.WIN_ANIMATION_SPEED;
            if (winAnimationProgress >= 1) {
                winAnimation = null;
                winAnimationProgress = 0;
            }
        }

        // Update particles
        particles = particles.filter(p => {
            p.x += p.vx || 0;
            p.y += p.vy || 0;
            p.life -= PARAMS.PARTICLE_DECAY_RATE;
            return p.life > 0;
        });

        // Limit the number of particles
        if (particles.length > PARAMS.MAX_PARTICLES) {
            particles.splice(0, particles.length - PARAMS.MAX_PARTICLES);
        }
    }

    node.onDrawBackground = function(ctx) {
        if (!this.flags.collapsed) {
            ctx.save();

            const nodeWidth = this.size[0];
            const baseXOffset = (nodeWidth - PARAMS.ANIMATION_WIDTH) / 2;
            ctx.translate(baseXOffset + PARAMS.X_OFFSET, -PARAMS.ANIMATION_HEIGHT + PARAMS.Y_OFFSET);

            drawModernSlotMachine(ctx);

            ctx.restore();

            updateAnimation();

            this.setDirtyCanvas(true);
            requestAnimationFrame(() => this.setDirtyCanvas(true));
        }
    };

    node.onMouseDown = function(event) {
        const nodeWidth = this.size[0];
        const baseXOffset = (nodeWidth - PARAMS.ANIMATION_WIDTH) / 2;
        const localX = event.canvasX - this.pos[0] - baseXOffset - PARAMS.X_OFFSET;
        const localY = event.canvasY - this.pos[1] + PARAMS.ANIMATION_HEIGHT - PARAMS.Y_OFFSET;

        if (localX >= PARAMS.ANIMATION_WIDTH / 2 - PARAMS.BUTTON_WIDTH / 2 &&
            localX <= PARAMS.ANIMATION_WIDTH / 2 + PARAMS.BUTTON_WIDTH / 2 &&
            localY >= PARAMS.ANIMATION_HEIGHT - PARAMS.BUTTON_Y_OFFSET &&
            localY <= PARAMS.ANIMATION_HEIGHT - PARAMS.BUTTON_Y_OFFSET + PARAMS.BUTTON_HEIGHT) {
            spin();
            return true;
        }
    };

    node.setDirtyCanvas(true);
}