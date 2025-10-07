import { app } from "../../../../scripts/app.js";

app.registerExtension({
    name: "Fill-Nodes.BulletHellGame",
    async nodeCreated(node) {
        if (node.comfyClass === "FL_BulletHellGame") {
            addBulletHellGame(node);
            node.size = [700, 500];
        }
    }
});

function addBulletHellGame(node) {
    const PADDING = 23;
    let gameWidth, gameHeight;
    let ship, enemies, playerBullets, enemyBullets, score, lives, gameOver;
    let level = 1;
    let flashTimer = 0;
    let lastMouseX, lastMouseY;

    // Add reset button
    node.addWidget("button", "Reset Game", "reset", () => {
        resetGame();
    });

    function resetGame() {
        gameWidth = node.size[0] - PADDING * 2;
        gameHeight = node.size[1] - PADDING * 2 - 30; // 30px for title

        ship = { x: gameWidth / 2, y: gameHeight - 20, size: gameHeight * 0.02, angle: -Math.PI / 2 };
        enemies = [];
        playerBullets = [];
        enemyBullets = [];
        score = 0;
        lives = 3;
        gameOver = false;
        level = 1;
        flashTimer = 0;

        spawnEnemies();
    }

    function spawnEnemies() {
        enemies = [];
        for (let i = 0; i < level + 4; i++) {
            const angle = Math.random() * Math.PI * 2;
            enemies.push({
                x: Math.random() * gameWidth,
                y: Math.random() * gameHeight * 0.5,
                size: gameHeight * 0.025,
                dx: Math.cos(angle) * gameWidth * 0.005,
                dy: Math.sin(angle) * gameHeight * 0.005,
                angle: angle,
                shootTimer: 0
            });
        }
    }

    // Initialize game
    resetGame();

    function drawArrow(ctx, x, y, size, angle, color) {
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(angle);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(size, 0);
        ctx.lineTo(-size, -size / 2);
        ctx.lineTo(-size, size / 2);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
    }

    node.onDrawBackground = function(ctx) {
        if (!this.flags.collapsed) {
            ctx.save();
            ctx.translate(PADDING, PADDING + 30); // Adjust for node padding and title

            // Clear the game area
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, gameWidth, gameHeight);

            // Flash effect
            if (flashTimer > 0) {
                ctx.fillStyle = `rgba(255, 255, 255, ${flashTimer / 10})`;
                ctx.fillRect(0, 0, gameWidth, gameHeight);
                flashTimer--;
            }

            // Draw ship
            drawArrow(ctx, ship.x, ship.y, ship.size, ship.angle, 'lime');

            // Draw enemies
            enemies.forEach(enemy => {
                drawArrow(ctx, enemy.x, enemy.y, enemy.size, enemy.angle, 'red');
            });

            // Draw player bullets
            ctx.fillStyle = 'cyan';
            playerBullets.forEach(bullet => {
                ctx.beginPath();
                ctx.arc(bullet.x, bullet.y, 3, 0, Math.PI * 2);
                ctx.fill();
            });

            // Draw enemy bullets
            ctx.fillStyle = 'yellow';
            enemyBullets.forEach(bullet => {
                ctx.beginPath();
                ctx.arc(bullet.x, bullet.y, 2, 0, Math.PI * 2);
                ctx.fill();
            });

            // Draw score, lives, and level
            ctx.fillStyle = 'white';
            ctx.font = `${Math.max(12, gameHeight * 0.05)}px Arial`;
            ctx.fillText(`Score: ${score}`, 10, 20);
            ctx.fillText(`Lives: ${lives}`, gameWidth - 80, 20);
            ctx.fillText(`Level: ${level}`, gameWidth / 2 - 30, 20);

            if (gameOver) {
                ctx.fillStyle = 'white';
                ctx.font = `${Math.max(16, gameHeight * 0.12)}px Arial`;
                ctx.fillText('Game Over', gameWidth / 2 - 60, gameHeight / 2);
            }

            ctx.restore();
        }

        this.setDirtyCanvas(true);
        requestAnimationFrame(() => this.setDirtyCanvas(true));
    };

    function updateGame() {
        if (gameOver) return;

        // Move enemies
        enemies.forEach(enemy => {
            enemy.x += enemy.dx;
            enemy.y += enemy.dy;

            // Wrap around screen
            if (enemy.x < -enemy.size) enemy.x = gameWidth + enemy.size;
            if (enemy.x > gameWidth + enemy.size) enemy.x = -enemy.size;
            if (enemy.y > gameHeight + enemy.size) enemy.y = -enemy.size;
            if (enemy.y < -enemy.size) enemy.y = gameHeight + enemy.size;

            // Update angle
            enemy.angle = Math.atan2(enemy.dy, enemy.dx);

            // Enemy shooting
            enemy.shootTimer++;
            if (enemy.shootTimer >= 60) { // Shoot every 60 frames
                enemyBullets.push(...createBulletPattern(enemy));
                enemy.shootTimer = 0;
            }
        });

        // Move and guide player bullets
        playerBullets = playerBullets.filter(bullet => {
            // Find the nearest enemy
            let nearestEnemy = null;
            let minDist = Infinity;
            enemies.forEach(enemy => {
                const dist = Math.hypot(enemy.x - bullet.x, enemy.y - bullet.y);
                if (dist < minDist) {
                    minDist = dist;
                    nearestEnemy = enemy;
                }
            });

            if (nearestEnemy) {
                // Calculate the angle to the nearest enemy
                const targetAngle = Math.atan2(nearestEnemy.y - bullet.y, nearestEnemy.x - bullet.x);

                // Calculate the current angle of the bullet
                let currentAngle = Math.atan2(bullet.dy, bullet.dx);

                // Calculate the difference between the angles
                let angleDiff = targetAngle - currentAngle;

                // Normalize the angle difference
                if (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
                if (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;

                // Adjust the current angle towards the target angle
                currentAngle += angleDiff * 0.2; // Increase this value for more aggressive turning

                // Update bullet velocity
                const speed = Math.hypot(bullet.dx, bullet.dy);
                bullet.dx = Math.cos(currentAngle) * speed;
                bullet.dy = Math.sin(currentAngle) * speed;

                // Increase bullet speed over time for more aggressive pursuit
                const speedIncrease = 1.01; // Increase this value for faster acceleration
                bullet.dx *= speedIncrease;
                bullet.dy *= speedIncrease;
            }

            bullet.x += bullet.dx;
            bullet.y += bullet.dy;
            return bullet.x > 0 && bullet.x < gameWidth && bullet.y > 0 && bullet.y < gameHeight;
        });

        // Move enemy bullets
        enemyBullets = enemyBullets.filter(bullet => {
            bullet.x += bullet.dx;
            bullet.y += bullet.dy;
            return bullet.y < gameHeight && bullet.x > 0 && bullet.x < gameWidth;
        });

        // Check player bullet collisions
        playerBullets.forEach(bullet => {
            enemies = enemies.filter(enemy => {
                if (Math.hypot(bullet.x - enemy.x, bullet.y - enemy.y) < enemy.size) {
                    score += 10;
                    return false;
                }
                return true;
            });
        });

        // Check enemy bullet collisions
        enemyBullets = enemyBullets.filter(bullet => {
            if (Math.hypot(bullet.x - ship.x, bullet.y - ship.y) < ship.size) {
                lives--;
                flashTimer = 10; // Start flash effect
                if (lives <= 0) {
                    gameOver = true;
                }
                return false;
            }
            return true;
        });

        // Check if level is cleared
        if (enemies.length === 0) {
            level++;
            spawnEnemies();
        }
    }

    function createBulletPattern(enemy) {
        const bullets = [];
        const bulletSpeed = gameHeight * 0.005;
        const patterns = [
            // Spiral pattern
            () => {
                for (let i = 0; i < 8; i++) {
                    const angle = (Math.PI * 2 / 8) * i;
                    bullets.push({
                        x: enemy.x,
                        y: enemy.y,
                        dx: Math.cos(angle) * bulletSpeed,
                        dy: Math.sin(angle) * bulletSpeed
                    });
                }
            },
            // Aimed shot
            () => {
                const dx = ship.x - enemy.x;
                const dy = ship.y - enemy.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                bullets.push({
                    x: enemy.x,
                    y: enemy.y,
                    dx: (dx / dist) * bulletSpeed,
                    dy: (dy / dist) * bulletSpeed
                });
            },
            // Spread shot
            () => {
                for (let i = -2; i <= 2; i++) {
                    bullets.push({
                        x: enemy.x,
                        y: enemy.y,
                        dx: Math.cos(enemy.angle + i * 0.2) * bulletSpeed,
                        dy: Math.sin(enemy.angle + i * 0.2) * bulletSpeed
                    });
                }
            }
        ];

        // Randomly choose a pattern
        patterns[Math.floor(Math.random() * patterns.length)]();
        return bullets;
    }

    // Update game state
    setInterval(updateGame, 1000 / 60);

    // Add mouse move event listener to the node
    node.onMouseMove = function(event) {
        const localX = event.canvasX - this.pos[0] - PADDING;
        const localY = event.canvasY - this.pos[1] - PADDING - 30;

        if (localX >= 0 && localX <= gameWidth) {
            ship.x = localX;
        }
        if (localY >= 0 && localY <= gameHeight) {
            ship.y = localY;
        }

        // Update ship angle to point in the direction of movement
        if (lastMouseX !== undefined && lastMouseY !== undefined) {
            ship.angle = Math.atan2(localY - lastMouseY, localX - lastMouseX);
        }
        lastMouseX = localX;
        lastMouseY = localY;
    };

    // Add click event listener to the node
    node.onMouseDown = function(event) {
        const localX = event.canvasX - this.pos[0] - PADDING;
        const localY = event.canvasY - this.pos[1] - PADDING - 30;

        if (localX >= 0 && localX <= gameWidth && localY >= 0 && localY <= gameHeight) {
            if (!gameOver) {
                playerBullets.push({
                    x: ship.x,
                    y: ship.y,
                    dx: Math.cos(ship.angle) * gameHeight * 0.02,
                    dy: Math.sin(ship.angle) * gameHeight * 0.02
                });
            } else {
                resetGame();
            }
        }
    };

    // Handle node resizing
    node.onResize = function() {
        resetGame();
    };

    node.setSize([300, 250]); // Set initial size
    node.setDirtyCanvas(true);
}