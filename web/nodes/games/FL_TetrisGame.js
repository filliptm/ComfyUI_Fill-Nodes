import { app } from "../../../../scripts/app.js";

app.registerExtension({
    name: "Fill-Nodes.TetrisGame",
    async nodeCreated(node) {
        if (node.comfyClass === "FL_TetrisGame") {
            addTetrisGame(node);
            node.size = [290, 594];
        }
    }
});

function addTetrisGame(node) {
    const PADDING = 10;
    const BORDER_WIDTH = 2;
    let canvas, context, grid, tetrominoSequence, playfield, tetrominos, colors;
    let count, tetromino, rAF, gameOver;

    // Add reset button
    node.addWidget("button", "Reset Game", "reset", () => {
        resetGame();
    });

    function resetGame() {
        const nodeWidth = node.size[0] - PADDING * 2;
        const nodeHeight = node.size[1] - PADDING * 2 - 30; // 30px for title
        grid = Math.floor(Math.min(nodeWidth / 10, nodeHeight / 20));
        canvas = document.createElement('canvas');
        canvas.width = grid * 10 + BORDER_WIDTH * 2;
        canvas.height = grid * 20 + BORDER_WIDTH * 2;
        context = canvas.getContext('2d');

        tetrominoSequence = [];
        playfield = [];
        for (let row = -2; row < 20; row++) {
            playfield[row] = [];
            for (let col = 0; col < 10; col++) {
                playfield[row][col] = 0;
            }
        }

        tetrominos = {
            'I': [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
            'J': [[1,0,0],[1,1,1],[0,0,0]],
            'L': [[0,0,1],[1,1,1],[0,0,0]],
            'O': [[1,1],[1,1]],
            'S': [[0,1,1],[1,1,0],[0,0,0]],
            'Z': [[1,1,0],[0,1,1],[0,0,0]],
            'T': [[0,1,0],[1,1,1],[0,0,0]]
        };

        colors = {
            'I': 'cyan', 'O': 'yellow', 'T': 'purple',
            'S': 'green', 'Z': 'red', 'J': 'blue', 'L': 'orange'
        };

        count = 0;
        tetromino = getNextTetromino();
        gameOver = false;

        if (rAF) {
            cancelAnimationFrame(rAF);
        }
        rAF = requestAnimationFrame(loop);
    }

    function getRandomInt(min, max) {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }

    function generateSequence() {
        const sequence = ['I', 'J', 'L', 'O', 'S', 'T', 'Z'];
        while (sequence.length) {
            const rand = getRandomInt(0, sequence.length - 1);
            const name = sequence.splice(rand, 1)[0];
            tetrominoSequence.push(name);
        }
    }

    function getNextTetromino() {
        if (tetrominoSequence.length === 0) {
            generateSequence();
        }
        const name = tetrominoSequence.pop();
        const matrix = tetrominos[name];
        const col = playfield[0].length / 2 - Math.ceil(matrix[0].length / 2);
        const row = name === 'I' ? -1 : -2;
        return { name: name, matrix: matrix, row: row, col: col };
    }

    function rotate(matrix) {
        const N = matrix.length - 1;
        const result = matrix.map((row, i) => row.map((val, j) => matrix[N - j][i]));
        return result;
    }

    function isValidMove(matrix, cellRow, cellCol) {
        for (let row = 0; row < matrix.length; row++) {
            for (let col = 0; col < matrix[row].length; col++) {
                if (matrix[row][col] && (
                    cellCol + col < 0 ||
                    cellCol + col >= playfield[0].length ||
                    cellRow + row >= playfield.length ||
                    playfield[cellRow + row][cellCol + col])
                ) {
                    return false;
                }
            }
        }
        return true;
    }

    function placeTetromino() {
        for (let row = 0; row < tetromino.matrix.length; row++) {
            for (let col = 0; col < tetromino.matrix[row].length; col++) {
                if (tetromino.matrix[row][col]) {
                    if (tetromino.row + row < 0) {
                        return showGameOver();
                    }
                    playfield[tetromino.row + row][tetromino.col + col] = tetromino.name;
                }
            }
        }

        for (let row = playfield.length - 1; row >= 0; ) {
            if (playfield[row].every(cell => !!cell)) {
                for (let r = row; r >= 0; r--) {
                    for (let c = 0; c < playfield[r].length; c++) {
                        playfield[r][c] = playfield[r-1][c];
                    }
                }
            }
            else {
                row--;
            }
        }

        tetromino = getNextTetromino();
    }

    function showGameOver() {
        cancelAnimationFrame(rAF);
        gameOver = true;

        context.fillStyle = 'black';
        context.globalAlpha = 0.75;
        context.fillRect(BORDER_WIDTH, canvas.height / 2 - 30, canvas.width - BORDER_WIDTH * 2, 60);

        context.globalAlpha = 1;
        context.fillStyle = 'white';
        context.font = '16px monospace';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText('GAME OVER!', canvas.width / 2, canvas.height / 2);
    }

    function drawBorder() {
        context.fillStyle = 'white';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = 'black';
        context.fillRect(BORDER_WIDTH, BORDER_WIDTH, canvas.width - BORDER_WIDTH * 2, canvas.height - BORDER_WIDTH * 2);
    }

    function loop() {
        rAF = requestAnimationFrame(loop);
        context.clearRect(0, 0, canvas.width, canvas.height);

        drawBorder();

        for (let row = 0; row < 20; row++) {
            for (let col = 0; col < 10; col++) {
                if (playfield[row][col]) {
                    const name = playfield[row][col];
                    context.fillStyle = colors[name];
                    context.fillRect(col * grid + BORDER_WIDTH, row * grid + BORDER_WIDTH, grid - 1, grid - 1);
                }
            }
        }

        if (tetromino) {
            if (++count > 35) {
                tetromino.row++;
                count = 0;
                if (!isValidMove(tetromino.matrix, tetromino.row, tetromino.col)) {
                    tetromino.row--;
                    placeTetromino();
                }
            }

            context.fillStyle = colors[tetromino.name];
            for (let row = 0; row < tetromino.matrix.length; row++) {
                for (let col = 0; col < tetromino.matrix[row].length; col++) {
                    if (tetromino.matrix[row][col]) {
                        context.fillRect((tetromino.col + col) * grid + BORDER_WIDTH, (tetromino.row + row) * grid + BORDER_WIDTH, grid - 1, grid - 1);
                    }
                }
            }
        }

        node.setDirtyCanvas(true);
    }

    node.onDrawBackground = function(ctx) {
        if (!this.flags.collapsed) {
            ctx.drawImage(canvas, PADDING, PADDING + 30);
        }
    };

    function handleKeyDown(e) {
        if (gameOver) return;

        if (e.which === 37 || e.which === 39) {
            const col = e.which === 37 ? tetromino.col - 1 : tetromino.col + 1;
            if (isValidMove(tetromino.matrix, tetromino.row, col)) {
                tetromino.col = col;
            }
        }

        if (e.which === 38) {
            const matrix = rotate(tetromino.matrix);
            if (isValidMove(matrix, tetromino.row, tetromino.col)) {
                tetromino.matrix = matrix;
            }
        }

        if (e.which === 40) {
            const row = tetromino.row + 1;
            if (!isValidMove(tetromino.matrix, row, tetromino.col)) {
                tetromino.row = row - 1;
                placeTetromino();
                return;
            }
            tetromino.row = row;
        }
    }

    // Handle node resizing
    node.onResize = function() {
        resetGame();
    };

    // Add event listener for keydown
    document.addEventListener('keydown', handleKeyDown);

    // Remove event listener when node is removed
    node.onRemoved = function() {
        document.removeEventListener('keydown', handleKeyDown);
    };

    node.setSize([300, 400]); // Set initial size
    resetGame();
}