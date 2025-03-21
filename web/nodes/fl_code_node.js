import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "Comfy.FL_CodeNode",
	async nodeCreated(node) {
		if (node.comfyClass === "FL_CodeNode") {
			addCodeEditor(node);
        }
	}
});

function addCodeEditor(node) {
	// Constants for UI layout
	const PADDING = 10;
	const LINE_HEIGHT = 20;
	const CHAR_WIDTH = 8;
	const GUTTER_WIDTH = 40;
	
	// Editor state
	let editorContent = "";
	let cursorPos = 0;
	let scrollOffset = 0;
	let lines = [""];
	let isEditing = false;

	// Find and hide the default code input widget
	const codeWidget = node.widgets.find(w => w.name === "code_input");
	if (codeWidget) {
		editorContent = codeWidget.value;
		lines = editorContent.split('\n');
		// Hide the original widget
		codeWidget.origComputeSize = codeWidget.computeSize;
		codeWidget.computeSize = () => [0, -4]; // Make it invisible
	}

	// Keywords for basic syntax highlighting
	const keywords = [
		"def", "class", "import", "from", "return", "if", "else", "elif",
		"for", "while", "try", "except", "with", "as", "in", "is", "not",
		"and", "or", "True", "False", "None"
	];

	function drawEditor(ctx) {
		if (!node.flags.collapsed) {
			const nodeWidth = node.size[0];
			const width = nodeWidth - PADDING * 2;
			const height = node.size[1] - 60; // Leave space for other widgets
			
			// Draw editor background
			ctx.fillStyle = "#1e1e1e";
			ctx.fillRect(PADDING, 40, width, height);
			
			// Draw gutter background
			ctx.fillStyle = "#252526";
			ctx.fillRect(PADDING, 40, GUTTER_WIDTH, height);

			// Draw lines
			ctx.font = "12px monospace";
			const visibleLines = Math.floor(height / LINE_HEIGHT);
			
			for (let i = 0; i < Math.min(lines.length, visibleLines); i++) {
				const y = 40 + (i + 1) * LINE_HEIGHT;
				
				// Draw line number
				ctx.fillStyle = "#858585";
				ctx.textAlign = "right";
				ctx.fillText((i + 1).toString(), PADDING + GUTTER_WIDTH - 5, y);
				
				// Draw line content with syntax highlighting
				ctx.textAlign = "left";
				const line = lines[i];
				let x = PADDING + GUTTER_WIDTH + 5;
				
				// Simple syntax highlighting
				const words = line.split(/(\s+)/);
				words.forEach(word => {
					if (keywords.includes(word)) {
						ctx.fillStyle = "#569cd6"; // Blue for keywords
					} else if (/^["'].*["']$/.test(word)) {
						ctx.fillStyle = "#ce9178"; // Brown for strings
					} else if (/^\d+$/.test(word)) {
						ctx.fillStyle = "#b5cea8"; // Green for numbers
					} else {
						ctx.fillStyle = "#d4d4d4"; // Default color
					}
					ctx.fillText(word, x, y);
					x += ctx.measureText(word).width;
				});
			}

			// Draw cursor if editing
			if (isEditing) {
				const cursorLine = Math.floor(cursorPos / (nodeWidth - GUTTER_WIDTH - PADDING * 2));
				const cursorX = (cursorPos % (nodeWidth - GUTTER_WIDTH - PADDING * 2)) * CHAR_WIDTH + PADDING + GUTTER_WIDTH + 5;
				const cursorY = 40 + cursorLine * LINE_HEIGHT;
				
				ctx.strokeStyle = "#ffffff";
				ctx.beginPath();
				ctx.moveTo(cursorX, cursorY);
				ctx.lineTo(cursorX, cursorY + LINE_HEIGHT);
				ctx.stroke();
			}
		}
	}

	// Override node's drawing
	const origDrawBackground = node.onDrawBackground;
	node.onDrawBackground = function(ctx) {
		if (origDrawBackground) {
			origDrawBackground.call(this, ctx);
		}
		drawEditor(ctx);
	};

	// Handle mouse events
	node.onMouseDown = function(evt) {
		const localX = evt.canvasX - this.pos[0];
		const localY = evt.canvasY - this.pos[1];
		
		if (localY >= 40 && localX >= PADDING && localX <= this.size[0] - PADDING) {
			isEditing = true;
			// Calculate cursor position from click
			const line = Math.floor((localY - 40) / LINE_HEIGHT);
			const char = Math.floor((localX - PADDING - GUTTER_WIDTH - 5) / CHAR_WIDTH);
			cursorPos = Math.max(0, Math.min(lines[line]?.length || 0, char));
			this.setDirtyCanvas(true);
			return true;
		}
		return false;
	};

	// Handle keyboard input
	document.addEventListener('keydown', function(evt) {
		if (isEditing && node.selected) {
			const nodeWidth = node.size[0];
			if (evt.key === 'Enter') {
				const currentLine = Math.floor(cursorPos / (nodeWidth - GUTTER_WIDTH - PADDING * 2));
				const currentLineContent = lines[currentLine] || "";
				const cursorPosInLine = cursorPos % (nodeWidth - GUTTER_WIDTH - PADDING * 2);
				
				// Split the current line at cursor position
				const newLine = currentLineContent.substring(cursorPosInLine);
				lines[currentLine] = currentLineContent.substring(0, cursorPosInLine);
				lines.splice(currentLine + 1, 0, newLine);
				
				cursorPos = (currentLine + 1) * (nodeWidth - GUTTER_WIDTH - PADDING * 2);
				evt.preventDefault();
			} else if (evt.key === 'Backspace') {
				const currentLine = Math.floor(cursorPos / (nodeWidth - GUTTER_WIDTH - PADDING * 2));
				if (cursorPos > 0 || currentLine > 0) {
					if (cursorPos % (nodeWidth - GUTTER_WIDTH - PADDING * 2) === 0 && currentLine > 0) {
						// At start of line, merge with previous line
						const previousLine = lines[currentLine - 1];
						lines[currentLine - 1] = previousLine + (lines[currentLine] || "");
						lines.splice(currentLine, 1);
						cursorPos = (currentLine - 1) * (nodeWidth - GUTTER_WIDTH - PADDING * 2) + previousLine.length;
					} else {
						// Normal backspace within line
						const linePos = cursorPos % (nodeWidth - GUTTER_WIDTH - PADDING * 2);
						lines[currentLine] = lines[currentLine].substring(0, linePos - 1) +
											lines[currentLine].substring(linePos);
						cursorPos--;
					}
				}
				evt.preventDefault();
			} else if (evt.key.length === 1 && !evt.ctrlKey && !evt.metaKey) {
				const currentLine = Math.floor(cursorPos / (nodeWidth - GUTTER_WIDTH - PADDING * 2));
				const linePos = cursorPos % (nodeWidth - GUTTER_WIDTH - PADDING * 2);
				
				// Ensure the line exists
				while (lines.length <= currentLine) {
					lines.push("");
				}
				
				// Insert character at cursor position
				lines[currentLine] = lines[currentLine].substring(0, linePos) +
									evt.key +
									lines[currentLine].substring(linePos);
				cursorPos++;
				evt.preventDefault();
			}
			
			// Update widget value
			if (codeWidget) {
				codeWidget.value = lines.join('\n');
			}
			
			node.setDirtyCanvas(true);
		}
	});

	node.onMouseUp = function(evt) {
		// Handle mouse up if needed
	};

	// Initial size
	node.setSize([400, 300]);
}