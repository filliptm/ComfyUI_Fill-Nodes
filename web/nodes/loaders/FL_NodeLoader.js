import { app } from "../../../../scripts/app.js";

class FL_NodeLoader {
    constructor() {
        this.serialize_widgets = true;
        this.addInput("trigger", "trigger");
        this.addOutput("complete", "trigger");
        this.columnCount = 3; // Default column count
    }

    onNodeCreated() {
        this.addLoadButton();
        this.updateNodeInfo();
    }

    addLoadButton() {
        this.addWidget(
            "button",
            "Load FL Nodes",
            "Load",
            (value) => {
                this.loadFLNodes();
            }
        );

        // Add column count selector with direct value
        this.addWidget(
            "number",
            "Columns",
            3, // Default value
            (value) => {
                this.columnCount = Math.max(1, Math.floor(value));
            },
            { min: 1, max: 10, step: 1, precision: 0, default: 3 }
        );

        this.addWidget(
            "text",
            "Node Count",
            "",
            (v) => {},
            { readonly: true }
        );

        this.addWidget(
            "text",
            "Node List",
            "",
            (v) => {},
            { readonly: true, multiline: true }
        );
    }

async loadFLNodes() {
    const nodeTypes = Object.keys(LiteGraph.registered_node_types)
        .filter(type => type.includes("FL_"))
        .sort((a, b) => a.localeCompare(b));

    if (nodeTypes.length === 0) {
        console.log("No FL_ nodes found");
        return;
    }

    // Clear existing nodes
    const existingNodes = app.graph.findNodesByType("FL_");
    existingNodes.forEach(node => {
        if (node !== this) {
            app.graph.remove(node);
        }
    });

    // First pass: Create nodes and get their sizes
    const nodeInfo = [];
    for (const nodeType of nodeTypes) {
        const node = LiteGraph.createNode(nodeType);
        if (node) {
            app.graph.add(node);
            if (node.onNodeCreated) {
                node.onNodeCreated();
            }

            // Wait for node initialization
            await new Promise(resolve => setTimeout(resolve, 10));

            nodeInfo.push({
                node: node,
                width: node.size[0],
                height: node.size[1]
            });
        }
    }

    // Calculate column metrics
    const numColumns = Math.max(1, Math.min(this.columnCount || 3, nodeTypes.length));
    const nodesPerColumn = Math.ceil(nodeInfo.length / numColumns);

    // Calculate maximum dimensions for each column
    const columnMetrics = Array(numColumns).fill().map(() => ({
        maxWidth: 0,
        maxHeight: 0
    }));

    // Find maximum dimensions for each column
    nodeInfo.forEach((info, index) => {
        const columnIndex = Math.floor(index / nodesPerColumn);
        if (columnIndex < numColumns) {
            columnMetrics[columnIndex].maxWidth = Math.max(
                columnMetrics[columnIndex].maxWidth,
                info.width
            );
            columnMetrics[columnIndex].maxHeight = Math.max(
                columnMetrics[columnIndex].maxHeight,
                info.height
            );
        }
    });

    // Calculate positions with proper spacing
    const padding = 50; // Minimum space between nodes
    let currentX = padding;

    // Position nodes using calculated metrics
    nodeInfo.forEach((info, index) => {
        const columnIndex = Math.floor(index / nodesPerColumn);
        const positionInColumn = index % nodesPerColumn;

        // Calculate X position based on previous columns' widths
        let x = padding;
        for (let i = 0; i < columnIndex; i++) {
            x += columnMetrics[i].maxWidth + padding;
        }

        // Center node within its column
        x += (columnMetrics[columnIndex].maxWidth - info.width) / 2;

        // Calculate Y position with proper spacing
        const y = padding + (positionInColumn * (Math.max(...columnMetrics.map(m => m.maxHeight)) + padding));

        // Set node position
        info.node.pos = [x, y];
    });

    // Update info widgets
    this.updateNodeInfo();

    // Center view on nodes
    const canvas = app.canvas;
    if (canvas) {
        // Calculate the bounds of all nodes
        let minX = Infinity;
        let minY = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;

        nodeInfo.forEach(info => {
            const node = info.node;
            minX = Math.min(minX, node.pos[0]);
            minY = Math.min(minY, node.pos[1]);
            maxX = Math.max(maxX, node.pos[0] + node.size[0]);
            maxY = Math.max(maxY, node.pos[1] + node.size[1]);
        });

        // Center the view on the nodes
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        canvas.offset = [
            canvas.width / 2 - centerX,
            canvas.height / 2 - centerY
        ];
        canvas.setZoom(0.8);
        canvas.setDirty(true, true);
    }
}

    updateNodeInfo() {
        const flNodes = Object.keys(LiteGraph.registered_node_types)
            .filter(type => type.includes("FL_"))
            .sort();

        if (this.widgets) {
            this.widgets[2].value = flNodes.length.toString();
            this.widgets[3].value = flNodes.join("\n");
        }
    }

    onExecute() {
        this.triggerSlot(0);
    }
}

// Register the node
app.registerExtension({
    name: "Comfy.FL_NodeLoader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FL_NodeLoader") {
            for (const key of Object.getOwnPropertyNames(FL_NodeLoader.prototype)) {
                if (key !== "constructor") {
                    nodeType.prototype[key] = FL_NodeLoader.prototype[key];
                }
            }
        }
    },
    nodeTypes: {
        FL_NodeLoader: {
            output: ["trigger"],
            input: ["trigger"],
            name: "FL Node Loader",
            category: "utils",
            color: "#2a363b"
        }
    }
});``