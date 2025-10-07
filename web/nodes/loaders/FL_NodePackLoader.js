import { app } from "../../../../scripts/app.js";

class FL_NodePackLoader {
    constructor() {
        this.serialize_widgets = true;
        this.addInput("trigger", "trigger");
        this.addOutput("complete", "trigger");
        this.columnCount = 3;
        this.selectedFolder = "";
    }

    onNodeCreated() {
        this.addWidgets();
        this.updateNodeInfo();
    }

    getNodePacks() {
        // Get all registered nodes
        const nodeTypes = Object.keys(LiteGraph.registered_node_types);

        // Group nodes by their custom node folder
        const nodePacks = new Map();

        nodeTypes.forEach(type => {
            const node = LiteGraph.registered_node_types[type];
            if (node && node.category) {
                const category = node.category;
                if (!nodePacks.has(category)) {
                    nodePacks.set(category, []);
                }
                nodePacks.get(category).push(type);
            }
        });

        return nodePacks;
    }

    addWidgets() {
        // Get available node packs
        const nodePacks = this.getNodePacks();
        const categories = Array.from(nodePacks.keys()).sort();

        // Add folder selector
        this.addWidget(
            "combo",
            "Node Pack",
            categories[0] || "",
            (value) => {
                this.selectedFolder = value;
                this.updateNodeInfo();
            },
            { values: categories }
        );

        this.addWidget(
            "button",
            "Load Nodes",
            "Load",
            (value) => {
                this.loadNodes();
            }
        );

        this.addWidget(
            "number",
            "Columns",
            3,
            (value) => {
                this.columnCount = Math.max(1, Math.floor(Number(value)));
            },
            {
                min: 1,
                max: 10,
                step: 1,
                precision: 0,
                default: 3
            }
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

    async loadNodes() {
        const nodePacks = this.getNodePacks();
        if (!this.selectedFolder || !nodePacks.has(this.selectedFolder)) {
            console.log("No folder selected or invalid folder");
            return;
        }

        const nodeTypes = nodePacks.get(this.selectedFolder).sort();

        if (nodeTypes.length === 0) {
            console.log(`No nodes found in category ${this.selectedFolder}`);
            return;
        }

        // Clear existing nodes from this category
        const existingNodes = app.graph._nodes.filter(node =>
            node.type && nodePacks.get(this.selectedFolder)?.includes(node.type)
        );

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
        const padding = 50;
        let currentX = padding;

        // Position nodes using calculated metrics
        nodeInfo.forEach((info, index) => {
            const columnIndex = Math.floor(index / nodesPerColumn);
            const positionInColumn = index % nodesPerColumn;

            let x = padding;
            for (let i = 0; i < columnIndex; i++) {
                x += columnMetrics[i].maxWidth + padding;
            }

            x += (columnMetrics[columnIndex].maxWidth - info.width) / 2;

            const y = padding + (positionInColumn * (Math.max(...columnMetrics.map(m => m.maxHeight)) + padding));

            info.node.pos = [x, y];
        });

        // Update info widgets
        this.updateNodeInfo();

        // Center view on nodes
        const canvas = app.canvas;
        if (canvas) {
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
        const nodePacks = this.getNodePacks();
        if (!this.selectedFolder) return;

        const nodes = nodePacks.get(this.selectedFolder) || [];

        if (this.widgets) {
            this.widgets[3].value = nodes.length.toString();
            this.widgets[4].value = nodes.join("\n");
        }
    }

    onExecute() {
        this.triggerSlot(0);
    }
}

// Register the node
app.registerExtension({
    name: "Comfy.FL_NodePackLoader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FL_NodePackLoader") {
            for (const key of Object.getOwnPropertyNames(FL_NodePackLoader.prototype)) {
                if (key !== "constructor") {
                    nodeType.prototype[key] = FL_NodePackLoader.prototype[key];
                }
            }
        }
    },
    nodeTypes: {
        FL_NodePackLoader: {
            output: ["trigger"],
            input: ["trigger"],
            name: "Node Pack Folder Loader",
            category: "utils",
            color: "#2a363b"
        }
    }
});