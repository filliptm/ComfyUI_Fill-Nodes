import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Fill-Nodes.appearance", // Extension name
    nodeCreated(node) {
        // Check if the node's comfyClass starts with "FL_"
        if (node.comfyClass.startsWith("FL_")) {
            // Apply styling
            node.color = "#16727c";
            node.bgcolor = "#4F0074";

            // Uncomment the following line if you want to set a specific size for all FL nodes
            // node.setSize([200, 58]);
        }
    }
});
