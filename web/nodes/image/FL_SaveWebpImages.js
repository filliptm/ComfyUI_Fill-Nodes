import { app } from "../../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.SaveWebPImage",
    async nodeCreated(node) {
        if (node.comfyClass === "SaveWebPImage") {
            node.addWidget("image", "preview", "");

            // Add a text widget to display the save status
            const statusWidget = node.addWidget("text", "status", "");
            node.onExecuted = function(output) {
                if (output && output.ui && output.ui.images && output.ui.images.length > 0) {
                    // Update the preview with the first saved image
                    node.widgets.find(w => w.name === "preview").value = output.ui.images[0].filename;
                }
                // Update the status text
                statusWidget.value = output.result[1];  // This is the message we returned
            }
        }
    }
});