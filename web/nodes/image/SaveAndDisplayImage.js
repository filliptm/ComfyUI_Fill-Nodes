import { app } from "../../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.SaveAndDisplayImage",
    async nodeCreated(node) {
        if (node.comfyClass === "SaveAndDisplayImage") {
            node.addWidget("image", "preview", "");
        }
    }
});