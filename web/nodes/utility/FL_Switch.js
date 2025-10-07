/**
 * File: FL_Switch.js
 * Project: ComfyUI_Fill-Nodes
 * 
 * A switch node that can choose between two processing paths based on a switch value
 */

import { app } from "../../../../scripts/app.js"

const _id = "FL_Switch"

app.registerExtension({
    name: 'fl.node.' + _id,
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }
        
        // Override the getExtraMenuOptions to add a "Toggle Switch" option
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            if (getExtraMenuOptions) {
                getExtraMenuOptions.apply(this, arguments);
            }
            
            options.push({
                content: "Toggle Switch",
                callback: () => {
                    // Get the current switch value
                    const switchWidget = this.widgets.find(w => w.name === "switch");
                    if (switchWidget) {
                        // Toggle between true and false
                        switchWidget.value = !switchWidget.value;
                        this.setDirtyCanvas(true, true);
                    }
                }
            });
            
            return options;
        };
    }
})