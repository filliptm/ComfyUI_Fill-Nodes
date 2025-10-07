/**
 * File: FL_Switch_Big.js
 * Project: ComfyUI_Fill-Nodes
 * 
 * A switch-case node that can choose between multiple processing paths based on a condition
 */

import { app } from "../../../../scripts/app.js"

const _id = "FL_Switch_Big"

app.registerExtension({
    name: 'fl.node.' + _id,
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }
        
        // Override the getExtraMenuOptions to add quick case selection options
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            if (getExtraMenuOptions) {
                getExtraMenuOptions.apply(this, arguments);
            }
            
            // Get the current switch condition and cases
            const switchWidget = this.widgets.find(w => w.name === "switch_condition");
            const case1Widget = this.widgets.find(w => w.name === "case_1");
            const case2Widget = this.widgets.find(w => w.name === "case_2");
            const case3Widget = this.widgets.find(w => w.name === "case_3");
            const case4Widget = this.widgets.find(w => w.name === "case_4");
            const case5Widget = this.widgets.find(w => w.name === "case_5");
            
            if (switchWidget && case1Widget && case2Widget && case3Widget && case4Widget && case5Widget) {
                // Add menu options to quickly set the switch condition to each case
                options.push({
                    content: "Set to Case 1",
                    callback: () => {
                        switchWidget.value = case1Widget.value;
                        this.setDirtyCanvas(true, true);
                    }
                });
                
                options.push({
                    content: "Set to Case 2",
                    callback: () => {
                        switchWidget.value = case2Widget.value;
                        this.setDirtyCanvas(true, true);
                    }
                });
                
                options.push({
                    content: "Set to Case 3",
                    callback: () => {
                        switchWidget.value = case3Widget.value;
                        this.setDirtyCanvas(true, true);
                    }
                });
                
                options.push({
                    content: "Set to Case 4",
                    callback: () => {
                        switchWidget.value = case4Widget.value;
                        this.setDirtyCanvas(true, true);
                    }
                });
                
                options.push({
                    content: "Set to Case 5",
                    callback: () => {
                        switchWidget.value = case5Widget.value;
                        this.setDirtyCanvas(true, true);
                    }
                });
                
                // Add a separator
                options.push(null);
                
                // Add option to clear the switch condition
                options.push({
                    content: "Clear Switch Condition",
                    callback: () => {
                        switchWidget.value = "";
                        this.setDirtyCanvas(true, true);
                    }
                });
            }
            
            return options;
        };
    }
})