import { app } from "../../../../scripts/app.js";

app.registerExtension({
	name: "FillNodes.GPTImage1ADV", // Unique name for the extension
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		// Check if this is the correct node we want to modify
		if (nodeData.name === "FL_GPT_Image1_ADV") {
			// This function is called when a new node of this type is created
			nodeType.prototype.onNodeCreated = function () {
				this._image_type = "IMAGE"; // For gpt-image-1 edits/variations
				this._prompt_type = "STRING";

				// Add the "Update inputs" button to this node's widget list
				this.addWidget("button", "Update inputs", null, () => {
					if (!this.inputs) {
						this.inputs = [];
					}
					
					const inputCountWidget = this.widgets.find(w => w.name === "inputcount");
					if (!inputCountWidget) {
						console.error("FL_GPT_Image1_ADV: 'inputcount' widget not found on this node!");
						return;
					}
					const target_prompts = parseInt(inputCountWidget.value);

                    // Current number of prompt inputs (prompt_1 is required and always present)
                    // Count additional prompts starting from prompt_2
                    let current_prompts = 1; // Start with 1 for the required prompt_1
                    for(let i = 0; i < this.inputs.length; i++) {
                        if (this.inputs[i].name === `prompt_${current_prompts + 1}`) {
                            current_prompts++;
                        }
                    }
                    
					if (target_prompts === current_prompts) {
						return; // No change needed
					}

					if (target_prompts < current_prompts) {
						// Reduce the number of prompt inputs
						const prompts_to_remove = current_prompts - target_prompts;
						for (let i = 0; i < prompts_to_remove; i++) {
                            const prompt_num_to_remove = current_prompts - i;
                            let last_prompt_index = -1;
                            // Optional: also remove corresponding image_X if it exists for this prompt number
                            let last_image_index = -1; 

                            for (let j = this.inputs.length - 1; j >= 0; j--) {
                                if (this.inputs[j].name === `prompt_${prompt_num_to_remove}`) {
                                    last_prompt_index = j;
                                } else if (this.inputs[j].name === `image_${prompt_num_to_remove}`) {
                                    last_image_index = j;
                                }
                            }
                            // Remove prompt first, then image if it was before prompt
                            if (last_prompt_index !== -1) this.removeInput(last_prompt_index);
                            
                            // Re-find image index if prompt was removed and shifted indices
                            if (last_image_index !== -1) {
                                let current_last_image_idx = -1;
                                for (let k=this.inputs.length -1; k>=0; k--) {
                                    if (this.inputs[k].name === `image_${prompt_num_to_remove}`) {
                                        current_last_image_idx = k;
                                        break;
                                    }
                                }
                                if (current_last_image_idx !== -1) this.removeInput(current_last_image_idx);
                            }
						}
					} else {
						// Increase the number of prompt inputs
                        // Start from current_prompts + 1 because prompt_1 up to prompt_{current_prompts} exist
						for (let i = current_prompts + 1; i <= target_prompts; ++i) {
						                      // For gpt-image-1, image inputs are optional and typically for edits/variations.
						                      // This ADV node supports adding an optional image and a required prompt per slot.
						                      // A mask input could also be added here if desired for each slot.
						                      this.addInput(`image_${i}`, this._image_type, { label: `image_${i} (opt)` }); // Optional image
						                      this.addInput(`prompt_${i}`, this._prompt_type, { multiline: true, default: `Describe image ${i}` });
						}
					}
					// Refresh the node's appearance
                    this.setDirtyCanvas(true, true);
				});

                // Initial call to sync inputs if loaded from workflow with different inputcount
                // Ensure widgets are available before calling
                if (this.widgets && this.widgets.find(w => w.name === "inputcount")) {
                    this.widgets.find(w => w.name === "Update inputs").callback();
                }
			};
		}
	},
});