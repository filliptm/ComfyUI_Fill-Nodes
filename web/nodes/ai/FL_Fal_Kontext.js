import { app } from "../../../../scripts/app.js";

app.registerExtension({
	name: "FillNodes.FL_Fal_Kontext", // Unique name for the extension
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		// Check if this is the correct node we want to modify
		if (nodeData.name === "FL_Fal_Kontext") {
			// This function is called when a new node of this type is created
			nodeType.prototype.onNodeCreated = function () {
				this._image_type = "IMAGE";
				this._prompt_type = "STRING";

				// Add the "Update inputs" button to this node's widget list
				this.addWidget("button", "Update inputs", null, () => {
					if (!this.inputs) {
						this.inputs = [];
					}
					
					const inputCountWidget = this.widgets.find(w => w.name === "inputcount");
					if (!inputCountWidget) {
						console.error("FL_Fal_Kontext: 'inputcount' widget not found on this node!");
						return;
					}
					const target_pairs = parseInt(inputCountWidget.value);

					// Current number of *pairs* (image_1/prompt_1 is one pair)
			                 // image_1 and prompt_1 are required and always present.
			                 // So, count additional pairs starting from image_2/prompt_2
			                 let current_pairs = 1; // Start with 1 for the required image_1/prompt_1
			                 for(let i = 0; i < this.inputs.length; i++) {
			                     if (this.inputs[i].name === `image_${current_pairs + 1}`) {
			                         current_pairs++;
			                     }
			                 }
			                 
					if (target_pairs === current_pairs) {
						return; // No change needed
					}

					if (target_pairs < current_pairs) {
						// Reduce the number of pairs
						const pairs_to_remove = current_pairs - target_pairs;
						for (let i = 0; i < pairs_to_remove; i++) {
			                         // Remove the last prompt and then the last image input of the highest index pair
			                         let last_prompt_index = -1;
			                         let last_image_index = -1;
			                         const pair_num_to_remove = current_pairs - i;

			                         for (let j = this.inputs.length - 1; j >= 0; j--) {
			                             if (this.inputs[j].name === `prompt_${pair_num_to_remove}`) {
			                                 last_prompt_index = j;
			                             } else if (this.inputs[j].name === `image_${pair_num_to_remove}`) {
			                                 last_image_index = j;
			                             }
			                         }
			                         if (last_prompt_index !== -1) this.removeInput(last_prompt_index);
			                         if (last_image_index !== -1 && last_image_index < this.inputs.length) { // Check if index is still valid after prompt removal
			                              // Need to re-check index if prompt was before image and got removed
			                              let current_last_image_index = -1;
			                              for (let k = this.inputs.length - 1; k >=0; k--) {
			                                  if (this.inputs[k].name === `image_${pair_num_to_remove}`) {
			                                      current_last_image_index = k;
			                                      break;
			                                  }
			                              }
			                              if(current_last_image_index !== -1) this.removeInput(current_last_image_index);
			                         } else if (last_image_index !== -1) { // If prompt was after image or not found
			                             this.removeInput(last_image_index);
			                         }
						}
					} else {
						// Increase the number of pairs
						                  // Start from current_pairs + 1 because image_1/prompt_1 up to image_{current_pairs}/prompt_{current_pairs} exist
						for (let i = current_pairs + 1; i <= target_pairs; ++i) {
							this.addInput(`image_${i}`, this._image_type);
						                      this.addInput(`prompt_${i}`, this._prompt_type, { multiline: true, default: `prompt for image ${i}` });
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