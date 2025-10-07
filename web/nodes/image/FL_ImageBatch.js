import { app } from "../../../../scripts/app.js";

app.registerExtension({
	name: "FillNodes.ImageBatch", // Unique name for this new node's extension
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		// Check if this is the "FL_ImageBatch" node
		if (nodeData.name === "FL_ImageBatch") {
			// This function is called when a new "FL_ImageBatch" node is created
			nodeType.prototype.onNodeCreated = function () {
				this._type = "IMAGE"; // Define the type of input we are managing

				// Add the "Update inputs" button to this node's widget list
				this.addWidget("button", "Update inputs", null, () => {
					if (!this.inputs) {
						this.inputs = [];
					}
					
					// Find the 'inputcount' widget (defined in Python) to get the target number
					const inputCountWidget = this.widgets.find(w => w.name === "inputcount");
					if (!inputCountWidget) {
						console.error("FL_ImageBatch: 'inputcount' widget not found on this node!");
						return;
					}
					const target_number_of_inputs = parseInt(inputCountWidget.value); // Ensure it's a number

					// Count current IMAGE inputs that match our naming convention
					const num_inputs = this.inputs.filter(input => input.type === this._type && input.name.startsWith("image_")).length;

					if (target_number_of_inputs === num_inputs) {
						return; // No change needed
					}

					if (target_number_of_inputs < num_inputs) {
						// Reduce the number of inputs
						const inputs_to_remove = num_inputs - target_number_of_inputs;
						for (let i = 0; i < inputs_to_remove; i++) {
							// Remove the last IMAGE input matching our convention
							for (let j = this.inputs.length - 1; j >= 0; j--) {
								if (this.inputs[j].type === this._type && this.inputs[j].name.startsWith("image_")) {
									this.removeInput(j);
									break; // Exit inner loop once an input is removed
								}
							}
						}
					} else {
						// Increase the number of inputs
						for (let i = num_inputs + 1; i <= target_number_of_inputs; ++i) {
							this.addInput(`image_${i}`, this._type);
						}
					}
					// Refresh the node's appearance
                    this.setDirtyCanvas(true, true);
				});
			};
		}
	},
});