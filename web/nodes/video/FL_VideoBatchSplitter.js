/**
 * File: FL_VideoBatchSplitter.js
 * Project: ComfyUI_Fill-Nodes
 * 
 * A video batch splitter that dynamically creates outputs based on output_count
 */

import { app } from "../../../../scripts/app.js";

app.registerExtension({
	name: "FillNodes.VideoBatchSplitter", // Unique name for the extension
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		// Check if this is the correct node we want to modify
		if (nodeData.name === "FL_VideoBatchSplitter") {
			// This function is called when a new node of this type is created
			nodeType.prototype.onNodeCreated = function () {
				this._output_type = "IMAGE";

				// Add the "Update outputs" button to this node's widget list
				this.addWidget("button", "Update outputs", null, () => {
					if (!this.outputs) {
						this.outputs = [];
					}
					
					const outputCountWidget = this.widgets.find(w => w.name === "output_count");
					if (!outputCountWidget) {
						console.error("FL_VideoBatchSplitter: 'output_count' widget not found on this node!");
						return;
					}
					const target_outputs = parseInt(outputCountWidget.value);

					// Current number of outputs
					let current_outputs = this.outputs ? this.outputs.length : 0;
					
					if (target_outputs === current_outputs) {
						return; // No change needed
					}

					if (target_outputs < current_outputs) {
						// Reduce the number of outputs
						const outputs_to_remove = current_outputs - target_outputs;
						for (let i = 0; i < outputs_to_remove; i++) {
							// Remove the last output
							let last_output_index = -1;
							const output_num_to_remove = current_outputs - i;

							for (let j = this.outputs.length - 1; j >= 0; j--) {
								if (this.outputs[j].name === `batch_${output_num_to_remove}`) {
									last_output_index = j;
									break;
								}
							}
							if (last_output_index !== -1) this.removeOutput(last_output_index);
						}
					} else {
						// Increase the number of outputs
						// Start from current_outputs + 1 because batch_1 up to batch_{current_outputs} exist
						for (let i = current_outputs + 1; i <= target_outputs; ++i) {
							this.addOutput(`batch_${i}`, this._output_type);
						}
					}
					// Refresh the node's appearance
					this.setDirtyCanvas(true, true);
				});

				// Initial call to sync outputs if loaded from workflow with different output_count
				// Ensure widgets are available before calling
				if (this.widgets && this.widgets.find(w => w.name === "output_count")) {
					this.widgets.find(w => w.name === "Update outputs").callback();
				}
			};

			// Override getExtraMenuOptions to add validation helper
			const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
			nodeType.prototype.getExtraMenuOptions = function(_, options) {
				if (getExtraMenuOptions) {
					getExtraMenuOptions.apply(this, arguments);
				}
				
				const framesPerBatchWidget = this.widgets.find(w => w.name === "frames_per_batch");
				const outputCountWidget = this.widgets.find(w => w.name === "output_count");
				
				if (framesPerBatchWidget && outputCountWidget) {
					options.push({
						content: "🔄 Refresh Outputs",
						callback: () => {
							const updateButton = this.widgets.find(w => w.name === "Update outputs");
							if (updateButton) {
								updateButton.callback();
							}
						}
					});
					
					options.push({
						content: "📊 Calculate Total Frames Needed",
						callback: () => {
							const framesPerBatch = framesPerBatchWidget.value;
							const outputCount = outputCountWidget.value;
							const totalFrames = framesPerBatch * outputCount;
							
							app.ui.dialog.show(`
								<div style="padding: 20px;">
									<h3>Frame Calculation</h3>
									<p><strong>Frames per batch:</strong> ${framesPerBatch}</p>
									<p><strong>Number of outputs:</strong> ${outputCount}</p>
									<p><strong>Total frames needed:</strong> ${totalFrames}</p>
									<p style="margin-top: 15px; color: #888;">
										Your input video must have exactly ${totalFrames} frames 
										to split evenly into ${outputCount} batches of ${framesPerBatch} frames each.
									</p>
								</div>
							`);
						}
					});
					
					options.push(null); // separator
				}
				
				return options;
			};
		}
	},
});