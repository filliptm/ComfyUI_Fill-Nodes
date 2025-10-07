/**
 * File: fl_code_node.js
 * Project: ComfyUI_Fill-Nodes
 *
 */

import { app } from "../../../../scripts/app.js"
import { node_add_dynamic } from "../widget.js"

const _id = "FL_CodeNode"
const _prefix = 'input'

app.registerExtension({
	name: 'fl.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }
        nodeType = node_add_dynamic(nodeType, _prefix);
	}
})