/**
 * Support for widgets in the UI
*/

const TypeSlot = {
    Input: 1,
    Output: 2,
};

const TypeSlotEvent = {
    Connect: true,
    Disconnect: false,
};

/**
 * Manage the slots on a node to allow a dynamic number of inputs
*/
export function node_add_dynamic(nodeType, prefix, dynamic_type='*', index_start=0, shape=LiteGraph.GRID_SHAPE, match_output=false) {
    /*
    this one should just put the "prefix" as the last empty entry.
    Means we have to pay attention not to collide key names in the
    input list.
    */
    index_start = Math.max(0, index_start);
    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
        const me = onNodeCreated?.apply(this);
        this.addInput(prefix, dynamic_type);
        if (match_output) {
		    this.addOutput(prefix, dynamic_type, { shape: shape });
        }
        return me;
    }

    const onConnectionsChange = nodeType.prototype.onConnectionsChange
    nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
        const me = onConnectionsChange?.apply(this, arguments)
        if (slotType === TypeSlot.Input) {
            if (slot_idx >= index_start && link_info) {
                if (event === TypeSlotEvent.Connect) {
                    const fromNode = this.graph._nodes.find(
                        (otherNode) => otherNode.id == link_info.origin_id
                    )
                    if (fromNode) {
                        const parent_link = fromNode.outputs[link_info.origin_slot];
                        node_slot.type = parent_link.type;
                        node_slot.name = `${slot_idx}_${parent_link.name.toLowerCase()}`;
                        if (match_output) {
                            const slot_out = this.outputs[slot_idx];
                            slot_out.type = parent_link.type;
                            slot_out.name = `${slot_idx}_${parent_link.type}`;
                        }
                    }
                }
            }
            // check that the last slot is a dynamic entry....
            let last = this.inputs[this.inputs.length-1];
            if (last.type != dynamic_type || last.name != prefix) {
                this.addInput(prefix, dynamic_type);
            }
            if (match_output) {
                last = this.outputs[this.outputs.length-1];
                if (last.type != dynamic_type || last.name != prefix) {
                    this.addOutput(prefix, dynamic_type);
                }
            }
        }

        setTimeout(() => {
            // clean off missing slot connects
            if (this.graph === undefined) {
                return;
            }
            let idx = index_start;
            while (idx < this.inputs.length-1) {
                const slot = this.inputs[idx];
                if (slot.link == null) {
                    if (match_output) {
                        this.removeOutput(slot_idx);
                    }
                    this.removeInput(slot_idx);
                } else {
                    idx += 1;
                }
            }
         }, 25);

        return me;
    }
    return nodeType;
}