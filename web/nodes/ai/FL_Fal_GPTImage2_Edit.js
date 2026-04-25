import { app } from "../../../../scripts/app.js";

// Dynamic-socket driver for FL_Fal_GPTImage2_Edit.
// Adds/removes `image_ref_N` IMAGE inputs based on the `image_count` widget.
// image_ref_1 is declared in Python INPUT_TYPES so it's always present.

app.registerExtension({
  name: "FillNodes.FL_Fal_GPTImage2_Edit",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "FL_Fal_GPTImage2_Edit") return;

    nodeType.prototype.onNodeCreated = function () {
      this.addWidget("button", "Update inputs", null, () => {
        this._flks_syncRefInputs();
      });

      // Sync once on creation so workflows loaded with a non-default
      // image_count get their sockets rebuilt automatically.
      this._flks_syncRefInputs();
    };

    nodeType.prototype._flks_syncRefInputs = function () {
      if (!this.inputs) this.inputs = [];

      const countWidget = this.widgets.find((w) => w.name === "image_count");
      if (!countWidget) {
        console.error("FL_Fal_GPTImage2_Edit: 'image_count' widget not found");
        return;
      }
      const target = Math.max(1, parseInt(countWidget.value, 10) || 1);

      // Count how many image_ref_N sockets currently exist.
      let current = 0;
      for (const inp of this.inputs) {
        if (inp && typeof inp.name === "string" && inp.name.startsWith("image_ref_")) {
          current++;
        }
      }
      // image_ref_1 is required (declared in INPUT_TYPES). If for any reason
      // current is 0 here, leave Python's required-input wiring in place.
      if (current === 0) current = 1;

      if (target === current) return;

      if (target < current) {
        // Remove from the highest index down, only the dynamic ones (N >= 2).
        for (let n = current; n > target && n >= 2; n--) {
          const name = `image_ref_${n}`;
          const idx = this.inputs.findIndex((i) => i && i.name === name);
          if (idx !== -1) this.removeInput(idx);
        }
      } else {
        for (let n = current + 1; n <= target; n++) {
          this.addInput(`image_ref_${n}`, "IMAGE");
        }
      }

      this.setDirtyCanvas(true, true);
    };
  },
});
