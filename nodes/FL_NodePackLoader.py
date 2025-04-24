class FL_NodePackLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trigger": "TRIGGER",
            }
        }

    RETURN_TYPES = ("TRIGGER",)
    FUNCTION = "load_nodes"
    CATEGORY = "🏵️Fill Nodes/Loaders"

    def load_nodes(self, trigger):
        return (trigger,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")  # Ensures the node always processes when triggered
