class FL_NodeLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"trigger": "TRIGGER"}}

    RETURN_TYPES = ("TRIGGER",)
    FUNCTION = "load_nodes"
    CATEGORY = "🏵️Fill Nodes/Loaders"

    def load_nodes(self, trigger):
        return (trigger,)