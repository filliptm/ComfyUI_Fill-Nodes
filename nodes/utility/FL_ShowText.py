class FL_ShowText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "show"
    OUTPUT_NODE = True
    CATEGORY = "🏵️Fill Nodes/Utility"

    def show(self, text, unique_id=None):
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        return {"ui": {"text": [text]}, "result": (text,)}
