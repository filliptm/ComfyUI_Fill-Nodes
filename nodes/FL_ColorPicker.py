class FL_ColorPicker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "selected_color": ("STRING", {"default": "#FF0000"})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_color"
    CATEGORY = "ui"

    def get_color(self, selected_color):
        return (selected_color,)