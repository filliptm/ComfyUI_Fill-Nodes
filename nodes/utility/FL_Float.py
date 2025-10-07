class FL_Float:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "output_float"
    CATEGORY = "🏵️Fill Nodes/Utility"

    def output_float(self, value):
        # Round to 2 decimal places to ensure consistent display
        value = round(value, 2)
        return (value,)