class FL_IntToFloat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "int_value": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float_value",)
    FUNCTION = "convert"
    CATEGORY = "🏵️Fill Nodes/Utility"

    def convert(self, int_value):
        return (float(int_value),)


class FL_FloatToInt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float_value": ("FLOAT", {"default": 0.0}),
                "rounding_mode": (["round", "floor", "ceil", "truncate"], {"default": "round"}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int_value",)
    FUNCTION = "convert"
    CATEGORY = "🏵️Fill Nodes/Utility"

    def convert(self, float_value, rounding_mode):
        import math

        if rounding_mode == "round":
            result = round(float_value)
        elif rounding_mode == "floor":
            result = math.floor(float_value)
        elif rounding_mode == "ceil":
            result = math.ceil(float_value)
        elif rounding_mode == "truncate":
            result = math.trunc(float_value)
        else:
            result = round(float_value)

        return (int(result),)
