class AlwaysEqualProxy(str):

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class FL_CodeNode:

    @classmethod
    def INPUT_TYPES(cls):
        required = {"code_input": ("STRING", {"multiline": True})}
        optional = {f"input_{i}": (AlwaysEqualProxy("*")) for i in range(4)}
        return {"required": required, "optional": optional}

    CATEGORY = "🏵️Fill Nodes"
    RETURN_TYPES = tuple(AlwaysEqualProxy("*") for _ in range(4))
    RETURN_NAMES = tuple(f"output_{i}" for i in range(4))

    FUNCTION = "execute"

    def execute(self, code_input, **kwargs):
        outputs = {i: None for i in range(4)}

        try:
            exec(code_input, {"inputs": kwargs, "outputs": outputs})
        except Exception as e:
            raise RuntimeError(f"Error executing user code: {e}")

        return tuple(outputs[i] for i in range(4))

