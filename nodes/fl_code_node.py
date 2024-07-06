
from pathlib import Path
from .sup import ROOT, AlwaysEqualProxy, parse_dynamic

class FL_CodeNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "code_input": ("STRING", {"default": "outputs[0] = 'hello, world!'", "multiline": True, "dynamicPrompts": False}),
                "file": ("STRING", {"default": "./res/hello.py", "multiline": False, "dynamicPrompts": False}),
                "use_file": ("BOOLEAN", {"default": False})
        }}

    CATEGORY = "🏵️Fill Nodes"
    RETURN_TYPES = tuple(AlwaysEqualProxy("*") for _ in range(4))
    RETURN_NAMES = tuple(f"output_{i}" for i in range(4))
    DESCRIPTION = """
FL_CodeNode is designed to execute custom user-provided Python code. The code can be directly entered as a string input or loaded from a specified file. This class processes dynamic inputs and provides four generic output slots. The execution environment includes predefined 'inputs' and 'outputs' dictionaries to facilitate interaction with the code. Proper error handling is included to ensure informative feedback in case of execution failures. This node is ideal for users needing to integrate custom logic or algorithms into their workflows.
"""
    FUNCTION = "execute"

    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("nan")

    def execute(self, code_input, file, use_file, **kwargs):
        outputs = {i: None for i in range(4)}
        inputs = kwargs.copy()
        inputs.update({i: v for i, v in enumerate(kwargs.values())})
        if use_file:
            # load the referenced file
            code_input = ""
            if not (fname := Path(ROOT / file)).is_file():
                if not (fname := Path(file)).is_file():
                    fname = None
            if fname is not None:
                try:
                    with open(str(fname), 'r') as f:
                        code_input = f.read()
                except Exception as e:
                    raise RuntimeError(f"[FL_CodeNode] error loading code file: {e}")

        # sanitize?
        # code_input = code_input

        try:
            exec(code_input, {"inputs": inputs, "outputs": outputs})
        except Exception as e:
            raise RuntimeError(f"Error executing user code: {e}")

        return tuple(outputs[i] for i in range(4))
