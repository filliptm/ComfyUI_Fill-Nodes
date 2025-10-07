import os
import pathlib

class FL_PathTypeChecker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("PATH",)
    FUNCTION = "check_path_type"
    CATEGORY = "🏵️Fill Nodes/Utility"

    def check_path_type(self, input_path):
        input_path = input_path.strip()  # Remove leading/trailing whitespace

        if not input_path:
            return ("Empty path provided.",)

        path = pathlib.Path(input_path)

        if path.is_absolute():
            return ("Absolute path",)
        elif path.is_relative_to(pathlib.Path.cwd()):
            return ("Relative path",)
        else:
            # Check if it might be a valid relative path
            try:
                path.relative_to(".")
                return ("Relative path",)
            except ValueError:
                pass

        # If it's not recognized as absolute or relative, it might be invalid or a special case
        if os.path.splitdrive(input_path)[0]:
            return ("Drive-specific path",)
        elif input_path.startswith('//') or input_path.startswith('\\\\'):
            return ("UNC path",)
        elif '://' in input_path:
            return ("URL-like path",)
        else:
            return ("Unrecognized or invalid path",)