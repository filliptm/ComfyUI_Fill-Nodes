import folder_paths


class FL_StringToLoraName:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = (folder_paths.get_filename_list("loras"),)
    RETURN_NAMES = ("lora_name",)
    FUNCTION = "execute"
    CATEGORY = "🏵️Fill Nodes/Utility"

    def execute(self, lora_name):
        return (lora_name,)
