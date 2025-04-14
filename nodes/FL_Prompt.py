class FL_PromptBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prepend": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "append": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "concatenate_prompts"
    CATEGORY = "🏵️Fill Nodes/Prompting"

    def concatenate_prompts(self, prepend, prompt, append):
        # Simply concatenate the strings without adding spaces
        # This preserves any spaces the user has explicitly added
        result = prepend + prompt + append
        
        return (result,)