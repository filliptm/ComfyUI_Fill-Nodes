class FL_PromptSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prepend_text": ("STRING", {"multiline": True, "default": ""}),
                "prompts": ("STRING", {"multiline": True}),
                "append_text": ("STRING", {"multiline": True, "default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "select_prompt"
    CATEGORY = "🏵️Fill Nodes"

    def select_prompt(self, prepend_text, prompts, append_text, index):
        prompt_lines = prompts.split("\n")
        num_prompts = len(prompt_lines)

        # will wrap around 0--Number of items
        index %= num_prompts

        # will clamp 0--Number of items
        # index = max(0, min(len(prompt_lines)-1, index))

        selected_prompt = prompt_lines[index].strip()
        selected_prompt = f"{prepend_text.strip()} {selected_prompt} {append_text.strip()}"

        return (selected_prompt,)