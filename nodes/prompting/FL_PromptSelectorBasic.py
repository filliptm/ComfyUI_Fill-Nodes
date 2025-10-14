class FL_PromptSelectorBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {"multiline": True}),
                "index": ("INT", {"default": 0, "min": 0, "max": 6969, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "select_prompt"
    CATEGORY = "🏵️Fill Nodes/Prompting"

    def select_prompt(self, prompts, index):
        prompt_lines = prompts.split("\n")

        num_prompts = len(prompt_lines)

        # Loop through prompts using modulo
        index = index % num_prompts

        selected_prompt = prompt_lines[index].strip()

        return (selected_prompt,)


NODE_CLASS_MAPPINGS = {
    "FL_PromptSelectorBasic": FL_PromptSelectorBasic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_PromptSelectorBasic": "Prompt Selector Basic 🏵️",
}
