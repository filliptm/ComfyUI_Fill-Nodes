class FL_PromptMulti:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_text": ("STRING", {"multiline": True, "default": ""}),
                "negative_text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": "prompt_"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "name")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "process_prompts"
    CATEGORY = "🏵️Fill Nodes/Prompting"

    def process_prompts(self, positive_text, negative_text, name_prefix="prompt_"):
        # Split the input texts by newlines
        positive_prompts = [p.strip() for p in positive_text.split('\n') if p.strip()]
        negative_prompts = [n.strip() for n in negative_text.split('\n') if n.strip()]
        
        # Ensure we have matching numbers of prompts
        # If not, repeat the last one or use empty string
        if len(positive_prompts) < len(negative_prompts):
            if len(positive_prompts) > 0:
                positive_prompts.extend([positive_prompts[-1]] * (len(negative_prompts) - len(positive_prompts)))
            else:
                positive_prompts = [""] * len(negative_prompts)
        elif len(negative_prompts) < len(positive_prompts):
            if len(negative_prompts) > 0:
                negative_prompts.extend([negative_prompts[-1]] * (len(positive_prompts) - len(negative_prompts)))
            else:
                negative_prompts = [""] * len(positive_prompts)
        
        # Create output lists
        positives = []
        negatives = []
        names = []
        
        for i, (pos, neg) in enumerate(zip(positive_prompts, negative_prompts)):
            name = f"{name_prefix}{i+1}"
            positives.append(pos)
            negatives.append(neg)
            names.append(name)
        
        return (positives, negatives, names)