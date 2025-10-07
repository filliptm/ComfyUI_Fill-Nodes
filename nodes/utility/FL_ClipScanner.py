import open_clip

class FL_ClipScanner:
    def __init__(self):
        self.current_model = None
        self.current_pretrained = None
        self.tokenizer = None
        
        # Model configurations
        self.model_configs = {
            "SDXL (ViT-G/14)": {
                "model": "ViT-g-14",
                "pretrained": "laion2b_s12b_b42k"
            },
            "SD 1.5 (ViT-L/14)": {
                "model": "ViT-L-14",
                "pretrained": "openai"
            },
            "FLUX (ViT-L/14)": {
                "model": "ViT-L-14",
                "pretrained": "openai"
            }
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (list(cls().model_configs.keys()),),
                "text": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze_tokens"
    CATEGORY = "🏵️Fill Nodes/Utility"
    OUTPUT_NODE = True

    def initialize_model(self, model_type):
        if (self.current_model != self.model_configs[model_type]["model"] or
            self.current_pretrained != self.model_configs[model_type]["pretrained"]):
            try:
                config = self.model_configs[model_type]
                self.tokenizer = open_clip.get_tokenizer(config["model"])
                self.current_model = config["model"]
                self.current_pretrained = config["pretrained"]
                return True
            except Exception as e:
                print(f"Error initializing CLIP tokenizer: {str(e)}")
                self.tokenizer = None
                return False
        return True

    def analyze_tokens(self, model_type: str, text: str) -> tuple[str]:
        if not self.initialize_model(model_type):
            return ("Error: Failed to initialize CLIP tokenizer.",)

        try:
            # Tokenize the input text
            tokens = self.tokenizer(text)
            
            # Convert token IDs to words
            decoded_tokens = [self.tokenizer.decoder.get(tok, f"[{tok}]")
                            for tok in tokens[0].tolist()]
            
            # Remove start, end, and padding tokens
            filtered_tokens = [t for t in decoded_tokens
                             if not t.startswith("<") and t != "!"]
            
            # Create formatted output
            output = []
            output.append("=" * 60)
            output.append(f"Model: {model_type}")
            output.append(f"CLIP: {self.current_model} ({self.current_pretrained})")
            output.append(f"Prompt: \"{text}\"")
            output.append(f"Tokenized Output: {filtered_tokens}")
            output.append(f"Token Count: {len(filtered_tokens)}")
            output.append("=" * 60)
            
            return ("\n".join(output),)

        except Exception as e:
            return (f"Error analyzing tokens: {str(e)}",)

    @classmethod
    def IS_CHANGED(cls, model_type, text):
        return float("NaN")