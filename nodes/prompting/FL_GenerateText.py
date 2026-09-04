QWEN_CHAT_START = "<|im_start|>"
QWEN_CHAT_END = "<|im_end|>"
QWEN_NO_THINK = "<think>\n\n</think>\n\n"


def format_qwen_chat(system_prompt, prompt, thinking=False):
    chat = (
        f"{QWEN_CHAT_START}system\n{system_prompt}{QWEN_CHAT_END}\n"
        f"{QWEN_CHAT_START}user\n{prompt}{QWEN_CHAT_END}\n"
        f"{QWEN_CHAT_START}assistant\n"
    )
    if not thinking:
        chat += QWEN_NO_THINK
    return chat


class FL_GenerateText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "A complete Qwen3 or Qwen3-VL text encoder with generation weights."}),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": True,
                }),
                "max_length": ("INT", {"default": 512, "min": 1, "max": 32768}),
                "sampling": (["on", "off"], {"default": "on"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 2.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 1000}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "thinking": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "🏵️Fill Nodes/Prompting"
    DESCRIPTION = "Generate text locally with a complete Qwen3 or Qwen3-VL text encoder. H3's truncated conditioning encoder cannot generate text."

    def generate(self, clip, system_prompt, prompt, max_length, sampling, temperature, top_k, top_p, min_p, repetition_penalty, presence_penalty, seed, thinking):
        chat = format_qwen_chat(system_prompt, prompt, thinking=thinking)
        tokens = clip.tokenize(chat, skip_template=True, min_length=1)
        generated_ids = clip.generate(
            tokens,
            do_sample=sampling == "on",
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
        )
        generated_text = clip.decode(generated_ids)
        return {"ui": {"generated_text": [generated_text]}, "result": (generated_text,)}
