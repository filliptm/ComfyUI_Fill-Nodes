"""Qwen Image Edit with Per-Image Strength Control"""

import node_helpers
import comfy.utils
import math
import torch


class FL_QwenImageEditStrength:
    """
    Enhanced version of TextEncodeQwenImageEditPlus that allows controlling
    the strength/weight of each individual image in the conditioning.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image1_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "image2": ("IMAGE",),
                "image2_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "image3": ("IMAGE",),
                "image3_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode_with_strength"
    CATEGORY = "🏵️Fill Nodes/WIP"

    def encode_with_strength(self, clip, prompt, vae=None,
                            image1=None, image1_strength=1.0,
                            image2=None, image2_strength=1.0,
                            image3=None, image3_strength=1.0):

        ref_latents = []
        ref_strengths = []
        images = [
            (image1, image1_strength),
            (image2, image2_strength),
            (image3, image3_strength)
        ]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, (image, strength) in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)

                # Resize for vision model (384x384)
                total = int(384 * 384)
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))

                # Encode to latents if VAE is provided
                if vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    latent = vae.encode(s.movedim(1, -1)[:, :, :, :3])

                    # Apply strength by scaling the latent
                    if strength != 1.0:
                        latent = latent * strength

                    ref_latents.append(latent)
                    ref_strengths.append(strength)

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        # Tokenize and encode
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        # Add reference latents with applied strengths
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": ref_latents},
                append=True
            )

        return (conditioning,)


NODE_CLASS_MAPPINGS = {
    "FL_QwenImageEditStrength": FL_QwenImageEditStrength,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_QwenImageEditStrength": "Qwen Image Edit with Strength 🏵️",
}
