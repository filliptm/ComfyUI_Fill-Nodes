"""Qwen Image Edit with Per-Image Strength Control"""

import node_helpers
import comfy.utils
import math
import torch


class FL_QwenImageEditStrength:
    """
    Enhanced version of TextEncodeQwenImageEditPlus that allows controlling
    the strength/weight of each individual image in the conditioning.

    Features:
    - Per-image strength control (range: -10.0 to 10.0)
    - Multiple interpolation methods for combining image latents
    - Flexible blending options for creative control
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "interpolation_method": (["weighted_sum", "average", "maximum", "minimum", "multiply"], {"default": "weighted_sum"}),
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

    def encode_with_strength(self, clip, prompt, interpolation_method, vae=None,
                            image1=None, image1_strength=1.0,
                            image2=None, image2_strength=1.0,
                            image3=None, image3_strength=1.0):

        individual_latents = []
        strengths = []
        images = [
            (image1, image1_strength),
            (image2, image2_strength),
            (image3, image3_strength)
        ]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""
        image_dimensions = None  # Track first image dimensions for validation

        for i, (image, strength) in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)

                # Validate that all images have the same dimensions when using VAE
                if vae is not None:
                    current_dims = (samples.shape[2], samples.shape[3])  # (height, width)
                    if image_dimensions is None:
                        image_dimensions = current_dims
                    elif current_dims != image_dimensions:
                        raise ValueError(
                            f"Image size mismatch: Image {i+1} has dimensions {current_dims[0]}x{current_dims[1]}, "
                            f"but the first image has dimensions {image_dimensions[0]}x{image_dimensions[1]}. "
                            f"All images must have the same dimensions when using VAE encoding. "
                            f"Please resize your images to match before connecting them to this node."
                        )

                # Resize for vision model (384x384)
                total = int(384 * 384)
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))

                # Encode to latents if VAE is provided (using original dimensions)
                if vae is not None:
                    latent = vae.encode(samples.movedim(1, -1)[:, :, :, :3])

                    individual_latents.append(latent)
                    strengths.append(strength)

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        # Tokenize and encode
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        # Combine latents using the selected interpolation method
        if len(individual_latents) > 0:
            combined_latents = self._interpolate_latents(individual_latents, strengths, interpolation_method)
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": combined_latents},
                append=True
            )

        return (conditioning,)

    def _interpolate_latents(self, latents, strengths, method):
        """
        Combine multiple latents using the specified interpolation method.

        Args:
            latents: List of latent tensors
            strengths: List of strength values for each latent
            method: Interpolation method to use

        Returns:
            List of combined latents (or original list for weighted_sum)
        """
        if len(latents) == 1:
            # Single latent, just apply strength
            return [latents[0] * strengths[0]]

        if method == "weighted_sum":
            # Keep latents separate but apply individual strengths
            # This is the original behavior
            return [latent * strength for latent, strength in zip(latents, strengths)]

        elif method == "average":
            # Average all latents, then apply combined strength
            stacked = torch.stack(latents, dim=0)
            averaged = torch.mean(stacked, dim=0)
            # Apply average strength
            avg_strength = sum(strengths) / len(strengths)
            return [averaged * avg_strength]

        elif method == "maximum":
            # Take maximum values across all latents
            stacked = torch.stack([latent * strength for latent, strength in zip(latents, strengths)], dim=0)
            maximum, _ = torch.max(stacked, dim=0)
            return [maximum]

        elif method == "minimum":
            # Take minimum values across all latents
            stacked = torch.stack([latent * strength for latent, strength in zip(latents, strengths)], dim=0)
            minimum, _ = torch.min(stacked, dim=0)
            return [minimum]

        elif method == "multiply":
            # Multiply all latents together (blend mode)
            result = latents[0] * strengths[0]
            for latent, strength in zip(latents[1:], strengths[1:]):
                result = result * (latent * strength)
            return [result]

        else:
            # Default to weighted_sum
            return [latent * strength for latent, strength in zip(latents, strengths)]


NODE_CLASS_MAPPINGS = {
    "FL_QwenImageEditStrength": FL_QwenImageEditStrength,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_QwenImageEditStrength": "Qwen Image Edit with Strength 🏵️",
}
