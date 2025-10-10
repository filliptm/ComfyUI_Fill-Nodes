import torch
import math
from nodes import common_ksampler, VAEDecode, VAEEncode
import comfy.samplers
import comfy.utils
import logging
import numpy as np
from PIL import Image
import torch.nn.functional as F
import latent_preview


class FL_KsamplerPlus:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "input_type": (["latent", "image"],),
                "x_slices": ("INT", {"default": 2, "min": 1, "max": 8}),
                "y_slices": ("INT", {"default": 2, "min": 1, "max": 8}),
                "overlap": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.9, "step": 0.01}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "use_sliced_conditioning": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "latent_image": ("LATENT",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "IMAGE")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "vae", "image")
    FUNCTION = "sample"
    CATEGORY = "🏵️Fill Nodes/Ksamplers"

    @staticmethod
    def crop_tensor(tensor, region):
        x1, y1, x2, y2 = region
        return tensor[:, :, y1:y2, x1:x2]

    @staticmethod
    def resize_tensor(tensor, size, mode="nearest-exact"):
        return F.interpolate(tensor, size=size, mode=mode)

    @staticmethod
    def resize_region(region, init_size, resize_size):
        x1, y1, x2, y2 = region
        init_width, init_height = init_size
        resize_width, resize_height = resize_size
        x1 = math.floor(x1 * resize_width / init_width)
        x2 = math.ceil(x2 * resize_width / init_width)
        y1 = math.floor(y1 * resize_height / init_height)
        y2 = math.ceil(y2 * resize_height / init_height)
        return (x1, y1, x2, y2)

    @classmethod
    def crop_controlnet(cls, cond_dict, region, init_size, canvas_size, tile_size):
        if "control" not in cond_dict:
            return
        c = cond_dict["control"]
        controlnet = c.copy()
        cond_dict["control"] = controlnet
        while c is not None:
            hint = controlnet.cond_hint_original
            resized_crop = cls.resize_region(region, canvas_size, hint.shape[2:])
            hint = cls.crop_tensor(hint, resized_crop)
            hint = cls.resize_tensor(hint, tile_size)
            controlnet.cond_hint_original = hint
            c = c.previous_controlnet
            controlnet.set_previous_controlnet(c.copy() if c is not None else None)
            controlnet = controlnet.previous_controlnet

    @classmethod
    def crop_cond(cls, cond, region, init_size, canvas_size, tile_size):
        cropped = []
        for emb, x in cond:
            cond_dict = x.copy()
            cls.crop_controlnet(cond_dict, region, init_size, canvas_size, tile_size)
            cropped.append([emb, cond_dict])
        return cropped

    @staticmethod
    def adjust_conditioning_strength(cond, strength_factor):
        return [(emb * strength_factor, x) for emb, x in cond]

    @staticmethod
    def create_blend_mask(height, width, overlap_h, overlap_w, is_top, is_left, is_bottom, is_right, device):
        mask = torch.ones((height, width), device=device)

        if overlap_h > 0:
            x = torch.linspace(0, math.pi, overlap_h, device=device)
            cos = 0.5 * (1 - torch.cos(x))
            if not is_top:
                mask[:overlap_h, :] *= cos[:, None]
            if not is_bottom:
                mask[-overlap_h:, :] *= cos.flip(0)[:, None]
        if overlap_w > 0:
            x = torch.linspace(0, math.pi, overlap_w, device=device)
            cos = 0.5 * (1 - torch.cos(x))
            if not is_left:
                mask[:, :overlap_w] *= cos[None, :]
            if not is_right:
                mask[:, -overlap_w:] *= cos.flip(0)[None, :]

        return mask

    def sample(self, model, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise, input_type,
               x_slices, y_slices, overlap, batch_size, use_sliced_conditioning, latent_image=None, image=None,
               vae=None):
        try:
            device = comfy.model_management.get_torch_device()

            if input_type == "image" and image is not None and vae is not None:
                latent_image = VAEEncode().encode(vae, image)[0]
            elif input_type == "latent" and latent_image is None:
                raise ValueError("Latent image is required when input type is set to latent")
            elif input_type == "image" and (image is None or vae is None):
                raise ValueError("Both image and VAE are required when input type is set to image")

            # Force batch size to 1 when sliced conditioning is used
            if use_sliced_conditioning:
                batch_size = 1

            # Handle variable tensor dimensions (4D or 5D)
            latent_shape = latent_image["samples"].shape
            if len(latent_shape) == 5:
                # 5D tensor: [batch, frames, channels, height, width]
                b, f, c, h, w = latent_shape
                logging.info(f"Processing 5D latent tensor with shape: {latent_shape}")
            elif len(latent_shape) == 4:
                # 4D tensor: [batch, channels, height, width]
                b, c, h, w = latent_shape
            else:
                raise ValueError(f"Unexpected latent tensor shape: {latent_shape}. Expected 4D or 5D tensor.")

            base_slice_height = h // y_slices
            base_slice_width = w // x_slices
            overlap_height = int(base_slice_height * overlap)
            overlap_width = int(base_slice_width * overlap)

            samples = None  # We'll initialize this later when we know the correct number of channels

            def process_slice(y, x):
                y_start = max(0, y * base_slice_height - overlap_height)
                y_end = min(h, (y + 1) * base_slice_height + overlap_height)
                x_start = max(0, x * base_slice_width - overlap_width)
                x_end = min(w, (x + 1) * base_slice_width + overlap_width)

                # Handle both 4D and 5D tensor slicing
                if len(latent_shape) == 5:
                    section = latent_image["samples"][:, :, :, y_start:y_end, x_start:x_end].to(device=device)
                else:
                    section = latent_image["samples"][:, :, y_start:y_end, x_start:x_end].to(device=device)

                if use_sliced_conditioning:
                    region = (x_start * 8, y_start * 8, x_end * 8, y_end * 8)
                    init_size = (w * 8, h * 8)
                    canvas_size = init_size
                    tile_size = ((x_end - x_start) * 8, (y_end - y_start) * 8)
                    strength_factor = 1.2  # Adjust this value to increase or decrease conditioning strength
                    cropped_positive = self.adjust_conditioning_strength(
                        self.crop_cond(positive, region, init_size, canvas_size, tile_size), strength_factor)
                    cropped_negative = self.adjust_conditioning_strength(
                        self.crop_cond(negative, region, init_size, canvas_size, tile_size), strength_factor)
                else:
                    cropped_positive = positive
                    cropped_negative = negative

                return section, y_start, y_end, x_start, x_end, cropped_positive, cropped_negative

            total_slices = x_slices * y_slices
            all_slices = [(y, x) for y in range(y_slices) for x in range(x_slices)]

            for i in range(0, total_slices, batch_size):
                batch_slices = all_slices[i:min(i + batch_size, total_slices)]
                batch_sections = [process_slice(y, x) for y, x in batch_slices]

                batch_latents = torch.cat([section for section, _, _, _, _, _, _ in batch_sections], dim=0)

                if use_sliced_conditioning:
                    batch_positive = batch_sections[0][
                        5]  # Since batch_size is 1, we can directly use the first (and only) element
                    batch_negative = batch_sections[0][6]
                else:
                    batch_positive = positive * len(batch_sections)
                    batch_negative = negative * len(batch_sections)

                processed_batch = common_ksampler(model, seed + i, steps, cfg, sampler_name, scheduler,
                                                  batch_positive, batch_negative,
                                                  {"samples": batch_latents}, denoise=denoise)[0]

                processed_sections = torch.split(processed_batch["samples"], b, dim=0)

                # Initialize samples tensor if it hasn't been initialized yet
                if samples is None:
                    if len(latent_shape) == 5:
                        processed_channels = processed_sections[0].shape[2]
                        samples = torch.zeros((b, f, processed_channels, h, w), device=device)
                    else:
                        processed_channels = processed_sections[0].shape[1]
                        samples = torch.zeros((b, processed_channels, h, w), device=device)

                for (_, y_start, y_end, x_start, x_end, _, _), processed_section in zip(batch_sections,
                                                                                        processed_sections):
                    is_top = y_start == 0
                    is_left = x_start == 0
                    is_bottom = y_end == h
                    is_right = x_end == w
                    blend_mask = self.create_blend_mask(y_end - y_start, x_end - x_start,
                                                        overlap_height, overlap_width,
                                                        is_top, is_left, is_bottom, is_right,
                                                        device)

                    blend_mask = blend_mask.unsqueeze(0).unsqueeze(0).expand_as(processed_section)

                    processed_section = processed_section.to(device=device)
                    blend_mask = blend_mask.to(device=device)

                    # Handle both 4D and 5D tensor blending
                    if len(latent_shape) == 5:
                        samples[:, :, :, y_start:y_end, x_start:x_end] = (
                                samples[:, :, :, y_start:y_end, x_start:x_end] * (1 - blend_mask) +
                                processed_section * blend_mask
                        )
                    else:
                        samples[:, :, y_start:y_end, x_start:x_end] = (
                                samples[:, :, y_start:y_end, x_start:x_end] * (1 - blend_mask) +
                                processed_section * blend_mask
                        )

                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            output_image = None
            if vae is not None:
                vae_decoder = VAEDecode()
                output_image = vae_decoder.decode(vae, {"samples": samples})[0]

            return (model, positive, negative, {"samples": samples}, vae, output_image)

        except Exception as e:
            logging.error(f"Error in FL_KsamplerPlus: {str(e)}")
            raise