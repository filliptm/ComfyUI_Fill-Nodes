import torch
from nodes import common_ksampler, VAEDecode, VAEEncode
import comfy.samplers
import comfy.utils
import torch.nn.functional as F
import logging


class FL_FractalKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale_factor": ("FLOAT", {"default": 1.25, "min": 1.1, "max": 2.0, "step": 0.05}),
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "fractal_levels": ("INT", {"default": 2, "min": 1, "max": 5}),
            },
            "optional": {
                "latent_image": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT", "VAE")
    FUNCTION = "sample"
    CATEGORY = "🏵️Fill Nodes/WIP"

    def recursive_sample(self, model, vae, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise,
                         scale_factor, blend_factor, latent, level, max_levels):
        if level >= max_levels:
            return latent

        # Upscale in latent space
        samples = latent['samples']
        _, _, h, w = samples.shape
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        upscaled_samples = F.interpolate(samples, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Slice the latent into 4 slices
        mid_h, mid_w = new_h // 2, new_w // 2
        slices = [
            upscaled_samples[:, :, :mid_h, :mid_w],
            upscaled_samples[:, :, :mid_h, mid_w:],
            upscaled_samples[:, :, mid_h:, :mid_w],
            upscaled_samples[:, :, mid_h:, mid_w:]
        ]

        # Render each slice with reduced denoise
        rendered_slices = []
        for i, slice_latent in enumerate(slices):
            rendered_slice = common_ksampler(model, seed + i + (level * 1000), steps, cfg, sampler_name, scheduler,
                                             positive, negative, {"samples": slice_latent},
                                             denoise=denoise * (0.75 ** level))[0]

            # Recursively process each slice
            recursive_result = self.recursive_sample(model, vae, positive, negative, seed + i + (level * 1000),
                                                     steps, cfg, sampler_name, scheduler, denoise,
                                                     scale_factor, blend_factor, rendered_slice,
                                                     level + 1, max_levels)

            rendered_slices.append(recursive_result["samples"])

        # Composite the slices onto the upscaled latent with a blend factor
        composite = upscaled_samples.clone()
        composite[:, :, :mid_h, :mid_w] = rendered_slices[0] * blend_factor + composite[:, :, :mid_h, :mid_w] * (
                    1 - blend_factor)
        composite[:, :, :mid_h, mid_w:] = rendered_slices[1] * blend_factor + composite[:, :, :mid_h, mid_w:] * (
                    1 - blend_factor)
        composite[:, :, mid_h:, :mid_w] = rendered_slices[2] * blend_factor + composite[:, :, mid_h:, :mid_w] * (
                    1 - blend_factor)
        composite[:, :, mid_h:, mid_w:] = rendered_slices[3] * blend_factor + composite[:, :, mid_h:, mid_w:] * (
                    1 - blend_factor)

        return {"samples": composite}

    def sample(self, model, vae, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise,
               scale_factor, blend_factor, fractal_levels, latent_image=None):
        try:
            device = comfy.model_management.get_torch_device()

            if latent_image is None:
                latent_image = {"samples": torch.zeros((1, 4, 64, 64), device=device)}

            # Initial sampling
            initial_sample = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                                             positive, negative, latent_image, denoise=denoise)[0]

            # Start recursive sampling
            final_latent = self.recursive_sample(model, vae, positive, negative, seed, steps, cfg, sampler_name,
                                                 scheduler,
                                                 denoise, scale_factor, blend_factor, initial_sample, 0, fractal_levels)

            return (final_latent, vae)

        except Exception as e:
            logging.error(f"Error in FL_FractalKSampler: {str(e)}")
            raise