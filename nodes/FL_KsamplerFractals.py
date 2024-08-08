import torch
from nodes import common_ksampler, VAEDecode, VAEEncode
import comfy.samplers
import comfy.utils
import torch.nn.functional as F


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
                "fractal_levels": ("INT", {"default": 3, "min": 1, "max": 5}),
            },
            "optional": {
                "latent_image": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "FL/Sampling"

    def sample(self, model, vae, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise, fractal_levels,
               latent_image=None):
        device = comfy.model_management.get_torch_device()

        if latent_image is None:
            latent_image = {"samples": torch.zeros((1, 4, 64, 64), device=device)}

        def fractal_sample(latent, level, current_denoise):
            # Decode to pixel space
            pixel_image = VAEDecode().decode(vae, latent)[0]

            # Upscale by 25%
            _, _, h, w = pixel_image.shape
            new_h, new_w = int(h * 1.25), int(w * 1.25)
            upscaled_image = F.interpolate(pixel_image, size=(new_h, new_w), mode='bicubic', align_corners=False)

            # Re-encode to latent space
            upscaled_latent = VAEEncode().encode(vae, upscaled_image)

            # Sample at current level
            current_sample = common_ksampler(model, seed + level, steps, cfg, sampler_name, scheduler,
                                             positive, negative, {"samples": upscaled_latent}, denoise=current_denoise)[
                0]

            if level == 0:
                return current_sample

            # Divide into quadrants
            h, w = current_sample["samples"].shape[2:]
            mid_h, mid_w = h // 2, w // 2
            quadrants = [
                {"samples": current_sample["samples"][:, :, :mid_h, :mid_w]},
                {"samples": current_sample["samples"][:, :, :mid_h, mid_w:]},
                {"samples": current_sample["samples"][:, :, mid_h:, :mid_w]},
                {"samples": current_sample["samples"][:, :, mid_h:, mid_w:]}
            ]

            # Recursively sample quadrants
            next_denoise = current_denoise * 0.5
            sampled_quadrants = [fractal_sample(quad, level - 1, next_denoise) for quad in quadrants]

            # Combine quadrants
            top = torch.cat((sampled_quadrants[0]["samples"], sampled_quadrants[1]["samples"]), dim=3)
            bottom = torch.cat((sampled_quadrants[2]["samples"], sampled_quadrants[3]["samples"]), dim=3)
            return {"samples": torch.cat((top, bottom), dim=2)}

        final_sample = fractal_sample(latent_image, fractal_levels, denoise)
        return (final_sample,)