import torch
from nodes import common_ksampler, VAEDecode
import comfy.samplers
import comfy.utils

class FL_UltimateUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("MODEL", "VAE", "CONDITIONING", "CONDITIONING", "LATENT", "IMAGE")
    RETURN_NAMES = ("model", "vae", "positive", "negative", "latent", "image")
    FUNCTION = "sample"
    CATEGORY = "FL/Sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, vae=None):
        samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise)[0]

        # Actual image decoding (if VAE is provided)
        output_image = None
        if vae is not None:
            vae_decoder = VAEDecode()
            output_image = vae_decoder.decode(vae, samples)[0]

        # Prepare the output
        return (model, vae, positive, negative, samples, output_image)

    @classmethod
    def IS_CHANGED(s, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, vae=None):
        # This method ensures the node always runs, even if outputs are not connected
        return float("NaN")