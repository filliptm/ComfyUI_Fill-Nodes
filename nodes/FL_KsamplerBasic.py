import torch
from nodes import common_ksampler, VAEDecode, VAEEncode
import comfy.samplers
import comfy.utils
import logging

class FL_KsamplerBasic:
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

    def sample(self, model, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise, latent_image=None, image=None, vae=None):
        try:
            device = comfy.model_management.get_torch_device()

            # Handle input image if provided
            if image is not None and vae is not None:
                latent_image = VAEEncode().encode(vae, image)[0]
            elif latent_image is None:
                latent_image = {"samples": torch.zeros((1, 4, 64, 64), device=device)}

            samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                                      positive, negative, latent_image, denoise=denoise)[0]

            output_image = None
            if vae is not None:
                vae_decoder = VAEDecode()
                output_image = vae_decoder.decode(vae, samples)[0]

            return (model, positive, negative, samples, vae, output_image)

        except Exception as e:
            logging.error(f"Error in FL_KsamplerBasic: {str(e)}")
            raise