import torch
import comfy.sd
import comfy.model_base
import comfy.samplers
import comfy.sample
import comfy.k_diffusion.sampling

class FL_TD_KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "conditioning_positive": ("CONDITIONING",),
                             "conditioning_negative": ("CONDITIONING",),
                             "latent_image": ("LATENT",),
                             "steps": ("INT", {"default": 20, "min": 1, "max": 1000, "step": 1}),
                             "seed": ("INT", {"default": 42, "min": 0, "max": 2 ** 32 - 1}),
                             "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                             "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                             "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                             "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "🏵️Fill Nodes/experiments"

    def sample(self, model, conditioning_positive, conditioning_negative, latent_image, steps, seed, cfg, sampler_name,
               scheduler, denoise):
        device = comfy.model_management.get_torch_device()
        latent = latent_image["samples"]
        original_shape = latent.shape

        # Set the seed for reproducibility
        torch.manual_seed(seed)

        # Setup noise
        noise = torch.randn_like(latent, device=device)

        # Setup sampler
        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name,
                                          scheduler=scheduler, denoise=denoise, model_options=model.model_options)

        # Setup progress bar
        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        try:
            samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler,
                                          conditioning_positive, conditioning_negative, latent,
                                          denoise=denoise, disable_noise=False, start_step=0, last_step=steps,
                                          force_full_denoise=True, noise_mask=None, callback=callback,
                                          disable_pbar=disable_pbar, seed=seed)
        except Exception as e:
            print('Custom KSampler error encountered:', e)
            raise e
        finally:
            if pbar:
                pbar.update_absolute(steps, steps)

        # Prepare the output in the expected format
        out = {
            "samples": samples,
            "original_shape": original_shape,
            "noise_seed": seed,
            "steps": steps
        }

        return out