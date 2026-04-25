import json
import logging
import torch
from aiohttp import web

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import latent_preview
from nodes import VAEDecode, VAEEncode
from server import PromptServer


class FL_KsamplerSigma:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sigma_max_hint": ("FLOAT", {"default": 14.6146, "min": 0.001, "max": 1000.0, "step": 0.01}),
                "sigma_min_hint": ("FLOAT", {"default": 0.0292, "min": 0.0, "max": 1000.0, "step": 0.001}),
                "schedule_curve": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "latent_image": ("LATENT",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "IMAGE", "SIGMAS")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "vae", "image", "sigmas")
    FUNCTION = "sample"
    CATEGORY = "🏵️Fill Nodes/Ksamplers"

    def sample(self, model, positive, negative, seed, steps, cfg, sampler_name, denoise,
               sigma_max_hint, sigma_min_hint, schedule_curve,
               latent_image=None, vae=None, image=None):
        try:
            device = comfy.model_management.get_torch_device()

            if image is not None and vae is not None:
                latent_image = VAEEncode().encode(vae, image)[0]
            elif latent_image is None:
                latent_image = {"samples": torch.zeros((1, 4, 64, 64), device=device)}

            sigmas = _build_sigmas(model, schedule_curve, steps, denoise)
            if sigmas is None or sigmas.numel() < 2:
                raise ValueError(
                    "FL_KsamplerSigma: schedule_curve is empty or invalid. "
                    "Draw a sigma curve on the node before sampling."
                )

            samples_dict = _sample_with_sigmas(
                model, seed, cfg, sampler_name, positive, negative,
                latent_image, sigmas,
            )
            sigmas_out = sigmas.detach().cpu()

            output_image = None
            if vae is not None:
                output_image = VAEDecode().decode(vae, samples_dict)[0]

            return (model, positive, negative, samples_dict, vae, output_image, sigmas_out)

        except Exception as e:
            logging.error(f"Error in FL_KsamplerSigma: {e}")
            raise


def _build_sigmas(model, curve_json, steps, denoise):
    """Parse the curve JSON, interpolate to (steps+1) samples, enforce monotonicity,
    clamp to the model's sigma range, and force the tail to zero."""
    if not curve_json:
        return None
    try:
        parsed = json.loads(curve_json)
    except (json.JSONDecodeError, TypeError):
        return None

    points = parsed.get("points") if isinstance(parsed, dict) else parsed
    if not isinstance(points, list) or len(points) < 2:
        return None

    interpolation = "monotone_cubic"
    if isinstance(parsed, dict):
        interpolation = parsed.get("interpolation", interpolation)
    if interpolation not in ("monotone_cubic", "linear"):
        interpolation = "monotone_cubic"

    norm = []
    for p in points:
        if isinstance(p, dict):
            t = p.get("t")
            s = p.get("sigma")
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            t, s = p[0], p[1]
        else:
            continue
        if t is None or s is None:
            continue
        try:
            norm.append((float(t), float(s)))
        except (TypeError, ValueError):
            continue
    if len(norm) < 2:
        return None

    norm.sort(key=lambda p: p[0])
    if norm[0][0] > 1e-6:
        norm.insert(0, (0.0, norm[0][1]))
    if norm[-1][0] < 1.0 - 1e-6:
        norm.append((1.0, norm[-1][1]))

    effective_steps = steps if (denoise is None or denoise > 0.9999) else max(1, int(steps / denoise))
    n_samples = effective_steps + 1
    ts = torch.linspace(0.0, 1.0, n_samples, dtype=torch.float64)
    xs = torch.tensor([p[0] for p in norm], dtype=torch.float64)
    ys = torch.tensor([p[1] for p in norm], dtype=torch.float64)

    if interpolation == "linear":
        sigmas_full = _interp_linear(ts, xs, ys)
    else:
        sigmas_full = _interp_monotone_cubic(ts, xs, ys)

    monotone = torch.cummin(sigmas_full, dim=0).values
    monotone = torch.clamp(monotone, min=0.0)

    model_sampling = model.get_model_object("model_sampling")
    sigma_max = float(model_sampling.sigma_max)
    monotone = torch.clamp(monotone, max=sigma_max)

    if denoise is not None and denoise <= 0.9999 and denoise > 0.0:
        monotone = monotone[-(steps + 1):]

    monotone[-1] = 0.0
    return monotone.to(dtype=torch.float32)


def _interp_linear(ts, xs, ys):
    idx = torch.searchsorted(xs, ts).clamp(1, len(xs) - 1)
    x0 = xs[idx - 1]
    x1 = xs[idx]
    y0 = ys[idx - 1]
    y1 = ys[idx]
    w = (ts - x0) / (x1 - x0).clamp(min=1e-12)
    return y0 + w * (y1 - y0)


def _interp_monotone_cubic(ts, xs, ys):
    """Fritsch-Carlson monotone cubic (PCHIP)."""
    n = xs.shape[0]
    if n == 2:
        return _interp_linear(ts, xs, ys)

    h = xs[1:] - xs[:-1]
    delta = (ys[1:] - ys[:-1]) / h.clamp(min=1e-12)

    m = torch.zeros(n, dtype=xs.dtype)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    for i in range(n - 1):
        if delta[i] == 0:
            m[i] = 0.0
            m[i + 1] = 0.0
        else:
            a = m[i] / delta[i]
            b = m[i + 1] / delta[i]
            s = a * a + b * b
            if s > 9.0:
                tau = 3.0 / torch.sqrt(s)
                m[i] = tau * a * delta[i]
                m[i + 1] = tau * b * delta[i]

    idx = torch.searchsorted(xs, ts).clamp(1, n - 1)
    x0 = xs[idx - 1]
    x1 = xs[idx]
    y0 = ys[idx - 1]
    y1 = ys[idx]
    m0 = m[idx - 1]
    m1 = m[idx]
    hi = (x1 - x0).clamp(min=1e-12)
    t = (ts - x0) / hi
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * y0 + h10 * hi * m0 + h01 * y1 + h11 * hi * m1


def _sample_with_sigmas(model, seed, cfg, sampler_name, positive, negative, latent, sigmas):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_image, latent.get("downscale_ratio_spacial", None)
    )

    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = latent.get("noise_mask", None)

    sampler = comfy.samplers.sampler_object(sampler_name)
    sigmas = sigmas.to(device=model.load_device)

    callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = comfy.sample.sample_custom(
        model, noise, cfg, sampler, sigmas, positive, negative, latent_image,
        noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed,
    )
    samples = samples.to(
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return out
