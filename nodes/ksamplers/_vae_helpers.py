"""Shared VAE-decode helper for FL Ksampler nodes.

The Plus / PlusV2 / Basic / Sigma / ContextWindow / XYZPlot nodes all offer a
convenience IMAGE output that auto-runs VAE Decode on the sampled LATENT. For
SD / SDXL / Flux this works because the sampler's latent shape matches what the
paired VAE expects. For packed-latent / video models (Wan, LTX, future
families) the sampler emits a packed [B, C*F, H, W] (or otherwise shape-
shifted) tensor and the VAE's first conv layer rejects it with an unhelpful
channel-mismatch error.

This helper does a cheap pre-flight check: it compares the latent's channel
count against ``vae.latent_channels`` (a public attribute ComfyUI's
``comfy.sd.VAE`` already sets) and skips the auto-decode with a user-facing
message when they disagree. The LATENT output continues to flow; the user
just wires an external VAE Decode for the image side.
"""

import logging

import torch

from nodes import VAEDecode


_logger = logging.getLogger("fl_fill_nodes")


def _placeholder_image(reference_tensor):
    """1x1 black IMAGE tensor in ComfyUI's [B, H, W, C] convention."""
    device = "cpu"
    if hasattr(reference_tensor, "device"):
        device = reference_tensor.device
    return torch.zeros((1, 1, 1, 3), device=device)


def safe_vae_decode(vae, latent_dict, *, node_name="FL Ksampler"):
    """Decode ``latent_dict`` via VAE with a pre-flight shape check.

    Returns the decoded IMAGE on success, or a 1x1 placeholder image when the
    latent's channel count doesn't match the VAE's expected ``latent_channels``.
    On mismatch, a single ``logger.warning`` line tells the user to wire an
    external VAE Decode.
    """
    samples = latent_dict.get("samples") if isinstance(latent_dict, dict) else latent_dict
    if samples is None or not hasattr(samples, "shape") or samples.dim() < 2:
        return _placeholder_image(samples)

    actual_c = int(samples.shape[1])
    expected_c = getattr(vae, "latent_channels", None)
    if isinstance(expected_c, int) and expected_c > 0 and expected_c != actual_c:
        _logger.warning(
            "[%s] Skipping auto VAE decode: latent has %d channel(s) but the VAE expects %d. "
            "This usually means a packed-latent / video model (Wan, LTX, etc.). "
            "Wire the LATENT output to an external VAE Decode node and ignore this node's IMAGE output.",
            node_name, actual_c, expected_c,
        )
        return _placeholder_image(samples)

    return VAEDecode().decode(vae, latent_dict)[0]
