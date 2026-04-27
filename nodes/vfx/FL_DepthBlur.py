# FL_DepthBlur: depth-aware lens-style blur for ComfyUI.
#
# Takes (image, depth_map) and produces a depth-of-field result. Items at the
# user-set focal_depth stay sharp; pixels deviating from that focal plane get
# progressively blurred according to a falloff curve.
#
# Implementation notes:
#   * Pre-compute N pre-blurred copies of the image (gaussian/box/disc kernels).
#   * Build a per-pixel "blur amount" map from the depth map + focal params.
#   * For each pixel sample two adjacent stack layers and lerp between them by
#     the fractional blur amount. Vectorized + GPU-friendly.

import io
import base64
import math

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from server import PromptServer

from comfy.utils import ProgressBar


FALLOFF_CURVES = ["linear", "quadratic", "smooth", "exponential"]
BLUR_MODES = ["gaussian", "box", "disc"]


class FL_DepthBlur:
    """Depth-of-field blur driven by a depth map."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "focal_depth": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "focal_range": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_blur": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 64.0, "step": 0.5}),
                "depth_invert": ("BOOLEAN", {"default": False}),
                "falloff_curve": (FALLOFF_CURVES, {"default": "quadratic"}),
                "near_blur_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "far_blur_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur_mode": (BLUR_MODES, {"default": "gaussian"}),
                "quality_steps": ("INT", {"default": 8, "min": 4, "max": 24, "step": 1}),
                "depth_smoothing": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "depth_remap_low": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "depth_remap_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bokeh_brightness_boost": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.05}),
                "mask_feather": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 32.0, "step": 0.5}),
                "show_preview": ("BOOLEAN", {"default": False, "label": "Show Preview on Node"}),
            },
            "optional": {
                "protect_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "blur_amount_viz", "in_focus_mask")
    FUNCTION = "apply"
    OUTPUT_NODE = True
    CATEGORY = "🏵️Fill Nodes/VFX"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def apply(self, image, depth_map, focal_depth, focal_range, max_blur,
              depth_invert, falloff_curve, near_blur_strength, far_blur_strength,
              blur_mode, quality_steps, depth_smoothing,
              depth_remap_low, depth_remap_high, bokeh_brightness_boost,
              mask_feather, show_preview=False, protect_mask=None):

        # Normalize batch dims, match depth to image batch.
        image = image.float()
        depth_map = depth_map.float()
        B, H, W, C = image.shape

        # If image is RGBA, keep alpha untouched at the end.
        had_alpha = (C == 4)
        rgb = image[..., :3]
        alpha = image[..., 3:4] if had_alpha else None

        # Prepare depth: make sure it has a batch matching the image, and
        # collapse to single channel.
        depth_single = self._prepare_depth(depth_map, H, W, B)
        if depth_invert:
            depth_single = 1.0 - depth_single

        # Optional remap to compress the active depth range.
        lo = min(depth_remap_low, depth_remap_high)
        hi = max(depth_remap_low, depth_remap_high)
        if hi - lo > 1e-6:
            depth_single = ((depth_single - lo) / (hi - lo)).clamp(0.0, 1.0)
        else:
            depth_single = torch.full_like(depth_single, 0.5)

        # Optional gaussian smoothing on the depth map itself.
        if depth_smoothing > 1e-3:
            depth_single = self._gaussian_blur_2d(
                depth_single.unsqueeze(1), depth_smoothing
            ).squeeze(1)

        # Build per-pixel blur amount in [0, max_blur].
        blur_map = self._compute_blur_map(
            depth_single, focal_depth, focal_range, max_blur,
            falloff_curve, near_blur_strength, far_blur_strength,
        )

        # Build in-focus mask (1 where blur ~ 0).
        if max_blur > 1e-6:
            in_focus = (1.0 - (blur_map / max_blur).clamp(0.0, 1.0))
        else:
            in_focus = torch.ones_like(blur_map)

        # Build the pre-blurred stack and composite using the blur map.
        result = self._composite_stack(
            rgb, blur_map, max_blur, blur_mode, quality_steps,
            bokeh_brightness_boost,
        )
        result = result.clamp(0.0, 1.0)

        # Protect-mask compositing: where the (optionally feathered) mask is
        # white, lerp back to the original sharp RGB. White=fully protected.
        protect = self._prepare_protect_mask(protect_mask, H, W, B,
                                             mask_feather, image.device)
        if protect is not None:
            m = protect.unsqueeze(-1)  # (B, H, W, 1) for broadcast over channels
            result = result * (1.0 - m) + rgb * m
            # Reflect protection in the diagnostic outputs too.
            blur_map = blur_map * (1.0 - protect)
            in_focus = torch.clamp(in_focus + protect, 0.0, 1.0)

        if had_alpha:
            result = torch.cat([result, alpha], dim=-1)

        # Build a 3-channel grayscale viz of the blur amount.
        if max_blur > 1e-6:
            viz = (blur_map / max_blur).clamp(0.0, 1.0)
        else:
            viz = torch.zeros_like(blur_map)
        viz_rgb = viz.unsqueeze(-1).expand(-1, -1, -1, 3)

        if show_preview:
            try:
                preview_data = self._prepare_preview(result)
                PromptServer.instance.send_sync(
                    "fl_depth_blur_preview",
                    {"image": preview_data},
                )
            except Exception as e:
                print(f"[FL_DepthBlur] preview send failed: {e}")

        return (result, viz_rgb, in_focus)

    # ------------------------------------------------------------------
    # Depth handling
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_depth(depth_map, target_h, target_w, target_b):
        """Coerce the depth_map IMAGE into a (B, H, W) single-channel tensor
        matching the target image's batch and spatial dims."""
        if depth_map.ndim == 3:
            depth_map = depth_map.unsqueeze(0)

        # Collapse channels: assume grayscale, but average if RGB(A).
        if depth_map.shape[-1] >= 3:
            d = depth_map[..., :3].mean(dim=-1)
        else:
            d = depth_map[..., 0]

        DB, DH, DW = d.shape
        if (DH, DW) != (target_h, target_w):
            d = F.interpolate(
                d.unsqueeze(1), size=(target_h, target_w),
                mode="bilinear", align_corners=False,
            ).squeeze(1)

        if DB != target_b:
            if DB == 1:
                d = d.repeat(target_b, 1, 1)
            elif target_b % DB == 0:
                d = d.repeat(target_b // DB, 1, 1)
            else:
                # Fall back to first frame.
                d = d[0:1].repeat(target_b, 1, 1)

        return d.clamp(0.0, 1.0)

    @classmethod
    def _prepare_protect_mask(cls, mask, target_h, target_w, target_b,
                              feather, device):
        """Coerce a Comfy MASK (or stray IMAGE) into (B, H, W) in [0, 1],
        matching target dims. Returns None if no usable mask was provided."""
        if mask is None:
            return None
        m = mask.float()
        # Comfy MASK is (B, H, W). IMAGE is (B, H, W, C); accept either.
        if m.ndim == 4:
            if m.shape[-1] >= 3:
                m = m[..., :3].mean(dim=-1)
            else:
                m = m[..., 0]
        elif m.ndim == 2:
            m = m.unsqueeze(0)
        elif m.ndim != 3:
            return None

        if m.numel() == 0:
            return None

        MB, MH, MW = m.shape
        if (MH, MW) != (target_h, target_w):
            m = F.interpolate(
                m.unsqueeze(1), size=(target_h, target_w),
                mode="bilinear", align_corners=False,
            ).squeeze(1)
        if MB != target_b:
            if MB == 1:
                m = m.repeat(target_b, 1, 1)
            elif target_b % MB == 0:
                m = m.repeat(target_b // MB, 1, 1)
            else:
                m = m[0:1].repeat(target_b, 1, 1)

        m = m.clamp(0.0, 1.0).to(device)

        if feather > 1e-3:
            m = cls._gaussian_blur_2d(m.unsqueeze(1), float(feather)).squeeze(1)
            m = m.clamp(0.0, 1.0)

        return m

    # ------------------------------------------------------------------
    # Blur amount map
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_blur_map(depth, focal_depth, focal_range, max_blur,
                          curve, near_strength, far_strength):
        """For every pixel produce a desired blur radius in pixels."""
        # Distance from focal plane, asymmetrically signed.
        delta = depth - focal_depth  # positive = farther, negative = nearer
        abs_delta = delta.abs()

        # Subtract half-range; anything inside the in-focus band gets 0.
        half = focal_range * 0.5
        d = (abs_delta - half).clamp(min=0.0)

        # Normalize against the worst case (1.0 - half) so d -> [0, 1].
        denom = max(1.0 - half, 1e-6)
        n = (d / denom).clamp(0.0, 1.0)

        if curve == "linear":
            shaped = n
        elif curve == "quadratic":
            shaped = n * n
        elif curve == "smooth":
            # smoothstep
            shaped = n * n * (3.0 - 2.0 * n)
        elif curve == "exponential":
            # Maps 0 -> 0 and 1 -> 1, but ramps fast near 1.
            shaped = (torch.exp(n * 3.0) - 1.0) / (math.e ** 3 - 1.0)
        else:
            shaped = n

        # Asymmetric near/far multiplier.
        side = torch.where(delta < 0,
                           torch.full_like(shaped, near_strength),
                           torch.full_like(shaped, far_strength))
        shaped = shaped * side

        return shaped * max_blur

    # ------------------------------------------------------------------
    # Stack-and-composite
    # ------------------------------------------------------------------

    def _composite_stack(self, rgb, blur_map, max_blur, mode, steps, bokeh_boost):
        """Pre-blur the image at `steps` discrete radii, then per-pixel pick
        and lerp between adjacent layers based on blur_map."""
        if max_blur < 1e-3 or steps < 2:
            return rgb

        # Permute to (B, C, H, W) for kernel ops.
        x = rgb.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape

        # Optional bokeh brightness pump: brighten before disc blur, then
        # gamma-renormalize. Subtle but gives bokeh more "pop".
        if mode == "disc" and bokeh_boost > 1.0:
            x_input = torch.pow(x.clamp(0.0, 1.0), 1.0 / bokeh_boost)
        else:
            x_input = x

        # Generate radii: 0, r1, r2, ..., max_blur (linearly spaced).
        radii = torch.linspace(0.0, float(max_blur), steps, device=x.device)

        # Build the stack. layers[k] has shape (B, C, H, W).
        layers = []
        for r in radii:
            r_val = float(r.item())
            if r_val < 1e-3:
                layers.append(x_input)
                continue
            if mode == "gaussian":
                blurred = self._gaussian_blur_2d(x_input, r_val)
            elif mode == "box":
                blurred = self._box_blur_2d(x_input, r_val)
            elif mode == "disc":
                blurred = self._disc_blur_2d(x_input, r_val)
            else:
                blurred = x_input
            layers.append(blurred)

        # Reverse the bokeh boost on the final layer only is wrong — we need
        # to do it per-layer so unblurred pixels also come back to normal.
        if mode == "disc" and bokeh_boost > 1.0:
            layers = [torch.pow(L.clamp(0.0, 1.0), bokeh_boost) for L in layers]

        # Stack along a new "level" axis: (steps, B, C, H, W).
        stack = torch.stack(layers, dim=0)

        # blur_map: (B, H, W) in [0, max_blur]. Map to fractional layer index.
        # idx = blur_map / max_blur * (steps - 1)
        idx = (blur_map / max(max_blur, 1e-6)) * (steps - 1)
        idx = idx.clamp(0.0, steps - 1 - 1e-4)
        lo_idx = idx.floor().long()
        hi_idx = (lo_idx + 1).clamp(max=steps - 1)
        frac = (idx - lo_idx.float()).unsqueeze(1)  # (B, 1, H, W) for broadcast over C

        # Gather: build (B, C, H, W) tensors by indexing the stack on dim 0.
        # Using torch.gather with expanded indices.
        # Move to (B, steps, C, H, W) for easier per-batch gather.
        stack_bhfirst = stack.permute(1, 0, 2, 3, 4).contiguous()  # (B, steps, C, H, W)

        # We need per-pixel index: expand lo_idx/hi_idx to (B, 1, C, H, W) → gather along dim=1.
        lo_g = lo_idx.unsqueeze(1).unsqueeze(1).expand(-1, 1, C, -1, -1)
        hi_g = hi_idx.unsqueeze(1).unsqueeze(1).expand(-1, 1, C, -1, -1)
        lo_layer = stack_bhfirst.gather(1, lo_g).squeeze(1)  # (B, C, H, W)
        hi_layer = stack_bhfirst.gather(1, hi_g).squeeze(1)

        out = lo_layer * (1.0 - frac) + hi_layer * frac

        # Back to (B, H, W, C).
        return out.permute(0, 2, 3, 1).contiguous()

    # ------------------------------------------------------------------
    # Kernel implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_kernel_1d(sigma, device, dtype):
        # Radius covering ~3 sigma.
        radius = max(1, int(math.ceil(sigma * 3.0)))
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
        k = k / k.sum()
        return k, radius

    @classmethod
    def _gaussian_blur_2d(cls, x, sigma):
        """Separable gaussian. x is (B, C, H, W) or (B, 1, H, W)."""
        if sigma < 1e-3:
            return x
        device, dtype = x.device, x.dtype
        k, r = cls._gaussian_kernel_1d(float(sigma), device, dtype)
        C = x.shape[1]
        kx = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
        ky = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
        x = F.pad(x, (r, r, 0, 0), mode="reflect")
        x = F.conv2d(x, kx, groups=C)
        x = F.pad(x, (0, 0, r, r), mode="reflect")
        x = F.conv2d(x, ky, groups=C)
        return x

    @staticmethod
    def _box_blur_2d(x, radius):
        if radius < 1e-3:
            return x
        r = max(1, int(round(radius)))
        size = 2 * r + 1
        device, dtype = x.device, x.dtype
        C = x.shape[1]
        # Separable box kernel for speed.
        k = torch.full((1, 1, 1, size), 1.0 / size, device=device, dtype=dtype).expand(C, 1, 1, size)
        x = F.pad(x, (r, r, 0, 0), mode="reflect")
        x = F.conv2d(x, k, groups=C)
        ky = torch.full((1, 1, size, 1), 1.0 / size, device=device, dtype=dtype).expand(C, 1, size, 1)
        x = F.pad(x, (0, 0, r, r), mode="reflect")
        x = F.conv2d(x, ky, groups=C)
        return x

    @staticmethod
    def _disc_kernel_2d(radius, device, dtype):
        """Build a circular (disc) kernel — flat inside the disc, zero outside."""
        r = max(1, int(math.ceil(radius)))
        y, x = torch.meshgrid(
            torch.arange(-r, r + 1, device=device, dtype=dtype),
            torch.arange(-r, r + 1, device=device, dtype=dtype),
            indexing="ij",
        )
        # Soft anti-aliased edge.
        d = torch.sqrt(x * x + y * y)
        k = (radius + 0.5 - d).clamp(0.0, 1.0)
        k = k / k.sum()
        return k, r

    @classmethod
    def _disc_blur_2d(cls, x, radius):
        if radius < 1e-3:
            return x
        k, r = cls._disc_kernel_2d(float(radius), x.device, x.dtype)
        C = x.shape[1]
        # Non-separable: one big 2D conv, grouped per channel.
        kernel = k.view(1, 1, *k.shape).expand(C, 1, -1, -1)
        x = F.pad(x, (r, r, r, r), mode="reflect")
        x = F.conv2d(x, kernel, groups=C)
        return x

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_preview(tensor_image):
        """Convert (B, H, W, C) → base64 PNG data URI of the first frame."""
        if tensor_image.shape[0] > 0:
            t = tensor_image[0]
        else:
            t = tensor_image
        arr = (t.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            pil = Image.fromarray(arr, mode="RGBA")
        else:
            pil = Image.fromarray(arr[..., :3], mode="RGB")
        pil.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
