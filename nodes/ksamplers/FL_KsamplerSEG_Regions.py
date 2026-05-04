# FL_KsamplerSEG_Regions: tessellate an image into N Voronoi cells with
# overlapping, feathered region masks so EVERY pixel is touched by at least
# one diffusion pass and seams blend smoothly. Optional Lloyd relaxation for
# uniform cell sizes; optional safe-zone subtraction.
#
# Mask design (post-fix):
#   - One soft mask per region. Built by dilating the hard Voronoi cell, then
#     gaussian-blurring the edge. The result has values in [0, 1] that taper
#     across the boundary into adjacent cells.
#   - The SAME soft mask is used as both the diffusion noise_mask AND the
#     composite blend mask. This guarantees: (a) the model partially updates
#     pixels in the overlap band, (b) the alpha-accumulator math is consistent
#     and normalizes to ~1.0 everywhere, (c) the union of all masks covers the
#     entire canvas (modulo safe_zone).
#
# Performance: Voronoi labeling, mask processing, and Lloyd relaxation all run
# on the user's torch device (GPU when available) using batched ops.

import io
import base64
import math

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from server import PromptServer

from .FL_KsamplerSEG_common import make_regions_dict


class FL_KsamplerSEG_Regions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_regions": ("INT", {"default": 12, "min": 4, "max": 64, "step": 1}),
                "relaxation_iterations": ("INT", {"default": 5, "min": 0, "max": 8, "step": 1}),
                "region_overlap_factor": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01}),
                "edge_softness": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 0.5, "step": 0.01}),
                "context_padding_factor": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
                "safe_zone_feather_px": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 32.0, "step": 0.5}),
                "downscale_ratio": ("INT", {"default": 8, "min": 1, "max": 16, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "show_preview": ("BOOLEAN", {"default": True}),
                "preview_mode": (["overlay", "coverage_heatmap"], {"default": "overlay"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "latent": ("LATENT",),
                "safe_zone_mask": ("MASK",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("SEG_REGIONS", "IMAGE", "INT")
    RETURN_NAMES = ("regions", "preview_image", "effective_count")
    FUNCTION = "build"
    CATEGORY = "🏵️Fill Nodes/Ksamplers"

    def build(self, num_regions, relaxation_iterations, region_overlap_factor,
              edge_softness, context_padding_factor, safe_zone_feather_px,
              downscale_ratio, seed, show_preview, preview_mode,
              image=None, latent=None, safe_zone_mask=None, unique_id=None):
        if image is None and latent is None:
            raise ValueError(
                "FL_KsamplerSEG_Regions: connect either 'image' or 'latent' to size the regions."
            )
        if image is not None and latent is not None:
            raise ValueError(
                "FL_KsamplerSEG_Regions: connect either 'image' or 'latent', not both."
            )

        if image is not None:
            H, W = int(image.shape[1]), int(image.shape[2])
            viz_source = (image[0].cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            samples = latent["samples"]
            H = int(samples.shape[-2]) * downscale_ratio
            W = int(samples.shape[-1]) * downscale_ratio
            viz_source = np.full((H, W, 3), 64, dtype=np.uint8)

        device = self._pick_device()

        # All heavy mask work runs at "work resolution" -- 2x latent res. The
        # sampler downsamples masks 8x to latent space anyway, so doing them
        # at 8x* of full means we redundantly compute ~64x more pixels than
        # needed. work_scale = downscale_ratio * 2 keeps a 2x margin for
        # crisp boundaries.
        work_scale = max(1, int(downscale_ratio) * 2)
        H_w = max(8, (H + work_scale - 1) // work_scale)
        W_w = max(8, (W + work_scale - 1) // work_scale)
        sx = W_w / float(W)
        sy = H_w / float(H)

        # Initial seed points on a regular aspect-matched grid (in work-res coords).
        # Pure random scatter gives ±50% area variance after labeling; grid init +
        # Lloyd relaxation converges to ~5% variance, so cells end up uniform.
        # A small jitter (10% of cell size) prevents exact alignment with image
        # edges and gives Lloyd a non-degenerate starting point.
        rng = np.random.default_rng(int(seed))
        seeds_np = self._grid_initial_seeds(int(num_regions), W_w, H_w, rng)
        seeds = torch.from_numpy(seeds_np).to(device=device)

        # Lloyd relaxation runs at work res (cheap because work res is small).
        labels_w = self._compute_labels_gpu(H_w, W_w, seeds, device)
        for _ in range(int(relaxation_iterations)):
            seeds = self._lloyd_step(labels_w, int(num_regions), H_w, W_w, device)
            labels_w = self._compute_labels_gpu(H_w, W_w, seeds, device)

        # Hard cell masks at work res: (N, H_w, W_w).
        idx = torch.arange(int(num_regions), device=device).view(-1, 1, 1)
        hard_masks_w = (labels_w.unsqueeze(0) == idx).to(torch.float32)

        non_empty = hard_masks_w.flatten(1).sum(dim=1) >= 1.0
        if non_empty.sum().item() == 0:
            raise RuntimeError(
                "FL_KsamplerSEG_Regions: all regions vanished after labeling."
            )
        hard_masks_w = hard_masks_w[non_empty]

        # Per-cell typical size in WORK-res pixels.
        cell_areas_w = hard_masks_w.flatten(1).sum(dim=1)
        avg_cell_side_w = float(cell_areas_w.mean().sqrt().item())

        # Dilation + feather radii in work-res pixels (proportionally smaller
        # than the full-res equivalents -- this is where the speedup comes from).
        dilate_w = max(1.0, float(region_overlap_factor) * avg_cell_side_w)
        feather_w = max(1.0, float(edge_softness) * avg_cell_side_w)

        # Build soft masks at work res. Both ops batched + grouped so N regions
        # cost roughly the same as 1.
        soft_masks_w = self._dilate_then_blur(hard_masks_w, dilate_w, feather_w)

        # Per-region renormalize so each soft mask peaks at 1.0.
        peaks = soft_masks_w.flatten(1).amax(dim=1).clamp(min=1e-6).view(-1, 1, 1)
        soft_masks_w = (soft_masks_w / peaks).clamp(0.0, 1.0)

        # Bbox extraction at WORK res, vectorized via row/col any (no Python loop).
        bboxes_w = self._extract_bboxes_vectorized(soft_masks_w, alpha_threshold=0.01)

        # Scale work-res bboxes back up to image-res. Round outward so we never
        # truncate the soft mask at the bbox edge.
        bboxes = []
        for (y0w, x0w, y1w, x1w) in bboxes_w:
            y0 = max(0, int(math.floor(y0w / sy)))
            x0 = max(0, int(math.floor(x0w / sx)))
            y1 = min(H, int(math.ceil(y1w / sy)))
            x1 = min(W, int(math.ceil(x1w / sx)))
            bboxes.append((y0, x0, y1, x1))

        # Padded bboxes (image-res, clipped).
        padded_bboxes = []
        for (y0, x0, y1, x1) in bboxes:
            max_side = max(y1 - y0, x1 - x0)
            pad = int(round(float(context_padding_factor) * max_side))
            padded_bboxes.append((
                max(0, y0 - pad),
                max(0, x0 - pad),
                min(H, y1 + pad),
                min(W, x1 + pad),
            ))

        # Safe-zone subtraction. Apply at work res (much cheaper) before the
        # final upsample.
        if safe_zone_mask is not None:
            safe_feathered_w = self._prepare_safe_zone(
                safe_zone_mask, H_w, W_w,
                # Scale feather px proportionally.
                float(safe_zone_feather_px) * (H_w / float(H)),
                device,
            )
            soft_masks_w = soft_masks_w * (1.0 - safe_feathered_w.unsqueeze(0))
            keep = soft_masks_w.flatten(1).sum(dim=1) >= 1.0
            if keep.sum().item() == 0:
                raise RuntimeError(
                    "FL_KsamplerSEG_Regions: all regions consumed by safe_zone_mask."
                )
            soft_masks_w = soft_masks_w[keep]
            padded_bboxes = [b for i, b in enumerate(padded_bboxes) if bool(keep[i].item())]
            bboxes = [b for i, b in enumerate(bboxes) if bool(keep[i].item())]

        # Coverage diagnostic at work res (cheap).
        coverage_w = soft_masks_w.sum(dim=0)
        min_cov = float(coverage_w.min().item())
        max_cov = float(coverage_w.max().item())
        # We don't fail on under-coverage (safe_zone is a legitimate cause), but
        # warn if a non-safe-zone area has weight < 0.01.
        if safe_zone_mask is None and min_cov < 0.01:
            print(
                f"[FL_KsamplerSEG_Regions] warning: coverage minimum is {min_cov:.4f}; "
                f"some pixels may not be diffused. Increase region_overlap_factor."
            )

        # Single full-res upsample of all soft masks at the end. Bilinear is
        # fine because the masks are already smooth (gaussian-blurred) at
        # work res; upsampling adds no aliasing artifacts.
        if (H_w, W_w) != (H, W):
            soft_masks = F.interpolate(
                soft_masks_w.unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(0).clamp(0.0, 1.0)
        else:
            soft_masks = soft_masks_w

        # Build the output dict. The soft mask serves both roles:
        # write_mask = soft mask (used as noise_mask in the sampler)
        # composite_mask = soft mask (used for blending in the sampler)
        # shape_mask = the same (kept for API compatibility)
        shape_t = soft_masks.detach().to(device="cpu", dtype=torch.float32)

        regions = make_regions_dict(
            shape_masks=shape_t,
            write_masks=shape_t,
            composite_masks=shape_t,
            padded_bboxes=padded_bboxes,
            image_size=(H, W),
            downscale_ratio=int(downscale_ratio),
            seed=int(seed),
        )

        # Visualization. Upsample the work-res coverage / labels to full res
        # for display. (The visualization is the only place we touch full res
        # outside of the final mask upsample.)
        if preview_mode == "coverage_heatmap":
            coverage_full = F.interpolate(
                coverage_w.unsqueeze(0).unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze().cpu().numpy()
            viz = self._build_coverage_viz(viz_source, coverage_full,
                                           min_cov, max_cov)
        else:
            labels_full = F.interpolate(
                labels_w.float().unsqueeze(0).unsqueeze(0), size=(H, W),
                mode="nearest",
            ).squeeze().long().cpu().numpy()
            viz = self._build_overlay_viz(
                viz_source, labels_full,
                padded_bboxes, len(padded_bboxes),
            )

        if show_preview:
            try:
                preview_uri = self._encode_preview(viz)
                PromptServer.instance.send_sync(
                    "fl_seg_regions_preview",
                    {
                        "node": str(unique_id) if unique_id is not None else "",
                        "image": preview_uri,
                        "count": len(padded_bboxes),
                        "size": [W, H],
                        "min_coverage": round(min_cov, 4),
                        "max_coverage": round(max_cov, 4),
                    },
                )
            except Exception as e:
                print(f"[FL_KsamplerSEG_Regions] preview send failed: {e}")

        viz_tensor = torch.from_numpy(viz.astype(np.float32) / 255.0).unsqueeze(0)
        return (regions, viz_tensor, len(padded_bboxes))

    # ------------------------------------------------------------------
    # GPU helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_device():
        try:
            import comfy.model_management as mm
            return mm.intermediate_device()
        except Exception:
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

    @staticmethod
    def _grid_initial_seeds(num_regions, W, H, rng):
        """Place seeds on an aspect-matched regular grid + small jitter.

        Combined with Lloyd relaxation, this produces near-uniform-area Voronoi
        cells. Pure random initialization at the same num_regions / iterations
        leaves much more area variance.
        """
        # Pick rows & cols so rows*cols >= num_regions, and the grid aspect ratio
        # roughly matches the canvas aspect ratio.
        aspect = float(W) / max(1.0, float(H))
        # cols / rows ~= aspect, and cols * rows >= N
        # -> rows = ceil(sqrt(N / aspect)); cols = ceil(N / rows)
        rows = max(1, int(np.ceil(np.sqrt(num_regions / max(aspect, 1e-6)))))
        cols = max(1, int(np.ceil(num_regions / rows)))

        # Generate (rows * cols) cell-center positions, then take the first
        # num_regions of them (truncating extras from the end).
        cell_w = W / cols
        cell_h = H / rows
        positions = []
        for r in range(rows):
            for c in range(cols):
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h
                positions.append((cx, cy))
                if len(positions) >= num_regions:
                    break
            if len(positions) >= num_regions:
                break

        seeds = np.array(positions, dtype=np.float32)

        # Jitter by 10% of cell size in each dimension. This prevents the
        # initial positions from sitting on a degenerate exact grid (which can
        # cause Lloyd to get stuck at a local minimum) without disturbing the
        # uniformity much.
        jitter_x = (rng.uniform(-0.5, 0.5, size=num_regions) * 0.1 * cell_w).astype(np.float32)
        jitter_y = (rng.uniform(-0.5, 0.5, size=num_regions) * 0.1 * cell_h).astype(np.float32)
        seeds[:, 0] = np.clip(seeds[:, 0] + jitter_x, 0.0, W - 1)
        seeds[:, 1] = np.clip(seeds[:, 1] + jitter_y, 0.0, H - 1)

        return seeds

    @staticmethod
    def _compute_labels_gpu(H, W, seeds, device):
        N = seeds.shape[0]
        slab_h = max(1, min(H, 16_000_000 // max(1, W * N)))
        ys_arange = torch.arange(H, device=device, dtype=torch.float32)
        xs_arange = torch.arange(W, device=device, dtype=torch.float32)
        labels = torch.empty(H, W, dtype=torch.long, device=device)
        sx = seeds[:, 0]
        sy = seeds[:, 1]
        for y0 in range(0, H, slab_h):
            y1 = min(H, y0 + slab_h)
            ys_slab = ys_arange[y0:y1]
            dx = xs_arange.view(1, W, 1) - sx.view(1, 1, N)
            dy = ys_slab.view(-1, 1, 1) - sy.view(1, 1, N)
            d2 = dx * dx + dy * dy
            labels[y0:y1] = d2.argmin(dim=-1)
        return labels

    @staticmethod
    def _lloyd_step(labels, N, H, W, device):
        flat = labels.flatten()
        counts = torch.zeros(N, device=device, dtype=torch.float32)
        counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))

        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        sum_x = torch.zeros(N, device=device, dtype=torch.float32)
        sum_y = torch.zeros(N, device=device, dtype=torch.float32)
        sum_x.scatter_add_(0, flat, xs.flatten())
        sum_y.scatter_add_(0, flat, ys.flatten())

        safe_counts = counts.clamp(min=1.0)
        cx = sum_x / safe_counts
        cy = sum_y / safe_counts

        empty = counts < 1.0
        if empty.any():
            cx = torch.where(empty, torch.rand_like(cx) * (W - 1), cx)
            cy = torch.where(empty, torch.rand_like(cy) * (H - 1), cy)

        return torch.stack([cx, cy], dim=1)

    @classmethod
    def _dilate_then_blur(cls, masks, dilate_px, feather_px):
        """Morphological dilation (via max-pool) then gaussian blur. Both batched
        across the N region dimension so this scales O(1) in N rather than O(N)."""
        # Dilation = max pool with kernel = 2*r+1, stride 1, padding r.
        r = max(1, int(round(float(dilate_px))))
        # max_pool2d wants (N, C, H, W). We treat each region as a batch item
        # with 1 channel.
        x = masks.unsqueeze(1)  # (N, 1, H, W)
        dilated = F.max_pool2d(x, kernel_size=2 * r + 1, stride=1, padding=r)
        # Now blur the dilated mask. Reuse the batched gaussian helper.
        blurred = cls._batched_gaussian_blur(dilated.squeeze(1), float(feather_px))
        return blurred

    @staticmethod
    def _batched_gaussian_blur(masks, sigma):
        """Separable gaussian blur on (N, H, W) tensor via grouped conv2d."""
        if sigma < 0.1:
            return masks
        radius = max(1, int(math.ceil(sigma * 3.0)))
        device = masks.device
        dtype = masks.dtype
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
        k = k / k.sum()
        N = masks.shape[0]
        x4 = masks.unsqueeze(0)
        kx = k.view(1, 1, 1, -1).expand(N, 1, 1, -1)
        ky = k.view(1, 1, -1, 1).expand(N, 1, -1, 1)
        x4 = F.pad(x4, (radius, radius, 0, 0), mode="reflect")
        x4 = F.conv2d(x4, kx, groups=N)
        x4 = F.pad(x4, (0, 0, radius, radius), mode="reflect")
        x4 = F.conv2d(x4, ky, groups=N)
        return x4.squeeze(0)

    @staticmethod
    def _extract_bboxes_vectorized(soft_masks, alpha_threshold=0.01):
        """Per-region (y0,x0,y1,x1) bboxes computed in batched torch ops.

        Avoids the per-region Python loop with .nonzero() (which forces a
        device->CPU sync per region). Instead does N argmin/argmax ops in
        parallel, then a single CPU sync for the integer extraction.
        """
        N, H, W = soft_masks.shape
        device = soft_masks.device
        thresholded = soft_masks > alpha_threshold

        row_any = thresholded.any(dim=2)  # (N, H)
        col_any = thresholded.any(dim=1)  # (N, W)

        # For each region, find first/last True row and column.
        # Trick: argmax on a bool tensor returns the first True (after cast to int).
        #        For "last True", argmax over the reversed tensor.
        #        Tag empty rows/cols so we can detect and clamp them.
        ys_arange = torch.arange(H, device=device)
        xs_arange = torch.arange(W, device=device)

        # First True row per region (returns 0 if all-False, but we'll mask).
        y0 = row_any.float().argmax(dim=1)  # (N,)
        # Last True row: argmax on the flipped row, then convert back.
        y1 = (H - 1) - row_any.flip(dims=[1]).float().argmax(dim=1)
        x0 = col_any.float().argmax(dim=1)
        x1 = (W - 1) - col_any.flip(dims=[1]).float().argmax(dim=1)

        # Detect empty regions (no row had any True).
        any_row = row_any.any(dim=1)  # (N,)

        # Clamp into valid bbox shape; use end-exclusive convention.
        y1_excl = (y1 + 1).clamp(min=1, max=H)
        x1_excl = (x1 + 1).clamp(min=1, max=W)

        # Stack everything into one (N, 4) tensor and CPU-sync once.
        stacked = torch.stack([y0, x0, y1_excl, x1_excl, any_row.long()], dim=1).cpu().tolist()

        bboxes = []
        for (y0_i, x0_i, y1_i, x1_i, any_i) in stacked:
            if not any_i:
                bboxes.append((0, 0, 1, 1))
            else:
                bboxes.append((int(y0_i), int(x0_i), int(y1_i), int(x1_i)))
        return bboxes

    @classmethod
    def _prepare_safe_zone(cls, mask, H, W, feather_px, device):
        m = mask.float().to(device=device)
        if m.ndim == 4:
            m = m.mean(dim=-1) if m.shape[-1] > 1 else m[..., 0]
        if m.ndim == 2:
            m = m.unsqueeze(0)
        m = m[0]
        if m.shape != (H, W):
            m = F.interpolate(
                m.unsqueeze(0).unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(0).squeeze(0)
        m = m.clamp(0.0, 1.0)
        if feather_px > 0.1:
            blurred = cls._batched_gaussian_blur(m.unsqueeze(0), float(feather_px))
            m = blurred.squeeze(0).clamp(0.0, 1.0)
        return m

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    @staticmethod
    def _build_overlay_viz(image_np, labels, bboxes, count):
        """Hard-cell-boundary overlay (the original viz). Used when preview_mode='overlay'."""
        H, W = labels.shape
        labels_t = torch.from_numpy(labels.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        ky = kx.transpose(-1, -2)
        ex = F.conv2d(F.pad(labels_t, (1, 1, 1, 1), mode="replicate"), kx)
        ey = F.conv2d(F.pad(labels_t, (1, 1, 1, 1), mode="replicate"), ky)
        edges = (ex.abs() + ey.abs()).squeeze().numpy() > 0.5

        viz = image_np.copy()
        viz[edges] = (viz[edges] * 0.3 + np.array([20, 220, 220], dtype=np.float32) * 0.7).astype(np.uint8)

        pil = Image.fromarray(viz)
        draw = ImageDraw.Draw(pil)
        font = None
        for name in ("arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"):
            try:
                font = ImageFont.truetype(name, 20)
                break
            except IOError:
                continue
        if font is None:
            font = ImageFont.load_default()

        for i, (y0, x0, y1, x1) in enumerate(bboxes):
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            label = str(i)
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                tw, th = draw.textsize(label, font=font)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    draw.text((cx - tw // 2 + dx, cy - th // 2 + dy),
                              label, fill=(0, 0, 0), font=font)
            draw.text((cx - tw // 2, cy - th // 2), label,
                      fill=(255, 240, 80), font=font)

        return np.array(pil)

    @staticmethod
    def _build_coverage_viz(image_np, coverage_np, min_cov, max_cov):
        """Coverage heatmap: visualizes the SUM of all soft masks per pixel.

        Color scheme:
          - black/dark red = under-covered (< 0.5) — bad, will not diffuse cleanly
          - dark blue = at coverage 1.0 (ideal — single region writes here)
          - cyan/white = high coverage (> 1.5) — multiple regions overlap (fine,
            the accumulator normalizes)
        """
        H, W = coverage_np.shape
        # Normalize for display: clamp to [0, 2.5] which spans realistic values.
        norm = np.clip(coverage_np, 0.0, 2.5) / 2.5

        # Build a tone-mapped color image: low coverage -> dark, ideal -> blue,
        # over-coverage -> cyan/white.
        r = np.where(norm < 0.4, (0.4 - norm) * 2.5, 0.0)  # red where under-covered
        b = np.clip(norm * 1.5, 0.0, 1.0)
        g = np.clip((norm - 0.4) * 2.0, 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)

        # Blend with the source image so you can still see what's underneath.
        blend = (rgb.astype(np.float32) * 0.65 + image_np.astype(np.float32) * 0.35)
        blend = blend.clip(0, 255).astype(np.uint8)

        pil = Image.fromarray(blend)
        draw = ImageDraw.Draw(pil)
        font = None
        for name in ("arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"):
            try:
                font = ImageFont.truetype(name, 16)
                break
            except IOError:
                continue
        if font is None:
            font = ImageFont.load_default()

        # Stamp coverage stats in the top-left.
        legend = f"coverage: min={min_cov:.2f}  max={max_cov:.2f}  (target: min >= 1.0)"
        try:
            bbox = draw.textbbox((0, 0), legend, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(legend, font=font)
        draw.rectangle([4, 4, 4 + tw + 12, 4 + th + 8], fill=(0, 0, 0))
        draw.text((10, 8), legend, fill=(255, 240, 80), font=font)

        return np.array(pil)

    @staticmethod
    def _encode_preview(image_np):
        pil = Image.fromarray(image_np)
        pil.thumbnail((768, 768), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
