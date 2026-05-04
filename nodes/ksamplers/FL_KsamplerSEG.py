# FL_KsamplerSEG: per-region segmented diffusion sampler.
#
# Cascade-paint compositing (single mode):
#   - Regions are painted onto a working canvas in a corner-anchored
#     nearest-neighbor traversal (each new region paints adjacent to
#     already-painted territory).
#   - Each region samples from the IN-PROGRESS canvas, so the model sees
#     previously-painted regions as part of its context. The model itself
#     does the seam-blending in feature space rather than us doing it
#     mechanically in pixel space.
#   - Per-region results are alpha-painted on top using the soft mask:
#     canvas[bbox] = result * mask + canvas[bbox] * (1 - mask)
#
# The soft masks come from FL_KsamplerSEG_Regions; they overlap and feather
# already, so coverage is guaranteed.

import logging

import torch
import torch.nn.functional as F

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import latent_preview

from .FL_KsamplerSEG_common import unwrap_regions, latent_bbox_from_image_bbox


CASCADE_START_CORNERS = ["top_left", "top_right", "bottom_left", "bottom_right", "center"]


class FL_KsamplerSEG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "regions": ("SEG_REGIONS",),
                "latent_image": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 6.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cascade_start_corner": (CASCADE_START_CORNERS, {"default": "top_left"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "🏵️Fill Nodes/Ksamplers"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def sample(self, model, regions, latent_image, positive, negative,
               seed, steps, cfg, sampler_name, scheduler, denoise,
               cascade_start_corner):
        regions = unwrap_regions(regions)

        latent_samples = latent_image["samples"]
        if latent_samples.is_nested:
            raise NotImplementedError(
                "FL_KsamplerSEG: nested-tensor latents not supported yet."
            )

        is_zero = bool(torch.count_nonzero(latent_samples) == 0)
        if is_zero and denoise < 0.99:
            raise ValueError(
                "FL_KsamplerSEG: per-region refinement requires denoise=1.0 with "
                "an empty latent, or a real latent input with denoise<1.0. "
                "An empty latent at low denoise produces noise."
            )

        H, W = regions["image_size"]
        latent_h = latent_samples.shape[-2]
        latent_w = latent_samples.shape[-1]

        # Auto-detect the actual spatial downscale from the model. Different
        # models use different ratios -- SD/SDXL=8, Flux=8, LTX-Video=32,
        # Hunyuan-Video=16, etc. Use the model's authoritative value rather
        # than the user's hint on the Regions node, which won't know.
        downscale = self._resolve_downscale(model, regions, latent_h, latent_w, H, W)

        expected_lh = (H + downscale - 1) // downscale
        expected_lw = (W + downscale - 1) // downscale
        if abs(latent_h - expected_lh) > 1 or abs(latent_w - expected_lw) > 1:
            logging.warning(
                f"[FL_KsamplerSEG] latent shape ({latent_h}x{latent_w}) doesn't match "
                f"regions image_size {H}x{W} / downscale={downscale}. Sampling may misalign."
            )

        # One-line diagnostic so users can confirm the right downscale was picked.
        print(f"[FL_KsamplerSEG] latent={tuple(latent_samples.shape)}  "
              f"regions={H}x{W}  downscale={downscale}")

        device = model.load_device

        latent_full = comfy.sample.fix_empty_latent_channels(
            model, latent_samples.to(device=device),
            latent_image.get("downscale_ratio_spacial", None),
        )

        N = regions["shape_masks"].shape[0]
        per_region_cond = regions.get("conditioning_per_region")
        if per_region_cond is not None and len(per_region_cond) != N:
            logging.warning(
                f"[FL_KsamplerSEG] conditioning_per_region length {len(per_region_cond)} "
                f"!= region count {N}; falling back to widget conditioning."
            )
            per_region_cond = None

        # Spatial-locality ordering: corner-anchored nearest-neighbor.
        # Each new region paints adjacent to already-painted territory so the
        # cascade flows naturally from the chosen corner outward.
        write_areas = regions["write_masks"].sum(dim=(1, 2))
        order = self._cascade_order(
            regions=regions,
            write_areas=write_areas,
            start_corner=cascade_start_corner,
        )

        # Pre-compute per-region (latent_bbox, write_mask_lat, comp_mask_lat,
        # pos_cond, neg_cond, seed). Skips regions too small for stable attention.
        region_specs = []
        for n in order:
            spec = self._build_region_spec(
                regions=regions, region_index=int(n), downscale=downscale,
                latent_h=latent_h, latent_w=latent_w, device=device,
                per_region_cond=per_region_cond,
                cond_pos_default=positive, cond_neg_default=negative,
                base_seed=int(seed),
            )
            if spec is not None:
                region_specs.append(spec)

        if not region_specs:
            logging.warning("[FL_KsamplerSEG] no usable regions; returning input latent unchanged.")
            return (latent_image,)

        # Run the cascade.
        canvas = latent_full.clone()
        # Pre-compute the broadcast shape for masks. For a 4D latent (B,C,H,W)
        # the mask broadcasts as (1,1,H,W). For a 5D video latent (B,C,T,H,W)
        # it must broadcast as (1,1,1,H,W) -- one extra leading dim per
        # non-spatial axis. PyTorch won't auto-align mismatched-rank tensors.
        latent_ndim = latent_full.ndim
        for spec in region_specs:
            samples = self._sample_one_region_full(
                model=model, source_latent=canvas, spec=spec,
                steps=steps, cfg=cfg, sampler_name=sampler_name,
                scheduler=scheduler, denoise=denoise,
                latent_ndim=latent_ndim,
            )
            by0, bx0, by1, bx1 = spec["latent_bbox"]
            comp_b = self._reshape_mask_for_broadcast(
                spec["comp_lat"], latent_ndim,
            ).to(dtype=canvas.dtype)
            samples_dev = samples.to(device=canvas.device, dtype=canvas.dtype)
            existing = canvas[..., by0:by1, bx0:bx1]
            canvas[..., by0:by1, bx0:bx1] = samples_dev * comp_b + existing * (1.0 - comp_b)

        out = canvas.to(
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        result = dict(latent_image)
        result["samples"] = out
        result.pop("noise_mask", None)
        return (result,)

    # ------------------------------------------------------------------
    # Per-region helpers
    # ------------------------------------------------------------------

    def _build_region_spec(self, *, regions, region_index, downscale,
                           latent_h, latent_w, device, per_region_cond,
                           cond_pos_default, cond_neg_default, base_seed):
        """Resolve geometry + masks + conditioning for one region. Returns None
        if the region is too small to sample."""
        bbox = regions["padded_bboxes"][region_index]
        latent_bbox = latent_bbox_from_image_bbox(bbox, downscale, latent_h, latent_w)
        by0, bx0, by1, bx1 = latent_bbox

        if (by1 - by0) < 8 or (bx1 - bx0) < 8:
            logging.warning(
                f"[FL_KsamplerSEG] region {region_index} latent bbox too small "
                f"({by1-by0}x{bx1-bx0}); skipping."
            )
            return None

        write_full = regions["write_masks"][region_index].to(device=device, dtype=torch.float32)
        comp_full = regions["composite_masks"][region_index].to(device=device, dtype=torch.float32)

        iy0 = by0 * downscale
        ix0 = bx0 * downscale
        iy1 = min(by1 * downscale, write_full.shape[0])
        ix1 = min(bx1 * downscale, write_full.shape[1])
        write_img_crop = write_full[iy0:iy1, ix0:ix1]
        comp_img_crop = comp_full[iy0:iy1, ix0:ix1]

        target_h_img = (by1 - by0) * downscale
        target_w_img = (bx1 - bx0) * downscale
        write_img_crop = self._pad_to(write_img_crop, target_h_img, target_w_img)
        comp_img_crop = self._pad_to(comp_img_crop, target_h_img, target_w_img)

        write_lat = F.avg_pool2d(
            write_img_crop.unsqueeze(0).unsqueeze(0), kernel_size=downscale,
        ).squeeze(0).squeeze(0)
        comp_lat = F.avg_pool2d(
            comp_img_crop.unsqueeze(0).unsqueeze(0), kernel_size=downscale,
        ).squeeze(0).squeeze(0)

        if write_lat.sum() < 1e-6:
            return None

        if per_region_cond is not None:
            pos_cond, neg_cond = per_region_cond[region_index]
        else:
            pos_cond, neg_cond = cond_pos_default, cond_neg_default

        return {
            "region_index": region_index,
            "latent_bbox": latent_bbox,
            "write_lat": write_lat,
            "comp_lat": comp_lat,
            "pos_cond": pos_cond,
            "neg_cond": neg_cond,
            "seed": base_seed + region_index,
        }

    def _sample_one_region_full(self, *, model, source_latent, spec,
                                steps, cfg, sampler_name, scheduler, denoise,
                                latent_ndim=None):
        """Run a complete sampler call (all steps) for one region, sourcing the
        crop from `source_latent` (the in-progress canvas)."""
        by0, bx0, by1, bx1 = spec["latent_bbox"]
        latent_crop = source_latent[..., by0:by1, bx0:bx1].contiguous()

        noise = comfy.sample.prepare_noise(latent_crop.cpu(), spec["seed"])

        # Mask must match the latent rank so it broadcasts. For 4D latent
        # (image): (1,1,H,W). For 5D latent (video): (1,1,1,H,W).
        if latent_ndim is None:
            latent_ndim = source_latent.ndim
        noise_mask = self._reshape_mask_for_broadcast(
            spec["write_lat"], latent_ndim,
        ).to(dtype=latent_crop.dtype)

        callback = latent_preview.prepare_callback(model, steps)

        try:
            samples = comfy.sample.sample(
                model, noise, steps, cfg, sampler_name, scheduler,
                spec["pos_cond"], spec["neg_cond"], latent_crop,
                denoise=denoise, noise_mask=noise_mask,
                callback=callback, disable_pbar=True, seed=spec["seed"],
            )
        except Exception as e:
            logging.error(
                f"[FL_KsamplerSEG] region {spec['region_index']} sample failed: {e}"
            )
            raise

        return samples

    @staticmethod
    def _cascade_order(*, regions, write_areas, start_corner):
        """Corner-anchored nearest-neighbor traversal of regions for cascade_paint.

        Computes each region's centroid (mass-weighted) from its write_mask,
        picks the region nearest the chosen corner as the starting point, then
        greedily appends the unprocessed region whose centroid is closest to
        any already-processed region's centroid. Size is used as a tiebreaker
        (larger regions preferred when distances are within 1% of each other).

        Returns a list of region indices in painting order.
        """
        write_masks = regions["write_masks"]  # (N, H, W) on CPU
        N, H, W = write_masks.shape

        ys = torch.arange(H, dtype=torch.float32).view(1, H, 1)
        xs = torch.arange(W, dtype=torch.float32).view(1, 1, W)
        masses = write_masks.sum(dim=(1, 2)).clamp(min=1e-6)
        cy = (write_masks * ys).sum(dim=(1, 2)) / masses
        cx = (write_masks * xs).sum(dim=(1, 2)) / masses
        centroids = torch.stack([cx, cy], dim=1)  # (N, 2) in (x, y) order

        if start_corner == "top_left":
            anchor = torch.tensor([0.0, 0.0])
        elif start_corner == "top_right":
            anchor = torch.tensor([float(W - 1), 0.0])
        elif start_corner == "bottom_left":
            anchor = torch.tensor([0.0, float(H - 1)])
        elif start_corner == "bottom_right":
            anchor = torch.tensor([float(W - 1), float(H - 1)])
        else:  # center
            anchor = torch.tensor([(W - 1) / 2.0, (H - 1) / 2.0])

        d_to_anchor = ((centroids - anchor) ** 2).sum(dim=1)
        first = int(d_to_anchor.argmin().item())

        order = [first]
        unvisited = [i for i in range(N) if i != first]
        sizes = write_areas.tolist() if isinstance(write_areas, torch.Tensor) else list(write_areas)

        c_np = centroids.numpy()
        nearest_d = {}
        for i in unvisited:
            dx = c_np[i, 0] - c_np[first, 0]
            dy = c_np[i, 1] - c_np[first, 1]
            nearest_d[i] = dx * dx + dy * dy

        while unvisited:
            min_d = min(nearest_d[i] for i in unvisited)
            tie_threshold = min_d * 1.01 + 1e-6
            candidates = [i for i in unvisited if nearest_d[i] <= tie_threshold]
            if len(candidates) == 1:
                pick = candidates[0]
            else:
                pick = max(candidates, key=lambda i: sizes[i])

            order.append(pick)
            unvisited.remove(pick)

            for i in unvisited:
                dx = c_np[i, 0] - c_np[pick, 0]
                dy = c_np[i, 1] - c_np[pick, 1]
                d = dx * dx + dy * dy
                if d < nearest_d[i]:
                    nearest_d[i] = d

        return order

    @staticmethod
    def _resolve_downscale(model, regions, latent_h, latent_w, image_h, image_w):
        """Determine the actual spatial downscale ratio.

        Order of preference:
          1. The model's latent_format.spacial_downscale_ratio if it matches
             the observed latent shape (this is the authoritative source --
             SD/SDXL=8, Flux=8, LTX-Video=32, Hunyuan-Video=16, Wan=16, etc.)
          2. Inferred from latent_shape / image_shape if (1) doesn't match
          3. The regions dict's downscale_ratio hint as a final fallback
        """
        regions_hint = int(regions.get("downscale_ratio", 8))

        # Try the model's authoritative value.
        try:
            latent_format = model.get_model_object("latent_format")
            model_ratio = int(getattr(latent_format, "spacial_downscale_ratio", 8))
            # Verify it matches the observed shape. Allow ±1 for rounding.
            expected_lh = (image_h + model_ratio - 1) // model_ratio
            expected_lw = (image_w + model_ratio - 1) // model_ratio
            if abs(latent_h - expected_lh) <= 1 and abs(latent_w - expected_lw) <= 1:
                if model_ratio != regions_hint:
                    print(
                        f"[FL_KsamplerSEG] model uses downscale={model_ratio} "
                        f"(regions hint was {regions_hint}); using model's value."
                    )
                return model_ratio
        except Exception:
            pass

        # Infer from observed shapes.
        if latent_h > 0 and latent_w > 0:
            inferred_h = round(image_h / latent_h)
            inferred_w = round(image_w / latent_w)
            if inferred_h == inferred_w and inferred_h >= 1:
                if inferred_h != regions_hint:
                    print(
                        f"[FL_KsamplerSEG] inferred downscale={inferred_h} from "
                        f"latent vs image dims (regions hint was {regions_hint})."
                    )
                return inferred_h

        # Last resort.
        return regions_hint

    @staticmethod
    def _reshape_mask_for_broadcast(mask_2d, target_ndim):
        """Take a 2D (h, w) mask and add leading 1-dims to match target_ndim.

        For target_ndim=4 (image B,C,H,W): returns (1, 1, h, w).
        For target_ndim=5 (video B,C,T,H,W): returns (1, 1, 1, h, w).
        """
        if mask_2d.ndim != 2:
            # Defensive: if already higher-dim, leave it.
            return mask_2d
        out = mask_2d
        leading = max(0, target_ndim - 2)
        for _ in range(leading):
            out = out.unsqueeze(0)
        return out

    @staticmethod
    def _pad_to(t, h, w):
        ch = t.shape[0]
        cw = t.shape[1]
        if ch == h and cw == w:
            return t
        pad_h = max(0, h - ch)
        pad_w = max(0, w - cw)
        return F.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
