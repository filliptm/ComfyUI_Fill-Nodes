# Shared types, dict factory, and validators for the FL_KsamplerSEG_* pipeline.
#
# A SEG_REGIONS dict propagates through Regions -> Captioner -> Encoder -> Sampler.
# Each downstream node may add fields without breaking upstream consumers.
#
# Custom socket types (Fill-Nodes convention: plain string types):
#   SEG_REGIONS          -- shape masks + bboxes only
#   SEG_REGIONS_PROMPTED -- adds captions + default_negative
#   SEG_REGIONS_ENCODED  -- adds conditioning_per_region

import math
import torch


SEG_LEVEL_REGIONS = "regions"
SEG_LEVEL_PROMPTED = "prompted"
SEG_LEVEL_ENCODED = "encoded"


def make_regions_dict(
    shape_masks: torch.Tensor,
    write_masks: torch.Tensor,
    composite_masks: torch.Tensor,
    padded_bboxes: list,
    image_size: tuple,
    downscale_ratio: int,
    seed: int,
) -> dict:
    """Construct a SEG_REGIONS dict. All masks must be (N, H, W)."""
    assert shape_masks.ndim == 3
    assert write_masks.shape == shape_masks.shape
    assert composite_masks.shape == shape_masks.shape
    assert len(padded_bboxes) == shape_masks.shape[0]

    return {
        "_seg_level": SEG_LEVEL_REGIONS,
        "shape_masks": shape_masks,
        "write_masks": write_masks,
        "composite_masks": composite_masks,
        "padded_bboxes": list(padded_bboxes),
        "image_size": tuple(image_size),
        "downscale_ratio": int(downscale_ratio),
        "seed": int(seed),
        "effective_count": int(shape_masks.shape[0]),
        "captions": None,
        "caption_prefix": "",
        "caption_suffix": "",
        "default_negative": "",
        "conditioning_per_region": None,
    }


def attach_captions(regions: dict, captions: list, prefix: str, suffix: str,
                    default_negative: str) -> dict:
    """Returns a shallow-cloned regions dict promoted to PROMPTED level."""
    out = dict(regions)
    out["_seg_level"] = SEG_LEVEL_PROMPTED
    out["captions"] = list(captions)
    out["caption_prefix"] = prefix or ""
    out["caption_suffix"] = suffix or ""
    out["default_negative"] = default_negative or ""
    return out


def attach_conditioning(regions: dict, conditioning_per_region: list) -> dict:
    """Returns a shallow-cloned regions dict promoted to ENCODED level."""
    out = dict(regions)
    out["_seg_level"] = SEG_LEVEL_ENCODED
    out["conditioning_per_region"] = list(conditioning_per_region)
    return out


def unwrap_regions(regions, expect_min_level: str = SEG_LEVEL_REGIONS):
    """Validate and return the regions dict.

    expect_min_level: minimum level required. Sampler accepts any level;
    Encoder accepts REGIONS or PROMPTED.
    """
    if not isinstance(regions, dict):
        raise TypeError(
            f"Expected SEG_REGIONS dict, got {type(regions).__name__}. "
            "Wire the output of FL_KsamplerSEG_Regions into this input."
        )
    level = regions.get("_seg_level")
    order = {SEG_LEVEL_REGIONS: 0, SEG_LEVEL_PROMPTED: 1, SEG_LEVEL_ENCODED: 2}
    if level not in order:
        raise ValueError(f"Unrecognized SEG region dict level: {level}")
    if order[level] < order[expect_min_level]:
        raise ValueError(
            f"This input requires at least '{expect_min_level}' level "
            f"but received '{level}'."
        )
    for required in ("shape_masks", "write_masks", "composite_masks",
                     "padded_bboxes", "image_size", "downscale_ratio"):
        if required not in regions:
            raise ValueError(f"SEG region dict missing required key: {required}")
    return regions


def latent_bbox_from_image_bbox(bbox: tuple, downscale: int,
                                latent_h: int, latent_w: int) -> tuple:
    """Convert an image-space bbox (y0,x0,y1,x1) to latent-space bbox,
    rounding outward and clipping to latent bounds."""
    y0, x0, y1, x1 = bbox
    by0 = max(0, y0 // downscale)
    bx0 = max(0, x0 // downscale)
    by1 = min(latent_h, math.ceil(y1 / downscale))
    bx1 = min(latent_w, math.ceil(x1 / downscale))
    return (int(by0), int(bx0), int(by1), int(bx1))
