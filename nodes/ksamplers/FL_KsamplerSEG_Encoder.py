# FL_KsamplerSEG_Encoder: turn per-region captions into per-region CONDITIONING.
#
# Accepts SEG_REGIONS (no captions -> uses default_positive for all) or
# SEG_REGIONS_PROMPTED (uses captions). Encodes each unique prompt once
# through CLIP and reuses the result.

import logging

import nodes  # for CLIPTextEncode

from .FL_KsamplerSEG_common import unwrap_regions, attach_conditioning


class FL_KsamplerSEG_Encoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "regions": ("SEG_REGIONS",),
                "clip": ("CLIP",),
                "default_positive": ("STRING", {
                    "default": "high quality, detailed",
                    "multiline": True,
                }),
                "default_negative": ("STRING", {
                    "default": "blurry, low quality, artifacts",
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("SEG_REGIONS",)
    RETURN_NAMES = ("regions",)
    FUNCTION = "encode"
    CATEGORY = "🏵️Fill Nodes/Ksamplers"

    def encode(self, regions, clip, default_positive, default_negative):
        regions = unwrap_regions(regions)

        N = regions["shape_masks"].shape[0]
        captions = regions.get("captions") or [None] * N
        prefix = regions.get("caption_prefix", "") or ""
        suffix = regions.get("caption_suffix", "") or ""

        # Build the list of prompt texts per region.
        pos_texts = []
        for i in range(N):
            cap = captions[i] if i < len(captions) else None
            if cap and not cap.startswith("[caption failed"):
                pos_texts.append(f"{prefix}{cap}{suffix}".strip())
            else:
                pos_texts.append(default_positive)

        # Cache CLIP encodes by exact text. The negative is constant across regions.
        encoder = nodes.CLIPTextEncode()
        cache = {}

        def encode_text(text):
            if text in cache:
                return cache[text]
            cond = encoder.encode(clip, text)[0]
            # Strip area/mask keys -- those would be in full-frame coords and
            # break when applied to region crops.
            cleaned = []
            for entry in cond:
                if isinstance(entry, list) and len(entry) == 2:
                    embedding, meta = entry
                    if isinstance(meta, dict):
                        meta = {k: v for k, v in meta.items()
                                if k not in ("area", "mask", "mask_strength")}
                    cleaned.append([embedding, meta])
                else:
                    cleaned.append(entry)
            cache[text] = cleaned
            return cleaned

        neg_cond = encode_text(default_negative)
        per_region = []
        for text in pos_texts:
            try:
                pos_cond = encode_text(text)
            except Exception as e:
                logging.warning(
                    f"[FL_KsamplerSEG_Encoder] CLIP encode failed for "
                    f"text {text!r}: {e}; using default."
                )
                pos_cond = encode_text(default_positive)
            per_region.append((pos_cond, neg_cond))

        out = attach_conditioning(regions, per_region)
        return (out,)
