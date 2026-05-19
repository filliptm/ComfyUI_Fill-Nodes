"""Masked LTXV IC-LoRA guide utilities for Fill Nodes."""

import torch
import torch.nn.functional as F

import comfy.utils
import comfy_extras.nodes_lt as nodes_lt
import node_helpers


_INPAINT_GREEN = (102.0 / 255.0, 1.0, 0.0)


class FL_LTXVMaskedICLoRAGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "frame_idx": (
                    "INT",
                    {
                        "default": 0,
                        "min": -9999,
                        "max": 9999,
                        "tooltip": "Pixel frame index where the guide starts. LTX video guides snap to the model's latent cadence.",
                    },
                ),
                "guide_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Strength for the IC-LoRA guide conditioning.",
                    },
                ),
                "denoise_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Maximum denoise mask value applied to masked target latent areas.",
                    },
                ),
                "attention_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Strength for the guide attention mask metadata.",
                    },
                ),
                "mask_mode": (
                    ["both", "latent_noise", "attention"],
                    {
                        "default": "both",
                        "tooltip": "Choose whether the mask affects latent denoising, guide attention, or both.",
                    },
                ),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "inpaint_preprocess": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Fill masked guide pixels with LTX's inpaint green before VAE encoding.",
                    },
                ),
                "write_source_to_latent": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Encode the source image/video into the target latent frames before applying the denoise mask.",
                    },
                ),
                "latent_downscale_factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 1.0,
                        "tooltip": "Downscale only the IC-LoRA guide latent. The source latent write remains full size.",
                    },
                ),
                "crop": (["disabled", "center"], {"default": "disabled"}),
                "use_tiled_encode": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 32}),
                "tile_overlap": ("INT", {"default": 64, "min": 16, "max": 512, "step": 16}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "apply"
    CATEGORY = "🏵️Fill Nodes/WIP"
    DESCRIPTION = (
        "Adds an image/video as an LTXV IC-LoRA guide while applying a video mask "
        "to latent denoising, guide attention, or both."
    )
    def apply(
        self,
        positive,
        negative,
        vae,
        latent,
        image,
        mask,
        frame_idx,
        guide_strength,
        denoise_strength,
        attention_strength,
        mask_mode,
        invert_mask,
        inpaint_preprocess,
        write_source_to_latent,
        latent_downscale_factor,
        crop,
        use_tiled_encode,
        tile_size,
        tile_overlap,
    ):
        latent_downscale_factor = max(1, int(round(latent_downscale_factor)))
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"].clone()
        uses_latent_noise_mask = mask_mode in ("both", "latent_noise")
        noise_mask = self._get_noise_mask(latent, latent_image, uses_latent_noise_mask)

        if latent_image.ndim != 5:
            raise ValueError("FL_LTXVMaskedICLoRAGuide expects a 5D LTX video latent.")
        if latent_image.shape[1] != 128:
            raise ValueError("Adding LTXV guides to combined audio/video latents is not supported.")

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        mask = self._normalize_mask(mask, image.shape[0], image.shape[1], image.shape[2], image.device, invert_mask)

        source_image = image
        guide_input = image
        if inpaint_preprocess:
            guide_input = self._green_fill_masked_pixels(image, mask)

        time_scale_factor = scale_factors[0]
        num_frames_to_keep = ((guide_input.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
        causal_fix = frame_idx == 0 or num_frames_to_keep == 1
        if not causal_fix:
            guide_input = torch.cat([guide_input[:1], guide_input], dim=0)
            source_image = torch.cat([source_image[:1], source_image], dim=0)
            mask = torch.cat([mask[:1], mask], dim=0)

        guide_image, guide_latent, source_latent = self._encode_guides(
            vae=vae,
            latent_width=latent_width,
            latent_height=latent_height,
            guide_image=guide_input,
            source_image=source_image,
            scale_factors=scale_factors,
            latent_downscale_factor=latent_downscale_factor,
            crop=crop,
            use_tiled_encode=use_tiled_encode,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            needs_source_latent=write_source_to_latent,
        )
        mask = mask[: guide_image.shape[0]]

        if not causal_fix:
            guide_latent = guide_latent[:, :, 1:, :, :]
            if source_latent is not None:
                source_latent = source_latent[:, :, 1:, :, :]
            guide_image = guide_image[1:]
            mask = mask[1:]

        guide_orig_shape = list(guide_latent.shape[2:])
        guide_mask = None
        if latent_downscale_factor > 1:
            guide_latent, guide_mask = self._dilate_small_guide(
                guide_latent,
                latent_width,
                latent_height,
                int(latent_downscale_factor),
            )

        frame_idx, latent_idx = nodes_lt.LTXVAddGuide.get_latent_index(
            positive,
            latent_length,
            guide_image.shape[0],
            frame_idx,
            scale_factors,
        )
        if latent_idx + guide_latent.shape[2] > latent_length:
            raise ValueError("Conditioning frames exceed the length of the latent sequence.")

        if write_source_to_latent and source_latent is not None:
            self._write_source_latent(latent_image, source_latent, latent_idx)

        if mask_mode in ("both", "latent_noise"):
            noise_mask = self._apply_target_noise_mask(
                noise_mask=noise_mask,
                source_mask=mask,
                latent_idx=latent_idx,
                cond_length=guide_latent.shape[2],
                latent_height=latent_height,
                latent_width=latent_width,
                time_scale_factor=scale_factors[0],
                denoise_strength=denoise_strength,
            )

        positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
            positive=positive,
            negative=negative,
            frame_idx=frame_idx,
            latent_image=latent_image,
            noise_mask=noise_mask,
            guiding_latent=guide_latent,
            strength=guide_strength,
            scale_factors=scale_factors,
            guide_mask=guide_mask,
            latent_downscale_factor=latent_downscale_factor,
            causal_fix=causal_fix,
        )

        pre_filter_count = guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
        attention_mask = None
        if mask_mode in ("both", "attention"):
            attention_mask = self._attention_mask(mask, guide_image.shape[0])

        positive = self._append_guide_attention_entry(
            positive,
            pre_filter_count,
            guide_orig_shape,
            attention_strength=attention_strength,
            attention_mask=attention_mask,
        )
        negative = self._append_guide_attention_entry(
            negative,
            pre_filter_count,
            guide_orig_shape,
            attention_strength=attention_strength,
            attention_mask=attention_mask,
        )

        return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

    def _encode_guides(
        self,
        vae,
        latent_width,
        latent_height,
        guide_image,
        source_image,
        scale_factors,
        latent_downscale_factor,
        crop,
        use_tiled_encode,
        tile_size,
        tile_overlap,
        needs_source_latent,
    ):
        time_scale_factor, width_scale_factor, height_scale_factor = scale_factors
        num_frames_to_keep = ((guide_image.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
        guide_image = guide_image[:num_frames_to_keep]
        source_image = source_image[:num_frames_to_keep]

        target_width = int(latent_width * width_scale_factor / latent_downscale_factor)
        target_height = int(latent_height * height_scale_factor / latent_downscale_factor)
        guide_pixels = comfy.utils.common_upscale(
            guide_image.movedim(-1, 1),
            target_width,
            target_height,
            "bilinear",
            crop=crop,
        ).movedim(1, -1).clamp(0, 1)
        guide_latent = self._vae_encode(vae, guide_pixels[:, :, :, :3], use_tiled_encode, tile_size, tile_overlap)

        source_latent = None
        if needs_source_latent:
            full_pixels = comfy.utils.common_upscale(
                source_image.movedim(-1, 1),
                latent_width * width_scale_factor,
                latent_height * height_scale_factor,
                "bilinear",
                crop=crop,
            ).movedim(1, -1).clamp(0, 1)
            source_latent = self._vae_encode(vae, full_pixels[:, :, :, :3], use_tiled_encode, tile_size, tile_overlap)

        return guide_pixels, guide_latent, source_latent

    def _get_noise_mask(self, latent, latent_image, uses_latent_noise_mask):
        existing = latent.get("noise_mask", None)
        if existing is not None:
            return existing.clone()

        batch_size, _, latent_length, latent_height, latent_width = latent_image.shape
        fill_value = 0.0 if uses_latent_noise_mask else 1.0
        spatial_size = (latent_height, latent_width) if uses_latent_noise_mask else (1, 1)
        return torch.full(
            (batch_size, 1, latent_length, spatial_size[0], spatial_size[1]),
            fill_value,
            dtype=torch.float32,
            device=latent_image.device,
        )

    def _vae_encode(self, vae, pixels, use_tiled_encode, tile_size, tile_overlap):
        if use_tiled_encode:
            return vae.encode_tiled(pixels, tile_x=tile_size, tile_y=tile_size, overlap=tile_overlap)
        return vae.encode(pixels)

    def _normalize_mask(self, mask, frames, height, width, device, invert_mask):
        if mask.ndim == 4:
            mask = mask.mean(dim=-1)
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim != 3:
            raise ValueError("Mask must have shape (H,W), (F,H,W), or (F,H,W,C).")

        mask = mask.to(device=device, dtype=torch.float32).clamp(0, 1)
        if invert_mask:
            mask = 1.0 - mask

        if mask.shape[0] == 1 and frames > 1:
            mask = mask.expand(frames, -1, -1)
        elif mask.shape[0] < frames:
            pad = mask[-1:].expand(frames - mask.shape[0], -1, -1)
            mask = torch.cat([mask, pad], dim=0)
        elif mask.shape[0] > frames:
            mask = mask[:frames]

        if mask.shape[1] != height or mask.shape[2] != width:
            mask = F.interpolate(
                mask.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        return mask

    def _green_fill_masked_pixels(self, image, mask):
        green = torch.tensor(_INPAINT_GREEN, device=image.device, dtype=image.dtype).view(1, 1, 1, 3)
        mask = mask.to(device=image.device, dtype=image.dtype).unsqueeze(-1)
        return image[:, :, :, :3] * (1.0 - mask) + green * mask

    def _dilate_small_guide(self, guide_latent, latent_width, latent_height, latent_downscale_factor):
        if latent_width % latent_downscale_factor != 0 or latent_height % latent_downscale_factor != 0:
            raise ValueError(
                f"Latent spatial size {latent_width}x{latent_height} must be divisible by "
                f"latent_downscale_factor {latent_downscale_factor}."
            )

        dilated_shape = guide_latent.shape[:3] + (
            guide_latent.shape[3] * latent_downscale_factor,
            guide_latent.shape[4] * latent_downscale_factor,
        )
        dilated_samples = torch.zeros(
            dilated_shape,
            device=guide_latent.device,
            dtype=guide_latent.dtype,
            requires_grad=False,
        )
        dilated_samples[..., ::latent_downscale_factor, ::latent_downscale_factor] = guide_latent

        guide_mask = torch.full(
            (guide_latent.shape[0], 1, guide_latent.shape[2], dilated_shape[3], dilated_shape[4]),
            -1.0,
            device=guide_latent.device,
            dtype=guide_latent.dtype,
            requires_grad=False,
        )
        guide_mask[..., ::latent_downscale_factor, ::latent_downscale_factor] = 1.0
        return dilated_samples, guide_mask

    def _write_source_latent(self, latent_image, source_latent, latent_idx):
        cond_length = source_latent.shape[2]
        if latent_idx + cond_length > latent_image.shape[2]:
            raise ValueError("Source latent frames exceed the length of the target latent.")
        if source_latent.shape[3:] != latent_image.shape[3:]:
            raise ValueError(
                f"Source latent spatial size {tuple(source_latent.shape[3:])} does not match "
                f"target latent size {tuple(latent_image.shape[3:])}."
            )
        latent_image[:, :, latent_idx : latent_idx + cond_length] = source_latent

    def _apply_target_noise_mask(
        self,
        noise_mask,
        source_mask,
        latent_idx,
        cond_length,
        latent_height,
        latent_width,
        time_scale_factor,
        denoise_strength,
    ):
        batch_size = noise_mask.shape[0]
        if noise_mask.shape[3] == 1 or noise_mask.shape[4] == 1:
            noise_mask = noise_mask.expand(-1, -1, -1, latent_height, latent_width).clone()
        else:
            noise_mask = noise_mask.clone()

        temporal_mask = self._downsample_mask_to_latent_frames(source_mask, cond_length, time_scale_factor)
        latent_mask = F.interpolate(
            temporal_mask.unsqueeze(1),
            size=(latent_height, latent_width),
            mode="bilinear",
            align_corners=False,
        ).to(device=noise_mask.device, dtype=noise_mask.dtype)

        latent_mask = latent_mask.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        latent_mask = latent_mask.permute(0, 2, 1, 3, 4) * denoise_strength
        noise_mask[:, :, latent_idx : latent_idx + cond_length] = torch.maximum(
            noise_mask[:, :, latent_idx : latent_idx + cond_length],
            latent_mask,
        )
        return noise_mask

    def _downsample_mask_to_latent_frames(self, mask, cond_length, time_scale_factor):
        latent_masks = []
        for latent_frame in range(cond_length):
            if latent_frame == 0:
                start = 0
                end = 1
            else:
                start = 1 + (latent_frame - 1) * time_scale_factor
                end = min(1 + latent_frame * time_scale_factor, mask.shape[0])

            if start >= mask.shape[0]:
                frame_mask = mask[-1]
            else:
                frame_mask = mask[start:end].amax(dim=0)
            latent_masks.append(frame_mask)

        return torch.stack(latent_masks, dim=0)

    def _attention_mask(self, mask, frames):
        if mask.shape[0] == 1 and frames > 1:
            mask = mask.expand(frames, -1, -1)
        elif mask.shape[0] < frames:
            pad = mask[-1:].expand(frames - mask.shape[0], -1, -1)
            mask = torch.cat([mask, pad], dim=0)
        else:
            mask = mask[:frames]
        return mask.unsqueeze(0).unsqueeze(0)

    def _append_guide_attention_entry(
        self,
        conditioning,
        pre_filter_count,
        latent_shape,
        attention_strength,
        attention_mask,
    ):
        existing_entries = []
        for item in conditioning:
            entries = item[1].get("guide_attention_entries", None)
            if entries is not None:
                existing_entries = entries
                break

        entries = [*existing_entries]
        entries.append(
            {
                "pre_filter_count": pre_filter_count,
                "strength": attention_strength,
                "pixel_mask": attention_mask,
                "latent_shape": latent_shape,
            }
        )
        return node_helpers.conditioning_set_values(conditioning, {"guide_attention_entries": entries})


NODE_CLASS_MAPPINGS = {
    "FL_LTXVMaskedICLoRAGuide": FL_LTXVMaskedICLoRAGuide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_LTXVMaskedICLoRAGuide": "FL LTXV Masked IC-LoRA Guide",
}
