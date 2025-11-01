import torch
import sys
import os

# Add comfy to path to access internal modules
comfy_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "comfy")
if comfy_path not in sys.path:
    sys.path.insert(0, comfy_path)

import comfy.model_management
import comfy.utils
import comfy.latent_formats
import node_helpers
import nodes


class FL_WanVaceToVideoMultiRef:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "max_reference_frames": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1, "tooltip": "Maximum number of reference frames to use from the reference_video"}),
            },
            "optional": {
                "control_video": ("IMAGE",),
                "control_masks": ("MASK",),
                "reference_video": ("IMAGE", {"tooltip": "Video frames to use as reference (instead of single image). All frames will be encoded and prepended."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent")
    FUNCTION = "process_video"
    CATEGORY = "🏵️Fill Nodes/WIP"

    def process_video(self, positive, negative, vae, width, height, length, batch_size, strength, max_reference_frames, control_video=None, control_masks=None, reference_video=None):
        latent_length = ((length - 1) // 4) + 1

        if control_video is not None:
            control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5)
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5

        # Modified reference handling to support multiple frames
        # Key: NO PADDING - just use N reference frames and prepend them
        if reference_video is not None:
            # Take up to max_reference_frames from the reference_video
            num_ref_frames = min(reference_video.shape[0], max_reference_frames)
            reference_video_resized = comfy.utils.common_upscale(reference_video[:num_ref_frames].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

            # Encode just the N reference frames (no padding!)
            reference_latent = vae.encode(reference_video_resized[:, :, :, :3])
            # Apply the same channel-doubling pattern as the original
            reference_image = torch.cat([reference_latent, comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_latent))], dim=1)
        else:
            reference_image = None

        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
            if mask.shape[0] < length:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0)

        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        inactive = vae.encode(inactive[:, :, :, :3])
        reactive = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive, reactive), dim=1)

        # Prepend N reference frames to the control video (same as original, just with N frames instead of 1)
        if reference_image is not None:
            control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact').squeeze(0)

        trim_latent = 0
        if reference_image is not None:
            # Add mask padding for the N reference frames we prepended
            # reference_image.shape[2] = number of reference frames in latent space
            mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += reference_image.shape[2]
            trim_latent = reference_image.shape[2]

        mask = mask.unsqueeze(0)

        positive = node_helpers.conditioning_set_values(positive, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)

        latent = torch.zeros([batch_size, 16, latent_length, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        out_latent = {}
        out_latent["samples"] = latent

        return (positive, negative, out_latent, trim_latent)
