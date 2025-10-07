import torch
import math
import numpy as np
import torch.nn.functional as F
from ..sup import ROOT, ROOT_COMFY

# ComfyUI core imports
import node_helpers
import comfy.model_management
import comfy.utils
import comfy.latent_formats
import comfy.clip_vision

class FL_WanFirstLastFrameToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Original WAN parameters
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                
                # NEW: Timing Control
                "keyframe_start_position": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "keyframe_end_position": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "interpolation_curve": (["linear", "ease_in", "ease_out", "ease_in_out", "bounce"], {"default": "ease_in_out"}),
                "temporal_buffer": ("INT", {"default": 3, "min": 0, "max": 10}),
            },
            "optional": {
                # Original optional parameters
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT", ),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT", ),
                "start_image": ("IMAGE", ),
                "end_image": ("IMAGE", ),
                
                # NEW: Custom Mask Controls
                "interpolation_mask": ("MASK", ),
                "mask_feather": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_invert": ("BOOLEAN", {"default": False}),
                "regional_blending": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "🏵️Fill Nodes/WIP"

    def __init__(self):
        pass

    def generate_interpolation_curve(self, curve_type: str, length: int) -> torch.Tensor:
        """Generate normalized timing curve [0,1] for frame interpolation"""
        t = torch.linspace(0, 1, length)
        
        if curve_type == "linear":
            return t
        elif curve_type == "ease_in":
            return t ** 2
        elif curve_type == "ease_out":
            return 1 - (1 - t) ** 2
        elif curve_type == "ease_in_out":
            return torch.where(t < 0.5, 2 * t**2, 1 - 2 * (1-t)**2)
        elif curve_type == "bounce":
            return self._bounce_curve(t)
        else:
            return t  # fallback to linear

    def _bounce_curve(self, t: torch.Tensor) -> torch.Tensor:
        """Professional bounce interpolation for natural motion"""
        # Bounce-out easing function
        return torch.where(t < 4/11, 7.5625 * t**2,
               torch.where(t < 8/11, 7.5625 * (t - 6/11)**2 + 0.75,
               torch.where(t < 10/11, 7.5625 * (t - 9/11)**2 + 0.9375,
                          7.5625 * (t - 21/22)**2 + 0.984375)))

    def calculate_keyframe_positions(self, start_pos: float, end_pos: float, 
                                   length: int, buffer: int) -> dict:
        """Calculate actual frame indices for keyframes with temporal buffering"""
        
        # Convert normalized positions to frame indices
        start_frame = int(start_pos * (length - 1))
        end_frame = int(end_pos * (length - 1))
        
        # Ensure valid frame order
        if start_frame >= end_frame:
            start_frame = max(0, end_frame - 1)
        
        # Apply temporal buffering for smooth transitions
        buffer_start = max(0, start_frame - buffer)
        buffer_end = min(length - 1, end_frame + buffer)
        
        return {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'buffer_start': buffer_start,
            'buffer_end': buffer_end,
            'total_transition_frames': end_frame - start_frame + 1
        }

    def process_custom_mask(self, mask: torch.Tensor, feather: float, 
                           invert: bool) -> torch.Tensor:
        """Process user-provided mask with advanced controls"""
        
        # Ensure mask is properly formatted
        processed_mask = mask.clone()
        
        # Apply mask inversion if requested
        if invert:
            processed_mask = 1.0 - processed_mask
        
        # Apply feathering for smooth transitions
        if feather > 0:
            processed_mask = self.apply_mask_feathering(processed_mask, feather)
        
        return processed_mask

    def apply_mask_feathering(self, mask: torch.Tensor, feather_amount: float) -> torch.Tensor:
        """Apply Gaussian blur-based feathering to mask edges"""
        
        # Calculate kernel size based on feather amount
        kernel_size = int(feather_amount * 20) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        sigma = feather_amount * 5
        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        
        # Ensure mask has correct dimensions for conv2d
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(0)
        
        # Apply feathering
        feathered = F.conv2d(mask, 
                            kernel.unsqueeze(0).unsqueeze(0), 
                            padding=kernel_size//2)
        
        # Return to original shape
        return feathered.squeeze()

    def create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel for feathering"""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        
        g = coords**2
        g = (-g / (2 * sigma**2)).exp()
        
        g2d = g[:, None] * g[None, :]
        return g2d / g2d.sum()



    def encode(self, positive, negative, vae, width, height, length, batch_size,
               keyframe_start_position, keyframe_end_position, interpolation_curve, temporal_buffer,
               start_image=None, end_image=None, clip_vision_start_image=None, clip_vision_end_image=None,
               interpolation_mask=None, mask_feather=0.1, mask_invert=False, regional_blending=True):
        
        # If all parameters are default, use original WAN logic
        if (keyframe_start_position == 0.0 and keyframe_end_position == 1.0 and
            interpolation_curve == "linear" and interpolation_mask is None and temporal_buffer == 3):
            return self.original_wan_encode(
                positive, negative, vae, width, height, length, batch_size,
                start_image, end_image, clip_vision_start_image, clip_vision_end_image
            )
        
        # Enhanced WAN implementation with timing control
        return self.enhanced_wan_encode(
            positive, negative, vae, width, height, length, batch_size,
            keyframe_start_position, keyframe_end_position, interpolation_curve, temporal_buffer,
            start_image, end_image, clip_vision_start_image, clip_vision_end_image,
            interpolation_mask, mask_feather, mask_invert, regional_blending
        )

    def original_wan_encode(self, positive, negative, vae, width, height, length, batch_size,
                           start_image=None, end_image=None, clip_vision_start_image=None, clip_vision_end_image=None):
        """Original WAN functionality integrated natively"""
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
                            device=comfy.model_management.intermediate_device())
        
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if end_image is not None:
            end_image = comfy.utils.common_upscale(end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

        image = torch.ones((length, height, width, 3)) * 0.5
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

        if start_image is not None:
            image[:start_image.shape[0]] = start_image
            mask[:, :, :start_image.shape[0] + 3] = 0.0

        if end_image is not None:
            image[-end_image.shape[0]:] = end_image
            mask[:, :, -end_image.shape[0]:] = 0.0

        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        clip_vision_output = None
        if clip_vision_start_image is not None:
            clip_vision_output = clip_vision_start_image

        if clip_vision_end_image is not None:
            if clip_vision_output is not None:
                states = torch.cat([clip_vision_output.penultimate_hidden_states, clip_vision_end_image.penultimate_hidden_states], dim=-2)
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states
            else:
                clip_vision_output = clip_vision_end_image

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {"samples": latent}
        return (positive, negative, out_latent)

    def enhanced_wan_encode(self, positive, negative, vae, width, height, length, batch_size,
                           keyframe_start_position, keyframe_end_position, interpolation_curve, temporal_buffer,
                           start_image=None, end_image=None, clip_vision_start_image=None, clip_vision_end_image=None,
                           interpolation_mask=None, mask_feather=0.1, mask_invert=False, regional_blending=True):
        
        # Create base latent tensor (same as original WAN)
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
                            device=comfy.model_management.intermediate_device())
        
        # Process and resize input images (same as original WAN)
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1),
                                                   width, height, "bilinear", "center").movedim(1, -1)
        if end_image is not None:
            end_image = comfy.utils.common_upscale(end_image[-length:].movedim(-1, 1),
                                                 width, height, "bilinear", "center").movedim(1, -1)
        
        # Create full video sequence (gray background)
        image = torch.ones((length, height, width, 3)) * 0.5
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))
        
        # Calculate keyframe positions
        positions = self.calculate_keyframe_positions(
            keyframe_start_position, keyframe_end_position, length, temporal_buffer
        )
        
        # Generate timing curve
        timing_curve = self.generate_interpolation_curve(interpolation_curve, length)
        
        # Place start image at calculated position (not at beginning!)
        if start_image is not None:
            start_frame = positions['start_frame']
            start_frames = min(start_image.shape[0], length - start_frame)
            image[start_frame:start_frame + start_frames] = start_image[:start_frames]
            
            # Apply timing-based mask with buffer
            buffer_start = positions['buffer_start']
            buffer_end = min(start_frame + start_frames + temporal_buffer, length)
            
            # Create gradient mask based on timing curve
            for i in range(buffer_start, buffer_end):
                if i < len(timing_curve):
                    alpha = 1.0 - timing_curve[i].item()
                    mask_idx = i // 4
                    if mask_idx < mask.shape[2]:
                        mask[:, :, mask_idx] = alpha
        
        # Place end image at calculated position (not at end!)
        if end_image is not None:
            end_frame = positions['end_frame']
            end_frames = min(end_image.shape[0], end_frame + 1)
            start_idx = max(0, end_frame - end_frames + 1)
            image[start_idx:start_idx + end_frames] = end_image[-end_frames:]
            
            # Apply timing-based mask with buffer
            buffer_start = max(0, end_frame - end_frames - temporal_buffer)
            buffer_end = positions['buffer_end']
            
            # Create gradient mask based on timing curve
            for i in range(buffer_start, buffer_end):
                if i < len(timing_curve):
                    alpha = timing_curve[i].item()
                    mask_idx = i // 4
                    if mask_idx < mask.shape[2]:
                        mask[:, :, mask_idx] = alpha
        
        # Apply custom interpolation mask if provided
        if interpolation_mask is not None:
            processed_mask = self.process_custom_mask(interpolation_mask, mask_feather, mask_invert)
            # Integrate custom mask with WAN mask
            mask = self.integrate_custom_mask(mask, processed_mask, width, height)
        
        # Encode the complete video sequence
        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        
        # Apply to conditioning (same as original WAN)
        positive = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask
        })
        negative = node_helpers.conditioning_set_values(negative, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask
        })
        
        # Handle CLIP vision (same as original WAN)
        clip_vision_output = None
        if clip_vision_start_image is not None:
            clip_vision_output = clip_vision_start_image
        
        if clip_vision_end_image is not None:
            if clip_vision_output is not None:
                states = torch.cat([clip_vision_output.penultimate_hidden_states,
                                  clip_vision_end_image.penultimate_hidden_states], dim=-2)
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states
            else:
                clip_vision_output = clip_vision_end_image
        
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})
        
        out_latent = {"samples": latent}
        return (positive, negative, out_latent)
    

# Node registration info
NODE_CLASS_MAPPINGS = {
    "FL_WanFirstLastFrameToVideo": FL_WanFirstLastFrameToVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_WanFirstLastFrameToVideo": "FL Wan First Frame Last Frame"
}