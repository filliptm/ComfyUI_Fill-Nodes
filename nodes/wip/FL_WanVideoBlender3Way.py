# Standard library imports
import logging

# Third-party library imports
import torch
from comfy.utils import common_upscale

# Configure logging
try:
    from ..utils import log
except ImportError:
    log = logging.getLogger(__name__)


class FL_WanVideoBlender3Way:
    """
    Blends three input videos with cross-fade effects at the overlap regions.
    Perfect for: Input Video → Continuation Video → End Video workflows.
    All videos are automatically resized to match the first video's resolution.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "overlap_frames": ("INT", {"default": 10, "min": 1, "max": 1000, "step": 1, "tooltip": "Number of frames to blend at each transition point using cross-fade."}),
                "video_1": ("IMAGE", {"tooltip": "First video (reference resolution) - typically your input/start video."}),
                "video_2": ("IMAGE", {"tooltip": "Second video (middle/continuation) - will be resized to match video_1."}),
                "video_3": ("IMAGE", {"tooltip": "Third video (end) - will be resized to match video_1."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_video_frames",)
    FUNCTION = "blend_three_videos"
    CATEGORY = "🏵️Fill Nodes/WIP"
    DESCRIPTION = "Blends three videos with cross-fade transitions. Perfect for input → continuation → end workflows. All videos resized to match first video."

    def _resize_video(self, video, target_height, target_width):
        """Resize a batch of frames (B,H,W,C) to (target_height,target_width) using Lanczos."""
        if video.shape[1] == target_height and video.shape[2] == target_width:
            return video
        # (B, H, W, C) -> (B, C, H, W)
        video_permuted = video.permute(0, 3, 1, 2)
        resized = common_upscale(video_permuted, target_width, target_height, "lanczos", "disabled")  # (B, C, H, W)
        return resized.permute(0, 2, 3, 1)

    def _cross_fade(self, tail, head, overlap_frames):
        """Blend two tensors of shape (overlap_frames,H,W,C) using linear alpha."""
        device, dtype = tail.device, tail.dtype
        alphas = torch.linspace(0, 1, overlap_frames, device=device, dtype=dtype).view(-1, 1, 1, 1)
        blended = tail * (1 - alphas) + head * alphas
        return blended

    def blend_three_videos(self, overlap_frames, video_1, video_2, video_3):
        if video_1 is None or video_2 is None or video_3 is None:
            raise ValueError("All three videos (video_1, video_2, video_3) are required.")

        # Reference dimensions from first video
        ref_h, ref_w = video_1.shape[1:3]

        # Ensure all videos match size
        video_2_resized = self._resize_video(video_2, ref_h, ref_w)
        video_3_resized = self._resize_video(video_3, ref_h, ref_w)

        # Validate overlap frames
        if video_1.shape[0] < overlap_frames:
            raise ValueError(f"video_1 has {video_1.shape[0]} frames, less than overlap_frames={overlap_frames}.")
        if video_2_resized.shape[0] < overlap_frames * 2:
            raise ValueError(f"video_2 has {video_2_resized.shape[0]} frames, needs at least {overlap_frames * 2} frames for two overlaps.")
        if video_3_resized.shape[0] < overlap_frames:
            raise ValueError(f"video_3 has {video_3_resized.shape[0]} frames, less than overlap_frames={overlap_frames}.")

        # ========================================
        # First Blend: Video 1 → Video 2
        # ========================================
        # Extract overlap region from video_1 tail and video_2 head
        tail_1 = video_1[-overlap_frames:]
        head_2 = video_2_resized[:overlap_frames]
        blend_1_2 = self._cross_fade(tail_1, head_2, overlap_frames)

        # Assemble video_1 + blend + video_2_middle
        partial_result = torch.cat([
            video_1[:-overlap_frames],      # Video 1 without last overlap_frames
            blend_1_2,                       # Blended transition
            video_2_resized[overlap_frames:] # Video 2 without first overlap_frames
        ], dim=0)

        log.info(f"First blend complete: video_1 ({video_1.shape[0]} frames) + video_2 ({video_2_resized.shape[0]} frames) = {partial_result.shape[0]} frames")

        # ========================================
        # Second Blend: Partial Result → Video 3
        # ========================================
        # Extract overlap region from partial_result tail and video_3 head
        tail_partial = partial_result[-overlap_frames:]
        head_3 = video_3_resized[:overlap_frames]
        blend_2_3 = self._cross_fade(tail_partial, head_3, overlap_frames)

        # Assemble final video
        final_video = torch.cat([
            partial_result[:-overlap_frames],  # Partial result without last overlap_frames
            blend_2_3,                          # Blended transition
            video_3_resized[overlap_frames:]    # Video 3 without first overlap_frames
        ], dim=0)

        log.info(f"Second blend complete: partial ({partial_result.shape[0]} frames) + video_3 ({video_3_resized.shape[0]} frames) = {final_video.shape[0]} frames")
        log.info(f"Final blended video: {final_video.shape[0]} frames total")

        return (final_video.cpu().float(),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "FL_WanVideoBlender3Way": FL_WanVideoBlender3Way
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_WanVideoBlender3Way": "FL Wan Video Blender 3-Way"
}
