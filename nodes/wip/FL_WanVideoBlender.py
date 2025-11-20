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


class FL_WanVideoBlender:
    """
    Blends two input videos with a cross-fade effect in the overlap region.
    The resolution of the second clip is automatically resized to match the first.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "overlap_frames": ("INT", {"default": 10, "min": 1, "max": 1000, "step": 1, "tooltip": "Number of frames to blend between the two videos using cross-fade."}),
                "video_1": ("IMAGE", {"tooltip": "First video (reference resolution)."}),
                "video_2": ("IMAGE", {"tooltip": "Second video (will be resized to match video_1)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_video_frames",)
    FUNCTION = "blend_videos"
    CATEGORY = "🏵️Fill Nodes/WIP"
    DESCRIPTION = "Blends two input videos with a cross-fade. The resolution of the second clip is resized to match the first."

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

    def blend_videos(self, overlap_frames, video_1, video_2):
        if video_1 is None or video_2 is None:
            raise ValueError("Both video_1 and video_2 are required.")

        # Reference dimensions and properties from first video
        ref_h, ref_w = video_1.shape[1:3]

        # Ensure second video matches size
        video_2_resized = self._resize_video(video_2, ref_h, ref_w)

        if video_1.shape[0] < overlap_frames or video_2_resized.shape[0] < overlap_frames:
            raise ValueError(f"One of the videos is shorter than overlap_frames={overlap_frames}.")

        # Extract segments for blending
        tail = video_1[-overlap_frames:]
        head = video_2_resized[:overlap_frames]
        blended = self._cross_fade(tail, head, overlap_frames)

        # Assemble new timeline
        final_video = torch.cat([
            video_1[:-overlap_frames],
            blended,
            video_2_resized[overlap_frames:]
        ], dim=0)

        return (final_video.cpu().float(),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "FL_WanVideoBlender": FL_WanVideoBlender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_WanVideoBlender": "FL Wan Video Blender"
}
