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


class FL_WanVideoContinuationBlender:
    """
    Specialized blender for videos created by FL_WanVideoContinue.
    Automatically strips the embedded overlap frames from the continuation video
    before blending to prevent duplication.

    Perfect for creating seamless loops: Video A → Continuation → Video B (or back to A)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "overlap_frames": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1, "tooltip": "Number of overlap frames used in FL_WanVideoContinue. Must match the value used to create the continuation video."}),
                "video_1": ("IMAGE", {"tooltip": "First video (reference resolution) - the 'input_video_frames' from FL_WanVideoContinue."}),
                "continuation_video": ("IMAGE", {"tooltip": "The rendered continuation video from FL_WanVideoContinue - has embedded overlaps that will be automatically stripped."}),
                "video_2": ("IMAGE", {"tooltip": "Second video - the 'end_video_frames' from FL_WanVideoContinue. Will be resized to match video_1."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_video_frames",)
    FUNCTION = "blend_continuation"
    CATEGORY = "🏵️Fill Nodes/WIP"
    DESCRIPTION = "Blends videos with a continuation video that has embedded overlaps. Automatically strips overlap frames to prevent duplication. Use this after FL_WanVideoContinue."

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

    def blend_continuation(self, overlap_frames, video_1, continuation_video, video_2):
        if video_1 is None or continuation_video is None or video_2 is None:
            raise ValueError("All three videos (video_1, continuation_video, video_2) are required.")

        # Reference dimensions from first video
        ref_h, ref_w = video_1.shape[1:3]

        # Ensure video_2 matches size
        video_2_resized = self._resize_video(video_2, ref_h, ref_w)

        # Validate continuation video has enough frames for overlaps
        continuation_frames = continuation_video.shape[0]
        expected_min_frames = overlap_frames * 2  # Need at least start + end overlaps

        if continuation_frames < expected_min_frames:
            raise ValueError(f"Continuation video has {continuation_frames} frames but needs at least {expected_min_frames} frames (overlap_frames={overlap_frames} × 2).")

        # ========================================
        # Strip Overlap Frames from Continuation
        # ========================================
        # The continuation video structure is:
        # [overlap_frames from video_1 end] + [middle generated frames] + [overlap_frames from video_2 start]
        # We only want the middle section for blending

        continuation_middle = continuation_video[overlap_frames:-overlap_frames]
        middle_frame_count = continuation_middle.shape[0]

        log.info(f"Stripped {overlap_frames} frames from start and end of continuation video.")
        log.info(f"Using middle {middle_frame_count} frames from continuation (original: {continuation_frames} frames).")

        # Resize continuation if needed
        continuation_middle_resized = self._resize_video(continuation_middle, ref_h, ref_w)

        # Validate overlap frames
        if video_1.shape[0] < overlap_frames:
            raise ValueError(f"video_1 has {video_1.shape[0]} frames, less than overlap_frames={overlap_frames}.")
        if video_2_resized.shape[0] < overlap_frames:
            raise ValueError(f"video_2 has {video_2_resized.shape[0]} frames, less than overlap_frames={overlap_frames}.")

        # ========================================
        # First Blend: Video 1 → Continuation Middle
        # ========================================
        tail_1 = video_1[-overlap_frames:]
        head_cont = continuation_middle_resized[:overlap_frames]
        blend_1_cont = self._cross_fade(tail_1, head_cont, overlap_frames)

        # Assemble: video_1 (without last overlap) + blend + continuation_middle (without first overlap)
        partial_result = torch.cat([
            video_1[:-overlap_frames],
            blend_1_cont,
            continuation_middle_resized[overlap_frames:]
        ], dim=0)

        log.info(f"First blend complete: video_1 ({video_1.shape[0]} frames) + continuation_middle ({middle_frame_count} frames) = {partial_result.shape[0]} frames")

        # ========================================
        # Second Blend: Partial Result → Video 2
        # ========================================
        tail_partial = partial_result[-overlap_frames:]
        head_2 = video_2_resized[:overlap_frames]
        blend_cont_2 = self._cross_fade(tail_partial, head_2, overlap_frames)

        # Assemble final video
        final_video = torch.cat([
            partial_result[:-overlap_frames],
            blend_cont_2,
            video_2_resized[overlap_frames:]
        ], dim=0)

        log.info(f"Second blend complete: partial ({partial_result.shape[0]} frames) + video_2 ({video_2_resized.shape[0]} frames) = {final_video.shape[0]} frames")
        log.info(f"Final blended video: {final_video.shape[0]} frames total")

        # ========================================
        # Frame Count Verification
        # ========================================
        expected_frames = video_1.shape[0] + middle_frame_count + video_2_resized.shape[0] - (2 * overlap_frames)
        actual_frames = final_video.shape[0]

        if expected_frames != actual_frames:
            log.warning(f"Frame count mismatch! Expected {expected_frames}, got {actual_frames}")
        else:
            log.info(f"✓ Frame count verified: {actual_frames} frames (as expected)")

        return (final_video.cpu().float(),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "FL_WanVideoContinuationBlender": FL_WanVideoContinuationBlender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_WanVideoContinuationBlender": "FL Wan Video Continuation Blender"
}
