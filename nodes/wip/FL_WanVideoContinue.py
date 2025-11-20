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


class FL_WanVideoContinue:
    """
    Creates a continuation video by placing overlap frames from the end of input video at the start,
    and overlap frames from the start of end video at the end, with optional control images for guided generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_video_frames": ("IMAGE", {"tooltip": "Input video frames to create continuation from."}),
                "total_output_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Total number of frames for the output continuation video. Must satisfy: (frames - 1) divisible by 4."}),
                "overlap_frames": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1, "tooltip": "Number of frames to use for overlap at both start (from end of input video) and end (from start of end video frames)."}),
                "empty_frame_fill_level": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Grayscale level (0.0 black, 1.0 white) for empty continuation frames."}),
            },
            "optional": {
                "end_video_frames": ("IMAGE", {"tooltip": "Optional video frames - first X frames (where X=overlap_frames) will be placed at the end of the continuation video."}),
                "control_images": ("IMAGE", {"tooltip": "Optional control images to fill the empty frames."}),
                "inpaint_mask": ("MASK", {"tooltip": "Optional inpaint mask to use for the empty frames, overriding the default mask."}),
                "how_to_use_control_images": (["start_sequence_at_beginning_and_prioritise_input_frames", "start_sequence_after_overlap_frames_and_prioritise_input_frames"], {"default": "start_sequence_at_beginning_and_prioritise_input_frames", "tooltip": "If start_sequence_at_beginning_and_prioritise_input_frames is selected, control images align with frame 0 but input overlap frames take priority, so control images become visible after the overlap period. If start_sequence_after_overlap_frames_and_prioritise_input_frames is selected, control images start being placed after the overlap frames from the input video."}),
                "how_to_use_inpaint_masks": (["start_sequence_at_beginning_and_prioritise_input_frames", "start_sequence_after_overlap_frames_and_prioritise_input_frames"], {"default": "start_sequence_at_beginning_and_prioritise_input_frames", "tooltip": "If start_sequence_at_beginning_and_prioritise_input_frames is selected, inpaint masks align with frame 0 but preserve input overlap frames as known. If start_sequence_after_overlap_frames_and_prioritise_input_frames is selected, inpaint masks only affect frames after the overlap period."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("continuation_video_frames", "continuation_frame_masks",)
    FUNCTION = "generate_continuation_video"
    CATEGORY = "🏵️Fill Nodes/WIP"
    DESCRIPTION = "Creates a continuation video with symmetrical overlap - uses last N frames from input video at start and first N frames from end video at end."

    def generate_continuation_video(self, input_video_frames, total_output_frames, overlap_frames, empty_frame_fill_level, end_video_frames=None, control_images=None, inpaint_mask=None, how_to_use_control_images="start_sequence_at_beginning_and_prioritise_input_frames", how_to_use_inpaint_masks="start_sequence_at_beginning_and_prioritise_input_frames"):
        # 1. Validation and Setup
        total_output_frames = int(total_output_frames)
        if (total_output_frames - 1) % 4 != 0:
            raise ValueError("total_output_frames must satisfy (frames - 1) divisible by 4")

        if input_video_frames is None or input_video_frames.shape[0] == 0:
            log.error("Input video_frames is empty. Cannot proceed.")
            dummy_height, dummy_width, dummy_channels = 64, 64, 3
            return (torch.zeros((total_output_frames, dummy_height, dummy_width, dummy_channels), dtype=torch.float32),
                    torch.ones((total_output_frames, dummy_height, dummy_width), dtype=torch.float32))

        device = input_video_frames.device
        dtype = input_video_frames.dtype
        batch_size_input, frame_height, frame_width, num_channels = input_video_frames.shape

        # 2. Prepare Start Frames (from overlap)
        actual_overlap_frames = min(overlap_frames, batch_size_input, total_output_frames)
        if actual_overlap_frames < overlap_frames:
            log.warning(f"Requested {overlap_frames} overlap frames but input video only has {batch_size_input} frames or total output is smaller. Using {actual_overlap_frames} instead.")

        overlap_start_idx = batch_size_input - actual_overlap_frames
        start_frames_part = input_video_frames[overlap_start_idx : overlap_start_idx + actual_overlap_frames].clone()

        # 3. Prepare End Frames (using same overlap count as start)
        end_frames_part = torch.empty((0, frame_height, frame_width, num_channels), device=device, dtype=dtype)
        num_end_frames = 0
        if end_video_frames is not None and end_video_frames.shape[0] > 0 and total_output_frames > actual_overlap_frames:
            # Use the same overlap_frames count for end frames
            num_end_frames = min(actual_overlap_frames, end_video_frames.shape[0], total_output_frames - actual_overlap_frames)

            # Take first num_end_frames from end_video_frames
            end_frames_selected = end_video_frames[:num_end_frames].clone().to(device=device, dtype=dtype)

            # Resize if dimensions don't match
            if end_frames_selected.shape[1:] != (frame_height, frame_width, num_channels):
                log.info(f"Resizing end_video_frames from {end_frames_selected.shape[1:]} to {(frame_height, frame_width, num_channels)}.")
                end_frames_resized_list = []
                for i in range(end_frames_selected.shape[0]):
                    frame_to_resize = end_frames_selected[i].unsqueeze(0).permute(0, 3, 1, 2)
                    resized_frame = common_upscale(frame_to_resize, frame_width, frame_height, "lanczos", "disabled")
                    end_frames_resized_list.append(resized_frame.permute(0, 2, 3, 1))
                end_frames_part = torch.cat(end_frames_resized_list, dim=0)
            else:
                end_frames_part = end_frames_selected

            log.info(f"Using {num_end_frames} frames from end_video_frames for end overlap.")

        # 4. Prepare Middle Frames
        num_middle_frames = total_output_frames - actual_overlap_frames - num_end_frames
        middle_frames_part = torch.empty((0, frame_height, frame_width, num_channels), device=device, dtype=dtype)

        if num_middle_frames > 0:
            if control_images is not None:
                log.info(f"Using 'control_images' to fill the {num_middle_frames} middle frames with '{how_to_use_control_images}' mode.")
                control_images_resized = common_upscale(control_images.movedim(-1, 1), frame_width, frame_height, "lanczos", "disabled").movedim(1, -1)

                if how_to_use_control_images == "start_sequence_at_beginning_and_prioritise_input_frames":
                    # Skip the first overlap_frames control images to avoid duplication
                    duplicate_count = min(actual_overlap_frames, control_images_resized.shape[0])
                    available_after_dup = control_images_resized.shape[0] - duplicate_count
                    if available_after_dup < num_middle_frames:
                        log.info(f"After skipping {duplicate_count} control frames, only {available_after_dup} remain; padding {num_middle_frames - available_after_dup} frames with 'empty_frame_fill_level'.")
                        selected_control = control_images_resized[duplicate_count:]
                        padding_needed = num_middle_frames - selected_control.shape[0]
                        padding = torch.ones((padding_needed, frame_height, frame_width, num_channels), device=device, dtype=dtype) * empty_frame_fill_level
                        middle_frames_part = torch.cat([selected_control, padding], dim=0)
                    else:
                        middle_frames_part = control_images_resized[duplicate_count:duplicate_count + num_middle_frames].clone()
                else:  # "start_sequence_after_overlap_frames_and_prioritise_input_frames"
                    # Use control frames from the beginning of the sequence (C0, C1, C2...)
                    if control_images_resized.shape[0] < num_middle_frames:
                        log.warning(f"Provided 'control_images' have {control_images_resized.shape[0]} frames, less than needed ({num_middle_frames}). Padding with 'empty_frame_fill_level'.")
                        padding_needed = num_middle_frames - control_images_resized.shape[0]
                        padding = torch.ones((padding_needed, frame_height, frame_width, num_channels), device=device, dtype=dtype) * empty_frame_fill_level
                        middle_frames_part = torch.cat([control_images_resized, padding], dim=0)
                    else:
                        middle_frames_part = control_images_resized[:num_middle_frames].clone()
            else:
                log.info(f"No 'control_images', filling {num_middle_frames} middle frames with level {empty_frame_fill_level}.")
                middle_frames_part = torch.ones((num_middle_frames, frame_height, frame_width, num_channels), device=device, dtype=dtype) * empty_frame_fill_level

        # 5. Assemble Final Video
        continuation_video_output = torch.cat([start_frames_part, middle_frames_part, end_frames_part], dim=0)

        # 6. Create Mask
        continuation_frame_masks = torch.ones((total_output_frames, frame_height, frame_width), device=device, dtype=dtype)

        # Apply mask logic based on how_to_use_inpaint_masks parameter
        if how_to_use_inpaint_masks == "start_sequence_at_beginning_and_prioritise_input_frames":
            # Set known frames (overlap and end) to 0.0, but also set middle section based on control frame logic
            if actual_overlap_frames > 0:
                continuation_frame_masks[0:actual_overlap_frames] = 0.0
            if num_end_frames > 0:
                continuation_frame_masks[-num_end_frames:] = 0.0

            # For middle section, follow the same logic as control frames
            if control_images is not None and num_middle_frames > 0:
                duplicate_count = min(actual_overlap_frames, control_images.shape[0])
                available_after_dup = control_images.shape[0] - duplicate_count
                if available_after_dup >= num_middle_frames:
                    # If we have enough control frames after skipping, set those middle frames as known (0.0)
                    middle_start = actual_overlap_frames
                    middle_end = middle_start + num_middle_frames
                    continuation_frame_masks[middle_start:middle_end] = 0.0
        else:  # "start_sequence_after_overlap_frames_and_prioritise_input_frames"
            # Set known frames (overlap and end) to 0.0, rest stay as 1.0 (inpaint)
            if actual_overlap_frames > 0:
                continuation_frame_masks[0:actual_overlap_frames] = 0.0
            if num_end_frames > 0:
                continuation_frame_masks[-num_end_frames:] = 0.0

        # 7. Handle optional inpaint_mask with how_to_use_inpaint_masks logic
        if inpaint_mask is not None:
            log.info(f"Processing provided 'inpaint_mask' with '{how_to_use_inpaint_masks}' timing.")
            processed_mask = common_upscale(inpaint_mask.unsqueeze(1), frame_width, frame_height, "nearest-exact", "disabled").squeeze(1).to(device)

            if processed_mask.shape[0] != total_output_frames:
                log.info(f"Adjusting inpaint_mask frame count from {processed_mask.shape[0]} to {total_output_frames}.")
                if processed_mask.shape[0] < total_output_frames:
                    num_repeats = (total_output_frames + processed_mask.shape[0] - 1) // processed_mask.shape[0]
                    processed_mask = processed_mask.repeat(num_repeats, 1, 1)[:total_output_frames]
                else:
                    processed_mask = processed_mask[:total_output_frames]

            # Apply how_to_use_inpaint_masks logic to the provided mask
            if how_to_use_inpaint_masks == "start_sequence_at_beginning_and_prioritise_input_frames":
                # Use the provided mask as-is, but preserve known frames (overlap and end)
                if actual_overlap_frames > 0:
                    processed_mask[0:actual_overlap_frames] = 0.0  # Keep overlap frames as known
                if num_end_frames > 0:
                    processed_mask[-num_end_frames:] = 0.0  # Keep end frame as known
            else:  # "start_sequence_after_overlap_frames_and_prioritise_input_frames"
                # Only apply the provided mask after overlap frames
                if actual_overlap_frames > 0:
                    processed_mask[0:actual_overlap_frames] = 0.0  # Keep overlap frames as known
                    # The provided mask affects frames starting after overlap
                if num_end_frames > 0:
                    processed_mask[-num_end_frames:] = 0.0  # Keep end frame as known

            continuation_frame_masks = processed_mask.to(dtype=dtype)

        log.info(f"Generated continuation video. Start: {actual_overlap_frames} frames, Middle: {num_middle_frames} frames, End: {num_end_frames} frames.")

        return (continuation_video_output.cpu().float(), continuation_frame_masks.cpu().float())


# Node registration
NODE_CLASS_MAPPINGS = {
    "FL_WanVideoContinue": FL_WanVideoContinue
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_WanVideoContinue": "FL Wan Video Continue"
}
