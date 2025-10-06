# FL_Audio_Beat_Visualizer: Generate beat visualization frames from sequence JSON
import torch
import numpy as np
from typing import Tuple


class FL_Audio_Beat_Visualizer:
    """
    A ComfyUI node for generating beat visualization frames.
    Creates frames that alternate between black and white on beat switches.
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate_frames"
    CATEGORY = "🏵️Fill Nodes/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence_json": ("STRING", {"description": "Sequence JSON from FL_Audio_Music_Video_Sequencer"}),
            },
            "optional": {
                "images": ("IMAGE", {"description": "Batch of images to cycle through on beats"}),
                "use_black_white": ("BOOLEAN", {
                    "default": True,
                    "description": "Use black/white switching instead of images"
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "description": "Frame width (images will be resized to this)"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "description": "Frame height (images will be resized to this)"
                }),
                "start_with_white": ("BOOLEAN", {
                    "default": False,
                    "description": "Start with white instead of black (only for black/white mode)"
                }),
            }
        }

    def generate_frames(
        self,
        sequence_json: str,
        images: torch.Tensor = None,
        use_black_white: bool = True,
        width: int = 512,
        height: int = 512,
        start_with_white: bool = False
    ) -> Tuple[torch.Tensor]:
        """
        Generate beat visualization frames

        Args:
            sequence_json: JSON string from FL_Audio_Music_Video_Sequencer
            images: Optional batch of images to cycle through
            use_black_white: Use black/white switching instead of images
            width: Frame width in pixels
            height: Frame height in pixels
            start_with_white: Start with white instead of black

        Returns:
            Tuple containing tensor of frames (batch, height, width, channels)
        """
        print(f"\n{'='*60}")
        print(f"[FL Audio Beat Visualizer] DEBUG: Function called")
        print(f"[FL Audio Beat Visualizer] DEBUG: Use black/white mode = {use_black_white}")
        print(f"[FL Audio Beat Visualizer] DEBUG: Frame size = {width}x{height}")
        print(f"[FL Audio Beat Visualizer] DEBUG: Start with white = {start_with_white}")
        if images is not None:
            print(f"[FL Audio Beat Visualizer] DEBUG: Images batch shape = {images.shape}")
        print(f"{'='*60}\n")

        try:
            # Parse sequence JSON
            import json
            try:
                sequence_data = json.loads(sequence_json)
                shots = sequence_data['shots']
                metadata = sequence_data['metadata']
                total_shots = len(shots)
                total_frames = metadata['total_frames']

                print(f"[FL Audio Beat Visualizer] DEBUG: Total shots = {total_shots}")
                print(f"[FL Audio Beat Visualizer] DEBUG: Total frames = {total_frames}")

            except (json.JSONDecodeError, KeyError) as e:
                error_msg = f"Error parsing sequence JSON: {e}"
                print(f"[FL Audio Beat Visualizer] ERROR: {error_msg}")
                # Return single black frame on error
                return (torch.zeros((1, height, width, 3), dtype=torch.float32),)

            # Calculate expected memory usage
            expected_memory_mb = (total_frames * height * width * 3 * 4) / (1024 * 1024)
            print(f"\n[FL Audio Beat Visualizer] MEMORY DEBUG:")
            print(f"[FL Audio Beat Visualizer]   Total frames to generate: {total_frames}")
            print(f"[FL Audio Beat Visualizer]   Frame dimensions: {height}x{width}x3")
            print(f"[FL Audio Beat Visualizer]   Expected memory usage: {expected_memory_mb:.2f} MB")
            print(f"[FL Audio Beat Visualizer]   Data type: float32 (4 bytes per value)")

            # Create frame buffer
            frames = []

            # Generate frames for each shot
            if use_black_white:
                # Black/white alternating mode
                for shot_idx, shot in enumerate(shots):
                    frame_count = shot['frame_count']

                    # Determine color for this shot (alternate between black and white)
                    if start_with_white:
                        is_white = (shot_idx % 2 == 0)
                    else:
                        is_white = (shot_idx % 2 == 1)

                    # Create color value (0.0 for black, 1.0 for white)
                    color_value = 1.0 if is_white else 0.0

                    # Create frames for this shot
                    shot_frames = torch.full(
                        (frame_count, height, width, 3),
                        color_value,
                        dtype=torch.float32
                    )

                    frames.append(shot_frames)

                    print(f"[FL Audio Beat Visualizer] Shot {shot_idx}: {frame_count} frames, color={'white' if is_white else 'black'}")

            else:
                # Image cycling mode
                if images is None:
                    print(f"[FL Audio Beat Visualizer] WARNING: use_black_white=False but no images provided, falling back to black frames")
                    for shot_idx, shot in enumerate(shots):
                        frame_count = shot['frame_count']
                        shot_frames = torch.zeros((frame_count, height, width, 3), dtype=torch.float32)
                        frames.append(shot_frames)
                else:
                    num_images = images.shape[0]
                    original_height, original_width = images.shape[1], images.shape[2]
                    print(f"[FL Audio Beat Visualizer] DEBUG: Cycling through {num_images} images")
                    print(f"[FL Audio Beat Visualizer] DEBUG: Original image size: {original_width}x{original_height}")
                    print(f"[FL Audio Beat Visualizer] DEBUG: Resizing to: {width}x{height}")

                    # Resize images to target dimensions
                    import torch.nn.functional as F
                    # Permute from (batch, height, width, channels) to (batch, channels, height, width)
                    images_resized = images.permute(0, 3, 1, 2)
                    # Resize
                    images_resized = F.interpolate(images_resized, size=(height, width), mode='bilinear', align_corners=False)
                    # Permute back to (batch, height, width, channels)
                    images_resized = images_resized.permute(0, 2, 3, 1)

                    for shot_idx, shot in enumerate(shots):
                        frame_count = shot['frame_count']

                        # Cycle through images
                        image_idx = shot_idx % num_images
                        selected_image = images_resized[image_idx]  # Shape: (height, width, channels)

                        # Repeat the selected image for all frames in this shot
                        shot_frames = selected_image.unsqueeze(0).repeat(frame_count, 1, 1, 1)

                        frames.append(shot_frames)

                        print(f"[FL Audio Beat Visualizer] Shot {shot_idx}: {frame_count} frames, using image {image_idx}")

            # Concatenate all frames
            all_frames = torch.cat(frames, dim=0)

            # Calculate actual memory usage
            actual_memory_mb = (all_frames.element_size() * all_frames.nelement()) / (1024 * 1024)

            print(f"\n{'='*60}")
            print(f"[FL Audio Beat Visualizer] Frame generation complete!")
            print(f"[FL Audio Beat Visualizer] Total frames generated: {all_frames.shape[0]}")
            print(f"[FL Audio Beat Visualizer] Frame shape: {all_frames.shape}")
            print(f"[FL Audio Beat Visualizer] Frame dtype: {all_frames.dtype}")
            print(f"[FL Audio Beat Visualizer] Actual tensor memory: {actual_memory_mb:.2f} MB")
            print(f"[FL Audio Beat Visualizer] Element size: {all_frames.element_size()} bytes")
            print(f"{'='*60}\n")

            return (all_frames,)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL Audio Beat Visualizer] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            # Return single black frame on error
            return (torch.zeros((1, height, width, 3), dtype=torch.float32),)
