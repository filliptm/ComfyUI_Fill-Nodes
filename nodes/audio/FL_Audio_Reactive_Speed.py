# FL_Audio_Reactive_Speed: Time remapping based on audio envelope
import torch
import torch.nn.functional as F
import json
import numpy as np
from PIL import Image
from typing import Tuple


class FL_Audio_Reactive_Speed:
    """
    A ComfyUI node for applying audio-reactive speed/time remapping to frames.
    Speeds up or slows down playback based on envelope values from drum detection.
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "apply_speed"
    CATEGORY = "🏵️Fill Nodes/Audio"

    def t2p(self, t):
        """Tensor to PIL"""
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return None

    def prepare_mask_batch(self, mask, total_images):
        """Prepare mask batch to match image batch size"""
        if mask is None:
            return None
        mask_images = [self.t2p(m) for m in mask]
        if len(mask_images) < total_images:
            mask_images = mask_images * (total_images // len(mask_images) + 1)
        return mask_images[:total_images]

    def process_mask(self, mask, target_size):
        """Resize and convert mask to grayscale"""
        mask = mask.resize(target_size, Image.LANCZOS)
        return mask.convert('L') if mask.mode != 'L' else mask

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"description": "Input frames"}),
                "envelope_json": ("STRING", {"description": "Envelope JSON from FL_Audio_Reactive_Envelope"}),
            },
            "optional": {
                "mask": ("IMAGE", {
                    "default": None,
                    "description": "Optional mask to control where effect is applied"
                }),
                "base_speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "description": "Base playback speed (1.0 = normal, 0.0 = freeze)"
                }),
                "speed_intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": -3.0,
                    "max": 3.0,
                    "step": 0.01,
                    "description": "Speed intensity (positive = faster on hits, negative = slower)"
                }),
                "interpolation": (["bilinear", "bicubic", "nearest"], {
                    "default": "bilinear",
                    "description": "Frame interpolation mode"
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "description": "Invert envelope (slow down on hits instead of speed up)"
                }),
            }
        }

    def apply_speed(
        self,
        frames: torch.Tensor,
        envelope_json: str,
        mask=None,
        base_speed: float = 1.0,
        speed_intensity: float = 0.5,
        interpolation: str = "bilinear",
        invert: bool = False
    ) -> Tuple[torch.Tensor]:
        """
        Apply audio-reactive speed/time remapping to frames

        Args:
            frames: Input frames tensor (batch, height, width, channels)
            envelope_json: JSON string with envelope data
            mask: Optional mask to control where effect is applied
            base_speed: Base playback speed (1.0 = normal)
            speed_intensity: How much envelope affects speed
            interpolation: Frame interpolation mode
            invert: Slow down on hits instead of speed up

        Returns:
            Tuple containing time-remapped frames
        """
        print(f"\n{'='*60}")
        print(f"[FL Audio Reactive Speed] DEBUG: Function called")
        print(f"[FL Audio Reactive Speed] DEBUG: Input frames shape = {frames.shape}")
        print(f"[FL Audio Reactive Speed] DEBUG: Base speed = {base_speed}")
        print(f"[FL Audio Reactive Speed] DEBUG: Speed intensity = {speed_intensity}")
        print(f"[FL Audio Reactive Speed] DEBUG: Invert = {invert}")
        print(f"{'='*60}\n")

        try:
            # Parse envelope JSON
            envelope_data = json.loads(envelope_json)
            envelope = envelope_data['envelope']

            batch_size, height, width, channels = frames.shape
            num_envelope_frames = len(envelope)

            print(f"[FL Audio Reactive Speed] Input frames: {batch_size}")
            print(f"[FL Audio Reactive Speed] Envelope frames: {num_envelope_frames}")

            # Use the envelope frame count as target output
            target_frames = num_envelope_frames

            # Prepare mask batch if provided
            mask_images = self.prepare_mask_batch(mask, target_frames) if mask is not None else None

            # Build time remapping based on envelope
            # Calculate speed multiplier for each frame based on envelope
            speed_multipliers = []
            for env_val in envelope:
                if invert:
                    # Invert: high envelope = slow down
                    speed = base_speed - (env_val * speed_intensity)
                else:
                    # Normal: high envelope = speed up
                    speed = base_speed + (env_val * speed_intensity)

                # Clamp to reasonable range (0.0 allows freeze frame)
                speed = max(0.0, min(speed, 3.0))
                speed_multipliers.append(speed)

            # Build cumulative time mapping for speed ramping
            # Higher speed = advance more frames, lower speed = advance less/freeze
            # This creates actual speed-up/slow-down effect without normalizing
            source_positions = []
            current_source_pos = 0.0

            for speed in speed_multipliers:
                source_positions.append(current_source_pos)
                # Advance through source based on speed
                # speed=1.0 means advance 1 frame, speed=2.0 means skip ahead 2 frames
                # speed=0.5 means advance 0.5 frames (slow motion)
                # speed=0.0 means freeze frame
                current_source_pos += speed

            # Clamp all positions to valid source frame range
            # Don't normalize - let speed actually control playback rate
            source_positions = [min(pos, batch_size - 1.0) for pos in source_positions]

            print(f"[FL Audio Reactive Speed] Source position range: {min(source_positions):.2f} to {max(source_positions):.2f}")
            print(f"[FL Audio Reactive Speed] Source frames available: {batch_size}")
            print(f"[FL Audio Reactive Speed] Output frames: {target_frames}")
            print(f"[FL Audio Reactive Speed] Speed range: {min(speed_multipliers):.3f} to {max(speed_multipliers):.3f}")

            # Check if we'll run out of frames
            final_pos = source_positions[-1] if source_positions else 0
            if final_pos >= batch_size - 1:
                print(f"[FL Audio Reactive Speed] WARNING: Ran out of source frames at position {final_pos:.1f}/{batch_size-1} - last frames will repeat")

            # Sample frames based on time remapping
            output_frames = []

            for frame_idx, source_pos in enumerate(source_positions):
                # Get fractional frame position
                source_frame_low = int(np.floor(source_pos))
                source_frame_high = int(np.ceil(source_pos))
                blend_factor = source_pos - source_frame_low

                # Clamp to valid range
                source_frame_low = max(0, min(source_frame_low, batch_size - 1))
                source_frame_high = max(0, min(source_frame_high, batch_size - 1))

                # Get frames
                frame_low = frames[source_frame_low]
                frame_high = frames[source_frame_high]

                # Interpolate between frames if needed
                if source_frame_low != source_frame_high and interpolation != 'nearest':
                    blended_frame = frame_low * (1 - blend_factor) + frame_high * blend_factor
                else:
                    # Nearest neighbor or same frame
                    blended_frame = frame_low if blend_factor < 0.5 else frame_high

                # Apply mask if provided
                # For speed effect, mask determines where time-remapping is applied
                # mask=1.0 shows time-remapped frame, mask=0.0 shows original timeline frame
                if mask_images is not None:
                    # Get the original frame at the current output timeline position
                    # Clamp frame_idx to available input frames
                    original_frame_idx = min(frame_idx, batch_size - 1)
                    original_frame = frames[original_frame_idx]

                    # Convert mask to tensor
                    mask_img = self.process_mask(mask_images[frame_idx], (width, height))
                    mask_array = np.array(mask_img).astype(np.float32) / 255.0

                    # Expand mask to 3 channels
                    if len(mask_array.shape) == 2:
                        mask_tensor = torch.from_numpy(mask_array).unsqueeze(-1).repeat(1, 1, 3)
                    else:
                        mask_tensor = torch.from_numpy(mask_array)

                    # Blend: mask=1.0 shows time-remapped effect, mask=0.0 shows original timeline
                    blended_frame = original_frame * (1.0 - mask_tensor) + blended_frame * mask_tensor

                output_frames.append(blended_frame)

                if frame_idx % 100 == 0 or frame_idx < 5:
                    print(f"[FL Audio Reactive Speed] Frame {frame_idx}: envelope={envelope[frame_idx]:.3f}, speed={speed_multipliers[frame_idx]:.3f}, source_pos={source_pos:.2f}")

            # Stack all frames
            output_tensor = torch.stack(output_frames, dim=0)

            print(f"\n{'='*60}")
            print(f"[FL Audio Reactive Speed] Processing complete!")
            print(f"[FL Audio Reactive Speed] Input frames: {batch_size}")
            print(f"[FL Audio Reactive Speed] Output frames: {output_tensor.shape[0]}")
            print(f"[FL Audio Reactive Speed] Output shape: {output_tensor.shape}")
            print(f"{'='*60}\n")

            return (output_tensor,)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL Audio Reactive Speed] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return (frames,)
