# FL_Audio_Reactive_Brightness: Control brightness/luminance based on audio envelope
import torch
import numpy as np
import json
from PIL import Image
from typing import Tuple


class FL_Audio_Reactive_Brightness:
    """
    A ComfyUI node for applying audio-reactive brightness/luminance changes to frames.
    Adjusts brightness based on envelope values from drum detection.
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "apply_brightness"
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
                "base_brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "description": "Base brightness multiplier (1.0 = normal)"
                }),
                "brightness_intensity": ("FLOAT", {
                    "default": 0.15,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "Brightness intensity (positive = brighter on hits, negative = darker)"
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "description": "Invert envelope (darken on hits instead of brighten)"
                }),
                "clamp_output": ("BOOLEAN", {
                    "default": True,
                    "description": "Clamp output to valid range [0,1] to prevent overexposure"
                }),
            }
        }

    def apply_brightness(
        self,
        frames: torch.Tensor,
        envelope_json: str,
        mask=None,
        base_brightness: float = 1.0,
        brightness_intensity: float = 0.15,
        invert: bool = False,
        clamp_output: bool = True
    ) -> Tuple[torch.Tensor]:
        """
        Apply audio-reactive brightness effect to frames

        Args:
            frames: Input frames tensor (batch, height, width, channels)
            envelope_json: JSON string with envelope data
            base_brightness: Base brightness multiplier (1.0 = normal)
            brightness_intensity: How much envelope affects brightness
            invert: Darken on hits instead of brighten
            clamp_output: Clamp to [0,1] range

        Returns:
            Tuple containing brightness-adjusted frames
        """
        print(f"\n{'='*60}")
        print(f"[FL Audio Reactive Brightness] DEBUG: Function called")
        print(f"[FL Audio Reactive Brightness] DEBUG: Input frames shape = {frames.shape}")
        print(f"[FL Audio Reactive Brightness] DEBUG: Base brightness = {base_brightness}")
        print(f"[FL Audio Reactive Brightness] DEBUG: Brightness intensity = {brightness_intensity}")
        print(f"[FL Audio Reactive Brightness] DEBUG: Invert = {invert}")
        print(f"[FL Audio Reactive Brightness] DEBUG: Clamp output = {clamp_output}")
        print(f"{'='*60}\n")

        try:
            # Parse envelope JSON
            envelope_data = json.loads(envelope_json)
            envelope = envelope_data['envelope']

            batch_size, height, width, channels = frames.shape
            num_envelope_frames = len(envelope)

            print(f"[FL Audio Reactive Brightness] Input frames: {batch_size}")
            print(f"[FL Audio Reactive Brightness] Envelope frames: {num_envelope_frames}")

            # Handle frame count mismatch
            if batch_size != num_envelope_frames:
                print(f"[FL Audio Reactive Brightness] WARNING: Frame count mismatch! Using min({batch_size}, {num_envelope_frames})")
                max_frames = min(batch_size, num_envelope_frames)
            else:
                max_frames = batch_size

            # Prepare mask batch if provided
            mask_images = self.prepare_mask_batch(mask, max_frames) if mask is not None else None

            # Process each frame
            output_frames = []

            for frame_idx in range(max_frames):
                # Get envelope value for this frame
                envelope_value = envelope[frame_idx]

                # Calculate brightness multiplier for this frame
                if invert:
                    # Invert: high envelope = darker
                    brightness = base_brightness - (envelope_value * brightness_intensity)
                else:
                    # Normal: high envelope = brighter
                    brightness = base_brightness + (envelope_value * brightness_intensity)

                # Get frame
                frame = frames[frame_idx]

                # Apply brightness multiplier
                brightened_frame = frame * brightness

                # Clamp to valid range if requested
                if clamp_output:
                    brightened_frame = torch.clamp(brightened_frame, 0.0, 1.0)

                # Apply mask if provided
                if mask_images is not None:
                    # Convert mask to tensor
                    mask_img = self.process_mask(mask_images[frame_idx], (width, height))
                    mask_array = np.array(mask_img).astype(np.float32) / 255.0

                    # Expand mask to 3 channels
                    if len(mask_array.shape) == 2:
                        mask_tensor = torch.from_numpy(mask_array).unsqueeze(-1).repeat(1, 1, 3)
                    else:
                        mask_tensor = torch.from_numpy(mask_array)

                    # Blend: mask=1.0 shows brightness effect, mask=0.0 shows original
                    brightened_frame = frame * (1.0 - mask_tensor) + brightened_frame * mask_tensor

                output_frames.append(brightened_frame)

                if frame_idx % 100 == 0 or frame_idx < 5:
                    print(f"[FL Audio Reactive Brightness] Frame {frame_idx}: envelope={envelope_value:.3f}, brightness={brightness:.3f}")

            # Stack all frames
            output_tensor = torch.stack(output_frames, dim=0)

            print(f"\n{'='*60}")
            print(f"[FL Audio Reactive Brightness] Processing complete!")
            print(f"[FL Audio Reactive Brightness] Output frames: {output_tensor.shape[0]}")
            print(f"[FL Audio Reactive Brightness] Output shape: {output_tensor.shape}")
            print(f"[FL Audio Reactive Brightness] Value range: [{output_tensor.min():.3f}, {output_tensor.max():.3f}]")
            print(f"{'='*60}\n")

            return (output_tensor,)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL Audio Reactive Brightness] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return (frames,)
