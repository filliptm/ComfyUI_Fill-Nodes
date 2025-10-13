# FL_Audio_Reactive_Saturation: Control saturation based on audio envelope
import torch
import numpy as np
import json
from PIL import Image
from typing import Tuple


class FL_Audio_Reactive_Saturation:
    """
    A ComfyUI node for applying audio-reactive saturation changes to frames.
    Adjusts color saturation based on envelope values from drum detection.
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "apply_saturation"
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
                "base_saturation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "description": "Base saturation multiplier (1.0 = normal, 0.0 = grayscale)"
                }),
                "saturation_intensity": ("FLOAT", {
                    "default": 0.2,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "Saturation intensity (positive = more saturated on hits, negative = desaturated)"
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "description": "Invert envelope (desaturate on hits instead of saturate)"
                }),
            }
        }

    def apply_saturation(
        self,
        frames: torch.Tensor,
        envelope_json: str,
        mask=None,
        base_saturation: float = 1.0,
        saturation_intensity: float = 0.2,
        invert: bool = False
    ) -> Tuple[torch.Tensor]:
        """
        Apply audio-reactive saturation effect to frames

        Args:
            frames: Input frames tensor (batch, height, width, channels)
            envelope_json: JSON string with envelope data
            base_saturation: Base saturation multiplier (1.0 = normal)
            saturation_intensity: How much envelope affects saturation
            invert: Desaturate on hits instead of saturate

        Returns:
            Tuple containing saturation-adjusted frames
        """
        print(f"\n{'='*60}")
        print(f"[FL Audio Reactive Saturation] DEBUG: Function called")
        print(f"[FL Audio Reactive Saturation] DEBUG: Input frames shape = {frames.shape}")
        print(f"[FL Audio Reactive Saturation] DEBUG: Base saturation = {base_saturation}")
        print(f"[FL Audio Reactive Saturation] DEBUG: Saturation intensity = {saturation_intensity}")
        print(f"[FL Audio Reactive Saturation] DEBUG: Invert = {invert}")
        print(f"{'='*60}\n")

        try:
            # Parse envelope JSON
            envelope_data = json.loads(envelope_json)
            envelope = envelope_data['envelope']

            batch_size, height, width, channels = frames.shape
            num_envelope_frames = len(envelope)

            print(f"[FL Audio Reactive Saturation] Input frames: {batch_size}")
            print(f"[FL Audio Reactive Saturation] Envelope frames: {num_envelope_frames}")

            # Handle frame count mismatch
            if batch_size != num_envelope_frames:
                print(f"[FL Audio Reactive Saturation] WARNING: Frame count mismatch! Using min({batch_size}, {num_envelope_frames})")
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

                # Calculate saturation multiplier for this frame
                if invert:
                    # Invert: high envelope = desaturated
                    saturation = base_saturation - (envelope_value * saturation_intensity)
                else:
                    # Normal: high envelope = more saturated
                    saturation = base_saturation + (envelope_value * saturation_intensity)

                # Clamp saturation to valid range
                saturation = max(0.0, saturation)

                # Get frame
                frame = frames[frame_idx]

                # Convert to grayscale (luminance)
                # Using Rec. 709 luma coefficients: Y = 0.2126*R + 0.7152*G + 0.0722*B
                grayscale = (
                    frame[:, :, 0] * 0.2126 +
                    frame[:, :, 1] * 0.7152 +
                    frame[:, :, 2] * 0.0722
                )
                # Expand to RGB
                grayscale_rgb = grayscale.unsqueeze(-1).repeat(1, 1, 3)

                # Blend between grayscale and original based on saturation
                # saturation=0: full grayscale
                # saturation=1: original colors
                # saturation>1: oversaturated
                saturated_frame = grayscale_rgb + saturation * (frame - grayscale_rgb)

                # Clamp to valid range
                saturated_frame = torch.clamp(saturated_frame, 0.0, 1.0)

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

                    # Blend: mask=1.0 shows saturation effect, mask=0.0 shows original
                    saturated_frame = frame * (1.0 - mask_tensor) + saturated_frame * mask_tensor

                output_frames.append(saturated_frame)

                if frame_idx % 100 == 0 or frame_idx < 5:
                    print(f"[FL Audio Reactive Saturation] Frame {frame_idx}: envelope={envelope_value:.3f}, saturation={saturation:.3f}")

            # Stack all frames
            output_tensor = torch.stack(output_frames, dim=0)

            print(f"\n{'='*60}")
            print(f"[FL Audio Reactive Saturation] Processing complete!")
            print(f"[FL Audio Reactive Saturation] Output frames: {output_tensor.shape[0]}")
            print(f"[FL Audio Reactive Saturation] Output shape: {output_tensor.shape}")
            print(f"[FL Audio Reactive Saturation] Value range: [{output_tensor.min():.3f}, {output_tensor.max():.3f}]")
            print(f"{'='*60}\n")

            return (output_tensor,)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL Audio Reactive Saturation] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return (frames,)
