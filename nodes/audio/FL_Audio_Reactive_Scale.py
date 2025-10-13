# FL_Audio_Reactive_Scale: Scale/zoom frames based on audio envelope
import torch
import torch.nn.functional as F
import json
import numpy as np
from PIL import Image
from typing import Tuple


class FL_Audio_Reactive_Scale:
    """
    A ComfyUI node for applying audio-reactive scale/zoom effect to frames.
    Scales frames based on envelope values from drum detection.
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "apply_scale"
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
                "base_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.01,
                    "description": "Base scale (1.0 = normal size)"
                }),
                "scale_intensity": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "description": "Scale intensity multiplier (envelope * intensity)"
                }),
                "interpolation": (["bilinear", "bicubic", "nearest"], {
                    "default": "bilinear",
                    "description": "Interpolation mode for scaling"
                }),
                "maintain_aspect": ("BOOLEAN", {
                    "default": True,
                    "description": "Maintain aspect ratio (crop to fit)"
                }),
            }
        }

    def apply_scale(
        self,
        frames: torch.Tensor,
        envelope_json: str,
        mask=None,
        base_scale: float = 1.0,
        scale_intensity: float = 0.2,
        interpolation: str = "bilinear",
        maintain_aspect: bool = True
    ) -> Tuple[torch.Tensor]:
        """
        Apply audio-reactive scale effect to frames

        Args:
            frames: Input frames tensor (batch, height, width, channels)
            envelope_json: JSON string with envelope data
            mask: Optional mask to control where effect is applied
            base_scale: Base scale value (1.0 = normal)
            scale_intensity: How much envelope affects scale
            interpolation: Interpolation mode
            maintain_aspect: Maintain aspect ratio and crop

        Returns:
            Tuple containing scaled frames
        """
        print(f"\n{'='*60}")
        print(f"[FL Audio Reactive Scale] DEBUG: Function called")
        print(f"[FL Audio Reactive Scale] DEBUG: Input frames shape = {frames.shape}")
        print(f"[FL Audio Reactive Scale] DEBUG: Base scale = {base_scale}")
        print(f"[FL Audio Reactive Scale] DEBUG: Scale intensity = {scale_intensity}")
        print(f"[FL Audio Reactive Scale] DEBUG: Interpolation = {interpolation}")
        print(f"{'='*60}\n")

        try:
            # Parse envelope JSON
            envelope_data = json.loads(envelope_json)
            envelope = envelope_data['envelope']

            batch_size, height, width, channels = frames.shape
            num_envelope_frames = len(envelope)

            print(f"[FL Audio Reactive Scale] Input frames: {batch_size}")
            print(f"[FL Audio Reactive Scale] Envelope frames: {num_envelope_frames}")

            # Handle frame count mismatch
            if batch_size != num_envelope_frames:
                print(f"[FL Audio Reactive Scale] WARNING: Frame count mismatch! Using min({batch_size}, {num_envelope_frames})")
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

                # Calculate scale for this frame
                scale = base_scale + (envelope_value * scale_intensity)

                # Get frame and convert to BCHW format for interpolation
                frame = frames[frame_idx:frame_idx+1]  # Keep batch dimension
                frame = frame.permute(0, 3, 1, 2)  # BHWC -> BCHW

                # Calculate scaled dimensions
                scaled_height = int(height * scale)
                scaled_width = int(width * scale)

                # Scale the frame
                scaled_frame = F.interpolate(
                    frame,
                    size=(scaled_height, scaled_width),
                    mode=interpolation,
                    align_corners=False if interpolation != 'nearest' else None
                )

                # Crop or pad to original size
                if maintain_aspect:
                    # Center crop if scaled up, pad if scaled down
                    if scale > 1.0:
                        # Crop from center
                        start_h = (scaled_height - height) // 2
                        start_w = (scaled_width - width) // 2
                        cropped_frame = scaled_frame[:, :, start_h:start_h+height, start_w:start_w+width]
                    else:
                        # Pad to original size
                        pad_h = (height - scaled_height) // 2
                        pad_w = (width - scaled_width) // 2
                        pad_h_extra = height - scaled_height - pad_h
                        pad_w_extra = width - scaled_width - pad_w
                        cropped_frame = F.pad(scaled_frame, (pad_w, pad_w_extra, pad_h, pad_h_extra), mode='constant', value=0)
                else:
                    # Just resize back to original dimensions
                    cropped_frame = F.interpolate(
                        scaled_frame,
                        size=(height, width),
                        mode=interpolation,
                        align_corners=False if interpolation != 'nearest' else None
                    )

                # Convert back to BHWC
                cropped_frame = cropped_frame.permute(0, 2, 3, 1)
                scaled_output = cropped_frame[0]

                # Apply mask if provided
                if mask_images is not None:
                    original_frame = frames[frame_idx]

                    # Convert mask to tensor
                    mask_img = self.process_mask(mask_images[frame_idx], (width, height))
                    mask_array = np.array(mask_img).astype(np.float32) / 255.0

                    # Expand mask to 3 channels
                    if len(mask_array.shape) == 2:
                        mask_tensor = torch.from_numpy(mask_array).unsqueeze(-1).repeat(1, 1, 3)
                    else:
                        mask_tensor = torch.from_numpy(mask_array)

                    # Blend: mask=1.0 shows scaled effect, mask=0.0 shows original
                    scaled_output = original_frame * (1.0 - mask_tensor) + scaled_output * mask_tensor

                output_frames.append(scaled_output)

                if frame_idx % 100 == 0 or frame_idx < 5:
                    print(f"[FL Audio Reactive Scale] Frame {frame_idx}: envelope={envelope_value:.3f}, scale={scale:.3f}")

            # Stack all frames
            output_tensor = torch.stack(output_frames, dim=0)

            print(f"\n{'='*60}")
            print(f"[FL Audio Reactive Scale] Processing complete!")
            print(f"[FL Audio Reactive Scale] Output frames: {output_tensor.shape[0]}")
            print(f"[FL Audio Reactive Scale] Output shape: {output_tensor.shape}")
            print(f"{'='*60}\n")

            return (output_tensor,)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL Audio Reactive Scale] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return (frames,)
