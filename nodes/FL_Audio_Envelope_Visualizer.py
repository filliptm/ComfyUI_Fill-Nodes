# FL_Audio_Envelope_Visualizer: Visualize envelopes as fading white frames
import torch
from typing import Tuple


class FL_Audio_Envelope_Visualizer:
    """
    A ComfyUI node for visualizing audio envelopes as frames.
    Creates white frames that fade to black based on envelope values.
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "visualize_envelope"
    CATEGORY = "🏵️Fill Nodes/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "envelope_json": ("STRING", {
                    "description": "Envelope JSON from FL_Audio_Reactive_Envelope"
                }),
            },
            "optional": {
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "description": "Frame width"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "description": "Frame height"
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "description": "Brightness multiplier"
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "description": "Invert colors (black flashes on white)"
                }),
            }
        }

    def visualize_envelope(
        self,
        envelope_json: str,
        width: int = 512,
        height: int = 512,
        intensity: float = 1.0,
        invert: bool = False
    ) -> Tuple[torch.Tensor]:
        """
        Visualize envelope as fading frames

        Args:
            envelope_json: JSON string with envelope data from FL_Audio_Reactive_Envelope
            width: Frame width in pixels
            height: Frame height in pixels
            intensity: Brightness multiplier
            invert: Invert colors (black on white instead of white on black)

        Returns:
            Tuple containing tensor of frames (batch, height, width, channels)
        """
        print(f"\n{'='*60}")
        print(f"[FL Audio Envelope Visualizer] DEBUG: Function called")
        print(f"{'='*60}\n")

        try:
            # Parse envelope JSON
            import json
            envelope_data = json.loads(envelope_json)
            envelope = envelope_data['envelope']
            num_frames = len(envelope)

            print(f"[FL Audio Envelope Visualizer] DEBUG: Envelope length = {num_frames} frames")
            print(f"[FL Audio Envelope Visualizer] DEBUG: Frame size = {width}x{height}")
            print(f"[FL Audio Envelope Visualizer] DEBUG: Intensity = {intensity}")
            print(f"[FL Audio Envelope Visualizer] DEBUG: Invert = {invert}")

            # Calculate expected memory
            expected_memory_mb = (num_frames * height * width * 3 * 4) / (1024 * 1024)
            print(f"[FL Audio Envelope Visualizer] MEMORY DEBUG:")
            print(f"[FL Audio Envelope Visualizer]   Total frames: {num_frames}")
            print(f"[FL Audio Envelope Visualizer]   Frame dimensions: {height}x{width}x3")
            print(f"[FL Audio Envelope Visualizer]   Expected memory: {expected_memory_mb:.2f} MB")

            # Create frames
            frames = []

            for frame_idx, envelope_value in enumerate(envelope):
                # Apply intensity
                value = min(envelope_value * intensity, 1.0)

                # Invert if requested
                if invert:
                    value = 1.0 - value

                # Create frame with uniform color
                frame = torch.full((height, width, 3), value, dtype=torch.float32)
                frames.append(frame)

            # Stack all frames
            all_frames = torch.stack(frames, dim=0)

            # Calculate actual memory
            actual_memory_mb = (all_frames.element_size() * all_frames.nelement()) / (1024 * 1024)

            print(f"\n{'='*60}")
            print(f"[FL Audio Envelope Visualizer] Visualization complete!")
            print(f"[FL Audio Envelope Visualizer] Total frames: {all_frames.shape[0]}")
            print(f"[FL Audio Envelope Visualizer] Frame shape: {all_frames.shape}")
            print(f"[FL Audio Envelope Visualizer] Actual tensor memory: {actual_memory_mb:.2f} MB")
            print(f"[FL Audio Envelope Visualizer] Value range: [{all_frames.min():.3f}, {all_frames.max():.3f}]")
            print(f"{'='*60}\n")

            return (all_frames,)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL Audio Envelope Visualizer] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            # Return single black frame on error
            return (torch.zeros((1, height, width, 3), dtype=torch.float32),)
