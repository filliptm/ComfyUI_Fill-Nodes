# FL_Audio_Reactive_Edge_Glow: Edge detection with glow based on audio envelope
import torch
import torch.nn.functional as F
import json
from typing import Tuple


class FL_Audio_Reactive_Edge_Glow:
    """
    A ComfyUI node for applying audio-reactive edge detection and glow effect.
    Detects edges and adds glowing outline that pulses with the audio.
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "apply_edge_glow"
    CATEGORY = "🏵️Fill Nodes/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"description": "Input frames"}),
                "envelope_json": ("STRING", {"description": "Envelope JSON from FL_Audio_Reactive_Envelope"}),
            },
            "optional": {
                "edge_frames": ("IMAGE", {"description": "Pre-computed edge frames (grayscale/mask). If not provided, edges will be auto-detected."}),
                "edge_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "Edge detection sensitivity (only used if edge_frames not provided)"
                }),
                "glow_intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "description": "Base glow intensity multiplier"
                }),
                "envelope_intensity": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "description": "How much envelope affects glow (0 = static, higher = more reactive)"
                }),
                "glow_color": (["white", "original", "cyan", "magenta", "yellow"], {
                    "default": "cyan",
                    "description": "Color of the glow effect"
                }),
                "blend_mode": (["add", "screen", "overlay"], {
                    "default": "add",
                    "description": "How to blend glow with original"
                }),
            }
        }

    def apply_edge_glow(
        self,
        frames: torch.Tensor,
        envelope_json: str,
        edge_frames: torch.Tensor = None,
        edge_threshold: float = 0.1,
        glow_intensity: float = 0.5,
        envelope_intensity: float = 0.3,
        glow_color: str = "cyan",
        blend_mode: str = "add"
    ) -> Tuple[torch.Tensor]:
        """
        Apply audio-reactive edge glow effect to frames

        Args:
            frames: Input frames tensor (batch, height, width, channels)
            envelope_json: JSON string with envelope data
            edge_threshold: Edge detection sensitivity
            glow_intensity: Base glow brightness
            envelope_intensity: How much envelope affects glow
            glow_color: Color of glow effect
            blend_mode: Blend mode for compositing

        Returns:
            Tuple containing edge-glowed frames
        """
        print(f"\n{'='*60}")
        print(f"[FL Audio Reactive Edge Glow] DEBUG: Function called")
        print(f"[FL Audio Reactive Edge Glow] DEBUG: Input frames shape = {frames.shape}")
        print(f"[FL Audio Reactive Edge Glow] DEBUG: Edge threshold = {edge_threshold}")
        print(f"[FL Audio Reactive Edge Glow] DEBUG: Glow intensity = {glow_intensity}")
        print(f"[FL Audio Reactive Edge Glow] DEBUG: Envelope intensity = {envelope_intensity}")
        print(f"[FL Audio Reactive Edge Glow] DEBUG: Glow color = {glow_color}")
        print(f"{'='*60}\n")

        try:
            # Parse envelope JSON
            envelope_data = json.loads(envelope_json)
            envelope = envelope_data['envelope']

            batch_size, height, width, channels = frames.shape
            num_envelope_frames = len(envelope)

            print(f"[FL Audio Reactive Edge Glow] Input frames: {batch_size}")
            print(f"[FL Audio Reactive Edge Glow] Envelope frames: {num_envelope_frames}")

            # Handle frame count mismatch
            if batch_size != num_envelope_frames:
                print(f"[FL Audio Reactive Edge Glow] WARNING: Frame count mismatch! Using min({batch_size}, {num_envelope_frames})")
                max_frames = min(batch_size, num_envelope_frames)
            else:
                max_frames = batch_size

            # Check if user provided edge frames
            use_custom_edges = edge_frames is not None

            if use_custom_edges:
                print(f"[FL Audio Reactive Edge Glow] Using custom edge frames")
                edge_batch_size = edge_frames.shape[0]
                if edge_batch_size != max_frames:
                    print(f"[FL Audio Reactive Edge Glow] WARNING: Edge frame count ({edge_batch_size}) != input frame count ({max_frames})")
                    max_frames = min(max_frames, edge_batch_size)
            else:
                print(f"[FL Audio Reactive Edge Glow] Auto-detecting edges with Sobel")
                # Sobel kernels for edge detection
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Process each frame
            output_frames = []

            for frame_idx in range(max_frames):
                # Get envelope value for this frame
                envelope_value = envelope[frame_idx]

                # Calculate glow strength for this frame
                glow_strength = glow_intensity + (envelope_value * envelope_intensity)

                # Get frame
                frame = frames[frame_idx]

                # Get or compute edges
                if use_custom_edges:
                    # Use provided edge frames
                    edge_frame = edge_frames[frame_idx]
                    # Convert to grayscale if RGB
                    if edge_frame.shape[-1] == 3:
                        edges = (
                            edge_frame[:, :, 0] * 0.2126 +
                            edge_frame[:, :, 1] * 0.7152 +
                            edge_frame[:, :, 2] * 0.0722
                        )
                    else:
                        edges = edge_frame[:, :, 0] if len(edge_frame.shape) == 3 else edge_frame
                else:
                    # Auto-detect edges with Sobel
                    # Convert to grayscale for edge detection
                    gray = (
                        frame[:, :, 0] * 0.2126 +
                        frame[:, :, 1] * 0.7152 +
                        frame[:, :, 2] * 0.0722
                    )

                    # Add batch and channel dimensions for convolution
                    gray_batch = gray.unsqueeze(0).unsqueeze(0)

                    # Apply Sobel edge detection
                    edges_x = F.conv2d(gray_batch, sobel_x, padding=1)
                    edges_y = F.conv2d(gray_batch, sobel_y, padding=1)

                    # Calculate edge magnitude
                    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

                    # Normalize and threshold
                    edges = edges.squeeze(0).squeeze(0)
                    edges = edges / (edges.max() + 1e-8)
                    edges = torch.clamp((edges - edge_threshold) / (1.0 - edge_threshold), 0.0, 1.0)

                # Apply glow strength
                edges = edges * glow_strength

                # Choose glow color
                if glow_color == "white":
                    glow_rgb = edges.unsqueeze(-1).repeat(1, 1, 3)
                elif glow_color == "cyan":
                    glow_rgb = torch.stack([edges * 0.3, edges, edges], dim=-1)
                elif glow_color == "magenta":
                    glow_rgb = torch.stack([edges, edges * 0.3, edges], dim=-1)
                elif glow_color == "yellow":
                    glow_rgb = torch.stack([edges, edges, edges * 0.3], dim=-1)
                elif glow_color == "original":
                    # Use original frame colors at edge locations
                    glow_rgb = frame * edges.unsqueeze(-1)
                else:
                    glow_rgb = edges.unsqueeze(-1).repeat(1, 1, 3)

                # Blend with original frame
                if blend_mode == "add":
                    result = frame + glow_rgb
                elif blend_mode == "screen":
                    # Screen blend: 1 - (1-a)(1-b)
                    result = 1.0 - (1.0 - frame) * (1.0 - glow_rgb)
                elif blend_mode == "overlay":
                    # Simple overlay approximation
                    result = frame * (1.0 + glow_rgb)
                else:
                    result = frame + glow_rgb

                # Clamp to valid range
                result = torch.clamp(result, 0.0, 1.0)

                output_frames.append(result)

                if frame_idx % 100 == 0 or frame_idx < 5:
                    print(f"[FL Audio Reactive Edge Glow] Frame {frame_idx}: envelope={envelope_value:.3f}, glow={glow_strength:.3f}")

            # Stack all frames
            output_tensor = torch.stack(output_frames, dim=0)

            print(f"\n{'='*60}")
            print(f"[FL Audio Reactive Edge Glow] Processing complete!")
            print(f"[FL Audio Reactive Edge Glow] Output frames: {output_tensor.shape[0]}")
            print(f"[FL Audio Reactive Edge Glow] Output shape: {output_tensor.shape}")
            print(f"{'='*60}\n")

            return (output_tensor,)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL Audio Reactive Edge Glow] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return (frames,)
