import torch
import numpy as np
import json
from PIL import Image
import math
from comfy.utils import ProgressBar

class FL_Ripple:
    def __init__(self):
        self.modulation_index = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                # Audio reactivity (optional - at top for visibility)
                "envelope_json": ("STRING", {
                    "default": "",
                    "description": "Optional: Envelope JSON for audio-reactive blending"
                }),
                "blend_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "description": "Audio-reactive blend intensity"
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "description": "Invert envelope (show ripple when quiet)"
                }),
                "mask": ("IMAGE", {
                    "default": None,
                    "description": "Optional mask to control where effect is applied"
                }),
                # Effect parameters
                "amplitude": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "frequency": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "phase": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "center_x": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "center_y": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "modulation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ripple"
    CATEGORY = "🏵️Fill Nodes/VFX"

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def p2t(self, p):
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            t = torch.from_numpy(i).unsqueeze(0)
        return t

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

    def ripple(self, images, envelope_json="", blend_intensity=1.0, invert=False, mask=None, amplitude=10.0, frequency=20.0, phase=0.0, center_x=50.0, center_y=50.0, modulation=0.0):
        # Check if audio-reactive mode is enabled
        use_audio_reactive = envelope_json and envelope_json.strip() != ""

        if use_audio_reactive:
            return self._apply_audio_reactive(images, envelope_json, blend_intensity, invert, mask, amplitude, frequency, phase, center_x, center_y, modulation)
        else:
            return self._apply_static(images, mask, amplitude, frequency, phase, center_x, center_y, modulation)

    def _apply_static(self, images, mask, amplitude, frequency, phase, center_x, center_y, modulation):
        """Static ripple effect without audio reactivity"""
        out = []
        total_images = len(images)

        # Prepare mask batch if provided
        mask_images = self.prepare_mask_batch(mask, total_images) if mask is not None else None

        pbar = ProgressBar(total_images)
        for i, img in enumerate(images, start=1):
            p = self.t2p(img)
            width, height = p.size
            center_x_pixel = int(center_x / 100 * width)
            center_y_pixel = int(center_y / 100 * height)

            x, y = np.meshgrid(np.arange(width), np.arange(height))
            dx = x - center_x_pixel
            dy = y - center_y_pixel
            distance = np.sqrt(dx ** 2 + dy ** 2)

            # Apply modulation to amplitude and frequency
            modulation_factor = 1 + modulation * math.sin(2 * math.pi * self.modulation_index / total_images)
            modulated_amplitude = amplitude * modulation_factor
            modulated_frequency = frequency * modulation_factor

            angle = distance / modulated_frequency * 2 * np.pi + np.radians(phase)
            offset_x = (modulated_amplitude * np.sin(angle)).astype(int)
            offset_y = (modulated_amplitude * np.cos(angle)).astype(int)

            sample_x = np.clip(x + offset_x, 0, width - 1)
            sample_y = np.clip(y + offset_y, 0, height - 1)

            p_array = np.array(p)
            rippled_array = p_array[sample_y, sample_x]

            # Apply mask if provided
            if mask_images is not None:
                mask_img = self.process_mask(mask_images[i-1], p.size)
                # Blend original and ripple based on mask
                original_array = np.array(p).astype(np.float32)
                mask_array = np.array(mask_img).astype(np.float32) / 255.0

                # Expand mask to 3 channels
                if len(mask_array.shape) == 2:
                    mask_array = np.stack([mask_array] * 3, axis=-1)

                # Blend: mask=1.0 shows ripple, mask=0.0 shows original
                rippled_array = rippled_array.astype(np.float32) * mask_array + original_array * (1.0 - mask_array)

            o = rippled_array.astype(np.float32) / 255.0
            o = torch.from_numpy(o).unsqueeze(0)
            out.append(o)

            self.modulation_index += 1
            pbar.update_absolute(i)

        out = torch.cat(out, 0)
        return (out,)

    def _apply_audio_reactive(self, images, envelope_json, blend_intensity, invert, mask, amplitude, frequency, phase, center_x, center_y, modulation):
        """Audio-reactive ripple effect with envelope-based blending"""
        print(f"\n{'='*60}")
        print(f"[FL_Ripple Audio Reactive] DEBUG: Function called")
        print(f"[FL_Ripple Audio Reactive] DEBUG: Input shape = {images.shape}")
        print(f"[FL_Ripple Audio Reactive] DEBUG: Blend intensity = {blend_intensity}")
        print(f"[FL_Ripple Audio Reactive] DEBUG: Invert = {invert}")
        print(f"{'='*60}\n")

        try:
            # Parse envelope JSON
            envelope_data = json.loads(envelope_json)
            envelope = envelope_data['envelope']

            batch_size = len(images)
            num_envelope_frames = len(envelope)

            print(f"[FL_Ripple Audio Reactive] Input frames: {batch_size}")
            print(f"[FL_Ripple Audio Reactive] Envelope frames: {num_envelope_frames}")

            # Handle frame count mismatch
            if batch_size != num_envelope_frames:
                print(f"[FL_Ripple Audio Reactive] WARNING: Frame count mismatch! Using min({batch_size}, {num_envelope_frames})")
                max_frames = min(batch_size, num_envelope_frames)
            else:
                max_frames = batch_size

            # Prepare mask batch if provided
            mask_images = self.prepare_mask_batch(mask, max_frames) if mask is not None else None

            # PASS 1: Generate ripple effect for all frames with static parameters
            print(f"[FL_Ripple Audio Reactive] PASS 1: Generating ripple effect...")
            ripple_frames = []

            pbar = ProgressBar(max_frames)
            for b in range(max_frames):
                p = self.t2p(images[b])
                width, height = p.size
                center_x_pixel = int(center_x / 100 * width)
                center_y_pixel = int(center_y / 100 * height)

                x, y = np.meshgrid(np.arange(width), np.arange(height))
                dx = x - center_x_pixel
                dy = y - center_y_pixel
                distance = np.sqrt(dx ** 2 + dy ** 2)

                # Apply modulation to amplitude and frequency
                modulation_factor = 1 + modulation * math.sin(2 * math.pi * self.modulation_index / max_frames)
                modulated_amplitude = amplitude * modulation_factor
                modulated_frequency = frequency * modulation_factor

                angle = distance / modulated_frequency * 2 * np.pi + np.radians(phase)
                offset_x = (modulated_amplitude * np.sin(angle)).astype(int)
                offset_y = (modulated_amplitude * np.cos(angle)).astype(int)

                sample_x = np.clip(x + offset_x, 0, width - 1)
                sample_y = np.clip(y + offset_y, 0, height - 1)

                p_array = np.array(p)
                rippled_array = p_array[sample_y, sample_x]

                # Apply mask if provided
                if mask_images is not None:
                    mask_img = self.process_mask(mask_images[b], p.size)
                    # Blend original and ripple based on mask
                    original_array = np.array(p).astype(np.float32)
                    mask_array = np.array(mask_img).astype(np.float32) / 255.0

                    # Expand mask to 3 channels
                    if len(mask_array.shape) == 2:
                        mask_array = np.stack([mask_array] * 3, axis=-1)

                    # Blend: mask=1.0 shows ripple, mask=0.0 shows original
                    rippled_array = rippled_array.astype(np.float32) * mask_array + original_array * (1.0 - mask_array)

                rippled_image = Image.fromarray(rippled_array.astype(np.uint8))
                ripple_tensor = self.p2t(rippled_image)
                ripple_frames.append(ripple_tensor)

                self.modulation_index += 1
                pbar.update_absolute(b + 1)

            # Stack ripple frames
            ripple_batch = torch.cat(ripple_frames, dim=0)

            print(f"[FL_Ripple Audio Reactive] PASS 2: Applying envelope-based blending...")

            # PASS 2: Blend original and ripple frames based on envelope
            output_frames = []

            for frame_idx in range(max_frames):
                # Get envelope value for this frame
                envelope_value = envelope[frame_idx]

                # Calculate blend amount
                if invert:
                    # Invert: high envelope = show original (less ripple)
                    blend_amount = (1.0 - envelope_value) * blend_intensity
                else:
                    # Normal: high envelope = more ripple
                    blend_amount = envelope_value * blend_intensity

                # Clamp blend amount
                blend_amount = max(0.0, min(1.0, blend_amount))

                # Blend original and ripple frames
                # blend_amount=0: original footage
                # blend_amount=1: full ripple effect
                original_frame = images[frame_idx]
                ripple_frame = ripple_batch[frame_idx]

                blended_frame = (1.0 - blend_amount) * original_frame + blend_amount * ripple_frame
                output_frames.append(blended_frame)

                if frame_idx % 100 == 0 or frame_idx < 5:
                    print(f"[FL_Ripple Audio Reactive] Frame {frame_idx}: envelope={envelope_value:.3f}, blend={blend_amount:.3f}")

            # Stack all frames
            output_tensor = torch.stack(output_frames, dim=0)

            print(f"\n{'='*60}")
            print(f"[FL_Ripple Audio Reactive] Processing complete!")
            print(f"[FL_Ripple Audio Reactive] Output frames: {output_tensor.shape[0]}")
            print(f"[FL_Ripple Audio Reactive] Output shape: {output_tensor.shape}")
            print(f"{'='*60}\n")

            return (output_tensor,)

        except Exception as e:
            error_msg = f"Error in audio-reactive mode: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL_Ripple Audio Reactive] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"[FL_Ripple Audio Reactive] Falling back to static mode...")
            print(f"{'='*60}\n")
            # Fallback to static mode
            return self._apply_static(images, mask, amplitude, frequency, phase, center_x, center_y, modulation)