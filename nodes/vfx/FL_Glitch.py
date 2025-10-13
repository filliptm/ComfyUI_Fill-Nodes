import torch
import numpy as np
import json
from PIL import Image
from glitch_this import ImageGlitcher

from comfy.utils import ProgressBar

class FL_Glitch:
    def __init__(self):
        self.seed_index = 0

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
                    "description": "Invert envelope (show glitch when quiet)"
                }),
                "mask": ("IMAGE", {
                    "default": None,
                    "description": "Optional mask to control where effect is applied"
                }),
                # Effect parameters
                "glitch_amount": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "color_offset": (["Disable", "Enable"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "glitch"
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

    def s2b(self, v):
        return v == "Enable"

    def glitch(self, images, envelope_json="", blend_intensity=1.0, invert=False, mask=None, glitch_amount=1, color_offset="Disable", seed=0):
        # Check if audio-reactive mode is enabled
        use_audio_reactive = envelope_json and envelope_json.strip() != ""

        if use_audio_reactive:
            return self._apply_audio_reactive(images, glitch_amount, color_offset, seed, envelope_json, blend_intensity, invert, mask)
        else:
            return self._apply_static(images, glitch_amount, color_offset, seed, mask)

    def _apply_static(self, images, glitch_amount, color_offset, seed, mask=None):
        """Static glitch effect without audio reactivity"""
        color_offset = self.s2b(color_offset)
        g = ImageGlitcher()
        out = []
        total_images = len(images)

        # Convert seed to a list if it's a single value
        if not isinstance(seed, list):
            seed = [seed] * total_images

        # Prepare mask batch if provided
        mask_images = self.prepare_mask_batch(mask, total_images) if mask is not None else None

        pbar = ProgressBar(total_images)
        for i, image in enumerate(images, start=1):
            p = self.t2p(image)

            # Get the current seed value
            current_seed = seed[i - 1]

            # Ensure current_seed is a single integer value
            if isinstance(current_seed, (int, float)):
                current_seed = int(current_seed)
            elif isinstance(current_seed, (list, tuple)):
                current_seed = current_seed[0]
            else:
                current_seed = current_seed.iloc[0]

            g1 = g.glitch_image(p, glitch_amount, color_offset=color_offset, seed=current_seed)
            r1 = g1.rotate(90, expand=True)
            g2 = g.glitch_image(r1, glitch_amount, color_offset=color_offset, seed=current_seed)
            f = g2.rotate(-90, expand=True)

            # Apply mask if provided
            if mask_images is not None:
                mask_img = self.process_mask(mask_images[i-1], f.size)
                # Blend original and glitch based on mask
                glitch_array = np.array(f.convert("RGB")).astype(np.float32)
                original_array = np.array(p).astype(np.float32)
                mask_array = np.array(mask_img).astype(np.float32) / 255.0

                # Expand mask to 3 channels
                if len(mask_array.shape) == 2:
                    mask_array = np.stack([mask_array] * 3, axis=-1)

                # Blend: mask=1.0 shows glitch, mask=0.0 shows original
                blended = glitch_array * mask_array + original_array * (1.0 - mask_array)
                f = Image.fromarray(blended.astype(np.uint8))

            o = np.array(f.convert("RGB")).astype(np.float32) / 255.0
            o = torch.from_numpy(o).unsqueeze(0)
            out.append(o)
            pbar.update_absolute(i)

        out = torch.cat(out, 0)
        return (out,)

    def _apply_audio_reactive(self, images, glitch_amount, color_offset, seed, envelope_json, blend_intensity, invert, mask=None):
        """Audio-reactive glitch effect with envelope-based blending"""
        print(f"\n{'='*60}")
        print(f"[FL_Glitch Audio Reactive] DEBUG: Function called")
        print(f"[FL_Glitch Audio Reactive] DEBUG: Input shape = {images.shape}")
        print(f"[FL_Glitch Audio Reactive] DEBUG: Blend intensity = {blend_intensity}")
        print(f"[FL_Glitch Audio Reactive] DEBUG: Invert = {invert}")
        print(f"{'='*60}\n")

        try:
            # Parse envelope JSON
            envelope_data = json.loads(envelope_json)
            envelope = envelope_data['envelope']

            batch_size = len(images)
            num_envelope_frames = len(envelope)

            print(f"[FL_Glitch Audio Reactive] Input frames: {batch_size}")
            print(f"[FL_Glitch Audio Reactive] Envelope frames: {num_envelope_frames}")

            # Handle frame count mismatch
            if batch_size != num_envelope_frames:
                print(f"[FL_Glitch Audio Reactive] WARNING: Frame count mismatch! Using min({batch_size}, {num_envelope_frames})")
                max_frames = min(batch_size, num_envelope_frames)
            else:
                max_frames = batch_size

            color_offset_bool = self.s2b(color_offset)
            g = ImageGlitcher()

            # Convert seed to a list if it's a single value
            if not isinstance(seed, list):
                seed = [seed] * max_frames

            # Prepare mask batch if provided
            mask_images = self.prepare_mask_batch(mask, max_frames) if mask is not None else None

            # PASS 1: Generate glitch effect for all frames with static parameters
            print(f"[FL_Glitch Audio Reactive] PASS 1: Generating glitch effect...")
            glitch_frames = []

            pbar = ProgressBar(max_frames)
            for b in range(max_frames):
                p = self.t2p(images[b])

                # Get the current seed value
                current_seed = seed[b]

                # Ensure current_seed is a single integer value
                if isinstance(current_seed, (int, float)):
                    current_seed = int(current_seed)
                elif isinstance(current_seed, (list, tuple)):
                    current_seed = current_seed[0]
                else:
                    current_seed = current_seed.iloc[0]

                g1 = g.glitch_image(p, glitch_amount, color_offset=color_offset_bool, seed=current_seed)
                r1 = g1.rotate(90, expand=True)
                g2 = g.glitch_image(r1, glitch_amount, color_offset=color_offset_bool, seed=current_seed)
                f = g2.rotate(-90, expand=True)

                # Apply mask if provided
                if mask_images is not None:
                    mask_img = self.process_mask(mask_images[b], f.size)
                    # Blend original and glitch based on mask
                    glitch_array = np.array(f.convert("RGB")).astype(np.float32)
                    original_array = np.array(p).astype(np.float32)
                    mask_array = np.array(mask_img).astype(np.float32) / 255.0

                    # Expand mask to 3 channels
                    if len(mask_array.shape) == 2:
                        mask_array = np.stack([mask_array] * 3, axis=-1)

                    # Blend: mask=1.0 shows glitch, mask=0.0 shows original
                    blended = glitch_array * mask_array + original_array * (1.0 - mask_array)
                    f = Image.fromarray(blended.astype(np.uint8))

                glitch_tensor = self.p2t(f)
                glitch_frames.append(glitch_tensor)
                pbar.update_absolute(b + 1)

            # Stack glitch frames
            glitch_batch = torch.cat(glitch_frames, dim=0)

            print(f"[FL_Glitch Audio Reactive] PASS 2: Applying envelope-based blending...")

            # PASS 2: Blend original and glitch frames based on envelope
            output_frames = []

            for frame_idx in range(max_frames):
                # Get envelope value for this frame
                envelope_value = envelope[frame_idx]

                # Calculate blend amount
                if invert:
                    # Invert: high envelope = show original (less glitch)
                    blend_amount = (1.0 - envelope_value) * blend_intensity
                else:
                    # Normal: high envelope = more glitch
                    blend_amount = envelope_value * blend_intensity

                # Clamp blend amount
                blend_amount = max(0.0, min(1.0, blend_amount))

                # Blend original and glitch frames
                # blend_amount=0: original footage
                # blend_amount=1: full glitch effect
                original_frame = images[frame_idx]
                glitch_frame = glitch_batch[frame_idx]

                blended_frame = (1.0 - blend_amount) * original_frame + blend_amount * glitch_frame
                output_frames.append(blended_frame)

                if frame_idx % 100 == 0 or frame_idx < 5:
                    print(f"[FL_Glitch Audio Reactive] Frame {frame_idx}: envelope={envelope_value:.3f}, blend={blend_amount:.3f}")

            # Stack all frames
            output_tensor = torch.stack(output_frames, dim=0)

            print(f"\n{'='*60}")
            print(f"[FL_Glitch Audio Reactive] Processing complete!")
            print(f"[FL_Glitch Audio Reactive] Output frames: {output_tensor.shape[0]}")
            print(f"[FL_Glitch Audio Reactive] Output shape: {output_tensor.shape}")
            print(f"{'='*60}\n")

            return (output_tensor,)

        except Exception as e:
            error_msg = f"Error in audio-reactive mode: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL_Glitch Audio Reactive] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"[FL_Glitch Audio Reactive] Falling back to static mode...")
            print(f"{'='*60}\n")
            # Fallback to static mode
            return self._apply_static(images, glitch_amount, color_offset, seed, mask)