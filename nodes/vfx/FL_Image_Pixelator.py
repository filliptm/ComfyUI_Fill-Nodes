import torch
import numpy as np
import json
from PIL import Image
from kornia.morphology import gradient
import comfy.model_management

from comfy.utils import ProgressBar

class FL_ImagePixelator:
    def __init__(self):
        self.modulation_index = 0

    def t2p(self, t):
        """Tensor to PIL"""
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return None

    def p2t(self, p):
        """PIL to Tensor"""
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            return torch.from_numpy(i).unsqueeze(0)
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
                "image": ("IMAGE", {}),
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
                    "description": "Invert envelope (show pixelation when quiet)"
                }),
                "mask": ("IMAGE", {
                    "default": None,
                    "description": "Optional mask to control where effect is applied"
                }),
                # Effect parameters
                "scale_factor": ("FLOAT", {"default": 0.0500, "min": 0.0100, "max": 0.2000, "step": 0.0100}),
                "kernel_size": ("INT", {"default": 3, "max": 10, "step": 1}),
                "modulation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pixelate_image"
    CATEGORY = "🏵️Fill Nodes/VFX"

    def pixelate_image(self, image, envelope_json="", blend_intensity=1.0, invert=False, mask=None, scale_factor=0.05, kernel_size=3, modulation=0.0):
        # Check if audio-reactive mode is enabled
        use_audio_reactive = envelope_json and envelope_json.strip() != ""

        if use_audio_reactive:
            return self._apply_audio_reactive(image, envelope_json, blend_intensity, invert, mask, scale_factor, kernel_size, modulation)
        else:
            return self._apply_static(image, mask, scale_factor, kernel_size, modulation)

    def _apply_static(self, image, mask, scale_factor, kernel_size, modulation):
        """Static pixelation effect without audio reactivity"""
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # Batch dimension is present
                output_images = []
                total_frames = image.shape[0]

                # Prepare mask batch if provided
                mask_images = self.prepare_mask_batch(mask, total_frames) if mask is not None else None

                pbar = ProgressBar(total_frames)
                for i, single_image in enumerate(image, start=1):
                    original_image = single_image.clone()
                    single_image = single_image.unsqueeze(0)  # Add batch dimension
                    modulated_scale_factor = self.apply_modulation(scale_factor, modulation, total_frames)
                    single_image = self.apply_pixelation_tensor(single_image, modulated_scale_factor)
                    single_image = self.process(single_image, kernel_size)

                    # Apply mask if provided
                    if mask_images is not None:
                        single_pil = self.t2p(single_image.squeeze(0))
                        original_pil = self.t2p(original_image)
                        mask_img = self.process_mask(mask_images[i-1], single_pil.size)

                        # Blend original and pixelated based on mask
                        pixelated_array = np.array(single_pil).astype(np.float32)
                        original_array = np.array(original_pil).astype(np.float32)
                        mask_array = np.array(mask_img).astype(np.float32) / 255.0

                        # Expand mask to 3 channels
                        if len(mask_array.shape) == 2:
                            mask_array = np.stack([mask_array] * 3, axis=-1)

                        # Blend: mask=1.0 shows pixelated, mask=0.0 shows original
                        blended = pixelated_array * mask_array + original_array * (1.0 - mask_array)
                        single_image = self.p2t(Image.fromarray(blended.astype(np.uint8)))

                    output_images.append(single_image)
                    pbar.update_absolute(i)

                image = torch.cat(output_images, dim=0)  # Concatenate processed images along batch dimension
            elif image.dim() == 3:  # No batch dimension, single image
                image = image.unsqueeze(0)  # Add batch dimension
                modulated_scale_factor = self.apply_modulation(scale_factor, modulation, 1)
                image = self.apply_pixelation_tensor(image, modulated_scale_factor)
                image = self.process(image, kernel_size)
                image = image.squeeze(0)  # Remove batch dimension
            else:
                return (None,)
        elif isinstance(image, Image.Image):
            modulated_scale_factor = self.apply_modulation(scale_factor, modulation, 1)
            image = self.apply_pixelation_pil(image, modulated_scale_factor)
            image = self.process(image, kernel_size)
        else:
            return (None,)

        return (image,)

    def _apply_audio_reactive(self, image, envelope_json, blend_intensity, invert, mask, scale_factor, kernel_size, modulation):
        """Audio-reactive pixelation effect with envelope-based blending"""
        print(f"\n{'='*60}")
        print(f"[FL_ImagePixelator Audio Reactive] DEBUG: Function called")
        print(f"[FL_ImagePixelator Audio Reactive] DEBUG: Input shape = {image.shape}")
        print(f"[FL_ImagePixelator Audio Reactive] DEBUG: Blend intensity = {blend_intensity}")
        print(f"[FL_ImagePixelator Audio Reactive] DEBUG: Invert = {invert}")
        print(f"{'='*60}\n")

        try:
            # Parse envelope JSON
            envelope_data = json.loads(envelope_json)
            envelope = envelope_data['envelope']

            batch_size = image.shape[0]
            num_envelope_frames = len(envelope)

            print(f"[FL_ImagePixelator Audio Reactive] Input frames: {batch_size}")
            print(f"[FL_ImagePixelator Audio Reactive] Envelope frames: {num_envelope_frames}")

            # Handle frame count mismatch
            if batch_size != num_envelope_frames:
                print(f"[FL_ImagePixelator Audio Reactive] WARNING: Frame count mismatch! Using min({batch_size}, {num_envelope_frames})")
                max_frames = min(batch_size, num_envelope_frames)
            else:
                max_frames = batch_size

            # Prepare mask batch if provided
            mask_images = self.prepare_mask_batch(mask, max_frames) if mask is not None else None

            # PASS 1: Generate pixelation effect for all frames with static parameters
            print(f"[FL_ImagePixelator Audio Reactive] PASS 1: Generating pixelation effect...")
            pixelated_frames = []

            pbar = ProgressBar(max_frames)
            for b in range(max_frames):
                single_image = image[b].unsqueeze(0)  # Add batch dimension
                modulated_scale_factor = self.apply_modulation(scale_factor, modulation, max_frames)
                pixelated_image = self.apply_pixelation_tensor(single_image, modulated_scale_factor)
                pixelated_image = self.process(pixelated_image, kernel_size)

                # Apply mask if provided
                if mask_images is not None:
                    pixelated_pil = self.t2p(pixelated_image.squeeze(0))
                    original_pil = self.t2p(image[b])
                    mask_img = self.process_mask(mask_images[b], pixelated_pil.size)

                    # Blend original and pixelated based on mask
                    pixelated_array = np.array(pixelated_pil).astype(np.float32)
                    original_array = np.array(original_pil).astype(np.float32)
                    mask_array = np.array(mask_img).astype(np.float32) / 255.0

                    # Expand mask to 3 channels
                    if len(mask_array.shape) == 2:
                        mask_array = np.stack([mask_array] * 3, axis=-1)

                    # Blend: mask=1.0 shows pixelated, mask=0.0 shows original
                    blended = pixelated_array * mask_array + original_array * (1.0 - mask_array)
                    pixelated_image = self.p2t(Image.fromarray(blended.astype(np.uint8)))

                pixelated_frames.append(pixelated_image)
                pbar.update_absolute(b + 1)

            # Stack pixelated frames
            pixelated_batch = torch.cat(pixelated_frames, dim=0)

            print(f"[FL_ImagePixelator Audio Reactive] PASS 2: Applying envelope-based blending...")

            # PASS 2: Blend original and pixelated frames based on envelope
            output_frames = []

            for frame_idx in range(max_frames):
                # Get envelope value for this frame
                envelope_value = envelope[frame_idx]

                # Calculate blend amount
                if invert:
                    # Invert: high envelope = show original (less pixelation)
                    blend_amount = (1.0 - envelope_value) * blend_intensity
                else:
                    # Normal: high envelope = more pixelation
                    blend_amount = envelope_value * blend_intensity

                # Clamp blend amount
                blend_amount = max(0.0, min(1.0, blend_amount))

                # Blend original and pixelated frames
                # blend_amount=0: original footage
                # blend_amount=1: full pixelation effect
                original_frame = image[frame_idx]
                pixelated_frame = pixelated_batch[frame_idx]

                blended_frame = (1.0 - blend_amount) * original_frame + blend_amount * pixelated_frame
                output_frames.append(blended_frame)

                if frame_idx % 100 == 0 or frame_idx < 5:
                    print(f"[FL_ImagePixelator Audio Reactive] Frame {frame_idx}: envelope={envelope_value:.3f}, blend={blend_amount:.3f}")

            # Stack all frames
            output_tensor = torch.stack(output_frames, dim=0)

            print(f"\n{'='*60}")
            print(f"[FL_ImagePixelator Audio Reactive] Processing complete!")
            print(f"[FL_ImagePixelator Audio Reactive] Output frames: {output_tensor.shape[0]}")
            print(f"[FL_ImagePixelator Audio Reactive] Output shape: {output_tensor.shape}")
            print(f"{'='*60}\n")

            return (output_tensor,)

        except Exception as e:
            error_msg = f"Error in audio-reactive mode: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL_ImagePixelator Audio Reactive] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"[FL_ImagePixelator Audio Reactive] Falling back to static mode...")
            print(f"{'='*60}\n")
            # Fallback to static mode
            return self._apply_static(image, mask, scale_factor, kernel_size, modulation)

    def apply_modulation(self, scale_factor, modulation, total_frames):
        modulation_factor = 1 + modulation * torch.sin(2 * torch.pi * torch.tensor(self.modulation_index / total_frames))
        modulated_scale_factor = scale_factor * modulation_factor.item()
        self.modulation_index += 1
        return modulated_scale_factor

    def apply_pixelation_pil(self, input_image, scale_factor):
        width, height = input_image.size
        new_size = (int(width * scale_factor), int(height * scale_factor))
        resized_image = input_image.resize(new_size, Image.NEAREST)
        pixelated_image = resized_image.resize((width, height), Image.NEAREST)
        return pixelated_image

    def apply_pixelation_tensor(self, input_image, scale_factor):
        _, num_channels, height, width = input_image.shape
        new_height, new_width = max(1, int(height * scale_factor)), max(1, int(width * scale_factor))
        resized_tensor = torch.nn.functional.interpolate(input_image, size=(new_height, new_width), mode='nearest')
        output_tensor = torch.nn.functional.interpolate(resized_tensor, size=(height, width), mode='nearest')
        return output_tensor

    def process(self, image, kernel_size):
        device = comfy.model_management.get_torch_device()
        kernel = torch.ones(kernel_size, kernel_size, device=device)
        image_k = image.to(device).movedim(-1, 1)
        output = gradient(image_k, kernel)
        img_out = output.to(comfy.model_management.intermediate_device()).movedim(1, -1)
        return img_out