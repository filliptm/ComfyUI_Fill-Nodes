import torch
import numpy as np
import json
from PIL import Image
from sklearn.cluster import KMeans

from comfy.utils import ProgressBar

class FL_PixelArtShader:
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
                    "description": "Invert envelope (show pixel art when quiet)"
                }),
                "mask": ("IMAGE", {"default": None, "description": "Optional mask to control where effect is applied"}),
                # Effect parameters
                "pixel_size": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 100.0, "step": 1.0}),
                "color_depth": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 255.0, "step": 1.0}),
                "use_aspect_ratio": ("BOOLEAN", {"default": True}),
                "palette_image": ("IMAGE", {"default": None}),
                "palette_colors": ("INT", {"default": 16, "min": 2, "max": 16, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_pixel_art_shader"
    CATEGORY = "🏵️Fill Nodes/VFX"

    def apply_pixel_art_shader(self, images, use_aspect_ratio, pixel_size, color_depth, palette_image=None,
                               palette_colors=16, mask=None, envelope_json="", blend_intensity=1.0, invert=False):
        # Check if audio-reactive mode is enabled
        use_audio_reactive = envelope_json and envelope_json.strip() != ""

        if use_audio_reactive:
            return self._apply_audio_reactive(images, use_aspect_ratio, pixel_size, color_depth, palette_image, palette_colors, mask, envelope_json, blend_intensity, invert)
        else:
            return self._apply_static(images, use_aspect_ratio, pixel_size, color_depth, palette_image, palette_colors, mask)

    def _apply_static(self, images, use_aspect_ratio, pixel_size, color_depth, palette_image=None, palette_colors=16, mask=None):
        """Static pixel art effect without audio reactivity"""
        result = []
        total_images = len(images)
        pbar = ProgressBar(total_images)

        if palette_image is not None:
            palette = extract_palette(self.t2p(palette_image[0]), palette_colors)
        else:
            palette = None

        mask_images = self.prepare_mask_batch(mask, total_images) if mask is not None else None

        for idx, image in enumerate(images):
            img = self.t2p(image)

            mask_img = self.process_mask(mask_images[idx], img.size) if mask_images is not None else None

            result_img = pixel_art_effect(img, pixel_size, color_depth, use_aspect_ratio, palette, mask_img)
            result_img = self.p2t(result_img)
            result.append(result_img)
            pbar.update_absolute(idx + 1)

        return (torch.cat(result, dim=0),)

    def _apply_audio_reactive(self, images, use_aspect_ratio, pixel_size, color_depth, palette_image, palette_colors, mask, envelope_json, blend_intensity, invert):
        """Audio-reactive pixel art effect with envelope-based blending"""
        print(f"\n{'='*60}")
        print(f"[FL_PixelArt Audio Reactive] DEBUG: Function called")
        print(f"[FL_PixelArt Audio Reactive] DEBUG: Input shape = {images.shape}")
        print(f"[FL_PixelArt Audio Reactive] DEBUG: Blend intensity = {blend_intensity}")
        print(f"[FL_PixelArt Audio Reactive] DEBUG: Invert = {invert}")
        print(f"{'='*60}\n")

        try:
            # Parse envelope JSON
            envelope_data = json.loads(envelope_json)
            envelope = envelope_data['envelope']

            batch_size = len(images)
            num_envelope_frames = len(envelope)

            print(f"[FL_PixelArt Audio Reactive] Input frames: {batch_size}")
            print(f"[FL_PixelArt Audio Reactive] Envelope frames: {num_envelope_frames}")

            # Handle frame count mismatch
            if batch_size != num_envelope_frames:
                print(f"[FL_PixelArt Audio Reactive] WARNING: Frame count mismatch! Using min({batch_size}, {num_envelope_frames})")
                max_frames = min(batch_size, num_envelope_frames)
            else:
                max_frames = batch_size

            # Prepare palette if provided
            if palette_image is not None:
                palette = extract_palette(self.t2p(palette_image[0]), palette_colors)
            else:
                palette = None

            # Prepare mask batch if provided
            mask_images = self.prepare_mask_batch(mask, max_frames) if mask is not None else None

            # PASS 1: Generate pixel art effect for all frames with static parameters
            print(f"[FL_PixelArt Audio Reactive] PASS 1: Generating pixel art effect...")
            pixel_art_frames = []

            pbar = ProgressBar(max_frames)
            for b in range(max_frames):
                img = self.t2p(images[b])

                # Process mask for this frame if provided
                mask_img = self.process_mask(mask_images[b], img.size) if mask_images is not None else None

                pixel_art_img = pixel_art_effect(img, pixel_size, color_depth, use_aspect_ratio, palette, mask_img)
                pixel_art_tensor = self.p2t(pixel_art_img)
                pixel_art_frames.append(pixel_art_tensor)
                pbar.update_absolute(b + 1)

            # Stack pixel art frames
            pixel_art_batch = torch.cat(pixel_art_frames, dim=0)

            print(f"[FL_PixelArt Audio Reactive] PASS 2: Applying envelope-based blending...")

            # PASS 2: Blend original and pixel art frames based on envelope
            output_frames = []

            for frame_idx in range(max_frames):
                # Get envelope value for this frame
                envelope_value = envelope[frame_idx]

                # Calculate blend amount
                if invert:
                    # Invert: high envelope = show original (less pixel art)
                    blend_amount = (1.0 - envelope_value) * blend_intensity
                else:
                    # Normal: high envelope = more pixel art
                    blend_amount = envelope_value * blend_intensity

                # Clamp blend amount
                blend_amount = max(0.0, min(1.0, blend_amount))

                # Blend original and pixel art frames
                # blend_amount=0: original footage
                # blend_amount=1: full pixel art effect
                original_frame = images[frame_idx]
                pixel_art_frame = pixel_art_batch[frame_idx]

                blended_frame = (1.0 - blend_amount) * original_frame + blend_amount * pixel_art_frame
                output_frames.append(blended_frame)

                if frame_idx % 100 == 0 or frame_idx < 5:
                    print(f"[FL_PixelArt Audio Reactive] Frame {frame_idx}: envelope={envelope_value:.3f}, blend={blend_amount:.3f}")

            # Stack all frames
            output_tensor = torch.stack(output_frames, dim=0)

            print(f"\n{'='*60}")
            print(f"[FL_PixelArt Audio Reactive] Processing complete!")
            print(f"[FL_PixelArt Audio Reactive] Output frames: {output_tensor.shape[0]}")
            print(f"[FL_PixelArt Audio Reactive] Output shape: {output_tensor.shape}")
            print(f"{'='*60}\n")

            return (output_tensor,)

        except Exception as e:
            error_msg = f"Error in audio-reactive mode: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL_PixelArt Audio Reactive] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"[FL_PixelArt Audio Reactive] Falling back to static mode...")
            print(f"{'='*60}\n")
            # Fallback to static mode
            return self._apply_static(images, use_aspect_ratio, pixel_size, color_depth, palette_image, palette_colors, mask)

    def t2p(self, t):
        i = 255.0 * t.cpu().numpy().squeeze()
        return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    def p2t(self, p):
        i = np.array(p).astype(np.float32) / 255.0
        return torch.from_numpy(i).unsqueeze(0)

    def prepare_mask_batch(self, mask, total_images):
        if mask is None:
            return None
        mask_images = [self.t2p(m) for m in mask]
        if len(mask_images) < total_images:
            mask_images = mask_images * (total_images // len(mask_images) + 1)
        return mask_images[:total_images]

    def process_mask(self, mask, target_size):
        mask = mask.resize(target_size, Image.LANCZOS)
        return mask.convert('L') if mask.mode != 'L' else mask

def extract_palette(image, n_colors):
    image = image.convert('RGB')
    pixels = np.array(image).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    return torch.from_numpy(colors.astype(np.float32) / 255.0).to("cuda")

def pixel_art_effect(image, pixel_size, color_depth, use_aspect_ratio, palette, mask=None):
    image = torch.tensor(np.array(image)).float().to("cuda") / 255.0
    height, width = image.shape[0], image.shape[1]

    if use_aspect_ratio:
        aspect_ratio = width / height
        pixel_size_x, pixel_size_y = pixel_size, pixel_size / aspect_ratio
    else:
        pixel_size_x = pixel_size_y = pixel_size

    new_width = int(width / pixel_size_x)
    new_height = int(height / pixel_size_y)

    # Resize the image to create the pixelated effect
    pixelated = image.permute(2, 0, 1).unsqueeze(0)
    pixelated = torch.nn.functional.interpolate(pixelated, size=(new_height, new_width), mode='nearest')
    pixelated = torch.nn.functional.interpolate(pixelated, size=(height, width), mode='nearest')
    pixelated = pixelated.squeeze(0).permute(1, 2, 0)

    # Apply color depth reduction
    pixelated = adjust_color(pixelated, color_depth)

    if palette is not None:
        pixelated = apply_palette(pixelated, palette)

    if mask is not None:
        mask_tensor = torch.tensor(np.array(mask)).float().to("cuda") / 255.0
        mask_tensor = mask_tensor.unsqueeze(-1).expand(-1, -1, 3)
        pixelated = pixelated * mask_tensor + image * (1 - mask_tensor)

    return Image.fromarray((pixelated.cpu().numpy() * 255).astype(np.uint8))

def adjust_color(color, color_depth):
    return torch.floor(color * color_depth) / color_depth

def apply_palette(image, palette):
    original_shape = image.shape
    pixels = image.reshape(-1, 3)
    distances = torch.cdist(pixels, palette)
    nearest_palette_indices = torch.argmin(distances, dim=1)
    new_pixels = palette[nearest_palette_indices]
    return new_pixels.reshape(original_shape)