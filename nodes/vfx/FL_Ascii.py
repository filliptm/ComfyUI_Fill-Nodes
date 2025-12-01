
import os
import torch
import numpy as np
import json
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from ..sup import ROOT_FONTS

from comfy.utils import ProgressBar

def parse_fonts() -> dict:
    mgr = font_manager.FontManager()
    return {f"{font.name[0].upper()}/{font.name}": font.fname for font in mgr.ttflist}

class FL_Ascii:
    # Retrieve the environment variable and convert to lowercase
    env_var_value = os.getenv("FL_USE_SYSTEM_FONTS", 'false').strip().lower()
    # Scan the fonts folder for available fonts
    if env_var_value.strip() in ('true', '1', 't'):
        FONTS = parse_fonts()
    else:
        FONTS = {f"{str(font)}": str(font) for font in ROOT_FONTS.glob("*.[to][tf][f]")}
        FONTS = {f"{str(font.stem)}": str(font) for font in ROOT_FONTS.glob("*.[to][tf][f]")}
    print(f"LOADED {len(FONTS)} FONTS")
    FONT_NAMES = sorted(FONTS.keys())
    FONT_NAMES.sort(key=lambda i: i.lower())
    DESCRIPTION = """
FL_Ascii is a class that converts an image into ASCII art using specified characters, font, spacing, and font size.
You can select either local or system fonts based on an environment variable. The class provides customization options
such as using a sequence of characters or mapping characters based on pixel intensity. The spacing and font size can
be specified as single values or lists to vary across the image. This tool is useful for creating stylized visual
representations of images with ASCII characters.
"""

    def __init__(self):
        self.spacing_index = 0
        self.font_size_index = 0

    def t2p(self, t):
        """Tensor to PIL"""
        i = 255.0 * t.cpu().numpy().squeeze()
        return Image.fromarray(np.clip(i, 0, 255).astype('uint8'))

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
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "spacing": ("INT", {
                    "default": 20,
                    "min": 1,
                    "step": 1,
                }),
                "font_size": ("INT", {
                    "default": 20,
                    "min": 1,
                    "step": 1,
                }),
                "characters": ("STRING", {
                    "default": "MachineDelusions",
                    "description": "characters to use"
                }),
                "font": (s.FONT_NAMES, {"default": "arial"}),
                "sequence_toggle": ("BOOLEAN", {
                    "default": False,
                    "description": "toggle to type characters in sequence"
                }),
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
                    "description": "Invert envelope (show ASCII when quiet)"
                }),
                "mask": ("IMAGE", {
                    "default": None,
                    "description": "Optional mask to control where effect is applied"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_ascii_art_effect"
    CATEGORY = "🏵️Fill Nodes/VFX"

    def apply_ascii_art_effect(self, image: torch.Tensor, spacing: int, font_size: int, characters, font: str, sequence_toggle: bool, envelope_json: str = "", blend_intensity: float = 1.0, invert: bool = False, mask=None):
        # Check if audio-reactive mode is enabled
        use_audio_reactive = envelope_json and envelope_json.strip() != ""

        if use_audio_reactive:
            return self._apply_audio_reactive(image, spacing, font_size, characters, font, sequence_toggle, envelope_json, blend_intensity, invert, mask)
        else:
            return self._apply_static(image, spacing, font_size, characters, font, sequence_toggle, mask)

    def _apply_static(self, image: torch.Tensor, spacing: int, font_size: int, characters, font: str, sequence_toggle: bool, mask=None):
        """Static ASCII effect without audio reactivity"""
        batch_size = image.shape[0]
        result = torch.zeros_like(image)

        # get the local folder or system font
        font_path = self.FONTS[font]

        # Prepare mask batch if provided
        mask_images = self.prepare_mask_batch(mask, batch_size) if mask is not None else None

        pbar = ProgressBar(batch_size)
        for b in range(batch_size):
            img_b = image[b] * 255.0
            img_b = Image.fromarray(img_b.numpy().astype('uint8'), 'RGB')

            # Check if spacing is a list and get the current value
            if isinstance(spacing, list):
                if self.spacing_index >= len(spacing):
                    print("Warning: Spacing list index out of range. Using the last value.")
                    self.spacing_index = len(spacing) - 1
                current_spacing = spacing[self.spacing_index]
                self.spacing_index = (self.spacing_index + 1) % len(spacing)
            else:
                current_spacing = spacing

            # Check if font_size is a list and get the current value
            if isinstance(font_size, list):
                if self.font_size_index >= len(font_size):
                    print("Warning: Font size list index out of range. Using the last value.")
                    self.font_size_index = len(font_size) - 1
                current_font_size = font_size[self.font_size_index]
                self.font_size_index = (self.font_size_index + 1) % len(font_size)
            else:
                current_font_size = font_size

            # Process mask for this frame if provided
            mask_img = self.process_mask(mask_images[b], img_b.size) if mask_images is not None else None

            # Apply ASCII effect
            result_b = ascii_art_effect(img_b, current_spacing, current_font_size, characters, font_path, sequence_toggle, mask_img)
            result_b = torch.tensor(np.array(result_b)) / 255.0
            result[b] = result_b
            pbar.update_absolute(b)
            print(f"[FL_Ascii] {b+1} of {batch_size}")

        return (result,)

    def _apply_audio_reactive(self, image: torch.Tensor, spacing: int, font_size: int, characters, font: str, sequence_toggle: bool, envelope_json: str, blend_intensity: float, invert: bool, mask=None):
        """Audio-reactive ASCII effect with envelope-based blending"""
        print(f"\n{'='*60}")
        print(f"[FL_Ascii Audio Reactive] DEBUG: Function called")
        print(f"[FL_Ascii Audio Reactive] DEBUG: Input shape = {image.shape}")
        print(f"[FL_Ascii Audio Reactive] DEBUG: Blend intensity = {blend_intensity}")
        print(f"[FL_Ascii Audio Reactive] DEBUG: Invert = {invert}")
        print(f"{'='*60}\n")

        try:
            # Parse envelope JSON
            envelope_data = json.loads(envelope_json)
            envelope = envelope_data['envelope']

            batch_size = image.shape[0]
            num_envelope_frames = len(envelope)

            print(f"[FL_Ascii Audio Reactive] Input frames: {batch_size}")
            print(f"[FL_Ascii Audio Reactive] Envelope frames: {num_envelope_frames}")

            # Handle frame count mismatch
            if batch_size != num_envelope_frames:
                print(f"[FL_Ascii Audio Reactive] WARNING: Frame count mismatch! Using min({batch_size}, {num_envelope_frames})")
                max_frames = min(batch_size, num_envelope_frames)
            else:
                max_frames = batch_size

            # Prepare mask batch if provided
            mask_images = self.prepare_mask_batch(mask, max_frames) if mask is not None else None

            # PASS 1: Generate ASCII effect for all frames with static parameters
            print(f"[FL_Ascii Audio Reactive] PASS 1: Generating ASCII effect...")
            ascii_frames = torch.zeros_like(image[:max_frames])
            font_path = self.FONTS[font]

            pbar = ProgressBar(max_frames)
            for b in range(max_frames):
                img_b = image[b] * 255.0
                img_b = Image.fromarray(img_b.numpy().astype('uint8'), 'RGB')

                # Use static parameters for effect generation
                current_spacing = spacing if not isinstance(spacing, list) else spacing[b % len(spacing)]
                current_font_size = font_size if not isinstance(font_size, list) else font_size[b % len(font_size)]

                # Process mask for this frame if provided
                mask_img = self.process_mask(mask_images[b], img_b.size) if mask_images is not None else None

                ascii_b = ascii_art_effect(img_b, current_spacing, current_font_size, characters, font_path, sequence_toggle, mask_img)
                ascii_b = torch.tensor(np.array(ascii_b)) / 255.0
                ascii_frames[b] = ascii_b
                pbar.update_absolute(b)

            print(f"[FL_Ascii Audio Reactive] PASS 2: Applying envelope-based blending...")

            # PASS 2: Blend original and ASCII frames based on envelope
            output_frames = []

            for frame_idx in range(max_frames):
                # Get envelope value for this frame
                envelope_value = envelope[frame_idx]

                # Calculate blend amount
                if invert:
                    # Invert: high envelope = show original (less ASCII)
                    blend_amount = (1.0 - envelope_value) * blend_intensity
                else:
                    # Normal: high envelope = more ASCII
                    blend_amount = envelope_value * blend_intensity

                # Clamp blend amount
                blend_amount = max(0.0, min(1.0, blend_amount))

                # Blend original and ASCII frames
                # blend_amount=0: original footage
                # blend_amount=1: full ASCII effect
                original_frame = image[frame_idx]
                ascii_frame = ascii_frames[frame_idx]

                blended_frame = (1.0 - blend_amount) * original_frame + blend_amount * ascii_frame
                output_frames.append(blended_frame)

                if frame_idx % 100 == 0 or frame_idx < 5:
                    print(f"[FL_Ascii Audio Reactive] Frame {frame_idx}: envelope={envelope_value:.3f}, blend={blend_amount:.3f}")

            # Stack all frames
            output_tensor = torch.stack(output_frames, dim=0)

            print(f"\n{'='*60}")
            print(f"[FL_Ascii Audio Reactive] Processing complete!")
            print(f"[FL_Ascii Audio Reactive] Output frames: {output_tensor.shape[0]}")
            print(f"[FL_Ascii Audio Reactive] Output shape: {output_tensor.shape}")
            print(f"{'='*60}\n")

            return (output_tensor,)

        except Exception as e:
            error_msg = f"Error in audio-reactive mode: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL_Ascii Audio Reactive] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"[FL_Ascii Audio Reactive] Falling back to static mode...")
            print(f"{'='*60}\n")
            # Fallback to static mode
            return self._apply_static(image, spacing, font_size, characters, font, sequence_toggle, mask)

def ascii_art_effect(image: torch.Tensor, spacing: int, font_size: int, characters, font_file, sequence_toggle: bool, mask=None):
    small_image = image.resize((image.size[0] // spacing, image.size[1] // spacing), Image.Resampling.NEAREST)
    ascii_image = Image.new('RGB', image.size, (0, 0, 0))

    try:
        font = ImageFont.truetype(font_file, font_size)
    except Exception as e:
        print(f"Error loading font '{font_file}' with size {font_size}: {str(e)}")
        # Fallback to a default font or font size
        font = ImageFont.load_default()

    draw_image = ImageDraw.Draw(ascii_image)

    char_index = 0
    for i in range(small_image.height):
        for j in range(small_image.width):
            r, g, b = small_image.getpixel((j, i))

            if sequence_toggle:
                char = characters[char_index % len(characters)]
            else:
                k = (r + g + b) // 3
                char = characters[k * len(characters) // 256]
            char_index += 1

            draw_image.text(
                (j * spacing, i * spacing),
                char,
                font=font,
                fill=(r, g, b)
            )

    # Apply mask if provided
    if mask is not None:
        # Convert images to numpy arrays for blending
        ascii_array = np.array(ascii_image).astype(np.float32)
        original_array = np.array(image).astype(np.float32)
        mask_array = np.array(mask).astype(np.float32) / 255.0

        # Expand mask to 3 channels
        if len(mask_array.shape) == 2:
            mask_array = np.stack([mask_array] * 3, axis=-1)

        # Blend: mask=1.0 shows ASCII, mask=0.0 shows original
        blended = ascii_array * mask_array + original_array * (1.0 - mask_array)
        ascii_image = Image.fromarray(blended.astype(np.uint8))

    return ascii_image
