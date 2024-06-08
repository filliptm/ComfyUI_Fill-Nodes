from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from pathlib import Path
import os

from comfy.utils import ProgressBar
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
DIR_FONTS = ROOT / "fonts"

def parse_fonts() -> dict:
    mgr = font_manager.FontManager()
    return {f"+{font.name[0].upper()}/{font.name}": font.fname for font in mgr.ttflist}

class FL_Ascii:
    # Retrieve the environment variable and convert to lowercase
    env_var_value = os.getenv("FL_USE_SYSTEM_FONTS", 'false').strip().lower()
    # Scan the fonts folder for available fonts
    if env_var_value.strip() in ('true', '1', 't'):
        FONTS = parse_fonts()
    else:
        FONTS = {f"{str(font)}": str(font) for font in DIR_FONTS.glob("*.[to][tf][f]")}
    print(f"LOADED {len(FONTS)} FONTS")
    FONT_NAMES = sorted(FONTS.keys())
    FONT_NAMES.sort(key=lambda i: i.lower())

    def __init__(self):
        self.spacing_index = 0
        self.font_size_index = 0

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
                    "default": "\._♥♦♣MachineDelusions♣♦♥_./",
                    "description": "characters to use"
                }),
                "font": (s.FONT_NAMES, ),
                "sequence_toggle": (["off", "on"], {
                    "default": "off",
                    "description": "toggle to type characters in sequence"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_ascii_art_effect"
    CATEGORY = "🏵️Fill Nodes"

    def apply_ascii_art_effect(self, image: torch.Tensor, spacing: int, font_size: int, characters, font: str, sequence_toggle: str):
        batch_size = image.shape[0]
        result = torch.zeros_like(image)

        # get the local folder or system font
        font = self.FONTS[font]

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

            result_b = ascii_art_effect(img_b, current_spacing, current_font_size, characters, font, sequence_toggle)
            result_b = torch.tensor(np.array(result_b)) / 255.0
            result[b] = result_b
            pbar.update_absolute(b)

        return (result,)

def ascii_art_effect(image: torch.Tensor, spacing: int, font_size: int, characters, font_file, sequence_toggle):
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

            if sequence_toggle == "on":
                char = characters[char_index % len(characters)]
                char_index += 1
            else:
                k = (r + g + b) // 3
                char = characters[k * len(characters) // 256]

            draw_image.text(
                (j * spacing, i * spacing),
                char,
                font=font,
                fill=(r, g, b)
            )
    return ascii_image
