from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import sys

class FL_Ascii:
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
                    "min": 4,
                    "max": 100,
                    "step": 2,
                }),
                "font_size": ("INT", {
                    "default": 20,
                    "min": 4,
                    "max": 100,
                    "step": 2,
                }),
                "characters": ("STRING", {
                    "default": "\._♥♦♣MachineDelusions♣♦♥_./",
                    "description": "characters to use"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_ascii_art_effect"
    CATEGORY = "🏵️Fill Nodes"

    def apply_ascii_art_effect(self, image: torch.Tensor, spacing: int, font_size: int, characters):
        batch_size, height, width, channels = image.shape
        result = torch.zeros_like(image)

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

            result_b = ascii_art_effect(img_b, current_spacing, current_font_size, characters)
            result_b = torch.tensor(np.array(result_b)) / 255.0
            result[b] = result_b

            # Update the print log
            progress = (b + 1) / batch_size * 100
            sys.stdout.write(f"\rProcessing images: {progress:.2f}%")
            sys.stdout.flush()

        # Print a new line after the progress log
        print()

        return (result,)


def ascii_art_effect(image: torch.Tensor, spacing: int, font_size: int, characters):
    chars = characters
    small_image = image.resize((image.size[0] // spacing, image.size[1] // spacing), Image.Resampling.NEAREST)

    def get_char(value):
        return chars[value * len(chars) // 256]

    ascii_image = Image.new('RGB', image.size, (0, 0, 0))
    font = ImageFont.truetype("arial.ttf", font_size)
    draw_image = ImageDraw.Draw(ascii_image)

    for i in range(small_image.height):
        for j in range(small_image.width):
            r, g, b = small_image.getpixel((j, i))
            k = (r + g + b) // 3
            draw_image.text(
                (j * spacing, i * spacing),
                get_char(k),
                font=font,
                fill=(r, g, b)
            )

    return ascii_image