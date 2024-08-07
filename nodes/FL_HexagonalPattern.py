import torch
import numpy as np
from PIL import Image, ImageDraw
import math

from comfy.utils import ProgressBar

class FL_HexagonalPattern:
    def __init__(self):
        self.hexagon_size_index = 0
        self.shadow_offset_index = 0
        self.shadow_color_index = 0
        self.background_color_index = 0
        self.rotation_index = 0
        self.spacing_index = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "hexagon_size": ("INT", {"default": 100, "min": 50, "max": 500, "step": 10}),
                "shadow_offset": ("INT", {"default": 5, "min": 0, "max": 20, "step": 1}),
                "shadow_color": ("STRING", {"default": "purple"}),
                "background_color": ("STRING", {"default": "black"}),
                "rotation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "spacing": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "hexagonal_pattern"
    CATEGORY = "🏵️Fill Nodes/VFX"

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def create_hexagon_mask(self, size):
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.regular_polygon((size // 2, size // 2, size // 2), 6, fill=255)
        return mask

    def process_input_value(self, value, index):
        if isinstance(value, list):
            if index >= len(value):
                print(f"Warning: Value list index out of range. Using the last value.")
                index = len(value) - 1
            current_value = value[index]
            index = (index + 1) % len(value)
        else:
            current_value = value

        if hasattr(current_value, 'values'):
            current_value = float(current_value.values[0])

        return current_value, index

    def hexagonal_pattern(self, images, hexagon_size=100, shadow_offset=5, shadow_color="black", shadow_opacity=0.5,
                          background_color="white", rotation=0.0, spacing=1.0):
        out = []
        total_images = len(images)
        pbar = ProgressBar(total_images)
        for i, img_tensor in enumerate(images, start=1):
            p = self.t2p(img_tensor)
            width, height = p.size

            current_hexagon_size, self.hexagon_size_index = self.process_input_value(hexagon_size, self.hexagon_size_index)
            current_shadow_offset, self.shadow_offset_index = self.process_input_value(shadow_offset, self.shadow_offset_index)
            current_shadow_color, self.shadow_color_index = self.process_input_value(shadow_color, self.shadow_color_index)
            current_background_color, self.background_color_index = self.process_input_value(background_color, self.background_color_index)
            current_rotation, self.rotation_index = self.process_input_value(rotation, self.rotation_index)
            current_spacing, self.spacing_index = self.process_input_value(spacing, self.spacing_index)

            hexagon_mask = self.create_hexagon_mask(current_hexagon_size)

            output_image = Image.new("RGBA", (width, height), current_background_color)

            for y in range(0, height, int(current_hexagon_size * current_spacing * math.sqrt(3) / 2)):
                for x in range(0, width, int(current_hexagon_size * current_spacing)):
                    if y % (2 * int(current_hexagon_size * current_spacing * math.sqrt(3) / 2)) == int(current_hexagon_size * current_spacing * math.sqrt(3) / 2):
                        x += int(current_hexagon_size * current_spacing) // 2

                    cropped_hexagon = p.crop((x, y, x + current_hexagon_size, y + current_hexagon_size)).rotate(current_rotation, expand=True)

                    shadow = Image.new("RGBA", cropped_hexagon.size, (0, 0, 0, 0))
                    shadow_mask = hexagon_mask.copy().resize(cropped_hexagon.size)
                    shadow.paste(current_shadow_color, (current_shadow_offset, current_shadow_offset), shadow_mask)
                    shadow.putalpha(int(255 * shadow_opacity))

                    output_image.paste(shadow, (x + current_shadow_offset, y + current_shadow_offset), shadow_mask)
                    output_image.paste(cropped_hexagon, (x, y), shadow_mask)

            o = np.array(output_image.convert("RGB")).astype(np.float32) / 255.0
            o = torch.from_numpy(o).unsqueeze(0)
            out.append(o)

            pbar.update_absolute(i)

        out = torch.cat(out, 0)
        return (out,)
