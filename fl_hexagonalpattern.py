import torch
import numpy as np
from PIL import Image, ImageDraw
import math
import sys

class FL_HexagonalPattern:
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
    CATEGORY = "🏵️Fill Nodes"

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

    def calculate_adjusted_hexagon_size(self, width, height, hexagon_size, spacing):
        horizontal_count = math.ceil(width / (hexagon_size * spacing))
        vertical_count = math.ceil(height / (hexagon_size * spacing * math.sqrt(3) / 2))

        adjusted_width = width / horizontal_count
        adjusted_height = height / (vertical_count * math.sqrt(3) / 2)

        return min(adjusted_width, adjusted_height) / spacing

    def hexagonal_pattern(self, images, hexagon_size=100, shadow_offset=5, shadow_color="black", shadow_opacity=0.5,
                          background_color="white", rotation=0.0, spacing=1.0):
        out = []
        total_images = len(images)
        for i, img_tensor in enumerate(images, start=1):
            p = self.t2p(img_tensor)
            width, height = p.size

            adjusted_hexagon_size = self.calculate_adjusted_hexagon_size(width, height, hexagon_size, spacing)
            hexagon_mask = self.create_hexagon_mask(int(adjusted_hexagon_size))

            output_image = Image.new("RGBA", (width, height), background_color)

            for y in range(0, height, int(adjusted_hexagon_size * spacing * math.sqrt(3) / 2)):
                for x in range(0, width, int(adjusted_hexagon_size * spacing)):
                    if y % (2 * int(adjusted_hexagon_size * spacing * math.sqrt(3) / 2)) == int(adjusted_hexagon_size * spacing * math.sqrt(3) / 2):
                        x += int(adjusted_hexagon_size * spacing) // 2

                    cropped_hexagon = p.crop((x, y, x + int(adjusted_hexagon_size), y + int(adjusted_hexagon_size))).rotate(rotation, expand=True)

                    shadow = Image.new("RGBA", cropped_hexagon.size, (0, 0, 0, 0))
                    shadow_mask = hexagon_mask.copy().resize(cropped_hexagon.size)
                    shadow.paste(shadow_color, (shadow_offset, shadow_offset), shadow_mask)
                    shadow.putalpha(int(255 * shadow_opacity))

                    output_image.paste(shadow, (x + shadow_offset, y + shadow_offset), shadow_mask)
                    output_image.paste(cropped_hexagon, (x, y), shadow_mask)

            o = np.array(output_image.convert("RGB")).astype(np.float32) / 255.0
            o = torch.from_numpy(o).unsqueeze(0)
            out.append(o)

            # Print progress update
            progress = i / total_images * 100
            sys.stdout.write(f"\rProcessing images: {progress:.2f}%")
            sys.stdout.flush()

        # Print a new line after the progress update
        print()

        out = torch.cat(out, 0)
        return (out,)