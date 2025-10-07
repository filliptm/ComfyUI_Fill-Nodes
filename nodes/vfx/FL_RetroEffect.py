import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter  # Added ImageFilter import
import sys

class FL_RetroEffect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "color_offset": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scanline_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vignette_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_retro_effect"
    CATEGORY = "🏵️Fill Nodes/VFX"

    def apply_retro_effect(self, images, color_offset, scanline_strength, vignette_strength, noise_strength):
        result = []
        total_images = len(images)

        for i, image in enumerate(images, start=1):
            img = self.t2p(image)
            result_img = self.process_image(img, color_offset, scanline_strength, vignette_strength, noise_strength)
            result_img = self.p2t(result_img)
            result.append(result_img)

            # Update the print log
            progress = i / total_images * 100
            sys.stdout.write(f"\rProcessing images: {progress:.2f}%")
            sys.stdout.flush()

        # Print a new line after the progress log
        print()

        return (torch.cat(result, dim=0),)

    def process_image(self, image, color_offset, scanline_strength, vignette_strength, noise_strength):
        # Apply color offset
        r, g, b = image.split()
        r = ImageEnhance.Brightness(r).enhance(1 + color_offset)
        b = ImageEnhance.Brightness(b).enhance(1 - color_offset)
        image = Image.merge("RGB", (r, g, b))

        # Apply scanlines
        scanline_mask = Image.new("L", image.size, 0)
        for y in range(0, image.size[1], 2):
            scanline_mask.paste(int(255 * scanline_strength), (0, y, image.size[0], y + 1))
        image.paste(image, mask=scanline_mask)

        # Apply vignette
        vignette_mask = Image.new("L", image.size, 0)
        vignette_mask.paste(255, (0, 0, image.size[0], image.size[1]))
        vignette_mask = ImageOps.invert(vignette_mask)
        vignette_mask = vignette_mask.filter(ImageFilter.GaussianBlur(radius=image.size[0] * vignette_strength))
        image.paste(image, mask=ImageOps.invert(vignette_mask))

        # Apply noise
        noise = Image.effect_noise(image.size, sigma=noise_strength * 255).convert("RGB")
        image = Image.blend(image, noise, noise_strength)

        return image

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
