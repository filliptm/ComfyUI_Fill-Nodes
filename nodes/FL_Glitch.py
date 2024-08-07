import torch
import numpy as np
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

    def s2b(self, v):
        return v == "Enable"

    def glitch(self, images, glitch_amount=1, color_offset="Disable", seed=0):
        color_offset = self.s2b(color_offset)
        g = ImageGlitcher()
        out = []
        total_images = len(images)

        # Convert seed to a list if it's a single value
        if not isinstance(seed, list):
            seed = [seed] * total_images

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

            o = np.array(f.convert("RGB")).astype(np.float32) / 255.0
            o = torch.from_numpy(o).unsqueeze(0)
            out.append(o)
            pbar.update_absolute(i)

        out = torch.cat(out, 0)
        return (out,)