import torch
import numpy as np
from PIL import Image
import math
import sys

class FL_Ripple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "amplitude": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "frequency": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "phase": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ripple"
    CATEGORY = "🏵️Fill Nodes"

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def ripple(self, images, amplitude=10.0, frequency=20.0, phase=0.0):
        out = []
        total_images = len(images)
        for i, img in enumerate(images, start=1):
            p = self.t2p(img)
            width, height = p.size
            center_x = width // 2
            center_y = height // 2

            x, y = np.meshgrid(np.arange(width), np.arange(height))
            dx = x - center_x
            dy = y - center_y
            distance = np.sqrt(dx ** 2 + dy ** 2)

            angle = distance / frequency * 2 * np.pi + np.radians(phase)
            offset_x = (amplitude * np.sin(angle)).astype(int)
            offset_y = (amplitude * np.cos(angle)).astype(int)

            sample_x = np.clip(x + offset_x, 0, width - 1)
            sample_y = np.clip(y + offset_y, 0, height - 1)

            p_array = np.array(p)
            rippled_array = p_array[sample_y, sample_x]

            o = rippled_array.astype(np.float32) / 255.0
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