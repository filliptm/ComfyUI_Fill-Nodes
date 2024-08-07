import torch
import numpy as np
from PIL import Image
import math
from comfy.utils import ProgressBar

class FL_Ripple:
    def __init__(self):
        self.modulation_index = 0

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
                "center_x": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "center_y": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "modulation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ripple"
    CATEGORY = "🏵️Fill Nodes/VFX"

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def ripple(self, images, amplitude=10.0, frequency=20.0, phase=0.0, center_x=50.0, center_y=50.0, modulation=0.0):
        out = []
        total_images = len(images)
        pbar = ProgressBar(total_images)
        for i, img in enumerate(images, start=1):
            p = self.t2p(img)
            width, height = p.size
            center_x_pixel = int(center_x / 100 * width)
            center_y_pixel = int(center_y / 100 * height)

            x, y = np.meshgrid(np.arange(width), np.arange(height))
            dx = x - center_x_pixel
            dy = y - center_y_pixel
            distance = np.sqrt(dx ** 2 + dy ** 2)

            # Apply modulation to amplitude and frequency
            modulation_factor = 1 + modulation * math.sin(2 * math.pi * self.modulation_index / total_images)
            modulated_amplitude = amplitude * modulation_factor
            modulated_frequency = frequency * modulation_factor

            angle = distance / modulated_frequency * 2 * np.pi + np.radians(phase)
            offset_x = (modulated_amplitude * np.sin(angle)).astype(int)
            offset_y = (modulated_amplitude * np.cos(angle)).astype(int)

            sample_x = np.clip(x + offset_x, 0, width - 1)
            sample_y = np.clip(y + offset_y, 0, height - 1)

            p_array = np.array(p)
            rippled_array = p_array[sample_y, sample_x]

            o = rippled_array.astype(np.float32) / 255.0
            o = torch.from_numpy(o).unsqueeze(0)
            out.append(o)

            self.modulation_index += 1

            pbar.update_absolute(i)

        out = torch.cat(out, 0)
        return (out,)