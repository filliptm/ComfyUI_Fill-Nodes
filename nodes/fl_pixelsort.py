import torch
import numpy as np
from PIL import Image
from colorsys import rgb_to_hsv

from comfy.utils import ProgressBar

class FL_PixelSort:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "direction": (["Horizontal", "Vertical"],),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "smoothing": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rotation": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pixel_sort_saturation"
    CATEGORY = "🏵️Fill Nodes/VFX"

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def saturation(self, pixel):
        r, g, b = pixel
        _, s, _ = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        return s

    def pixel_sort_saturation(self, images, direction="Horizontal", threshold=0.5, smoothing=0.1, rotation=0):
        out = []
        total_images = len(images)
        pbar = ProgressBar(total_images)
        for i, img in enumerate(images, start=1):
            p = self.t2p(img)
            sorted_image = self.sort_pixels(p, self.saturation, threshold, smoothing, rotation)
            o = np.array(sorted_image.convert("RGB")).astype(np.float32) / 255.0
            o = torch.from_numpy(o).unsqueeze(0)
            out.append(o)
            pbar.update_absolute(i)
        out = torch.cat(out, 0)
        return (out,)

    def sort_pixels(self, image, value, threshold, smoothing, rotation=0):
        pixels = np.rot90(np.array(image), rotation)
        values = np.apply_along_axis(value, 2, pixels)
        edges = np.apply_along_axis(lambda row: np.convolve(row, [-1, 1], 'same'), 0, values > threshold)
        edges = np.maximum(edges, 0)
        edges = np.minimum(edges, 1)
        edges = np.convolve(edges.flatten(), np.ones(int(smoothing * pixels.shape[1])), 'same').reshape(edges.shape)

        intervals = [np.flatnonzero(row) for row in edges]

        pbar = ProgressBar(len(values))
        for row, key in enumerate(values):
            order = np.split(key, intervals[row])
            for index, interval in enumerate(order[1:]):
                order[index + 1] = np.argsort(interval) + intervals[row][index]
            order[0] = range(order[0].size)
            order = np.concatenate(order)

            for channel in range(3):
                pixels[row, :, channel] = pixels[row, order.astype('uint32'), channel]

            pbar.update_absolute(row)

        return Image.fromarray(np.rot90(pixels, -rotation))