import torch
import torch.nn.functional as F
from comfy.utils import ProgressBar


class FL_Dither:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "dither_method": (["Floyd-Steinberg", "Random", "Ordered", "Bayer"],),
                "num_colors": ("INT", {"default": 2, "min": 2, "max": 256, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_dither"
    CATEGORY = "🏵️Fill Nodes/VFX"

    def apply_dither(self, images, dither_method, num_colors):
        device = images.device
        total_images = images.shape[0]
        pbar = ProgressBar(total_images)

        result = []
        for idx in range(total_images):
            img = images[idx].unsqueeze(0)  # Add batch dimension
            dithered_img = self.dither_image(img, dither_method, num_colors, device)
            result.append(dithered_img)
            pbar.update_absolute(idx + 1)

        return (torch.cat(result, dim=0),)

    def dither_image(self, image, method, num_colors, device):
        if method == "Floyd-Steinberg":
            return self.floyd_steinberg_dither(image, num_colors, device)
        elif method == "Random":
            return self.random_dither(image, num_colors, device)
        elif method == "Ordered":
            return self.ordered_dither(image, num_colors, device)
        elif method == "Bayer":
            return self.bayer_dither(image, num_colors, device)
        else:
            return image

    def floyd_steinberg_dither(self, img, num_colors, device):
        img = img.clone()
        h, w = img.shape[2], img.shape[3]
        for y in range(h):
            for x in range(w):
                old_pixel = img[:, :, y, x].clone()
                new_pixel = torch.round(old_pixel * (num_colors - 1)) / (num_colors - 1)
                img[:, :, y, x] = new_pixel
                error = old_pixel - new_pixel
                if x + 1 < w:
                    img[:, :, y, x + 1] += error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        img[:, :, y + 1, x - 1] += error * 3 / 16
                    img[:, :, y + 1, x] += error * 5 / 16
                    if x + 1 < w:
                        img[:, :, y + 1, x + 1] += error * 1 / 16
        return img

    def random_dither(self, img, num_colors, device):
        noise = torch.rand_like(img) / (num_colors * 2)
        img = torch.floor((img + noise) * (num_colors - 1)) / (num_colors - 1)
        return img

    def ordered_dither(self, img, num_colors, device):
        bayer_matrix = torch.tensor([
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5]
        ], device=device).float() / 16.0

        h, w = img.shape[2], img.shape[3]
        bayer_tiled = bayer_matrix.repeat(h // 4 + 1, w // 4 + 1)[:h, :w]
        thresholds = bayer_tiled.unsqueeze(0).unsqueeze(0).repeat(img.shape[0], img.shape[1], 1, 1)

        img = torch.floor((img + thresholds / num_colors) * (num_colors - 1)) / (num_colors - 1)
        return img

    def bayer_dither(self, img, num_colors, device):
        bayer_matrix = torch.tensor([
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5]
        ], device=device).float() / 16.0

        h, w = img.shape[2], img.shape[3]
        bayer_tiled = bayer_matrix.repeat(h // 4 + 1, w // 4 + 1)[:h, :w]
        thresholds = bayer_tiled.unsqueeze(0).unsqueeze(0).repeat(img.shape[0], img.shape[1], 1, 1)

        quantized = torch.round(img * (num_colors - 1)) / (num_colors - 1)
        dithered = torch.where(img > thresholds, quantized + 1 / (num_colors - 1), quantized)
        return torch.clamp(dithered, 0, 1)