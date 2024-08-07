import torch
import numpy as np

from comfy.utils import ProgressBar

class FL_HalftonePattern:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "dot_size": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "dot_spacing": ("INT", {"default": 10, "min": 5, "max": 50, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "halftone_pattern"
    CATEGORY = "🏵️Fill Nodes/VFX"

    def halftone_pattern(self, images, dot_size=5, dot_spacing=10):
        out = []
        total_images = len(images)
        pbar = ProgressBar(total_images)
        for i, img in enumerate(images, start=1):
            img_np = img.cpu().numpy().squeeze()
            grayscale_image = np.dot(img_np[..., :3], [0.299, 0.587, 0.114])
            height, width = grayscale_image.shape

            halftone_image = np.ones((height, width), dtype=np.float32)

            for y in range(0, height, dot_spacing):
                for x in range(0, width, dot_spacing):
                    box = (x, y, x + dot_spacing, y + dot_spacing)
                    region_mean = np.mean(grayscale_image[box[1]:box[3], box[0]:box[2]])
                    dot_radius = int((1 - region_mean) * dot_size / 2)
                    dot_position = (x + dot_spacing // 2, y + dot_spacing // 2)

                    # Create a circular mask for the dot
                    y_grid, x_grid = np.ogrid[-dot_radius:dot_radius + 1, -dot_radius:dot_radius + 1]
                    mask = x_grid ** 2 + y_grid ** 2 <= dot_radius ** 2

                    # Apply the dot mask to the halftone image
                    y_start = max(0, dot_position[1] - dot_radius)
                    y_end = min(height, dot_position[1] + dot_radius + 1)
                    x_start = max(0, dot_position[0] - dot_radius)
                    x_end = min(width, dot_position[0] + dot_radius + 1)

                    # Ensure the mask dimensions match the sliced halftone image dimensions
                    mask_height = y_end - y_start
                    mask_width = x_end - x_start
                    mask = mask[:mask_height, :mask_width]

                    try:
                        halftone_image[y_start:y_end, x_start:x_end][mask] = 0
                    except Exception:
                        pass

            o = np.stack((halftone_image,) * 3, axis=-1)
            o = torch.from_numpy(o).unsqueeze(0)
            out.append(o)

            pbar.update_absolute(i)

        out = torch.cat(out, 0)
        return (out,)