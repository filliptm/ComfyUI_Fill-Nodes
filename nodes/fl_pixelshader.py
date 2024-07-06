import torch
import numpy as np
from PIL import Image

from comfy.utils import ProgressBar

class FL_PixelArtShader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "pixel_size": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 1000.0, "step": 1.0}),
                "color_depth": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 255.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_pixel_art_shader"
    CATEGORY = "🏵️Fill Nodes/VFX"

    def apply_pixel_art_shader(self, images, pixel_size, color_depth):
        result = []
        total_images = len(images)
        pbar = ProgressBar(total_images)
        for idx, image in enumerate(images, start=1):
            img = self.t2p(image)
            result_img = pixel_art_effect(img, pixel_size, color_depth)
            result_img = self.p2t(result_img)
            result.append(result_img)
            pbar.update_absolute(idx)

        return (torch.cat(result, dim=0),)

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

def pixel_art_effect(image, pixel_size, color_depth):
    # Move the input image to the GPU
    image = torch.tensor(np.array(image)).float().to("cuda") / 255.0

    # Get the input image dimensions
    height, width = image.shape[0], image.shape[1]

    # Create a grid of UV coordinates
    uv_x = torch.linspace(0, 1, width, device="cuda")
    uv_y = torch.linspace(0, 1, height, device="cuda")
    uv_grid = torch.stack(torch.meshgrid(uv_y, uv_x), dim=-1)  # Swap uv_y and uv_x

    # Evaluate the shader code for each pixel
    output_tensor = evaluate_shader(image, uv_grid, pixel_size, color_depth)

    # Convert the output tensor to a PIL image
    output_image = Image.fromarray((output_tensor.cpu().numpy() * 255).astype(np.uint8))

    return output_image

def evaluate_shader(image, uv_grid, pixel_size, color_depth):
    # Sample the input image at the pixelated UV coordinates
    pixelUV = torch.floor(uv_grid * pixel_size) / pixel_size
    color = texture_lookup(image, pixelUV)

    # Apply color adjustments
    color = adjust_color(color, color_depth)

    return color

def adjust_color(color, color_depth):
    # Apply color depth reduction
    color = torch.floor(color * color_depth) / color_depth

    return color

def texture_lookup(image, uv):
    # Clamp the UV coordinates to [0, 1]
    uv = torch.clamp(uv, 0.0, 1.0)

    # Calculate the pixel coordinates
    y = (uv[..., 0] * (image.shape[0] - 1)).long()  # Use uv[..., 0] for y
    x = (uv[..., 1] * (image.shape[1] - 1)).long()  # Use uv[..., 1] for x

    # Perform texture lookup using the pixel coordinates
    return image[y, x]