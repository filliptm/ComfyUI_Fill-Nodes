import torch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from comfy.utils import ProgressBar

class FL_PixelArtShader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "pixel_size": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 100.0, "step": 1.0}),
                "color_depth": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 255.0, "step": 1.0}),
                "use_aspect_ratio": ("BOOLEAN", {"default": True}),
                "palette_image": ("IMAGE", {"default": None}),
                "palette_colors": ("INT", {"default": 16, "min": 2, "max": 16, "step": 1}),
                "mask": ("IMAGE", {"default": None}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_pixel_art_shader"
    CATEGORY = "🏵️Fill Nodes/VFX"

    def apply_pixel_art_shader(self, images, use_aspect_ratio, pixel_size, color_depth, palette_image=None,
                               palette_colors=16, mask=None):
        result = []
        total_images = len(images)
        pbar = ProgressBar(total_images)

        if palette_image is not None:
            palette = extract_palette(self.t2p(palette_image[0]), palette_colors)
        else:
            palette = None

        mask_images = self.prepare_mask_batch(mask, total_images) if mask is not None else None

        for idx, image in enumerate(images):
            img = self.t2p(image)

            mask_img = self.process_mask(mask_images[idx], img.size) if mask_images is not None else None

            result_img = pixel_art_effect(img, pixel_size, color_depth, use_aspect_ratio, palette, mask_img)
            result_img = self.p2t(result_img)
            result.append(result_img)
            pbar.update_absolute(idx + 1)

        return (torch.cat(result, dim=0),)

    def t2p(self, t):
        i = 255.0 * t.cpu().numpy().squeeze()
        return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    def p2t(self, p):
        i = np.array(p).astype(np.float32) / 255.0
        return torch.from_numpy(i).unsqueeze(0)

    def prepare_mask_batch(self, mask, total_images):
        if mask is None:
            return None
        mask_images = [self.t2p(m) for m in mask]
        if len(mask_images) < total_images:
            mask_images = mask_images * (total_images // len(mask_images) + 1)
        return mask_images[:total_images]

    def process_mask(self, mask, target_size):
        mask = mask.resize(target_size, Image.LANCZOS)
        return mask.convert('L') if mask.mode != 'L' else mask

def extract_palette(image, n_colors):
    image = image.convert('RGB')
    pixels = np.array(image).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    return torch.from_numpy(colors.astype(np.float32) / 255.0).to("cuda")

def pixel_art_effect(image, pixel_size, color_depth, use_aspect_ratio, palette, mask=None):
    image = torch.tensor(np.array(image)).float().to("cuda") / 255.0
    height, width = image.shape[0], image.shape[1]

    if use_aspect_ratio:
        aspect_ratio = width / height
        pixel_size_x, pixel_size_y = pixel_size, pixel_size / aspect_ratio
    else:
        pixel_size_x = pixel_size_y = pixel_size

    new_width = int(width / pixel_size_x)
    new_height = int(height / pixel_size_y)

    # Resize the image to create the pixelated effect
    pixelated = image.permute(2, 0, 1).unsqueeze(0)
    pixelated = torch.nn.functional.interpolate(pixelated, size=(new_height, new_width), mode='nearest')
    pixelated = torch.nn.functional.interpolate(pixelated, size=(height, width), mode='nearest')
    pixelated = pixelated.squeeze(0).permute(1, 2, 0)

    # Apply color depth reduction
    pixelated = adjust_color(pixelated, color_depth)

    if palette is not None:
        pixelated = apply_palette(pixelated, palette)

    if mask is not None:
        mask_tensor = torch.tensor(np.array(mask)).float().to("cuda") / 255.0
        mask_tensor = mask_tensor.unsqueeze(-1).expand(-1, -1, 3)
        pixelated = pixelated * mask_tensor + image * (1 - mask_tensor)

    return Image.fromarray((pixelated.cpu().numpy() * 255).astype(np.uint8))

def adjust_color(color, color_depth):
    return torch.floor(color * color_depth) / color_depth

def apply_palette(image, palette):
    original_shape = image.shape
    pixels = image.reshape(-1, 3)
    distances = torch.cdist(pixels, palette)
    nearest_palette_indices = torch.argmin(distances, dim=1)
    new_pixels = palette[nearest_palette_indices]
    return new_pixels.reshape(original_shape)