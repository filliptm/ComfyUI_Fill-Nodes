import os
import numpy as np
import torch
from PIL import Image, ImageOps

from comfy.utils import ProgressBar

class FL_DirectoryCrawl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)  # Output a batch of images
    FUNCTION = "load_image_batch"
    CATEGORY = "🏵️Fill Nodes/utility"

    def load_image_batch(self, directory_path):
        if not directory_path:
            raise ValueError("Directory path is not provided.")

        image_paths = self.crawl_directories(directory_path)
        if not image_paths:
            raise ValueError("No images found in the specified directory and its subdirectories.")

        batch_images = []
        pbar = ProgressBar(len(image_paths))
        for idx, img_path in enumerate(image_paths):
            image = Image.open(img_path)
            image = ImageOps.exif_transpose(image)  # Correct orientation
            image = image.convert("RGB")
            image_np = np.array(image).astype(np.float32) / 255.0
            batch_images.append(image_np)
            pbar.update_absolute(idx)

        batch_images_np = np.stack(batch_images, axis=0)  # Create a numpy array batch
        batch_images_tensor = torch.from_numpy(batch_images_np)  # Convert to tensor

        return (batch_images_tensor,)

    def crawl_directories(self, directory):
        supported_formats = ["jpg", "jpeg", "png", "bmp", "gif", "txt"]
        image_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.split('.')[-1].lower() in supported_formats:
                    full_path = os.path.join(root, file)
                    image_paths.append(full_path)
        return image_paths
