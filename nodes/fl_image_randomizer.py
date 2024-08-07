import os
import numpy as np
import torch
from PIL import Image, ImageOps

class FL_ImageRandomizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", "PATH")
    FUNCTION = "select_image"
    CATEGORY = "🏵️Fill Nodes/utility"

    def select_image(self, directory_path, seed):
        if not directory_path:
            raise ValueError("Directory path is not provided.")

        images = self.load_images(directory_path)
        if not images:
            raise ValueError("No images found in the specified directory.")

        num_images = len(images)
        selected_index = seed % num_images

        selected_image_path = images[selected_index]

        image = Image.open(selected_image_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        return (image_tensor, selected_image_path)

    def load_images(self, directory):
        supported_formats = ["jpg", "jpeg", "png", "bmp", "gif"]
        return sorted([os.path.join(directory, f) for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f)) and f.split('.')[-1].lower() in supported_formats])