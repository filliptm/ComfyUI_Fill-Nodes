import os
import random
import numpy as np
import torch
from PIL import Image, ImageOps

class FL_ImageRandomizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": ""}),
                "randomize": ("BOOLEAN", {"default": True}),  # Toggle for randomization
                "run_trigger": ("INT", {"default": 0}),  # Dummy input for caching issue
            }
        }

    RETURN_TYPES = ("IMAGE", "PATH")  # Adjusted to include image path for preview
    FUNCTION = "select_image"
    CATEGORY = "🏵️Fill Nodes/utility"  # Adjusted to appear under "Fill Nodes"

    def __init__(self):
        self.last_index = -1

    def select_image(self, directory_path, randomize, run_trigger):
        if not directory_path:
            raise ValueError("Directory path is not provided.")

        images = self.load_images(directory_path)
        if not images:
            raise ValueError("No images found in the specified directory.")

        if randomize:
            selected_image_path = random.choice(images)
        else:
            self.last_index = (self.last_index + 1) % len(images)
            selected_image_path = images[self.last_index]

        image = Image.open(selected_image_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        return (image_tensor, selected_image_path)  # Return both data points

    def load_images(self, directory):
        supported_formats = ["jpg", "jpeg", "png", "bmp", "gif"]
        return [os.path.join(directory, f) for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f)) and f.split('.')[-1].lower() in supported_formats]
