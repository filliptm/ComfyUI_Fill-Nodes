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
                "file_type": (["images", "text"],),
                "max_files": ("INT", {"default": 100, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")  # Output a batch of images or list of text contents
    FUNCTION = "load_batch"
    CATEGORY = "🏵️Fill Nodes/utility"

    def load_batch(self, directory_path, file_type, max_files):
        if not directory_path:
            raise ValueError("Directory path is not provided.")

        file_paths = self.crawl_directories(directory_path, file_type)
        if not file_paths:
            raise ValueError(f"No {file_type} found in the specified directory and its subdirectories.")

        file_paths = file_paths[:max_files]  # Limit the number of files

        if file_type == "images":
            return self.load_image_batch(file_paths)
        else:
            return self.load_text_batch(file_paths)

    def load_image_batch(self, image_paths):
        batch_images = []
        pbar = ProgressBar(len(image_paths))
        for idx, img_path in enumerate(image_paths):
            image = Image.open(img_path)
            image = ImageOps.exif_transpose(image)  # Correct orientation
            image = image.convert("RGB")
            image_np = np.array(image).astype(np.float32) / 255.0
            batch_images.append(image_np)
            pbar.update_absolute(idx)

        # Pad images to the largest dimensions
        max_h = max(img.shape[0] for img in batch_images)
        max_w = max(img.shape[1] for img in batch_images)

        padded_images = []
        for img in batch_images:
            h, w, c = img.shape
            padded = np.zeros((max_h, max_w, c), dtype=np.float32)
            padded[:h, :w, :] = img
            padded_images.append(padded)

        batch_images_np = np.stack(padded_images, axis=0)
        batch_images_tensor = torch.from_numpy(batch_images_np)

        return (batch_images_tensor, "")

    def load_text_batch(self, text_paths):
        text_contents = []
        pbar = ProgressBar(len(text_paths))
        for idx, txt_path in enumerate(text_paths):
            with open(txt_path, 'r', encoding='utf-8') as file:
                content = file.read()
                text_contents.append(content)
            pbar.update_absolute(idx)

        return (torch.zeros(1), "\n---\n".join(text_contents))  # Return empty tensor for IMAGE type

    def crawl_directories(self, directory, file_type):
        if file_type == "images":
            supported_formats = ["jpg", "jpeg", "png", "bmp", "gif"]
        else:
            supported_formats = ["txt"]

        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.split('.')[-1].lower() in supported_formats:
                    full_path = os.path.join(root, file)
                    file_paths.append(full_path)
        return file_paths