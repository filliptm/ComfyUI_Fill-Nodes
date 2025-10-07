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

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("image_batch", "text_content", "image_list")
    OUTPUT_IS_LIST = (False, False, True)
    FUNCTION = "load_batch"
    CATEGORY = "🏵️Fill Nodes/Utility"

    def load_batch(self, directory_path, file_type, max_files):
        if not directory_path:
            raise ValueError("Directory path is not provided.")

        file_paths = self.crawl_directories(directory_path, file_type)
        if not file_paths:
            if file_type == "images":
                return (torch.empty(0), "", []) # Return empty for all if no files
            else: # text
                return (torch.empty(0), "", [])


        file_paths = file_paths[:max_files]  # Limit the number of files

        if file_type == "images":
            batch_tensor, image_list_tensors = self.load_images_data(file_paths)
            return (batch_tensor, "", image_list_tensors)
        else: # text
            text_content = self.load_text_data(file_paths)
            return (torch.empty(0), text_content, [])

    def load_images_data(self, image_paths):
        individual_images_np = []
        individual_image_tensors = []
        pbar = ProgressBar(len(image_paths))

        if not image_paths:
            return torch.empty(0), []

        for idx, img_path in enumerate(image_paths):
            image = Image.open(img_path)
            image = ImageOps.exif_transpose(image)  # Correct orientation
            image = image.convert("RGB")
            image_np = np.array(image).astype(np.float32) / 255.0
            individual_images_np.append(image_np)
            # Create tensor for the list output (B, H, W, C) -> (1, H, W, C)
            individual_image_tensors.append(torch.from_numpy(image_np)[None,])
            pbar.update_absolute(idx)

        # Pad images for batch output
        max_h = max(img.shape[0] for img in individual_images_np)
        max_w = max(img.shape[1] for img in individual_images_np)

        padded_images_for_batch = []
        for img_np in individual_images_np:
            h, w, c = img_np.shape
            padded = np.zeros((max_h, max_w, c), dtype=np.float32)
            padded[:h, :w, :] = img_np
            padded_images_for_batch.append(padded)

        batch_images_np = np.stack(padded_images_for_batch, axis=0)
        batch_images_tensor = torch.from_numpy(batch_images_np)

        return batch_images_tensor, individual_image_tensors

    def load_text_data(self, text_paths):
        text_contents = []
        if not text_paths:
            return ""
        pbar = ProgressBar(len(text_paths))
        for idx, txt_path in enumerate(text_paths):
            try:
                with open(txt_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    text_contents.append(content)
            except Exception as e:
                print(f"Warning: Could not read text file {txt_path}: {e}")
                text_contents.append(f"Error reading file: {txt_path}")
            pbar.update_absolute(idx)

        return "\n---\n".join(text_contents)

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