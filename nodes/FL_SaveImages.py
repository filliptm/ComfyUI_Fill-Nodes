import os
import torch
from PIL import Image
import numpy as np
import json

class FL_SaveImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "base_directory": ("STRING", {"default": "./output"}),
                "folder_structure": ("STRING", {"default": "[]"}),
                "file_name_template": ("STRING", {"default": "image_{index}.png"}),
                "start_index": ("INT", {"default": 1, "min": 0, "max": 1000000}),
            },
            "optional": {
                "metadata": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "🏵️Fill Nodes/Image"

    def save_images(self, images, base_directory, folder_structure, file_name_template, start_index, metadata=""):
        saved_paths = []
        folder_structure = json.loads(folder_structure)

        # Ensure base directory exists
        os.makedirs(base_directory, exist_ok=True)

        for i, image in enumerate(images):
            # Convert the image tensor to a PIL Image
            img = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))

            # Create the full folder path based on the folder structure
            full_folder_path = self.create_folder_path(base_directory, folder_structure)
            os.makedirs(full_folder_path, exist_ok=True)

            # Create the file name and ensure it doesn't overwrite existing files
            index = start_index + i
            while True:
                # Format index with zero padding (4 digits: 0001, 0002, etc.)
                padded_index = f"{index:04d}"
                full_file_name = file_name_template.format(index=padded_index)
                full_file_path = os.path.join(full_folder_path, full_file_name)
                if not os.path.exists(full_file_path):
                    break
                index += 1

            # Save the image
            img.save(full_file_path)

            # Save metadata if provided
            if metadata:
                metadata_file_name = f"{os.path.splitext(full_file_name)[0]}_metadata.txt"
                metadata_file_path = os.path.join(full_folder_path, metadata_file_name)
                with open(metadata_file_path, 'w') as f:
                    f.write(metadata)

            saved_paths.append(full_file_path)

        return (", ".join(saved_paths),)

    def create_folder_path(self, base_directory, folder_structure):
        path = base_directory
        for folder in folder_structure:
            path = os.path.join(path, folder['name'])
            for child in folder['children']:
                path = self.create_folder_path(path, [child])
        return path