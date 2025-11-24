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

        batch_size = len(images)
        has_index_placeholder = "{index}" in file_name_template

        for i, image in enumerate(images):
            # Convert the image tensor to a PIL Image
            img = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))

            # Create the full folder path based on the folder structure
            full_folder_path = self.create_folder_path(base_directory, folder_structure)
            os.makedirs(full_folder_path, exist_ok=True)

            # Determine the filename based on batch size and placeholder
            if batch_size > 1:
                # For batches, always use an index
                current_index = start_index + i
                padded_index = f"{current_index:04d}"

                if has_index_placeholder:
                    # Replace {index} with the current counter
                    full_file_name = file_name_template.format(index=padded_index)
                else:
                    # Append index before extension
                    base_name, ext = os.path.splitext(file_name_template)
                    full_file_name = f"{base_name}_{padded_index}{ext}"
            else:
                # Single image
                if has_index_placeholder:
                    # Replace {index} with start_index
                    padded_index = f"{start_index:04d}"
                    full_file_name = file_name_template.format(index=padded_index)
                else:
                    # Use exact filename
                    full_file_name = file_name_template

            full_file_path = os.path.join(full_folder_path, full_file_name)

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