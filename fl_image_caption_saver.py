import os
import torch
from PIL import Image


class FL_ImageCaptionSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "folder_name": ("STRING", {"default": "output_folder"}),
                "caption_text": ("STRING", {"default": "Your caption here"})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_images_with_captions"
    CATEGORY = "🏵️Fill Nodes"  # Adjusted to appear under "Fill Nodes"

    def save_images_with_captions(self, images, folder_name, caption_text):
        # Ensure output directory exists
        os.makedirs(folder_name, exist_ok=True)

        saved_files = []
        for i, image_tensor in enumerate(images):
            # Convert tensor to image
            image = Image.fromarray((image_tensor.numpy() * 255).astype('uint8'), 'RGB')
            image_file_name = f"{folder_name}/image_{i}.png"
            text_file_name = f"{folder_name}/image_{i}.txt"

            # Save image
            image.save(image_file_name)
            saved_files.append(image_file_name)

            # Save text file
            with open(text_file_name, "w") as text_file:
                text_file.write(caption_text)

        return (f"Saved {len(images)} images and captions in '{folder_name}'",)
