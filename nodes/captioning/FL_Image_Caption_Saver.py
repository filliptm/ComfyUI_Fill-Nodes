import os
import re
from PIL import Image
import numpy as np

from comfy.utils import ProgressBar

class FL_ImageCaptionSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "folder_name": ("STRING", {"default": "output_folder"}),
                "caption_text": ("STRING", {"default": "Your caption here"}),
                "overwrite": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_images_with_captions"
    CATEGORY = "🏵️Fill Nodes/Captioning"
    OUTPUT_NODE = True

    def sanitize_text(self, text):
        return re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)

    def save_images_with_captions(self, images, folder_name, caption_text, overwrite):
        os.makedirs(folder_name, exist_ok=True)
        sanitized_caption = self.sanitize_text(caption_text)

        saved_files = []
        pbar = ProgressBar(len(images))
        for i, image_tensor in enumerate(images):
            base_name = f"image_{i}"
            image_file_name = f"{folder_name}/{base_name}.png"
            text_file_name = f"{folder_name}/{base_name}.txt"

            if not overwrite:
                counter = 1
                while os.path.exists(image_file_name) or os.path.exists(text_file_name):
                    image_file_name = f"{folder_name}/{base_name}_{counter}.png"
                    text_file_name = f"{folder_name}/{base_name}_{counter}.txt"
                    counter += 1

            # Convert tensor to numpy array
            image_np = image_tensor.cpu().numpy()

            # Ensure the image is in the correct shape (height, width, channels)
            if image_np.shape[0] == 1:  # If the first dimension is 1, squeeze it
                image_np = np.squeeze(image_np, axis=0)

            # If the image is grayscale (2D), convert to RGB
            if len(image_np.shape) == 2:
                image_np = np.stack((image_np,) * 3, axis=-1)
            elif image_np.shape[2] == 1:  # If it's (height, width, 1)
                image_np = np.repeat(image_np, 3, axis=2)

            # Ensure values are in 0-255 range
            image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

            # Convert to PIL Image
            image = Image.fromarray(image_np)

            # Save image
            image.save(image_file_name)
            saved_files.append(image_file_name)

            with open(text_file_name, "w") as text_file:
                text_file.write(sanitized_caption)

            pbar.update_absolute(i)

        return (f"Saved {len(images)} images and sanitized captions in '{folder_name}'",)
