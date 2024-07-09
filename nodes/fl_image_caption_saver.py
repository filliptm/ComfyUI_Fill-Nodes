import os
from PIL import Image

from comfy.utils import ProgressBar

class FL_ImageCaptionSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "folder_name": ("STRING", {"default": "output_folder"}),
                "caption_text": ("STRING", {"default": "Your caption here"}),
                "overwrite": ("BOOLEAN", {"default": True})  # New overwrite toggle
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_images_with_captions"
    CATEGORY = "🏵️Fill Nodes/utility"
    OUTPUT_NODE = True

    def save_images_with_captions(self, images, folder_name, caption_text, overwrite):
        # Ensure output directory exists
        os.makedirs(folder_name, exist_ok=True)

        saved_files = []
        pbar = ProgressBar(len(images))
        for i, image_tensor in enumerate(images):
            base_name = f"image_{i}"
            image_file_name = f"{folder_name}/{base_name}.png"
            text_file_name = f"{folder_name}/{base_name}.txt"

            # Check if overwrite is disabled and file exists
            if not overwrite:
                counter = 1
                while os.path.exists(image_file_name) or os.path.exists(text_file_name):
                    image_file_name = f"{folder_name}/{base_name}_{counter}.png"
                    text_file_name = f"{folder_name}/{base_name}_{counter}.txt"
                    counter += 1

            # Convert tensor to image
            image = Image.fromarray((image_tensor.numpy() * 255).astype('uint8'), 'RGB')

            # Save image
            image.save(image_file_name)
            saved_files.append(image_file_name)

            # Save text file
            with open(text_file_name, "w") as text_file:
                text_file.write(caption_text)

            pbar.update_absolute(i)

        return (f"Saved {len(images)} images and captions in '{folder_name}'",)
