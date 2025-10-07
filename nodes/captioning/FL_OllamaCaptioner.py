import os
import re
from PIL import Image
import numpy as np
from comfy.utils import ProgressBar
from ollama import Client
from io import BytesIO
import base64

class FL_OllamaCaptioner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "folder_name": ("STRING", {"default": "output_folder"}),
                "use_llm": ("BOOLEAN", {"default": True}),
                "url": ("STRING", {"default": "http://127.0.0.1:11434"}),  # Default Ollama URL
                "model": ("STRING", {"default": "default_model"}),  # Replace with your model name
                "overwrite": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_images_with_captions"
    CATEGORY = "🏵️Fill Nodes/Captioning"
    OUTPUT_NODE = True

    def sanitize_text(self, text):
        return re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)

    def generate_caption_with_ollama(self, image_tensor, url, model):
        # Convert tensor to numpy array
        image_np = image_tensor.cpu().numpy()
        # Ensure the image is in the correct shape (height, width, channels)
        if image_np.shape[0] == 1:  # If the first dimension is 1, squeeze it
            image_np = np.squeeze(image_np, axis=0)
        if len(image_np.shape) == 2:
            image_np = np.stack((image_np,) * 3, axis=-1)
        elif image_np.shape[2] == 1:  # If it's (height, width, 1)
            image_np = np.repeat(image_np, 3, axis=2)
        # Ensure values are in 0-255 range
        image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
        # Convert to PIL Image
        image = Image.fromarray(image_np)

        # Encode image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = base64.b64encode(buffered.getvalue()).decode('utf-8')

        client = Client(host=url)
        response = client.generate(model=model, prompt="describe the image", images=[img_bytes])

        # Extract the caption from the response
        return response['response']

    def save_images_with_captions(self, images, folder_name, use_llm, url, model, overwrite):
        os.makedirs(folder_name, exist_ok=True)

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

            if use_llm:
                caption = self.generate_caption_with_ollama(image_tensor, url, model)
            else:
                caption = "Default Caption"

            sanitized_caption = self.sanitize_text(caption)

            # Convert tensor to numpy array and save the image as in the previous code
            image_np = image_tensor.cpu().numpy()
            if image_np.shape[0] == 1:
                image_np = np.squeeze(image_np, axis=0)
            if len(image_np.shape) == 2:
                image_np = np.stack((image_np,) * 3, axis=-1)
            elif image_np.shape[2] == 1:
                image_np = np.repeat(image_np, 3, axis=2)
            image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(image_np)
            image.save(image_file_name)
            saved_files.append(image_file_name)

            with open(text_file_name, "w") as text_file:
                text_file.write(sanitized_caption)

            pbar.update_absolute(i)

        return (f"Saved {len(images)} images and generated captions in '{folder_name}'",)
