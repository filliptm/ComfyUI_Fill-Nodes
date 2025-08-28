import os
import re
from PIL import Image
import numpy as np
from comfy.utils import ProgressBar


class FL_CaptionSaver_V2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_type": (["Image Input", "Directory Input"], {"default": "Image Input"}),
                "caption_input_type": (["Single Caption", "Multiple Captions"], {"default": "Single Caption"}),
                "folder_name": ("STRING", {"default": "output_folder"}),
                "overwrite": ("BOOLEAN", {"default": True}),
                "downsize_factor": ([1, 2, 3], {"default": 1})
            },
            "optional": {
                "images": ("IMAGE", {}),
                "mask_image": ("IMAGE", {}),
                "input_directory": ("STRING", {"default": ""}),
                "single_caption": ("STRING", {"default": "Your caption here"}),
                "multiple_captions": ("STRING", {"multiline": True, "default": ""})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_images_with_captions"
    CATEGORY = "🏵️Fill Nodes/Captioning"
    OUTPUT_NODE = True

    def sanitize_text(self, text):
        return re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)

    def save_images_with_captions(self, input_type, caption_input_type, folder_name, overwrite, downsize_factor,
                                  images=None, mask_image=None, input_directory=None, single_caption="", multiple_captions=""):
        os.makedirs(folder_name, exist_ok=True)

        if input_type == "Image Input" and images is not None:
            image_list = images
            use_original_names = False
        elif input_type == "Directory Input" and input_directory:
            image_list = [f for f in os.listdir(input_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            use_original_names = True
        else:
            return ("No valid input provided.",)

        if caption_input_type == "Single Caption":
            captions = [self.sanitize_text(single_caption)] * len(image_list)
        else:
            captions = [self.sanitize_text(cap.strip()) for cap in multiple_captions.split('\n') if cap.strip()]
            if len(captions) < len(image_list):
                captions.extend([captions[-1]] * (len(image_list) - len(captions)))
            elif len(captions) > len(image_list):
                captions = captions[:len(image_list)]

        saved_files = []
        pbar = ProgressBar(len(image_list))
        for i, (image_item, caption) in enumerate(zip(image_list, captions)):
            if use_original_names:
                base_name = os.path.splitext(image_item)[0]
                image_path = os.path.join(input_directory, image_item)
                image = Image.open(image_path)
            else:
                base_name = f"image_{i}"
                image_np = image_item.cpu().numpy()
                image_np = self.process_image_tensor(image_np)
                image = Image.fromarray(image_np)

            # Downsize the image
            if downsize_factor > 1:
                new_size = (image.width // downsize_factor, image.height // downsize_factor)
                image = image.resize(new_size, Image.LANCZOS)

            image_file_name = f"{folder_name}/{base_name}.png"
            text_file_name = f"{folder_name}/{base_name}.txt"

            if not overwrite:
                image_file_name, text_file_name = self.get_unique_filenames(folder_name, base_name)

            image.save(image_file_name)
            saved_files.append(image_file_name)

            # Save mask image if provided
            if mask_image is not None:
                if use_original_names:
                    mask_base_name = f"mask_{base_name}"
                else:
                    mask_base_name = f"mask_{i}"
                mask_file_name = f"{folder_name}/{mask_base_name}.png"
                
                if not overwrite and os.path.exists(mask_file_name):
                    counter = 1
                    while os.path.exists(f"{folder_name}/{mask_base_name}_{counter}.png"):
                        counter += 1
                    mask_file_name = f"{folder_name}/{mask_base_name}_{counter}.png"
                
                # Process mask image tensor
                if i < len(mask_image):
                    mask_np = mask_image[i].cpu().numpy()
                    mask_np = self.process_image_tensor(mask_np)
                    mask_pil = Image.fromarray(mask_np)
                    
                    # Downsize mask image with same factor
                    if downsize_factor > 1:
                        mask_new_size = (mask_pil.width // downsize_factor, mask_pil.height // downsize_factor)
                        mask_pil = mask_pil.resize(mask_new_size, Image.LANCZOS)
                    
                    mask_pil.save(mask_file_name)

            with open(text_file_name, "w") as text_file:
                text_file.write(caption)

            pbar.update_absolute(i)

        return (f"Saved {len(saved_files)} images (downsized by factor {downsize_factor}) and captions in '{folder_name}'",)

    def process_image_tensor(self, image_np):
        if image_np.shape[0] == 1:
            image_np = np.squeeze(image_np, axis=0)
        if len(image_np.shape) == 2:
            image_np = np.stack((image_np,) * 3, axis=-1)
        elif image_np.shape[2] == 1:
            image_np = np.repeat(image_np, 3, axis=2)
        return (image_np * 255).clip(0, 255).astype(np.uint8)

    def get_unique_filenames(self, folder_name, base_name):
        counter = 1
        image_file_name = f"{folder_name}/{base_name}.png"
        text_file_name = f"{folder_name}/{base_name}.txt"
        while os.path.exists(image_file_name) or os.path.exists(text_file_name):
            image_file_name = f"{folder_name}/{base_name}_{counter}.png"
            text_file_name = f"{folder_name}/{base_name}_{counter}.txt"
            counter += 1
        return image_file_name, text_file_name