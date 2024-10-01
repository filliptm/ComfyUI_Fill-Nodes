import os
import folder_paths
from nodes import SaveImage
from PIL import Image
import io


class FL_SaveWebPImage(SaveImage):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE",),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                     "quality": ("INT", {"default": 80, "min": 1, "max": 100, "step": 1})},
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
                }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "save_webp_images"
    OUTPUT_NODE = True
    CATEGORY = "🏵️Fill Nodes/Image"

    def save_webp_images(self, images, filename_prefix="ComfyUI", quality=80, prompt=None, extra_pnginfo=None):
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir)
        results = []

        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(i.astype('uint8'))
            file = f"{filename}_{counter:05}_.webp"
            fullpath = os.path.join(full_output_folder, file)

            img.save(fullpath, 'WEBP', quality=quality)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "output"
            })
            counter += 1

        return {
            "ui": {
                "images": results
            },
            "result": (images, f"Saved {len(results)} WebP image(s)")
        }


NODE_CLASS_MAPPINGS = {
    "SaveWebPImage": FL_SaveWebPImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveWebPImage": "Save WebP Image"
}