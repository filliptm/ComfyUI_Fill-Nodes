import os
import folder_paths
from nodes import SaveImage


class FL_SaveAndDisplayImage(SaveImage):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE",),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save_and_display_images"
    OUTPUT_NODE = True
    CATEGORY = "🏵️Fill Nodes/Image"

    def save_and_display_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        results = self.save_images(images, filename_prefix, prompt, extra_pnginfo)

        return {
            "ui": {
                "images": results['ui']['images']
            },
            "result": (images,)
        }