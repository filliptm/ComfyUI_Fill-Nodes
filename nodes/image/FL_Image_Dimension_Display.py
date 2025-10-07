import torch
from PIL import Image

class FL_ImageDimensionDisplay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "display_dimensions"
    CATEGORY = "🏵️Fill Nodes/Image"

    def display_dimensions(self, image):
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # Batch dimension is present
                _, height, width, _ = image.shape
            elif image.dim() == 3:  # No batch dimension, single image
                height, width, _ = image.shape
            else:
                return ("Unsupported tensor format",)
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            return ("Unsupported image format",)

        dimensions = f"Width: {width}, Height: {height}"
        return (dimensions,)

    @classmethod
    def IS_CHANGED(cls, image):
        return float("NaN")  # This ensures the node always updates