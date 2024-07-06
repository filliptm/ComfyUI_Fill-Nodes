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
    CATEGORY = "🏵️Fill Nodes/utility"

    def display_dimensions(self, image):
        # Check the number of dimensions in the image tensor to correctly unpack the dimensions
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # Batch dimension is present
                _, height, width, _, = image.shape
            elif image.dim() == 3:  # No batch dimension, single image
                _, height, width = image.shape
            else:
                return ("Unsupported tensor format",)
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            return ("Unsupported image format",)

        # Correctly assign width and height
        dimensions = f"Width: {width}, Height: {height}"

        # Display dimensions in the UI. This might need to be adapted.
        print(dimensions)

        # Return the dimensions as a string.
        return (dimensions,)
