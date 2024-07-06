import torch
import numpy as np

class FL_SDUltimate_Slices:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "slicing": (["1x1", "1x2", "1x3", "1x4",
                             "2x1", "2x2", "2x3", "2x4",
                             "3x1", "3x2", "3x3", "3x4",
                             "4x1", "4x2", "4x3", "4x4"],),
                "multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.25
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("image", "slice_width", "slice_height", "multiplier")
    FUNCTION = "slice_image"
    CATEGORY = "🏵️Fill Nodes/utility"

    def slice_image(self, image: torch.Tensor, slicing: str, multiplier: float):
        # Get the image dimensions
        _, height, width, _ = image.shape

        # Parse the slicing option
        slices_x, slices_y = map(int, slicing.split('x'))

        # Calculate the slice dimensions
        slice_width = width // slices_x
        slice_height = height // slices_y

        # Apply the multiplier
        slice_width = int(slice_width * multiplier)
        slice_height = int(slice_height * multiplier)

        return (image, slice_width, slice_height, multiplier)