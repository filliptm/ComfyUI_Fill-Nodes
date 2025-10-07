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
    CATEGORY = "🏵️Fill Nodes/Utility"

    def slice_image(self, image: torch.Tensor, slicing: str, multiplier: float):
        _, height, width, _ = image.shape
        slices_x, slices_y = map(int, slicing.split('x'))
        slice_width = int((width // slices_x) * multiplier)
        slice_height = int((height // slices_y) * multiplier)
        return (image, slice_width, slice_height, multiplier)

    @classmethod
    def IS_CHANGED(cls, image, slicing, multiplier):
        return float("NaN")
    