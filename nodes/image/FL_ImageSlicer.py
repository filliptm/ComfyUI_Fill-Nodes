import torch
from ..utils import tensor_to_pil, pil_to_tensor, resize_tensor


class FL_ImageSlicer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width_subdivisions": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1}),
                "height_subdivisions": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "slice_image"
    CATEGORY = "🏵️Fill Nodes/Image"

    def slice_image(self, image, width_subdivisions, height_subdivisions):
        # Convert from torch tensor to PIL Image
        pil_image = tensor_to_pil(image)

        # Get image dimensions
        width, height = pil_image.size

        # Calculate slice dimensions
        slice_width = width // width_subdivisions
        slice_height = height // height_subdivisions

        # Slice the image
        slices = []
        for y in range(height_subdivisions):
            for x in range(width_subdivisions):
                left = x * slice_width
                upper = y * slice_height
                right = left + slice_width
                lower = upper + slice_height

                slice_img = pil_image.crop((left, upper, right, lower))
                slice_tensor = pil_to_tensor(slice_img)
                slices.append(slice_tensor)

        # Stack all slices into a single tensor
        output_tensor = torch.cat(slices, dim=0)

        return (output_tensor,)