import torch
import torch.nn.functional as F


class FL_Padding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "padding_left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "red": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "green": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "include_alpha": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_padding"
    CATEGORY = "🏵️Fill Nodes/Utility"

    def tensor_to_rgba(self, tensor):
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] == 1:
            tensor = tensor.expand(-1, -1, -1, 4)
        elif tensor.shape[-1] == 3:
            tensor = torch.cat([tensor, torch.ones_like(tensor[:, :, :, :1])], dim=-1)
        return tensor

    def add_padding(self, image, padding_left, padding_right, padding_top, padding_bottom,
                    red, green, blue, include_alpha):
        # Convert image to RGBA
        image = self.tensor_to_rgba(image)

        # Get batch size and image dimensions
        B, H, W, C = image.shape

        # Calculate new dimensions
        new_height = H + padding_top + padding_bottom
        new_width = W + padding_left + padding_right

        # Create padding color with alpha
        padding_color = torch.tensor([red, green, blue, 255], device=image.device) / 255.0

        # Create new tensor with padding color
        padded_image = padding_color.view(1, 1, 1, 4).expand(B, new_height, new_width, 4).clone()

        # Copy the original image into the padded tensor
        padded_image[:, padding_top:padding_top + H, padding_left:padding_left + W, :] = image

        # Remove alpha channel if not included
        if not include_alpha:
            padded_image = padded_image[..., :3]

        return (padded_image,)