import torch


class FL_ApplyMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_mask"
    CATEGORY = "🏵️Fill Nodes/Image"

    def apply_mask(self, image, mask):
        # Ensure the image is in the correct format (B, H, W, C)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # Ensure the mask is in the correct format (B, H, W)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        elif len(mask.shape) == 3 and mask.shape[2] == 1:
            mask = mask.squeeze(-1)

        # Ensure mask and image have the same batch size
        if mask.shape[0] == 1 and image.shape[0] > 1:
            mask = mask.repeat(image.shape[0], 1, 1)

        # Ensure mask and image have the same spatial dimensions
        if mask.shape[1:] != image.shape[1:3]:
            mask = torch.nn.functional.interpolate(mask.unsqueeze(1).float(), size=image.shape[1:3],
                                                   mode='nearest').squeeze(1)

        # Normalize mask to 0-1 range if it's not already
        if mask.max() > 1:
            mask = mask / 255.0

        # Add alpha channel to the image if it doesn't exist
        if image.shape[3] == 3:
            alpha = torch.ones((image.shape[0], image.shape[1], image.shape[2], 1), device=image.device)
            image = torch.cat([image, alpha], dim=3)

        # Apply the mask to the alpha channel
        image[:, :, :, 3] = mask

        return (image,)