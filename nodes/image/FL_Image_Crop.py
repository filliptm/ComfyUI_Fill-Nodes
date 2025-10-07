import torch

class FL_ImageCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "crop_direction": (["Top", "Bottom", "Left", "Right", "Top and Bottom", "Left and Right"], {"default": "Top"}),
                "crop_amount": ("INT", {"default": 50, "min": 1, "max": 2048, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_image"
    CATEGORY = "🏵️Fill Nodes/Image"

    def crop_image(self, image, crop_direction, crop_amount):
        if isinstance(image, torch.Tensor):
            # Handle batch of images (4D tensor) or single image (3D tensor)
            if image.dim() == 4:  # Batch dimension is present [batch, height, width, channels]
                batch_size, height, width, channels = image.shape
            elif image.dim() == 3:  # No batch dimension, single image [height, width, channels]
                height, width, channels = image.shape
                # Add batch dimension for consistent processing
                image = image.unsqueeze(0)
                batch_size = 1
            else:
                raise ValueError("Unsupported tensor format")
        else:
            raise ValueError("Input must be a torch tensor")

        # Validate crop amount doesn't exceed image dimensions
        if crop_direction in ["Top", "Bottom", "Top and Bottom"]:
            max_crop = height // 2 if crop_direction == "Top and Bottom" else height - 1
            if crop_amount > max_crop:
                crop_amount = max_crop
        elif crop_direction in ["Left", "Right", "Left and Right"]:
            max_crop = width // 2 if crop_direction == "Left and Right" else width - 1
            if crop_amount > max_crop:
                crop_amount = max_crop

        # Apply cropping based on direction
        if crop_direction == "Top":
            cropped_image = image[:, crop_amount:, :, :]
        elif crop_direction == "Bottom":
            cropped_image = image[:, :-crop_amount, :, :]
        elif crop_direction == "Left":
            cropped_image = image[:, :, crop_amount:, :]
        elif crop_direction == "Right":
            cropped_image = image[:, :, :-crop_amount, :]
        elif crop_direction == "Top and Bottom":
            cropped_image = image[:, crop_amount:-crop_amount, :, :]
        elif crop_direction == "Left and Right":
            cropped_image = image[:, :, crop_amount:-crop_amount, :]

        # If original input was 3D (single image), remove batch dimension
        if batch_size == 1 and len(cropped_image.shape) == 4:
            cropped_image = cropped_image.squeeze(0)

        return (cropped_image,)

    @classmethod
    def IS_CHANGED(cls, image, crop_direction, crop_amount):
        return float("NaN")  # Always update when parameters change