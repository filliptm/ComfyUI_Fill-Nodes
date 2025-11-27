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
        # ComfyUI images are always [B, H, W, C] format
        batch_size = image.shape[0]
        height = image.shape[1]
        width = image.shape[2]

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

        return (cropped_image,)

    @classmethod
    def IS_CHANGED(cls, image, crop_direction, crop_amount):
        return float("NaN")  # Always update when parameters change