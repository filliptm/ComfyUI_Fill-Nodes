import torch


class FL_ApplyMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "crop_to_mask": ("BOOLEAN", {"default": False, "description": "Crop image to mask bounding box"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_mask"
    CATEGORY = "🏵️Fill Nodes/Image"

    def apply_mask(self, image, mask, crop_to_mask=False):
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

        # Crop to mask bounding box if enabled
        if crop_to_mask:
            # Process each image in the batch
            cropped_images = []
            for b in range(image.shape[0]):
                # Find bounding box of non-zero mask values
                mask_b = mask[b]
                nonzero = torch.nonzero(mask_b > 0)

                if nonzero.numel() > 0:
                    # Get min/max coordinates
                    y_min = nonzero[:, 0].min().item()
                    y_max = nonzero[:, 0].max().item() + 1
                    x_min = nonzero[:, 1].min().item()
                    x_max = nonzero[:, 1].max().item() + 1

                    # Crop the image
                    cropped = image[b, y_min:y_max, x_min:x_max, :]
                    cropped_images.append(cropped.unsqueeze(0))
                else:
                    # If mask is empty, keep original image
                    cropped_images.append(image[b].unsqueeze(0))

            # Stack cropped images (note: may have different sizes if masks differ)
            if len(cropped_images) == 1:
                image = cropped_images[0]
            else:
                # If all images have the same size, stack them
                shapes = [img.shape for img in cropped_images]
                if all(s == shapes[0] for s in shapes):
                    image = torch.cat(cropped_images, dim=0)
                else:
                    # Different sizes - return first one or handle differently
                    # For now, return the first cropped image
                    image = cropped_images[0]

        return (image,)