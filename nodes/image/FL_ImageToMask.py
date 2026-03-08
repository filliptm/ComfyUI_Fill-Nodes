import torch


class FL_ImageToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": (["luminance", "red", "green", "blue", "alpha"], {
                    "default": "luminance",
                    "description": "Which channel to extract as the mask"
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "image_to_mask"
    CATEGORY = "🏵️Fill Nodes/Image"

    def image_to_mask(self, image, channel):
        # image shape: (B, H, W, C)
        if channel == "luminance":
            # Standard luminance weights
            mask = 0.2989 * image[:, :, :, 0] + 0.5870 * image[:, :, :, 1] + 0.1140 * image[:, :, :, 2]
        elif channel == "red":
            mask = image[:, :, :, 0]
        elif channel == "green":
            mask = image[:, :, :, 1]
        elif channel == "blue":
            mask = image[:, :, :, 2]
        elif channel == "alpha":
            if image.shape[3] > 3:
                mask = image[:, :, :, 3]
            else:
                mask = torch.ones(image.shape[0], image.shape[1], image.shape[2], device=image.device)

        return (mask,)
