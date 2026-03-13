import torch
import comfy.utils

class FL_ImageListToImageBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "images": ("IMAGE",),
                }}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"
    CATEGORY = "🏵️Fill Nodes/Image"

    def doit(self, images):
        if len(images) <= 1:
            return (images[0],)

        target_shape = images[0].shape[1:]  # (H, W, C)
        processed = []
        for img in images:
            if img.shape[1:] != target_shape:
                img = comfy.utils.common_upscale(
                    img.movedim(-1, 1),
                    target_shape[1],
                    target_shape[0],
                    "lanczos",
                    "center"
                ).movedim(1, -1)
            processed.append(img)

        return (torch.cat(processed, dim=0),)


class FL_ImageBatchToImageList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"
    CATEGORY = "🏵️Fill Nodes/Image"

    def doit(self, image):
        images = [image[i:i + 1, ...] for i in range(image.shape[0])]
        return (images,)