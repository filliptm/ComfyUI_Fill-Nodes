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
        else:
            image1 = images[0]
            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = comfy.utils.common_upscale(
                        image2.movedim(-1, 1),
                        image1.shape[2],
                        image1.shape[1],
                        "lanczos",
                        "center"
                    ).movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)
            return (image1,)


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