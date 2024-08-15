import torch
import comfy
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel


class FL_UpscaleModel:
    rescale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    precision_options = ["16", "32"]

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "🏵️Fill Nodes/Loaders"

    def __init__(self):
        self.__imageScaler = ImageUpscaleWithModel()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "downscale_by": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.25,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "rescale_method": (cls.rescale_methods,),
                "precision": (cls.precision_options,),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                }),
            }
        }

    def upscale(self, upscale_model, image, downscale_by, rescale_method, precision, batch_size):
        original_device = image.device
        original_dtype = image.dtype

        if precision == "16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        upscale_model = upscale_model.to(dtype).to(original_device)

        # Split the input batch into a list of individual images
        image_list = list(torch.split(image, 1))
        total_images = len(image_list)

        upscaled_list = []

        for i in range(0, total_images, batch_size):
            batch = torch.cat(image_list[i:i + batch_size]).to(dtype)

            with torch.no_grad():
                if dtype == torch.float16:
                    with torch.autocast(device_type=original_device.type, dtype=dtype):
                        upscaled_batch = self.__imageScaler.upscale(upscale_model, batch)[0]
                else:
                    upscaled_batch = self.__imageScaler.upscale(upscale_model, batch)[0]

            if downscale_by < 1.0:
                target_height = round(upscaled_batch.shape[1] * downscale_by)
                target_width = round(upscaled_batch.shape[2] * downscale_by)

                upscaled_batch = upscaled_batch.permute(0, 3, 1, 2)

                upscaled_batch = torch.nn.functional.interpolate(
                    upscaled_batch,
                    size=(target_height, target_width),
                    mode=rescale_method if rescale_method != "lanczos" else "bicubic",
                    align_corners=False if rescale_method in ["bilinear", "bicubic"] else None
                )

                upscaled_batch = upscaled_batch.permute(0, 2, 3, 1)

            if dtype != original_dtype or downscale_by < 1.0:
                upscaled_batch = upscaled_batch.clamp(0, 1).to(original_dtype).to(original_device)

            upscaled_list.extend(list(torch.split(upscaled_batch, 1)))

        # Combine all processed images back into a single batch
        final_upscaled = torch.cat(upscaled_list)

        return (final_upscaled,)