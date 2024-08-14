import torch
import comfy
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

class FL_UpscaleModel:
    rescale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    precision_options = ["16", "32"]  # Removed "8" as it's not standard

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
            }
        }

    def upscale(self, upscale_model, image, downscale_by, rescale_method, precision):
        original_device = image.device
        original_dtype = image.dtype

        if precision == "16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        upscale_model = upscale_model.to(dtype).to(original_device)
        image = image.to(dtype)

        with torch.no_grad():
            if dtype == torch.float16:
                with torch.autocast(device_type=original_device.type, dtype=dtype):
                    upscaled = self.__imageScaler.upscale(upscale_model, image)[0]
            else:
                upscaled = self.__imageScaler.upscale(upscale_model, image)[0]

        if downscale_by < 1.0:
            target_height = round(upscaled.shape[1] * downscale_by)
            target_width = round(upscaled.shape[2] * downscale_by)

            # upscaled is already in [B, H, W, C] format
            # We need to change it to [B, C, H, W] for interpolate
            upscaled = upscaled.permute(0, 3, 1, 2)

            upscaled = torch.nn.functional.interpolate(
                upscaled,
                size=(target_height, target_width),
                mode=rescale_method if rescale_method != "lanczos" else "bicubic",
                align_corners=False if rescale_method in ["bilinear", "bicubic"] else None
            )

            # Change back to [B, H, W, C]
            upscaled = upscaled.permute(0, 2, 3, 1)

        # Only clamp and convert if necessary
        if dtype != original_dtype or downscale_by < 1.0:
            upscaled = upscaled.clamp(0, 1).to(original_dtype).to(original_device)

        return (upscaled,)