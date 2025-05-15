import torch
import comfy
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from tqdm import tqdm

class FL_UpscaleModel:
    rescale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    precision_options = ["auto", "32", "16", "bfloat16"]

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
                    "max": 100,
                    "step": 1,
                }),
            }
        }

    def upscale(self, upscale_model, image, downscale_by, rescale_method, precision, batch_size):
        original_device = image.device
        original_dtype = image.dtype

        # Determine the appropriate dtype based on precision and device
        if precision == "auto":
            dtype = torch.float16 if original_device.type == "cuda" else torch.float32
        elif precision == "16":
            dtype = torch.float16 if original_device.type == "cuda" else torch.bfloat16
        elif precision == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Ensure the chosen dtype is supported on the current device
        if dtype == torch.float16 and original_device.type != "cuda":
            print("Warning: float16 is not supported on CPU. Falling back to bfloat16.")
            dtype = torch.bfloat16

        upscale_model = upscale_model.to(dtype).to(original_device)

        # Split the input batch into a list of individual images
        image_list = list(torch.split(image, 1))
        total_images = len(image_list)

        upscaled_list = []

        # Create a tqdm progress bar
        pbar = tqdm(total=total_images, desc="Processing frames", unit="frame")

        for i in range(0, total_images, batch_size):
            batch = torch.cat(image_list[i:i + batch_size]).to(dtype).contiguous()

            with torch.no_grad():
                if dtype in [torch.float16, torch.bfloat16]:
                    with torch.autocast(device_type=original_device.type, dtype=dtype):
                        upscaled_batch = self.__imageScaler.upscale(upscale_model, batch)[0]
                else:
                    upscaled_batch = self.__imageScaler.upscale(upscale_model, batch)[0]

            if downscale_by < 1.0:
                target_height = round(upscaled_batch.shape[1] * downscale_by)
                target_width = round(upscaled_batch.shape[2] * downscale_by)

                upscaled_batch = upscaled_batch.permute(0, 3, 1, 2).contiguous()

                upscaled_batch = torch.nn.functional.interpolate(
                    upscaled_batch,
                    size=(target_height, target_width),
                    mode=rescale_method if rescale_method != "lanczos" else "bicubic",
                    align_corners=False if rescale_method in ["bilinear", "bicubic"] else None
                )

                upscaled_batch = upscaled_batch.permute(0, 2, 3, 1).contiguous()
            else:
                # Ensure contiguity if no permute operations (which now include .contiguous()) were performed.
                # This handles cases where __imageScaler.upscale might return a non-contiguous tensor.
                upscaled_batch = upscaled_batch.contiguous()

            if dtype != original_dtype or downscale_by < 1.0:
                upscaled_batch = upscaled_batch.clamp(0, 1).to(original_dtype).to(original_device)

            # upscaled_batch is now guaranteed to be contiguous before splitting.
            upscaled_list.extend(list(torch.split(upscaled_batch, 1)))

            # Update the progress bar
            pbar.update(len(batch))

        # Close the progress bar
        pbar.close()

        # Combine all processed images back into a single batch
        final_upscaled = torch.cat(upscaled_list)

        print(f"Upscaling complete. Processed {total_images} frames in batches of {batch_size}.")

        return (final_upscaled,)