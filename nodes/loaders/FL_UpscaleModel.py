import torch
from comfy import model_management
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from tqdm import tqdm


class _DtypeUpscaleModel:
    def __init__(self, model, dtype):
        self.model = model
        self.dtype = dtype
        self.patcher = model.patcher
        self.scale = model.scale

    def __call__(self, image):
        return self.model(image.to(dtype=self.dtype))


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
        model_device = upscale_model.patcher.load_device

        if precision == "auto":
            dtype = torch.float16 if upscale_model.supports_half and model_management.should_use_fp16(model_device) else torch.float32
        elif precision == "16":
            if upscale_model.supports_half and model_management.should_use_fp16(model_device):
                dtype = torch.float16
            elif upscale_model.supports_bfloat16 and model_management.should_use_bf16(model_device):
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        elif precision == "bfloat16":
            if not upscale_model.supports_bfloat16 or not model_management.should_use_bf16(model_device):
                raise ValueError(f"bfloat16 precision is not supported by this upscale model on {model_device.type}.")
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        upscale_model.to(dtype=dtype)
        scaler_model = _DtypeUpscaleModel(upscale_model, dtype)

        # Split the input batch into a list of individual images
        image_list = list(torch.split(image, 1))
        total_images = len(image_list)

        upscaled_list = []

        # Create a tqdm progress bar
        pbar = tqdm(total=total_images, desc="Processing frames", unit="frame")

        for i in range(0, total_images, batch_size):
            batch = torch.cat(image_list[i:i + batch_size]).contiguous()
            upscaled_batch = self.__imageScaler.upscale(scaler_model, batch)[0]

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
