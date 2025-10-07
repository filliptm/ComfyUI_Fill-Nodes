import os
import torch
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import comfy.model_management
from comfy.utils import ProgressBar


class FL_HunyuanDelight:
    def __init__(self):
        self.pipe = None
        self.model_path = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cfg_image": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "loops": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "🏵️Fill Nodes/AI"

    def download_model(self, model_name="hunyuan3d-delight-v2-0"):
        download_path = os.path.join("models", "diffusers")
        model_path = os.path.join(download_path, model_name)

        if not os.path.exists(model_path):
            print(f"Downloading Hunyuan3D model to: {model_path}")
            os.makedirs("models/diffusers", exist_ok=True)
            snapshot_download(
                repo_id="tencent/Hunyuan3D-2",
                allow_patterns=[f"*{model_name}*"],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )
        return model_path

    def load_model(self):
        if self.pipe is None:
            device = comfy.model_management.get_torch_device()
            model_path = self.download_model()

            self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                safety_checker=None,
            )
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(device)

            if device.type == "cuda":
                self.pipe.enable_model_cpu_offload()

    def process_image(self, image, cfg_image, steps, seed, loops):
        try:
            self.load_model()
            device = comfy.model_management.get_torch_device()

            # Store original batch dimension state
            original_batch = len(image.shape) == 4
            if not original_batch:
                image = image.unsqueeze(0)

            # Calculate total steps for progress bar
            total_images = image.shape[0]
            total_steps = steps * total_images * loops
            pbar = ProgressBar(total_steps)
            current_step = 0

            # Current working image (will be updated each loop)
            current_image = image

            # Process for each loop
            for loop in range(loops):
                output_images = []
                print(f"Processing loop {loop + 1}/{loops}")

                for batch_idx in range(total_images):
                    # Prepare single image
                    single_image = current_image[batch_idx]

                    # Ensure correct channel order (HWC -> CHW)
                    if single_image.shape[-1] == 3:
                        single_image = single_image.permute(2, 0, 1)

                    # Add batch dimension and normalize
                    image_tensor = single_image.unsqueeze(0)
                    if image_tensor.max() > 1.0:
                        image_tensor = image_tensor / 255.0

                    image_tensor = image_tensor.to(device)

                    # Get original dimensions
                    _, _, height, width = image_tensor.shape

                    def callback_fn(step, timestep, latents):
                        if step is not None:
                            # Calculate the global step across all loops
                            global_step = (loop * steps * total_images) + (batch_idx * steps) + step
                            pbar.update_absolute(global_step)

                    try:
                        with torch.no_grad():
                            output = self.pipe(
                                prompt="",
                                image=image_tensor,
                                generator=torch.manual_seed(seed + (loop * 1000)),
                                height=height,
                                width=width,
                                num_inference_steps=steps,
                                image_guidance_scale=cfg_image,
                                guidance_scale=1.0,  # Fixed value for cfg_text
                                output_type="pt",
                                callback=callback_fn,
                                callback_steps=1
                            ).images[0]

                        output = output.cpu()
                        if output.shape[0] == 3:
                            output = output.permute(1, 2, 0)

                        if len(output.shape) == 3:
                            output = output.unsqueeze(0)

                        output_images.append(output)

                    except Exception as e:
                        print(f"Error processing batch {batch_idx} in loop {loop + 1}: {str(e)}")
                        output_images.append(current_image[batch_idx].unsqueeze(0))

                if not output_images:
                    print(f"No images processed in loop {loop + 1}")
                    break

                # Update current_image for next loop
                current_image = torch.cat(output_images, dim=0)

            # Final output processing
            final_output = current_image
            if not original_batch:
                final_output = final_output.squeeze(0)

            print(f"Final output shape: {final_output.shape}")

            # Clear CUDA cache
            if device.type == "cuda":
                torch.cuda.empty_cache()

            return (final_output,)

        except Exception as e:
            print(f"Error in Hunyuan Delight processing: {str(e)}")
            return (image,)