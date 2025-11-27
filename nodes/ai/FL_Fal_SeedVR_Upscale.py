import io
import numpy as np
import os
import requests
from PIL import Image
import torch
from tqdm import tqdm
import fal_client


class FL_Fal_SeedVR_Upscale:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Enter your Fal AI API key"
                }),
                "image": ("IMAGE",),
                "upscale_factor": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5,
                    "description": "Upscaling factor (multiplies dimensions)"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 999999999,
                    "description": "Random seed (-1 for random)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("upscaled_image", "image_url", "width", "height")
    FUNCTION = "upscale_image"
    CATEGORY = "🏵️Fill Nodes/AI"

    def upload_tensor_to_fal(self, tensor_image):
        """Upload tensor image to fal.media CDN and return the URL."""
        # ComfyUI tensors are in [B, H, W, C] or [H, W, C] format
        # Remove batch dimension if present
        if tensor_image.dim() == 4:
            tensor_image = tensor_image.squeeze(0)

        # Ensure values are in [0, 1] range and convert to [0, 255]
        tensor_image = (tensor_image * 255.0).clamp(0, 255)

        # Convert to uint8 and then to PIL Image
        np_image = tensor_image.cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(np_image)

        # Upload to fal.media CDN
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        url = fal_client.upload(image_bytes, content_type="image/png")
        print(f"[Fal SeedVR Upscale] Uploaded image to CDN: {url[:80]}...")
        return url

    def download_image(self, image_url):
        """Download image from URL and convert to tensor"""
        try:
            print(f"[Fal SeedVR Upscale] Downloading upscaled image...")

            # Download image with progress bar
            response = requests.get(image_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            image_data = io.BytesIO()
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading image") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        image_data.write(chunk)
                        pbar.update(len(chunk))

            # Load image
            image_data.seek(0)
            pil_image = Image.open(image_data).convert('RGB')

            # Convert to tensor
            tensor_image = torch.from_numpy(np.array(pil_image)).float() / 255.0
            tensor_image = tensor_image.unsqueeze(0)  # Add batch dimension

            print(f"[Fal SeedVR Upscale] Image downloaded successfully: {pil_image.width}x{pil_image.height}")

            return tensor_image, pil_image.width, pil_image.height

        except Exception as e:
            print(f"[Fal SeedVR Upscale] Error downloading image: {str(e)}")
            # Return a dummy black image
            dummy_image = torch.zeros(1, 512, 512, 3)
            return dummy_image, 512, 512

    def upscale_image(self, api_key, image, upscale_factor, seed):
        """Upscale image using Fal AI SeedVR API"""
        if not api_key or api_key.strip() == "":
            # Create dummy outputs
            dummy_image = torch.zeros(1, 512, 512, 3)
            return (dummy_image, "", 512, 512)

        try:
            # Clean and set API key FIRST - needed for CDN upload
            clean_api_key = api_key.strip()
            key_preview = clean_api_key[:8] + "..." if len(clean_api_key) > 8 else clean_api_key
            print(f"[Fal SeedVR Upscale] Using API key starting with: {key_preview}")

            # Clear and set the environment variable
            if "FAL_KEY" in os.environ:
                del os.environ["FAL_KEY"]
            os.environ["FAL_KEY"] = clean_api_key

            # Convert the first image from the batch
            if image.dim() == 4 and image.shape[0] > 0:
                first_image = image[0]
            else:
                first_image = image

            # Get original dimensions
            if image.dim() == 4:
                original_height, original_width = image.shape[1], image.shape[2]
            else:
                original_height, original_width = image.shape[0], image.shape[1]

            print(f"[Fal SeedVR Upscale] Original image size: {original_width}x{original_height}")
            print(f"[Fal SeedVR Upscale] Upscale factor: {upscale_factor}x")
            print(f"[Fal SeedVR Upscale] Expected output size: {int(original_width * upscale_factor)}x{int(original_height * upscale_factor)}")

            # Upload image to fal.media CDN
            print(f"[Fal SeedVR Upscale] Uploading image to fal.media CDN...")
            image_url = self.upload_tensor_to_fal(first_image)

            # Prepare arguments
            arguments = {
                "image_url": image_url,
                "upscale_factor": upscale_factor
            }

            # Add seed if specified
            if seed >= 0:
                arguments["seed"] = seed
                print(f"[Fal SeedVR Upscale] Using seed: {seed}")

            # Print arguments without exposing potentially large base64 data
            safe_arguments = {k: v if not (isinstance(v, str) and v.startswith('data:')) else f"<data_uri_{len(v)}_chars>" for k, v in arguments.items()}
            print(f"[Fal SeedVR Upscale] Making API call with arguments: {safe_arguments}")

            # Force reload the fal_client module to avoid caching issues
            import sys
            if 'fal_client' in sys.modules:
                del sys.modules['fal_client']
            import fal_client

            # Submit the request synchronously
            print(f"[Fal SeedVR Upscale] Submitting upscale request...")
            result = fal_client.subscribe(
                "fal-ai/seedvr/upscale/image",
                arguments=arguments,
                with_logs=True
            )

            print(f"[Fal SeedVR Upscale] API call completed successfully")

            # Extract result
            if result and "image" in result:
                image_data = result["image"]
                image_url = image_data.get("url", "")
                output_width = image_data.get("width", 0)
                output_height = image_data.get("height", 0)
                used_seed = result.get("seed", seed)

                print(f"[Fal SeedVR Upscale] Upscaled image URL: {image_url}")
                print(f"[Fal SeedVR Upscale] Output dimensions: {output_width}x{output_height}")
                print(f"[Fal SeedVR Upscale] Used seed: {used_seed}")

                # Download the upscaled image
                upscaled_tensor, actual_width, actual_height = self.download_image(image_url)

                status_message = f"Successfully upscaled image {upscale_factor}x using Fal AI SeedVR. Output: {actual_width}x{actual_height}"
                print(f"[Fal SeedVR Upscale] {status_message}")

                return (upscaled_tensor, image_url, actual_width, actual_height)

            else:
                error_msg = "No image data in API response"
                print(f"[Fal SeedVR Upscale] {error_msg}")
                print(f"[Fal SeedVR Upscale] Full response: {result}")
                # Return dummy outputs
                dummy_image = torch.zeros(1, 512, 512, 3)
                return (dummy_image, "", 512, 512)

        except Exception as e:
            error_msg = f"Error in image upscaling: {str(e)}"
            print(f"[Fal SeedVR Upscale] {error_msg}")

            # Print traceback for debugging
            import traceback
            traceback.print_exc()

            # Return dummy outputs
            dummy_image = torch.zeros(1, 512, 512, 3)
            return (dummy_image, "", 512, 512)


# Node registration
NODE_CLASS_MAPPINGS = {
    "FL_Fal_SeedVR_Upscale": FL_Fal_SeedVR_Upscale
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_Fal_SeedVR_Upscale": "FL Fal SeedVR Upscale"
}
