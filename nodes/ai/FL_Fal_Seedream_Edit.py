# FL_Fal_Seedream_Edit: Fal AI ByteDance Seedream v4 Edit API Node
import os
import uuid
import json
import time
import io
import requests
import torch
import numpy as np
import base64
import fal_client
import asyncio
import concurrent.futures
import random
from typing import Tuple, List, Dict, Union, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from comfy.utils import ProgressBar


class FL_Fal_Seedream_Edit:
    """
    A ComfyUI node for the Fal AI ByteDance Seedream v4 Edit API.
    Takes multiple images and a prompt to edit them using Seedream's capabilities.
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "image_urls", "seed", "status_msg")
    FUNCTION = "edit_images"
    CATEGORY = "🏵️Fill Nodes/AI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False,
                                     "description": "Fal AI API key"}),
                "prompt": ("STRING", {"default": "Dress the model in the clothes and shoes.",
                                    "multiline": True, "forceInput": True,
                                    "description": "The prompt for image editing"}),
                "image_size": (["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"], 
                             {"default": "square_hd", "description": "The size of the generated image"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1,
                                     "description": "Number of images to generate"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647,
                               "description": "Random seed for image generation (0 = random)"}),
                "sync_mode": ("BOOLEAN", {"default": False,
                                        "description": "Wait for image generation before returning"}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
            },
            "optional": {
                "image_1": ("IMAGE", {"description": "First input image to edit"}),
                "image_2": ("IMAGE", {"description": "Second input image to edit"}),
                "image_3": ("IMAGE", {"description": "Third input image to edit"}),
                "image_4": ("IMAGE", {"description": "Fourth input image to edit"}),
                "image_5": ("IMAGE", {"description": "Fifth input image to edit"}),
                "image_6": ("IMAGE", {"description": "Sixth input image to edit"}),
                "image_7": ("IMAGE", {"description": "Seventh input image to edit"}),
                "image_8": ("IMAGE", {"description": "Eighth input image to edit"}),
                "image_9": ("IMAGE", {"description": "Ninth input image to edit"}),
                "image_10": ("IMAGE", {"description": "Tenth input image to edit"}),
                "retry_indefinitely": ("BOOLEAN", {"default": False}),
                "use_custom_resolution": ("BOOLEAN", {"default": False,
                                                        "description": "Override preset image size with custom width/height"}),
                "auto_scale_to_minimum": ("BOOLEAN", {"default": True,
                                                      "description": "Auto-scale dimensions below 1024 to meet minimum while preserving aspect ratio"}),
                "custom_width": ("INT", {"default": 1280, "min": 256, "max": 4096, "step": 8,
                                       "description": "Custom width (auto-scaled to meet API minimums if needed)"}),
                "custom_height": ("INT", {"default": 1280, "min": 256, "max": 4096, "step": 8,
                                        "description": "Custom height (auto-scaled to meet API minimums if needed)"}),
            }
        }

    def __init__(self):
        self.log_messages = []

    def _log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[FL_Fal_Seedream_Edit] {timestamp}: {message}"
        print(formatted_message)
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message

    def _create_error_image(self, error_message="API Error", width=1024, height=1024):
        image = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        font = None
        try:
            font_options = ['arial.ttf', 'DejaVuSans.ttf', 'FreeSans.ttf', 'NotoSans-Regular.ttf']
            for font_name in font_options:
                try:
                    font = ImageFont.truetype(font_name, 24)
                    break
                except IOError:
                    continue
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Calculate text position (centered)
        try:
            text_bbox = draw.textbbox((0,0), error_message, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(error_message, font=font)

        text_x = (width - text_width) / 2
        text_y = (height - text_height) / 2
        draw.text((text_x, text_y), error_message, fill=(255, 0, 0), font=font)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        self._log(f"Created error image: '{error_message}'")
        return img_tensor

    def _process_tensor_to_pil_list(self, tensor_image: Optional[torch.Tensor], image_name_prefix: str = "Image") -> Optional[List[Image.Image]]:
        if tensor_image is None:
            self._log(f"{image_name_prefix} input is None, skipping PIL conversion.")
            return None
        if not isinstance(tensor_image, torch.Tensor):
            self._log(f"{image_name_prefix} is not a tensor (type: {type(tensor_image)}), skipping.")
            return None

        pil_images = []
        if tensor_image.ndim == 4:  # Batch of images (B, H, W, C)
            if tensor_image.shape[0] == 0:
                self._log(f"{image_name_prefix} batch is empty (shape: {tensor_image.shape}).")
                return None
            for i in range(tensor_image.shape[0]):
                img_np = tensor_image[i].cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                pil_image = Image.fromarray(img_np)
                self._log(f"Converted {image_name_prefix} batch item {i} (original shape: {tensor_image.shape}) to PIL Image (size: {pil_image.size}).")
                pil_images.append(pil_image)
        elif tensor_image.ndim == 3:  # Single image (H, W, C)
            img_np = tensor_image.cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            self._log(f"Converted single {image_name_prefix} (original shape: {tensor_image.shape}) to PIL Image (size: {pil_image.size}).")
            pil_images.append(pil_image)
        else:
            self._log(f"Cannot convert {image_name_prefix} with ndim {tensor_image.ndim} (shape: {tensor_image.shape}) to PIL Image(s).")
            return None
        
        return pil_images if pil_images else None

    def _convert_image_to_url(self, pil_image: Image.Image) -> str:
        """Convert PIL image to base64 data URI with size optimization"""
        try:
            # First, try to compress the image to reduce payload size
            max_dimension = 2048  # Reduce from potentially 4096 to 2048 max
            original_size = pil_image.size
            
            # Convert RGBA to RGB if necessary (JPEG doesn't support transparency)
            if pil_image.mode == 'RGBA':
                # Create a white background and composite the RGBA image onto it
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                background.paste(pil_image, mask=pil_image.split()[-1])  # Use alpha channel as mask
                pil_image = background
                self._log(f"Converted RGBA image to RGB with white background")
            elif pil_image.mode != 'RGB':
                # Convert any other mode to RGB
                pil_image = pil_image.convert('RGB')
                self._log(f"Converted image mode to RGB")
            
            # Resize if image is too large
            if max(pil_image.size) > max_dimension:
                ratio = max_dimension / max(pil_image.size)
                new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                self._log(f"Resized image from {original_size} to {new_size} to reduce payload size")
            
            # Try JPEG first for smaller file size
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG", quality=85, optimize=True)
            jpeg_size = len(buffered.getvalue())
            
            # If JPEG is still too large (>800KB per image to stay well under 4MB total), reduce quality
            if jpeg_size > 800 * 1024:
                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG", quality=70, optimize=True)
                self._log(f"Reduced JPEG quality to 70% to manage payload size")
            
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            img_data_uri = f"data:image/jpeg;base64,{img_base64}"
            
            # Log the approximate size
            payload_size = len(img_data_uri)
            self._log(f"Image payload size: ~{payload_size // 1024}KB")
            
            return img_data_uri
        except Exception as e:
            self._log(f"Error converting image to base64: {str(e)}")
            raise

    async def _edit_images_async(self, api_key, prompt, input_images, image_size, num_images, seed, sync_mode, max_retries, retry_indefinitely, use_custom_resolution, auto_scale_to_minimum, custom_width, custom_height):
        try:
            # Calculate seed
            actual_seed = seed if seed != 0 else random.randint(1, 2147483647)
            
            self._log(f"Starting image editing with seed {actual_seed} and prompt: '{prompt[:50]}...'")
            
            # Prepare image URLs from input images
            image_urls = []
            if input_images:
                total_payload_size = 0
                for i, pil_image in enumerate(input_images):
                    try:
                        img_url = self._convert_image_to_url(pil_image)
                        image_urls.append(img_url)
                        total_payload_size += len(img_url)
                        self._log(f"Successfully converted image {i+1} to data URI")
                    except Exception as e:
                        self._log(f"Error converting image {i+1} to data URI: {str(e)}")
                        error_msg = f"Error: Failed to convert image {i+1}: {str(e)}"
                        return self._create_error_image(error_msg), "", str(actual_seed), error_msg
                
                # Check if total payload size is approaching the 4MB limit
                max_payload_size = 3.5 * 1024 * 1024  # 3.5MB to leave room for other data
                if total_payload_size > max_payload_size:
                    self._log(f"Warning: Total payload size ({total_payload_size // 1024}KB) is approaching API limits")
                else:
                    self._log(f"Total payload size: ~{total_payload_size // 1024}KB (within limits)")
            else:
                error_msg = "Error: No images provided for editing"
                return self._create_error_image(error_msg), "", str(actual_seed), error_msg
            
            # Clear any existing FAL_KEY environment variable to prevent caching issues
            if "FAL_KEY" in os.environ:
                del os.environ["FAL_KEY"]
            
            # Prepare the API request
            clean_api_key = api_key.strip()
            
            # Prepare image size parameter
            image_size_param = image_size
            # Check if custom dimensions are requested and valid
            if use_custom_resolution:
                if custom_width and custom_height and custom_width >= 1024 and custom_height >= 1024:
                    image_size_param = {
                        "width": custom_width,
                        "height": custom_height
                    }
                    self._log(f"Using custom image size: {custom_width}x{custom_height}")
                elif custom_width and custom_height:
                    # Custom dimensions provided but below minimum
                    aspect_ratio = custom_width / custom_height
                    self._log(f"Custom dimensions {custom_width}x{custom_height} below minimum. Aspect ratio: {aspect_ratio:.2f}")
                    
                    if auto_scale_to_minimum:
                        # Scale up dimensions to meet minimum while preserving aspect ratio
                        min_dimension = min(custom_width, custom_height)
                        if min_dimension < 1024:
                            scale_factor = 1024 / min_dimension
                            scaled_width = int(custom_width * scale_factor)
                            scaled_height = int(custom_height * scale_factor)
                            
                            # Ensure we don't exceed maximum dimensions
                            if scaled_width > 4096 or scaled_height > 4096:
                                max_scale = min(4096 / custom_width, 4096 / custom_height)
                                scaled_width = int(custom_width * max_scale)
                                scaled_height = int(custom_height * max_scale)
                            
                            # Round to nearest 64 pixels for better compatibility
                            scaled_width = (scaled_width // 64) * 64
                            scaled_height = (scaled_height // 64) * 64
                            
                            # Ensure minimums after rounding
                            scaled_width = max(1024, scaled_width)
                            scaled_height = max(1024, scaled_height)
                            
                            image_size_param = {
                                "width": scaled_width,
                                "height": scaled_height
                            }
                            self._log(f"Auto-scaled from {custom_width}x{custom_height} to {scaled_width}x{scaled_height} preserving aspect ratio")
                    else:
                        # Map aspect ratios to presets (with tolerance)
                        if aspect_ratio >= 2.1:  # ~21:9 ratio
                            image_size_param = "landscape_16_9"  # Closest available wide option
                            self._log(f"Selected landscape_16_9 based on wide aspect ratio")
                        elif aspect_ratio >= 1.6:  # ~16:9 ratio
                            image_size_param = "landscape_16_9"
                            self._log(f"Selected landscape_16_9 based on 16:9 aspect ratio")
                        elif aspect_ratio >= 1.2:  # ~4:3 ratio
                            image_size_param = "landscape_4_3"
                            self._log(f"Selected landscape_4_3 based on 4:3 aspect ratio")
                        elif aspect_ratio >= 0.9:  # ~1:1 ratio
                            image_size_param = "square_hd" if "hd" in image_size else "square"
                            self._log(f"Selected square based on 1:1 aspect ratio")
                        elif aspect_ratio >= 0.7:  # ~3:4 ratio
                            image_size_param = "portrait_4_3"
                            self._log(f"Selected portrait_4_3 based on 3:4 aspect ratio")
                        elif aspect_ratio >= 0.5:  # ~9:16 ratio
                            image_size_param = "portrait_16_9"
                            self._log(f"Selected portrait_16_9 based on 9:16 aspect ratio")
                        else:  # Very tall
                            image_size_param = "portrait_16_9"
                            self._log(f"Selected portrait_16_9 based on tall aspect ratio")
                else:
                    self._log("Custom resolution requested but dimensions are invalid or missing. Using preset image size.")
                    image_size_param = image_size
                    self._log(f"Using preset image size: {image_size}")
            else:
                self._log(f"Using preset image size: {image_size}")
            
            # Prepare the arguments for fal_client
            arguments = {
                "prompt": prompt,
                "image_urls": image_urls,
                "image_size": image_size_param,
                "num_images": num_images,
                "seed": actual_seed,
                "sync_mode": sync_mode
            }
            
            # Set the API key as an environment variable for fal_client
            os.environ["FAL_KEY"] = clean_api_key
            
            self._log(f"Calling Fal AI Seedream Edit API with {len(image_urls)} images...")
            
            # Define a callback for queue updates
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        self._log(f"API Log: {log['message']}")
            
            # Make the API call in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def make_fal_call():
                try:
                    # Force reload the fal_client module to avoid caching issues
                    import sys
                    if 'fal_client' in sys.modules:
                        del sys.modules['fal_client']
                    import fal_client
                    
                    # Make the API call using fal_client.subscribe
                    result = fal_client.subscribe(
                        "fal-ai/bytedance/seedream/v4/edit",
                        arguments=arguments,
                        with_logs=True,
                        on_queue_update=on_queue_update,
                    )
                    return result
                except Exception as e:
                    self._log(f"API call error: {str(e)}")
                    return None
            
            result = await loop.run_in_executor(None, make_fal_call)
            
            if result is None:
                error_msg = "Error: API call failed"
                return self._create_error_image(error_msg), "", str(actual_seed), error_msg
            
            self._log("API call completed successfully")
            
            # Extract image URLs and seed from the result
            output_image_urls = []
            returned_seed = actual_seed
            
            if "images" in result and len(result["images"]) > 0:
                for img_info in result["images"]:
                    if "url" in img_info:
                        output_image_urls.append(img_info["url"])
                self._log(f"Found {len(output_image_urls)} edited images in response")
            else:
                self._log("Warning: No images found in result")
                error_msg = "Error: No images in API response"
                return self._create_error_image(error_msg), "", str(actual_seed), error_msg
            
            # Extract returned seed
            if "seed" in result:
                returned_seed = result["seed"]
                self._log(f"API returned seed: {returned_seed}")
            
            # Download and process all generated images
            try:
                self._log(f"Downloading {len(output_image_urls)} edited images...")
                
                processed_images = []
                url_list = []
                
                for i, image_url in enumerate(output_image_urls):
                    # Download image
                    dl_response = requests.get(image_url)
                    dl_response.raise_for_status()
                    
                    # Convert to PIL Image
                    pil_image = Image.open(io.BytesIO(dl_response.content))
                    
                    # Convert to RGB if necessary
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # Convert to numpy array
                    np_image = np.array(pil_image).astype(np.float32) / 255.0
                    
                    # Convert to tensor (add batch dimension)
                    image_tensor = torch.from_numpy(np_image).unsqueeze(0)
                    processed_images.append(image_tensor)
                    url_list.append(image_url)
                    
                    self._log(f"Processed image {i+1}/{len(output_image_urls)} with shape {image_tensor.shape}")
                
                # Concatenate all images
                if len(processed_images) > 1:
                    # Handle multiple images - ensure same dimensions
                    max_height = max(img.shape[1] for img in processed_images)
                    max_width = max(img.shape[2] for img in processed_images)
                    
                    resized_images = []
                    for img_tensor in processed_images:
                        current_h, current_w = img_tensor.shape[1], img_tensor.shape[2]
                        
                        if current_h == max_height and current_w == max_width:
                            resized_images.append(img_tensor)
                        else:
                            # Pad the image
                            pad_h = max_height - current_h
                            pad_w = max_width - current_w
                            pad_left = pad_w // 2
                            pad_right = pad_w - pad_left
                            pad_top = pad_h // 2
                            pad_bottom = pad_h - pad_top
                            
                            padded_img = torch.nn.functional.pad(
                                img_tensor.permute(0, 3, 1, 2),
                                (pad_left, pad_right, pad_top, pad_bottom),
                                mode='constant',
                                value=0
                            ).permute(0, 2, 3, 1)
                            
                            resized_images.append(padded_img)
                    
                    combined_tensor = torch.cat(resized_images, dim=0)
                else:
                    combined_tensor = processed_images[0]
                
                combined_urls = " | ".join(url_list)
                
                self._log(f"Successfully processed {len(processed_images)} images with final shape {combined_tensor.shape}")
                
                return combined_tensor, combined_urls, str(returned_seed), f"Success: {len(processed_images)} images edited successfully"
                
            except Exception as e:
                error_msg = f"Download Error: {str(e)}"
                self._log(error_msg)
                return self._create_error_image(error_msg), "", str(actual_seed), error_msg
            
        except Exception as e:
            self._log(f"Error in async image editing: {str(e)}")
            error_msg = f"Error: {str(e)}"
            return self._create_error_image(error_msg), "", str(actual_seed), error_msg

    def edit_images(self, api_key, prompt, image_size="square_hd", num_images=1, seed=0, sync_mode=False, max_retries=3,
                   image_1=None, image_2=None, image_3=None, image_4=None, image_5=None,
                   image_6=None, image_7=None, image_8=None, image_9=None, image_10=None,
                   retry_indefinitely=False, use_custom_resolution=False, auto_scale_to_minimum=True,
                   custom_width=1280, custom_height=1280, **kwargs):
        self.log_messages = []
        if not api_key:
            error_msg = "API key not provided."
            self._log(error_msg)
            error_img_instance = self._create_error_image(error_msg)
            return (error_img_instance, "", "0", error_msg)

        # Collect all input images (up to 10 as per API limit)
        input_images = []
        input_tensors = [image_1, image_2, image_3, image_4, image_5, 
                        image_6, image_7, image_8, image_9, image_10]
        
        for i, tensor in enumerate(input_tensors):
            if tensor is not None:
                pil_images = self._process_tensor_to_pil_list(tensor, f"Image{i+1}")
                if pil_images:
                    input_images.extend(pil_images)
                    
                # Limit to 10 images as per API specification
                if len(input_images) >= 10:
                    input_images = input_images[:10]
                    self._log("Limiting to 10 images as per API specification")
                    break
        
        if not input_images:
            error_msg = "No valid input images provided."
            self._log(error_msg)
            error_img_instance = self._create_error_image(error_msg)
            return (error_img_instance, "", "0", error_msg)

        self._log(f"Processing {len(input_images)} input images for editing")

        # Run async processing using thread pool to avoid event loop conflicts
        def run_sync_edit():
            """Run async edit in a new thread with its own event loop"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._edit_images_async(
                    api_key, prompt, input_images, image_size, num_images, seed, 
                    sync_mode, max_retries, retry_indefinitely, use_custom_resolution, 
                    auto_scale_to_minimum, custom_width, custom_height
                ))
            finally:
                loop.close()
        
        result = None
        try:
            # Use thread pool executor to run async code in separate thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_sync_edit)
                result = future.result(timeout=300)  # 5 minute timeout
        except concurrent.futures.TimeoutError:
            self._log("Processing timed out after 5 minutes")
            error_img = self._create_error_image("Processing timeout")
            return (error_img, "", "0", "Processing timed out after 5 minutes")
        except Exception as e:
            self._log(f"Error in processing: {str(e)}")
            error_img = self._create_error_image(f"Processing error: {str(e)}")
            return (error_img, "", "0", f"Processing error: {str(e)}")
        
        if result is None:
            error_img = self._create_error_image("Processing failed to produce results")
            return (error_img, "", "0", "Processing failed to produce results")
        
        # Extract results
        images, image_urls, returned_seed, status_msg = result
        
        # Combine log messages with status
        final_log_output = "Processing Logs:\n" + "\n".join(self.log_messages) + "\n\n" + status_msg

        return (images, image_urls, returned_seed, final_log_output)