# FL_Fal_Gemini_ImageEdit: Fal AI Gemini 2.5 Flash Image Edit API Node
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


class FL_Fal_Gemini_ImageEdit:
    """
    A ComfyUI node for the Fal AI Gemini 2.5 Flash Image Edit API.
    Takes multiple images and a prompt to edit them using Gemini's multimodal capabilities.
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "image_urls", "description", "status_msg")
    FUNCTION = "edit_images"
    CATEGORY = "🏵️Fill Nodes/AI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False,
                                     "description": "Fal AI API key"}),
                "prompt": ("STRING", {"default": "make a photo of the man driving the car down the california coastline",
                                    "multiline": True, "forceInput": True,
                                    "description": "The prompt for image editing"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 666666,
                               "description": "Random seed for image generation (0 = random)"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1,
                                     "description": "Number of images to generate"}),
                "output_format": (["jpeg", "png"], {"default": "jpeg",
                                                   "description": "Output image format"}),
                "sync_mode": ("BOOLEAN", {"default": False,
                                        "description": "When true, images returned as data URIs instead of URLs"}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
            },
            "optional": {
                "image_1": ("IMAGE", {"description": "First input image to edit"}),
                "image_2": ("IMAGE", {"description": "Second input image to edit"}),
                "image_3": ("IMAGE", {"description": "Third input image to edit"}),
                "image_4": ("IMAGE", {"description": "Fourth input image to edit"}),
                "image_5": ("IMAGE", {"description": "Fifth input image to edit"}),
                "retry_indefinitely": ("BOOLEAN", {"default": False}),
            }
        }

    def __init__(self):
        self.log_messages = []

    def _log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[FL_Fal_Gemini_ImageEdit] {timestamp}: {message}"
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
        """Convert PIL image to base64 data URI"""
        try:
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            img_data_uri = f"data:image/png;base64,{img_base64}"
            return img_data_uri
        except Exception as e:
            self._log(f"Error converting image to base64: {str(e)}")
            raise

    async def _edit_images_async(self, api_key, prompt, input_images, seed, num_images, output_format, sync_mode, max_retries, retry_indefinitely):
        try:
            # Calculate seed
            actual_seed = seed if seed != 0 else random.randint(1, 666666)
            
            self._log(f"Starting image editing with seed {actual_seed} and prompt: '{prompt[:50]}...'")
            
            # Prepare image URLs from input images
            image_urls = []
            if input_images:
                for i, pil_image in enumerate(input_images):
                    try:
                        img_url = self._convert_image_to_url(pil_image)
                        image_urls.append(img_url)
                        self._log(f"Successfully converted image {i+1} to data URI")
                    except Exception as e:
                        self._log(f"Error converting image {i+1} to data URI: {str(e)}")
                        error_msg = f"Error: Failed to convert image {i+1}: {str(e)}"
                        return self._create_error_image(error_msg), "", "", error_msg
            else:
                error_msg = "Error: No images provided for editing"
                return self._create_error_image(error_msg), "", "", error_msg
            
            # Clear any existing FAL_KEY environment variable to prevent caching issues
            if "FAL_KEY" in os.environ:
                del os.environ["FAL_KEY"]
            
            # Prepare the API request
            clean_api_key = api_key.strip()
            
            # Prepare the arguments for fal_client
            arguments = {
                "prompt": prompt,
                "image_urls": image_urls,
                "seed": actual_seed,
                "num_images": num_images,
                "output_format": output_format,
                "sync_mode": sync_mode
            }
            
            # Set the API key as an environment variable for fal_client
            os.environ["FAL_KEY"] = clean_api_key
            
            self._log(f"Calling Fal AI Gemini Image Edit API with {len(image_urls)} images...")
            
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
                        "fal-ai/gemini-25-flash-image/edit",
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
                return self._create_error_image(error_msg), "", "", error_msg
            
            self._log("API call completed successfully")
            
            # Extract image URLs and description from the result
            output_image_urls = []
            description = ""
            
            if "images" in result and len(result["images"]) > 0:
                for img_info in result["images"]:
                    if "url" in img_info:
                        output_image_urls.append(img_info["url"])
                self._log(f"Found {len(output_image_urls)} edited images in response")
            else:
                self._log("Warning: No images found in result")
                error_msg = "Error: No images in API response"
                return self._create_error_image(error_msg), "", "", error_msg
            
            # Extract description
            if "description" in result:
                description = result["description"]
                self._log(f"Received description: {description[:100]}...")
            else:
                description = "No description provided"
            
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
                
                return combined_tensor, combined_urls, description, f"Success: {len(processed_images)} images edited successfully"
                
            except Exception as e:
                error_msg = f"Download Error: {str(e)}"
                self._log(error_msg)
                return self._create_error_image(error_msg), "", "", error_msg
            
        except Exception as e:
            self._log(f"Error in async image editing: {str(e)}")
            error_msg = f"Error: {str(e)}"
            return self._create_error_image(error_msg), "", "", error_msg

    def edit_images(self, api_key, prompt, seed=0, num_images=1, output_format="jpeg", sync_mode=False, max_retries=3,
                   image_1=None, image_2=None, image_3=None, image_4=None, image_5=None, retry_indefinitely=False, **kwargs):
        self.log_messages = []
        if not api_key:
            error_msg = "API key not provided."
            self._log(error_msg)
            error_img_instance = self._create_error_image(error_msg)
            return (error_img_instance, "", "", error_msg)

        # Collect all input images
        input_images = []
        input_tensors = [image_1, image_2, image_3, image_4, image_5]
        
        for i, tensor in enumerate(input_tensors):
            if tensor is not None:
                pil_images = self._process_tensor_to_pil_list(tensor, f"Image{i+1}")
                if pil_images:
                    input_images.extend(pil_images)
        
        if not input_images:
            error_msg = "No valid input images provided."
            self._log(error_msg)
            error_img_instance = self._create_error_image(error_msg)
            return (error_img_instance, "", "", error_msg)

        self._log(f"Processing {len(input_images)} input images for editing")

        # Run async processing using thread pool to avoid event loop conflicts
        def run_sync_edit():
            """Run async edit in a new thread with its own event loop"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._edit_images_async(
                    api_key, prompt, input_images, seed, num_images, output_format,
                    sync_mode, max_retries, retry_indefinitely
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
            return (error_img, "", "", "Processing timed out after 5 minutes")
        except Exception as e:
            self._log(f"Error in processing: {str(e)}")
            error_img = self._create_error_image(f"Processing error: {str(e)}")
            return (error_img, "", "", f"Processing error: {str(e)}")
        
        if result is None:
            error_img = self._create_error_image("Processing failed to produce results")
            return (error_img, "", "", "Processing failed to produce results")
        
        # Extract results
        images, image_urls, description, status_msg = result
        
        # Combine log messages with status
        final_log_output = "Processing Logs:\n" + "\n".join(self.log_messages) + "\n\n" + status_msg

        return (images, image_urls, description, final_log_output)