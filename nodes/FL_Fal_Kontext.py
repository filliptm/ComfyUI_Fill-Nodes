# FL_Fal_Kontext: Fal AI Flux Pro Kontext API Node with async support
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


class FL_Fal_Kontext:
    """
    A ComfyUI node for the Fal AI Flux Pro Kontext API.
    Takes multiple image/prompt pairs and generates new images using Fal AI's flux-pro/kontext endpoint.
    Supports async processing for multiple inputs.
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "image_urls", "status_msg")
    FUNCTION = "generate_images_advanced"
    CATEGORY = "🏵️Fill Nodes/AI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "api_key": ("STRING", {"multiline": False,
                                      "description": "Fal AI API key"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647,
                                "description": "Random seed for image generation (0 = random)"}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
                "prompt_1": ("STRING", {"default": "Put a donut next to the flour.",
                                      "multiline": True, "forceInput": True,
                                      "description": "Text prompt describing what to add or modify in the image"}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1,
                                           "description": "CFG scale - how closely to follow the prompt"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1,
                                     "description": "Number of images to generate per input"}),
                "aspect_ratio": (["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"],
                               {"default": "1:1", "description": "Aspect ratio of generated images"}),
                "output_format": (["jpeg", "png"], {"default": "jpeg",
                                                   "description": "Output image format"}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2",
                                                                     "description": "Safety tolerance (1=strict, 6=permissive)"}),
            },
            "optional": {
                "image_1": ("IMAGE", {"description": "Input image to modify"}),
                "retry_indefinitely": ("BOOLEAN", {"default": False}),
                "sync_mode": ("BOOLEAN", {"default": False,
                                        "description": "Wait for image generation before returning (higher latency)"}),
            }
        }

    def __init__(self):
        self.log_messages = []

    def _log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[FL_Fal_Kontext] {timestamp}: {message}"
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

    async def _generate_single_image_async(self, api_key, prompt_text, input_pil_images: Optional[List[Image.Image]], seed_val, max_retries, retry_indefinitely, guidance_scale, num_images, aspect_ratio, output_format, safety_tolerance, sync_mode, call_id):
        try:
            # Calculate seed
            actual_seed = seed_val if seed_val != 0 else random.randint(1, 2147483647)
            
            self._log(f"[Call {call_id}] Generating image with seed {actual_seed} for prompt: '{prompt_text[:50]}...'")
            
            # Convert image to base64
            img_data_uri = None
            if input_pil_images and len(input_pil_images) > 0:
                pil_image = input_pil_images[0]  # Take first image
                try:
                    # Convert PIL image to base64
                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    img_data_uri = f"data:image/png;base64,{img_base64}"
                    self._log(f"[Call {call_id}] Successfully converted image to base64")
                except Exception as e:
                    self._log(f"[Call {call_id}] Error converting image to base64: {str(e)}")
                    error_msg = f"Call {call_id} Error: Failed to convert image: {str(e)}"
                    return self._create_error_image(error_msg), "", error_msg, call_id
            else:
                error_msg = f"Call {call_id} Error: No image provided"
                return self._create_error_image(error_msg), "", error_msg, call_id
            
            # Clear any existing FAL_KEY environment variable to prevent caching issues
            if "FAL_KEY" in os.environ:
                del os.environ["FAL_KEY"]
            
            # Prepare the API request
            clean_api_key = api_key.strip()
            
            # Prepare the arguments for fal_client
            arguments = {
                "prompt": prompt_text,
                "image_url": img_data_uri,
                "seed": actual_seed,
                "guidance_scale": guidance_scale,
                "num_images": num_images,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "safety_tolerance": safety_tolerance,
                "sync_mode": sync_mode
            }
            
            # Set the API key as an environment variable for fal_client
            os.environ["FAL_KEY"] = clean_api_key
            
            self._log(f"[Call {call_id}] Calling Fal AI API with fal_client...")
            
            # Define a callback for queue updates
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        self._log(f"[Call {call_id}] API Log: {log['message']}")
            
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
                        "fal-ai/flux-pro/kontext",
                        arguments=arguments,
                        with_logs=True,
                        on_queue_update=on_queue_update,
                    )
                    return result
                except Exception as e:
                    self._log(f"[Call {call_id}] API call error: {str(e)}")
                    return None
            
            result = await loop.run_in_executor(None, make_fal_call)
            
            if result is None:
                error_msg = f"Call {call_id} Error: API call failed"
                return self._create_error_image(error_msg), "", error_msg, call_id
            
            self._log(f"[Call {call_id}] API call completed successfully")
            
            # Extract image URLs from the result
            image_urls = []
            if "images" in result and len(result["images"]) > 0:
                for img_info in result["images"]:
                    if "url" in img_info:
                        image_urls.append(img_info["url"])
                self._log(f"[Call {call_id}] Found {len(image_urls)} images in response")
            elif "image" in result and "url" in result["image"]:
                image_urls.append(result["image"]["url"])
                self._log(f"[Call {call_id}] Found 1 image in response (legacy format)")
            else:
                self._log(f"[Call {call_id}] Warning: No image URLs found in result")
                error_msg = f"Call {call_id} Error: No image URLs in API response"
                return self._create_error_image(error_msg), "", error_msg, call_id
            
            # Download and process all generated images
            try:
                self._log(f"[Call {call_id}] Downloading {len(image_urls)} generated images...")
                
                processed_images = []
                url_list = []
                
                for i, image_url in enumerate(image_urls):
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
                    
                    self._log(f"[Call {call_id}] Processed image {i+1}/{len(image_urls)} with shape {image_tensor.shape}")
                
                # Concatenate all images from this API call
                if len(processed_images) > 1:
                    # Handle multiple images - need to ensure same dimensions
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
                
                self._log(f"[Call {call_id}] Successfully processed {len(processed_images)} images with final shape {combined_tensor.shape}")
                
                return combined_tensor, combined_urls, f"Call {call_id} Success: {len(processed_images)} images generated successfully", call_id
                
            except Exception as e:
                error_msg = f"Call {call_id} Download Error: {str(e)}"
                self._log(error_msg)
                return self._create_error_image(error_msg), "", error_msg, call_id
            
        except Exception as e:
            self._log(f"[Call {call_id}] Error in async generation: {str(e)}")
            error_msg = f"Call {call_id} Error: {str(e)}"
            return self._create_error_image(error_msg), "", error_msg, call_id

    def generate_images_advanced(self, inputcount, api_key, seed=0, max_retries=3, prompt_1="Put a donut next to the flour.", guidance_scale=3.5, num_images=1, aspect_ratio="1:1", output_format="jpeg", safety_tolerance="2", image_1=None, retry_indefinitely=False, sync_mode=False, **kwargs):
        self.log_messages = []
        if not api_key:
            error_msg = "API key not provided."
            self._log(error_msg)
            error_img_instance = self._create_error_image(error_msg)
            return ([error_img_instance] * inputcount, "", error_msg)

        pbar = ProgressBar(inputcount)  # Initialize progress bar

        # Setup async tasks for each input
        async def run_batch():
            tasks = []
            
            for slot_idx in range(1, inputcount + 1):
                current_prompt = prompt_1 if slot_idx == 1 else kwargs.get(f"prompt_{slot_idx}", f"Default prompt for image {slot_idx}")
                
                current_image_tensor_for_slot = None
                if slot_idx == 1:
                    current_image_tensor_for_slot = image_1
                else:
                    current_image_tensor_for_slot = kwargs.get(f"image_{slot_idx}")
                
                pil_images_for_this_slot = self._process_tensor_to_pil_list(current_image_tensor_for_slot, f"InputSlot{slot_idx}")
                
                current_task_seed = seed + (slot_idx - 1) if seed != 0 else 0
                task_call_id = str(slot_idx)

                tasks.append(self._generate_single_image_async(
                    api_key, current_prompt, pil_images_for_this_slot,
                    current_task_seed, max_retries, retry_indefinitely,
                    guidance_scale, num_images, aspect_ratio, output_format,
                    safety_tolerance, sync_mode, task_call_id
                ))
                pbar.update_absolute(slot_idx)  # Update progress bar after task is added

            if not tasks:
                self._log("No tasks were created. This might indicate an issue with inputcount or logic.")
                return []

            # Run all tasks concurrently
            return await asyncio.gather(*tasks)

        # Run the async batch processing using thread pool to avoid event loop conflicts
        def run_sync_batch():
            """Run async batch in a new thread with its own event loop"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(run_batch())
            finally:
                loop.close()
        
        results_with_id = None  # Initialize results
        try:
            # Use thread pool executor to run async code in separate thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_sync_batch)
                results_with_id = future.result(timeout=300)  # 5 minute timeout
        except concurrent.futures.TimeoutError:
            self._log("Async processing timed out after 5 minutes")
            error_imgs = [self._create_error_image("Processing timeout")] * inputcount
            return (error_imgs, "", "Processing timed out after 5 minutes")
        except Exception as e:
            self._log(f"Error in async processing: {str(e)}")
            # Create batch of error images
            error_imgs = [self._create_error_image(f"Async processing error: {str(e)}")] * inputcount
            return (error_imgs, "", f"Async processing error: {str(e)}")
        
        # Process results (ensure results is not None if an error occurred before assignment)
        if results_with_id is None:
            self._log("Async processing did not yield results, possibly due to an earlier error before gather.")
            error_imgs = [self._create_error_image("Async processing failed to produce results")] * inputcount
            return (error_imgs, "", "Async processing failed to produce results")
        
        results_with_id.sort(key=lambda x: int(x[3]))  # Sort by call_id
        
        output_images = []
        output_urls = []
        output_texts = []

        for img_tensor, image_url, response_text, call_id_res in results_with_id:
            output_images.append(img_tensor)
            output_urls.append(f"Input {call_id_res}: {image_url}")
            output_texts.append(f"Response for Input {call_id_res}: {response_text}")

        # Handle variable image sizes by finding the maximum dimensions
        if output_images:
            # Find max dimensions across all images
            max_height = max(img.shape[1] for img in output_images)
            max_width = max(img.shape[2] for img in output_images)
            
            self._log(f"Max dimensions found: {max_height}x{max_width}")
            
            # Resize all images to max dimensions with padding
            resized_images = []
            for i, img_tensor in enumerate(output_images):
                current_h, current_w = img_tensor.shape[1], img_tensor.shape[2]
                
                if current_h == max_height and current_w == max_width:
                    # Image is already the right size
                    resized_images.append(img_tensor)
                    self._log(f"Image {i+1} already correct size: {current_h}x{current_w}")
                else:
                    # Need to pad the image
                    self._log(f"Resizing image {i+1} from {current_h}x{current_w} to {max_height}x{max_width}")
                    
                    # Calculate padding
                    pad_h = max_height - current_h
                    pad_w = max_width - current_w
                    
                    # Pad with zeros (black) - format: (pad_left, pad_right, pad_top, pad_bottom)
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                    pad_top = pad_h // 2
                    pad_bottom = pad_h - pad_top
                    
                    # PyTorch pad format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
                    # For images (B, H, W, C), we only pad H and W dimensions
                    padded_img = torch.nn.functional.pad(
                        img_tensor.permute(0, 3, 1, 2),  # Convert to (B, C, H, W)
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode='constant',
                        value=0
                    ).permute(0, 2, 3, 1)  # Convert back to (B, H, W, C)
                    
                    resized_images.append(padded_img)
            
            # Now concatenate all resized images
            batched_images = torch.cat(resized_images, dim=0)
            self._log(f"Successfully batched {len(resized_images)} images with final shape: {batched_images.shape}")
        else:
            batched_images = self._create_error_image("No images generated")
            
        combined_urls = " | ".join(output_urls)
        combined_responses = "\n\n".join(output_texts)
        
        final_log_output = "Processing Logs:\n" + "\n".join(self.log_messages) + "\n\n" + combined_responses

        return (batched_images, combined_urls, final_log_output)