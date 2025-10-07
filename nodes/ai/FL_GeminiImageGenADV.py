import os
import base64
import io
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import tempfile
from io import BytesIO
from google import genai
from google.genai import types
import time
import traceback
import asyncio
import concurrent.futures
import random
from typing import List, Tuple, Optional

from comfy.utils import ProgressBar

# Assuming ImageBatch is still needed if we are batching results, or can be removed if Gemini returns a batch
# from nodes import ImageBatch

class FL_GeminiImageGenADV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "api_key": ("STRING", {"default": os.getenv("GEMINI_API_KEY", ""), "multiline": False}),
                "model": (["models/gemini-2.0-flash-exp", "models/gemini-2.0-flash-preview-image-generation", "models/gemini-2.5-flash-image-preview"], {"default": "models/gemini-2.5-flash-image-preview"}),
                "always_square": ("BOOLEAN", {"default": False, "description": "When enabled, pads images to square dimensions. When disabled, outputs original resolution as image list."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
                "prompt_1": ("STRING", {"multiline": False, "default": "Describe image 1", "forceInput": True}),
            },
            "optional": {
                "image_1": ("IMAGE", {}), # Moved image_1 to optional. Default will be None if not connected.
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffff}), # Restored full seed range
                "retry_indefinitely": ("BOOLEAN", {"default": False}),
                # Subsequent image_i and prompt_i will be handled by **kwargs based on inputcount
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "API_responses")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "generate_images_advanced"
    CATEGORY = "🏵️Fill Nodes/AI"
    DESCRIPTION = """
Generates images using Gemini based on multiple image/prompt pairs.
Each pair triggers an asynchronous API call. Results are batched.
"""

    def __init__(self):
        self.log_messages = []
        self.min_size = 1024 # Minimum size from Editor
        try:
            import importlib.metadata
            genai_version = importlib.metadata.version('google-genai')
            self._log(f"Current google-genai version: {genai_version}")
            from packaging import version # Ensure packaging is imported
            if version.parse(genai_version) < version.parse('0.8.0'): # Example version, check Gemini docs
                self._log("Warning: google-genai version is too low, recommend upgrading to the latest version.")
                self._log("Suggested: pip install -q -U google-genai")
        except Exception as e:
            self._log(f"Unable to check google-genai version: {e}")

    def _log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[FL_GeminiImageGenADV] {timestamp}: {message}"
        print(formatted_message)
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message

    def _pad_image_to_minimum_size(self, pil_image):
        width, height = pil_image.size
        if width >= self.min_size and height >= self.min_size:
            return pil_image
        new_width = max(width, self.min_size)
        new_height = max(height, self.min_size)
        new_image = Image.new('RGB', (new_width, new_height), color=(255, 255, 255)) # White padding from Editor
        paste_x = (new_width - width) // 2
        paste_y = (new_height - height) // 2
        new_image.paste(pil_image, (paste_x, paste_y))
        self._log(f"Padded image from {width}x{height} to {new_width}x{new_height} with white borders")
        return new_image

    def _create_error_image(self, error_message="API Error", width=1024, height=1024): # Default size from Editor
        image = Image.new('RGB', (width, height), color=(0, 0, 0)) # Black error image from Editor
        draw = ImageDraw.Draw(image)
        font = None
        try:
            # Try to find a font that exists on most systems
            font_options = ['arial.ttf', 'DejaVuSans.ttf', 'FreeSans.ttf', 'NotoSans-Regular.ttf']
            for font_name in font_options:
                try:
                    font = ImageFont.truetype(font_name, 24) # Font size from Editor
                    break
                except IOError:
                    continue
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Calculate text position (centered)
        try: # Newer PIL versions
            text_bbox = draw.textbbox((0,0), error_message, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError: # Older PIL versions
            text_width, text_height = draw.textsize(error_message, font=font)

        text_x = (width - text_width) / 2
        text_y = (height - text_height) / 2
        draw.text((text_x, text_y), error_message, fill=(255, 0, 0), font=font) # Red text from Editor
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
        if tensor_image.ndim == 4: # Batch of images (B, H, W, C)
            if tensor_image.shape[0] == 0:
                self._log(f"{image_name_prefix} batch is empty (shape: {tensor_image.shape}).")
                return None
            for i in range(tensor_image.shape[0]):
                img_np = tensor_image[i].cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                pil_image = Image.fromarray(img_np)
                self._log(f"Converted {image_name_prefix} batch item {i} (original shape: {tensor_image.shape}) to PIL Image (size: {pil_image.size}).")
                pil_images.append(pil_image)
        elif tensor_image.ndim == 3: # Single image (H, W, C)
            img_np = tensor_image.cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            self._log(f"Converted single {image_name_prefix} (original shape: {tensor_image.shape}) to PIL Image (size: {pil_image.size}).")
            pil_images.append(pil_image)
        else:
            self._log(f"Cannot convert {image_name_prefix} with ndim {tensor_image.ndim} (shape: {tensor_image.shape}) to PIL Image(s).")
            return None
        
        return pil_images if pil_images else None

    def _call_gemini_api(self, client_instance, model_name_full, contents, gen_config_obj, retry_indefinitely, retry_count=0, max_retries=3, call_id="0"):
        try:
            self._log(f"[Call {call_id}] API call attempt #{retry_count + 1} to {model_name_full}{' (retrying indefinitely)' if retry_indefinitely else ''}")
            
            # Using client.models.generate_content like in FL_GeminiImageEditor
            response = client_instance.models.generate_content(
                model=model_name_full, # FL_GeminiImageEditor passes the full model string here
                contents=contents,
                config=gen_config_obj # FL_GeminiImageEditor uses 'config' for GenerateContentConfig
            )

            # Validate response structure (adapted from FL_GeminiImageEditor)
            if not hasattr(response, 'candidates') or not response.candidates:
                self._log(f"[Call {call_id}] Empty response: No candidates found")
                if retry_indefinitely or retry_count < max_retries - 1:
                    self._log(f"[Call {call_id}] Retrying in 2 seconds... (Attempt {retry_count + 2 if not retry_indefinitely else 'N/A'}/{max_retries if not retry_indefinitely else 'inf'})")
                    time.sleep(2)
                    return self._call_gemini_api(client_instance, model_name_full, contents, gen_config_obj, retry_indefinitely, retry_count + 1, max_retries, call_id)
                else:
                    self._log(f"[Call {call_id}] Maximum retries ({max_retries}) reached. Returning empty response.")
                    return None

            if not hasattr(response.candidates[0], 'content') or response.candidates[0].content is None:
                self._log(f"[Call {call_id}] Invalid response: candidates[0].content is missing")
                if retry_indefinitely or retry_count < max_retries - 1:
                    self._log(f"[Call {call_id}] Retrying in 2 seconds... (Attempt {retry_count + 2 if not retry_indefinitely else 'N/A'}/{max_retries if not retry_indefinitely else 'inf'})")
                    time.sleep(2)
                    return self._call_gemini_api(client_instance, model_name_full, contents, gen_config_obj, retry_indefinitely, retry_count + 1, max_retries, call_id)
                else:
                    self._log(f"[Call {call_id}] Maximum retries ({max_retries}) reached. Returning empty response.")
                    return None

            if not hasattr(response.candidates[0].content, 'parts') or response.candidates[0].content.parts is None:
                self._log(f"[Call {call_id}] Invalid response: candidates[0].content.parts is missing")
                if retry_indefinitely or retry_count < max_retries - 1:
                    self._log(f"[Call {call_id}] Retrying in 2 seconds... (Attempt {retry_count + 2 if not retry_indefinitely else 'N/A'}/{max_retries if not retry_indefinitely else 'inf'})")
                    time.sleep(2)
                    return self._call_gemini_api(client_instance, model_name_full, contents, gen_config_obj, retry_indefinitely, retry_count + 1, max_retries, call_id)
                else:
                    self._log(f"[Call {call_id}] Maximum retries ({max_retries}) reached. Returning empty response.")
                    return None
            
            self._log(f"[Call {call_id}] Valid API response received.")
            return response
            
        except Exception as e:
            self._log(f"[Call {call_id}] API call error: {str(e)}")
            if retry_indefinitely or retry_count < max_retries - 1:
                wait_time = 2 * (retry_count + 1) # Progressive backoff
                self._log(f"[Call {call_id}] Retrying in {wait_time}s... (Attempt {retry_count + 2 if not retry_indefinitely else 'N/A'}/{max_retries if not retry_indefinitely else 'inf'})")
                time.sleep(wait_time)
                return self._call_gemini_api(client_instance, model_name_full, contents, gen_config_obj, retry_indefinitely, retry_count + 1, max_retries, call_id)
            else:
                self._log(f"[Call {call_id}] Max retries ({max_retries}) reached. Giving up.")
                return None

    def _process_api_response(self, response, call_id="0", always_square=False):
        if response is None: # Simplified check from Editor
            self._log(f"[Call {call_id}] No valid response to process.")
            error_msg = "API Error: No content in response"
            return self._create_error_image(error_msg), error_msg

        response_text_parts = [] # Changed from response_text to response_text_parts to match ADV logic initially
        image_tensor = None

        if not hasattr(response, 'candidates') or not response.candidates: # Check from Editor
            self._log(f"[Call {call_id}] No candidates in API response")
            error_msg = "API returned an empty response"
            return self._create_error_image(error_msg), error_msg

        # Iterate through response parts (similar to Editor, but adapted for ADV's single image focus per call)
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text is not None:
                text_content = part.text
                response_text_parts.append(text_content)
                self._log(
                    f"[Call {call_id}] API returned text: {text_content[:100]}..." if len(
                        text_content) > 100 else text_content)

            elif hasattr(part, 'inline_data') and part.inline_data is not None:
                self._log(f"[Call {call_id}] API returned image data")
                try:
                    image_data = part.inline_data.data
                    
                    if not image_data or len(image_data) < 100: # Check from Editor
                        self._log(f"[Call {call_id}] Warning: Image data is empty or too small")
                        continue

                    pil_image = None
                    try:
                        pil_image = Image.open(BytesIO(image_data))
                        self._log(
                            f"[Call {call_id}] Direct PIL open successful, size: {pil_image.width}x{pil_image.height}")
                    except Exception as e1:
                        self._log(f"[Call {call_id}] Direct PIL open failed: {str(e1)}")
                        # Optional: Add temp file saving from Editor if direct open fails often
                        try:
                            temp_dir = tempfile.gettempdir()
                            # Ensure temp_dir is writable, or fall back
                            if not os.access(temp_dir, os.W_OK):
                                temp_dir = "." # Current directory as fallback
                                self._log(f"[Call {call_id}] Temp directory {tempfile.gettempdir()} not writable, using current directory.")

                            temp_file_path = os.path.join(temp_dir, f"gemini_image_adv_{call_id}_{int(time.time())}.png")
                            with open(temp_file_path, "wb") as f:
                                f.write(image_data)
                            pil_image = Image.open(temp_file_path)
                            self._log(f"[Call {call_id}] Opening via temp file {temp_file_path} successful")
                            try:
                                os.remove(temp_file_path) # Clean up temp file
                            except Exception as e_remove:
                                self._log(f"[Call {call_id}] Could not remove temp file {temp_file_path}: {e_remove}")
                        except Exception as e2:
                            self._log(f"[Call {call_id}] Opening via temp file failed: {str(e2)}")


                    if pil_image is None:
                        self._log(f"[Call {call_id}] Cannot open image, skipping")
                        continue
                        
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                        self._log(f"[Call {call_id}] Image converted to RGB mode")
                    
                    # Store original dimensions for logging
                    width, height = pil_image.size
                    self._log(f"[Call {call_id}] Original image size: {width}x{height}")

                    # Apply padding if always_square is enabled and image needs it
                    if always_square and (width < self.min_size or height < self.min_size):
                        self._log(
                            f"[Call {call_id}] Image size {width}x{height} is smaller than minimum {self.min_size}x{self.min_size}, padding needed")
                        pil_image = self._pad_image_to_minimum_size(pil_image)
                    elif always_square:
                        self._log(f"[Call {call_id}] Always square enabled but image already meets minimum size")
                    else:
                        self._log(f"[Call {call_id}] Always square disabled, keeping original size: {width}x{height}")

                    img_array = np.array(pil_image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(img_array).unsqueeze(0) # Batch dimension
                    self._log(f"[Call {call_id}] Image processed from API response. Shape: {image_tensor.shape}")
                    break # Assuming one image per response for ADV node
                except Exception as e:
                    self._log(f"[Call {call_id}] Error processing image from API response: {e}")
                    traceback.print_exc()
        
        final_response_text = "\n".join(response_text_parts)
        if image_tensor is None:
            self._log(f"[Call {call_id}] No image found in API response parts.")
            error_msg = "API Error: No image data in response" # More specific than Editor's default
            image_tensor = self._create_error_image(error_msg) # Use the updated _create_error_image
            if not final_response_text: final_response_text = error_msg # Keep this logic
        
        return image_tensor, final_response_text

    async def _generate_single_image_async(self, api_key, model_name_full, prompt_text, input_pil_images: Optional[List[Image.Image]], temperature, max_retries, retry_indefinitely, seed_val, call_id, always_square=False):
        try:
            try:
                client_instance = genai.Client(api_key=api_key)
            except TypeError:
                genai.configure(api_key=api_key)
                client_instance = genai.Client()
            except AttributeError:
                 self._log(f"[Call {call_id}] CRITICAL: genai.Client not found. Please check google-genai SDK installation and version.")
                 error_msg = f"Call {call_id} Error: genai.Client not found."
                 return self._create_error_image(error_msg), error_msg, call_id

            actual_seed = seed_val if seed_val != 0 else random.randint(1, 0xffffffffffffffff)
            self._log(f"[Call {call_id}] Using seed: {actual_seed} for prompt: '{prompt_text[:50]}...'")

            gen_config_params = {
                "temperature": temperature,
                "response_modalities": ['Text', 'Image']
            }
            if actual_seed != 0:
                 gen_config_params["seed"] = actual_seed
            
            gen_config_obj = types.GenerateContentConfig(**gen_config_params)
            
            if actual_seed != 0:
                current_seed_in_config = getattr(gen_config_obj, 'seed', None)
                if current_seed_in_config != actual_seed:
                    self._log(f"[Call {call_id}] Warning: Seed {actual_seed} was specified. GenerateContentConfig has seed: {current_seed_in_config}. Ensure model supports seed via this config.")

            contents = [prompt_text] # Start with the prompt
            if input_pil_images: # If the list of PIL images is provided and not empty
                for pil_img in input_pil_images:
                    if pil_img: # Ensure the image itself is not None
                        contents.append(pil_img)
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._call_gemini_api(client_instance, model_name_full, contents, gen_config_obj, retry_indefinitely, 0, max_retries, call_id)
            )
            
            img_tensor, response_text = self._process_api_response(response, call_id, always_square)
            return img_tensor, response_text, call_id # Return call_id to map results

        except Exception as e:
            self._log(f"[Call {call_id}] Error in async generation: {str(e)}")
            error_msg = f"Call {call_id} Error: {str(e)}"
            return self._create_error_image(error_msg), error_msg, call_id

    def generate_images_advanced(self, inputcount, api_key, model, always_square, temperature, max_retries, prompt_1, image_1=None, seed=0, retry_indefinitely=False, **kwargs):
        self.log_messages = []
        if not api_key:
            error_msg = "API key not provided."
            self._log(error_msg)
            error_img_instance = self._create_error_image(error_msg)
            # API key error should return 'inputcount' error images if we can determine it,
            # otherwise, a single error image is a reasonable fallback.
            # Since each slot is one API call now, inputcount is the number of expected results.
            return ([error_img_instance] * inputcount, error_msg)

        pbar = ProgressBar(inputcount) # Initialize progress bar

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
                    api_key, model, current_prompt, pil_images_for_this_slot,
                    temperature, max_retries, retry_indefinitely, current_task_seed, task_call_id, always_square
                ))
                pbar.update_absolute(slot_idx) # Update progress bar after task is added

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
            return (error_imgs, "Processing timed out after 5 minutes")
        except Exception as e:
            self._log(f"Error in async processing: {str(e)}")
            # Create batch of error images
            error_imgs = [self._create_error_image(f"Async processing error: {str(e)}")] * inputcount
            return (error_imgs, f"Async processing error: {str(e)}")
        
        # Process results (ensure results is not None if an error occurred before assignment)
        if results_with_id is None:
            self._log("Async processing did not yield results, possibly due to an earlier error before gather.")
            error_imgs = [self._create_error_image("Async processing failed to produce results")] * inputcount
            return (error_imgs, "Async processing failed to produce results")
        
        results_with_id.sort(key=lambda x: int(x[2]))
        
        output_images = []
        output_texts = []

        for img_tensor, response_text, call_id_res in results_with_id:
            output_images.append(img_tensor)
            output_texts.append(f"Response for Input {call_id_res}:\n{response_text}")

        # Return as list of individual tensors instead of batched tensor to match FL_GeminiImageEditor behavior
        combined_responses = "\n\n".join(output_texts)
        
        final_log_output = "Processing Logs:\n" + "\n".join(self.log_messages) + "\n\n" + combined_responses

        return (output_images if output_images else [self._create_error_image("No images generated")], final_log_output)