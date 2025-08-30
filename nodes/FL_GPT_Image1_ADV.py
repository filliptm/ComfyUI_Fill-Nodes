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
import time
import traceback
import asyncio
import concurrent.futures
import random
from typing import List, Tuple, Optional

class FL_GPT_Image1_ADV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}), # Max 10 concurrent calls
                "api_key": ("STRING", {"default": os.getenv("OPENAI_API_KEY", ""), "multiline": False}),
                # Global settings from FL_GPT_Image1.py, with _setting suffix
                "size_setting": (["1024x1024", "1536x1024", "1024x1536"], {"default": "1024x1024"}), # Adjusted for gpt-image-1 supported sizes
                "quality_setting": (["auto", "high", "medium", "low"], {"default": "auto"}), # Adjusted for gpt-image-1
                "background_setting": (["auto", "transparent", "opaque"], {"default": "auto"}),
                "output_format_setting": (["png", "jpeg", "webp"], {"default": "png"}),
                "prompt_1": ("STRING", {"multiline": True, "default": "Describe image 1", "forceInput": True}),
            },
            "optional": {
                "image_1": ("IMAGE", {}), # For edits/variations for prompt_1
                "seed_setting": ("INT", {"default": 0, "min": 0, "max": 2147483647}), # From FL_GPT_Image1
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "API_responses")
    FUNCTION = "generate_images_advanced"
    CATEGORY = "🏵️Fill Nodes/GPT"
    DESCRIPTION = """
Generates images using OpenAI's "gpt-image-1" model based on multiple prompts.
Each prompt (and optional image/mask for edits) triggers an asynchronous API call.
Uses global settings for size, quality, etc., for all generations/edits.
"""

    def __init__(self):
        self.log_messages = []

    def _log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[FL_GPT_Image1_ADV] {timestamp}: {message}"
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
            if font is None: font = ImageFont.load_default()
        except Exception: font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((0,0), error_message, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (width - text_width) / 2
        text_y = (height - text_height) / 2
        draw.text((text_x, text_y), error_message, fill=(255, 0, 0), font=font)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        self._log(f"Created error image: '{error_message}'")
        return img_tensor

    def _process_tensor_to_pil(self, tensor_image: Optional[torch.Tensor], image_name_prefix: str = "Image") -> Optional[Image.Image]:
        try:
            if tensor_image is None:
                self._log(f"{image_name_prefix} input is None, skipping PIL conversion.")
                return None
            if not isinstance(tensor_image, torch.Tensor):
                self._log(f"{image_name_prefix} is not a tensor (type: {type(tensor_image)}), skipping.")
                return None

            # Ensure tensor is in correct format [B, H, W, C] or [H, W, C]
            if tensor_image.ndim == 4 and tensor_image.shape[0] == 1: # Batch of 1 image
                img_np = tensor_image[0].cpu().numpy()
            elif tensor_image.ndim == 3: # Single image
                img_np = tensor_image.cpu().numpy()
            elif tensor_image.ndim == 4 and tensor_image.shape[0] > 1: # Batch of multiple images
                 self._log(f"{image_name_prefix} is a batch of {tensor_image.shape[0]} images. Using the first one for this slot.")
                 img_np = tensor_image[0].cpu().numpy()
            elif tensor_image.ndim == 4 and tensor_image.shape[0] == 0 : # Empty batch
                self._log(f"{image_name_prefix} is an empty batch (shape: {tensor_image.shape}).")
                return None
            else: # Other unexpected shapes
                self._log(f"{image_name_prefix} format incorrect or unhandled: {tensor_image.shape}")
                return None
            
            image_np = (img_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            self._log(f"{image_name_prefix} processed successfully, size: {pil_image.width}x{pil_image.height}")
            return pil_image
        except Exception as e:
            self._log(f"Error processing {image_name_prefix} tensor to PIL: {str(e)}")
            return None

    # Adapted from FL_GPT_Image1.py
    def _call_openai_api(self, api_key, payload_for_slot, endpoint="generations", call_id="0", retry_count=0, max_retries=3):
        try:
            self._log(f"[Call {call_id}] OpenAI API call attempt #{retry_count + 1} to endpoint: {endpoint} with prompt: '{payload_for_slot.get('prompt', '')[:50]}...'")
            
            url = f"https://api.openai.com/v1/images/{endpoint}"
            
            headers = {"Authorization": f"Bearer {api_key}"}
            
            if endpoint == "edits":
                self._log(f"[Call {call_id}] Using multipart/form-data for edits endpoint")
                multipart_data = {}
                files_to_send = {}

                for key, value in payload_for_slot.items():
                    if key == "image" and value is not None: # value is PIL Image
                        img_byte_arr = BytesIO()
                        value.save(img_byte_arr, format='PNG') # Save PIL to bytes
                        files_to_send["image"] = ("image.png", img_byte_arr.getvalue(), "image/png")
                        self._log(f"[Call {call_id}] Added image file to multipart request")
                    # Mask logic removed from payload construction as it's no longer an input
                    # elif key == "mask" and value is not None: ...
                    elif key not in ["image", "mask"]: # Other params are form data (mask would be caught here if passed, but it won't be)
                        multipart_data[key] = (None, str(value))
                
                self._log(f"[Call {call_id}] Sending multipart request with data: {list(multipart_data.keys())}, files: {list(files_to_send.keys())}")
                response = requests.post(url, headers=headers, data=multipart_data, files=files_to_send, timeout=120)

            else: # generations endpoint
                headers["Content-Type"] = "application/json"
                self._log(f"[Call {call_id}] Sending JSON request with payload keys: {list(payload_for_slot.keys())}")
                response = requests.post(url, headers=headers, json=payload_for_slot, timeout=120)
            
            if response.status_code == 200:
                self._log(f"[Call {call_id}] OpenAI API success.")
                return response.json()
            else:
                error_msg = f"[Call {call_id}] OpenAI API error: {response.status_code} - {response.text}"
                self._log(error_msg)
                # ... (retry logic as before)
                if retry_count < max_retries - 1:
                    wait_time = 2 * (retry_count + 1)
                    self._log(f"[Call {call_id}] Retrying in {wait_time}s... (Attempt {retry_count + 2}/{max_retries})")
                    time.sleep(wait_time)
                    return self._call_openai_api(api_key, payload_for_slot, endpoint, call_id, retry_count + 1, max_retries)
                else:
                    self._log(f"[Call {call_id}] Max retries ({max_retries}) reached. Returning error.")
                    # Ensure the error format is consistent for _process_openai_response
                    try: error_detail = response.json().get("error", {"message": response.text})
                    except: error_detail = {"message": response.text}
                    return {"error": error_detail}

        except Exception as e:
            self._log(f"[Call {call_id}] OpenAI API call exception: {str(e)}")
            traceback.print_exc()
            # ... (retry logic as before)
            if retry_count < max_retries - 1:
                wait_time = 2 * (retry_count + 1)
                self._log(f"[Call {call_id}] Retrying in {wait_time}s... (Attempt {retry_count + 2}/{max_retries})")
                time.sleep(wait_time)
                return self._call_openai_api(api_key, payload_for_slot, endpoint, call_id, retry_count + 1, max_retries)
            else:
                self._log(f"[Call {call_id}] Max retries ({max_retries}) reached due to exception. Giving up.")
                return {"error": {"message": f"Max retries reached. Last exception: {str(e)}"}}

    # Adapted from FL_GPT_Image1.py's _process_api_response
    def _process_openai_response(self, response_json, call_id="0", target_size_str="1024x1024"):
        target_width, target_height = map(int, target_size_str.split('x'))

        if response_json is None or "error" in response_json:
            error_content = response_json.get("error") if response_json else {"message": "No response from API"}
            error_msg = "Unknown API error"
            if isinstance(error_content, dict):
                error_msg = error_content.get("message", "Unknown API error")
            elif isinstance(error_content, str):
                error_msg = error_content

            self._log(f"[Call {call_id}] API Error: {error_msg}")
            # Check for specific errors like in FL_GPT_Image1
            if "organization verification" in error_msg.lower():
                error_msg = "OpenAI organization verification required"
            return self._create_error_image(f"API Error: {error_msg[:60]}", target_width, target_height), json.dumps(response_json if response_json else {"error": error_msg})

        if "data" not in response_json or not response_json["data"]:
            self._log(f"[Call {call_id}] No data in API response.")
            return self._create_error_image("No image data in response", target_width, target_height), json.dumps(response_json)

        try:
            # ADV node makes n=1 calls, so response.data should have 1 item
            img_data_entry = response_json["data"][0]
            img_tensor = None
            pil_image = None

            if "b64_json" in img_data_entry:
                img_bytes = base64.b64decode(img_data_entry["b64_json"])
                pil_image = Image.open(BytesIO(img_bytes))
            elif "url" in img_data_entry:
                self._log(f"[Call {call_id}] Downloading image from URL: {img_data_entry['url']}")
                dl_response = requests.get(img_data_entry["url"], timeout=30)
                if dl_response.status_code == 200:
                    pil_image = Image.open(BytesIO(dl_response.content))
                else:
                    self._log(f"[Call {call_id}] Failed to download image: HTTP {dl_response.status_code}")
                    error_msg = f"Failed to download image (HTTP {dl_response.status_code})"
                    return self._create_error_image(error_msg, target_width, target_height), json.dumps(response_json)
            
            if pil_image is None:
                 self._log(f"[Call {call_id}] Could not load image from response.")
                 return self._create_error_image("Corrupt image data", target_width, target_height), json.dumps(response_json)

            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            self._log(f"[Call {call_id}] Image successfully decoded/downloaded. Original size: {pil_image.size}")
            
            # Resize if necessary (though API should provide correct size based on 'size' param)
            # if pil_image.size != (target_width, target_height):
            #     self._log(f"[Call {call_id}] Warning: API returned size {pil_image.size}, expected {target_width}x{target_height}. Using returned size.")
                # pil_image = pil_image.resize((target_width, target_height), Image.LANCZOS)

            img_array = np.array(pil_image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            self._log(f"[Call {call_id}] Image processed. Shape: {img_tensor.shape}")
            
            # Extract revised_prompt if available (DALL-E 3 feature, gpt-image-1 might provide it)
            revised_prompt = img_data_entry.get("revised_prompt", "N/A")
            response_text_details = f"Revised Prompt (if any): {revised_prompt}\n"
            # Add other relevant info from response_json if needed, e.g. usage, id
            response_text_details += f"Full API Data (first item): {json.dumps(img_data_entry)}"

            return img_tensor, response_text_details

        except Exception as e:
            self._log(f"[Call {call_id}] Error processing OpenAI API response content: {e}")
            traceback.print_exc()
            return self._create_error_image(f"Response processing error: {str(e)[:60]}", target_width, target_height), json.dumps(response_json)


    async def _generate_single_image_async(self, api_key, payload_for_slot, endpoint, call_id, max_api_retries, target_size_str):
        try:
            loop = asyncio.get_event_loop()
            response_json = await loop.run_in_executor(
                None,
                lambda: self._call_openai_api(api_key, payload_for_slot, endpoint, call_id, 0, max_api_retries)
            )
            
            img_tensor, response_text = self._process_openai_response(response_json, call_id, target_size_str)
            return img_tensor, response_text, call_id

        except Exception as e:
            self._log(f"[Call {call_id}] Error in async generation for OpenAI (gpt-image-1): {str(e)}")
            traceback.print_exc()
            err_w, err_h = 1024, 1024
            try: err_w, err_h = map(int, target_size_str.split('x'))
            except: pass
            error_msg = f"Call {call_id} Async Error: {str(e)}"
            return self._create_error_image(error_msg[:60], err_w, err_h), error_msg, call_id

    def generate_images_advanced(self, inputcount, api_key,
                                 size_setting, quality_setting, background_setting, output_format_setting,
                                 prompt_1, image_1=None, # mask_1 removed
                                 seed_setting=0, **kwargs):
        self.log_messages = []
        # Hardcoded values
        hardcoded_moderation = "low"
        # output_compression is not directly sent to OpenAI API for b64_json, but kept for consistency if logic changes
        # hardcoded_output_compression = 100

        if not api_key:
            error_msg = "API key not provided for OpenAI (gpt-image-1)."
            self._log(error_msg)
            err_w, err_h = 1024,1024
            try:
                err_w, err_h = map(int, size_setting.split('x'))
            except:
                pass # Keep default if size_setting is invalid
            error_img_instance = self._create_error_image(error_msg, err_w, err_h)
            return ([error_img_instance] * inputcount, error_msg)

        self._log(f"Note: 'gpt-image-1' model may require OpenAI organization verification. See OpenAI docs if errors occur.")

        max_api_retries = 3

        # Setup async tasks for each input
        async def run_batch():
            tasks = []
            
            for slot_idx in range(1, inputcount + 1):
                current_prompt = prompt_1 if slot_idx == 1 else kwargs.get(f"prompt_{slot_idx}", f"Default prompt for slot {slot_idx}")
                current_image_tensor = image_1 if slot_idx == 1 else kwargs.get(f"image_{slot_idx}")
                # current_mask_tensor removed
                
                task_call_id = str(slot_idx)
                
                pil_image_for_slot = self._process_tensor_to_pil(current_image_tensor, f"InputSlot{slot_idx}_Image")
                # pil_mask_for_slot removed

                endpoint = "generations"
                # Base payload for generations
                payload = {
                    "model": "gpt-image-1",
                    "prompt": current_prompt,
                    "n": 1,
                    "size": size_setting,
                    "response_format": "b64_json" # Crucial for getting image data to process into tensor
                }

                # Add common optional parameters
                if quality_setting != "auto": payload["quality"] = quality_setting
                if background_setting != "auto": payload["background"] = background_setting
                payload["moderation"] = hardcoded_moderation # Always use hardcoded "low"
                # output_format_setting from UI primarily informs how FL_GPT_Image1 saves/handles,
                # for ADV node returning tensors, b64_json is the key API request.
                # The original FL_GPT_Image1 sends 'output_format' if not png.

                # Seed: OpenAI API for DALL-E generations doesn't typically use a 'seed' parameter in the request.
                # We log it here for tracking and consistency with other ADV nodes.
                current_slot_seed = 0
                if seed_setting != 0:
                    # Ensure seed is within a reasonable range if OpenAI ever supports it,
                    # or just for logging consistency. The original FL_GPT_Image1 uses a 32-bit int range.
                    # For ADV, simple incrementing is fine for logging.
                    current_slot_seed = seed_setting + (slot_idx - 1)
                
                self._log(f"[Call {task_call_id}] Using effective seed for logging/tracking: {current_slot_seed if current_slot_seed != 0 else 'Random (seed_setting was 0)'}. (Note: OpenAI API for gpt-image-1 does not use this seed directly in the request for generation.)")

                if pil_image_for_slot:
                    endpoint = "edits" # Or "variations" if no prompt is desired for image-only input. "edits" usually implies a prompt.
                    # Modify payload for edits/variations
                    payload["image"] = pil_image_for_slot
                    # Mask logic removed
                    # if pil_mask_for_slot:
                    #     payload["mask"] = pil_mask_for_slot
                    
                    # Parameters not typically used or differently handled for edits:
                    # FL_GPT_Image1.py *does* send 'size' for edits. To align, we will NOT delete it.
                    # The API documentation says output matches input size for edits, so 'size' might be ignored or validated.
                    # if "size" in payload: del payload["size"] # Removing this line to match FL_GPT_Image1's behavior
                    if "response_format" in payload: del payload["response_format"] # Remove, will use API default or 'output_format'
                    # Based on FL_GPT_Image1.py, if output_format_setting is not 'png' or 'auto',
                    # it sends 'output_format: <value>' for edits.
                    if output_format_setting not in ["png", "auto"]:
                        payload["output_format"] = output_format_setting
                    # Note: 'quality', 'background', 'style', 'moderation' might behave differently or be ignored by 'gpt-image-1' for edits.
                    # The current error is only about 'response_format'. We keep others for now.

                tasks.append(self._generate_single_image_async(
                    api_key, payload, endpoint, task_call_id, max_api_retries, size_setting
                ))

            if not tasks:
                self._log("No tasks were created for OpenAI (gpt-image-1).")
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
            err_w, err_h = 1024,1024
            try:
                err_w, err_h = map(int, size_setting.split('x'))
            except:
                pass
            error_imgs = [self._create_error_image("Processing timeout", err_w, err_h)] * inputcount
            return (error_imgs, "Processing timed out after 5 minutes")
        except Exception as e:
            self._log(f"Error in async processing: {str(e)}")
            err_w, err_h = 1024,1024
            try:
                err_w, err_h = map(int, size_setting.split('x'))
            except:
                pass
            # Create batch of error images
            error_imgs = [self._create_error_image(f"Async processing error: {str(e)}", err_w, err_h)] * inputcount
            return (error_imgs, f"Async processing error: {str(e)}")
        
        # Process results (ensure results is not None if an error occurred before assignment)
        if results_with_id is None:
            self._log("Async processing did not yield results, possibly due to an earlier error before gather.")
            err_w, err_h = 1024,1024
            try:
                err_w, err_h = map(int, size_setting.split('x'))
            except:
                pass
            error_imgs = [self._create_error_image("Async processing failed to produce results", err_w, err_h)] * inputcount
            return (error_imgs, "Async processing failed to produce results")
        
        results_with_id.sort(key=lambda x: int(x[2]))
        
        output_images = []
        output_texts = []

        for img_tensor, response_text, call_id_res in results_with_id:
            output_images.append(img_tensor)
            output_texts.append(f"Response for Input {call_id_res}:\n{response_text}")

        if not output_images or all(img is None for img in output_images): # Check if all are None
            err_w, err_h = 1024,1024
            try:
                err_w, err_h = map(int, size_setting.split('x'))
            except:
                pass
            batched_images = self._create_error_image("No images generated by OpenAI (gpt-image-1)", err_w, err_h)
            if inputcount > 1 and not all(img is None for img in output_images) : # if some generated, but then failed to batch
                 # Create a list of error images if batching fails but some individual images were okay
                 batched_images_list = []
                 for i, img in enumerate(output_images):
                     if img is not None: batched_images_list.append(img)
                     else: batched_images_list.append(self._create_error_image(f"Slot {i+1} failed", err_w, err_h))
                 if batched_images_list: batched_images = torch.cat(batched_images_list, dim=0)

        else:
            valid_images = [img for img in output_images if img is not None]
            if not valid_images: # All were None after filtering
                err_w, err_h = 1024,1024
                try:
                    err_w, err_h = map(int, size_setting.split('x'))
                except:
                    pass
                batched_images = self._create_error_image("All image slots failed", err_w, err_h)
                if inputcount > 1: # Create multiple error images for the batch
                    batched_images = torch.cat([self._create_error_image(f"Slot {i+1} failed", err_w, err_h) for i in range(inputcount)], dim=0)

            else:
                try:
                    batched_images = torch.cat(valid_images, dim=0)
                except Exception as e:
                    self._log(f"Error batching images: {e}. Creating error images for failed slots.")
                    batched_images_list = []
                    err_w, err_h = 1024,1024
                    try:
                        err_w, err_h = map(int, size_setting.split('x'))
                    except:
                        pass
                    for i in range(inputcount):
                        if i < len(output_images) and output_images[i] is not None:
                            batched_images_list.append(output_images[i])
                        else:
                            batched_images_list.append(self._create_error_image(f"Slot {i+1} processing error", err_w, err_h))
                    if batched_images_list:
                         batched_images = torch.cat(batched_images_list, dim=0)
                    else: # Should not happen if valid_images was not empty
                         batched_images = self._create_error_image("Batching failed catastrophically", err_w, err_h)


        combined_responses = "\n\n".join(output_texts)
        final_log_output = "Processing Logs (OpenAI gpt-image-1 ADV):\n" + "\n".join(self.log_messages) + "\n\n" + combined_responses

        return (batched_images, final_log_output)

# NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS are typically in __init__.py
# For standalone testing, you might include them here.
# For ComfyUI integration, they should be in the main __init__.py of your custom node pack.
# Example:
# NODE_CLASS_MAPPINGS = {
#     "FL_GPT_Image1_ADV": FL_GPT_Image1_ADV
# }
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "FL_GPT_Image1_ADV": "FL GPT Image1 ADV (gpt-image-1)"
# }