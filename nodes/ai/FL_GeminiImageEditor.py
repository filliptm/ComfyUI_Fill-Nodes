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


class FL_GeminiImageEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["models/gemini-2.0-flash-exp", "models/gemini-2.0-flash-preview-image-generation", "models/gemini-2.5-flash-image-preview", "models/gemini-2.5-flash-image", "models/gemini-3-pro-image-preview"], {"default": "models/gemini-2.5-flash-image"}),
                "aspect_ratio": (["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {"default": "1:1"}),
                "image_size": (["1K", "2K", "4K"], {"default": "1K", "tooltip": "Resolution size (2K/4K only supported by gemini-3-pro-image-preview)"}),
                "always_square": ("BOOLEAN", {"default": False, "description": "When enabled, pads images to square dimensions. When disabled, outputs original resolution as image list."}),
                "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                "seed": ("INT", {"default": 66666666, "min": 0, "max": 66666666}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Respond")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "generate_image"
    CATEGORY = "🏵️Fill Nodes/AI"

    def __init__(self):
        """Initialize logging system"""
        self.log_messages = []  # Global log message storage
        self.min_size = 1024  # Minimum size for both width and height

        # Check google-genai version
        try:
            import importlib.metadata
            genai_version = importlib.metadata.version('google-genai')
            self._log(f"Current google-genai version: {genai_version}")

            # Check if version meets minimum requirements
            from packaging import version
            if version.parse(genai_version) < version.parse('0.8.0'):
                self._log("Warning: google-genai version is too low, recommend upgrading to the latest version")
                self._log("Suggested: pip install -q -U google-genai")
        except Exception as e:
            self._log(f"Unable to check google-genai version: {e}")

    def _log(self, message):
        """Global logging function: record to log list"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[FL_GeminiImageGenerator] {timestamp}: {message}"
        print(formatted_message)
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message

    def _pad_image_to_minimum_size(self, pil_image):
        """Pad image with white to ensure it's at least min_size x min_size while preserving aspect ratio"""
        width, height = pil_image.size

        # If already meets minimum size requirements, return unchanged
        if width >= self.min_size and height >= self.min_size:
            return pil_image

        # Calculate new dimensions (preserve aspect ratio)
        new_width = max(width, self.min_size)
        new_height = max(height, self.min_size)

        # Create new white canvas
        new_image = Image.new('RGB', (new_width, new_height), color=(255, 255, 255))

        # Calculate position to paste the original image (centered)
        paste_x = (new_width - width) // 2
        paste_y = (new_height - height) // 2

        # Paste original image onto the white canvas
        new_image.paste(pil_image, (paste_x, paste_y))

        self._log(f"Padded image from {width}x{height} to {new_width}x{new_height} with white borders")
        return new_image

    def _create_error_image(self, error_message="API Failed to return an image", width=1024, height=1024):
        """Create black image with error text"""
        # Create black image
        image = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Try to use a system font
        try:
            # Try to find a font that exists on most systems
            font_options = ['Arial.ttf', 'DejaVuSans.ttf', 'FreeSans.ttf', 'NotoSans-Regular.ttf']
            font = None

            for font_name in font_options:
                try:
                    font = ImageFont.truetype(font_name, 24)
                    break
                except IOError:
                    continue

            if font is None:
                # Fall back to default font
                font = ImageFont.load_default()
        except Exception:
            # If everything fails, use default
            font = ImageFont.load_default()

        # Calculate text position (centered)
        text_width = draw.textlength(error_message, font=font) if hasattr(draw, 'textlength') else \
        font.getsize(error_message)[0]
        text_x = (width - text_width) / 2
        text_y = height / 2 - 12  # Vertically centered

        # Draw text
        draw.text((text_x, text_y), error_message, fill=(255, 0, 0), font=font)

        # Convert to tensor format [1, H, W, 3]
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        self._log(f"Created error image with message: '{error_message}'")
        return img_tensor

    def _generate_empty_image(self, width=1024, height=1024):
        """Generate standard format empty RGB image tensor - ensure ComfyUI compatible format [B,H,W,C]"""
        # Now just use error image with default message
        return self._create_error_image(width=width, height=height)

    def _process_tensor_to_pil(self, tensor, name="Image"):
        """Convert a tensor to a PIL image for API submission"""
        try:
            if tensor is None:
                self._log(f"{name} is None, skipping")
                return None

            # Ensure tensor is in correct format [1, H, W, 3]
            if len(tensor.shape) == 4 and tensor.shape[0] == 1:
                # Get first frame image
                image_np = tensor[0].cpu().numpy()

                # Convert to uint8 format for PIL
                image_np = (image_np * 255).astype(np.uint8)

                # Create PIL image
                pil_image = Image.fromarray(image_np)

                self._log(f"{name} processed successfully, size: {pil_image.width}x{pil_image.height}")
                return pil_image
            else:
                self._log(f"{name} format incorrect: {tensor.shape}")
                return None
        except Exception as e:
            self._log(f"Error processing {name}: {str(e)}")
            return None

    def _call_gemini_api(self, client, model, contents, gen_config, retry_count=0, max_retries=3, batch_id=0):
        """Call Gemini API with retry logic"""
        try:
            self._log(f"[Batch {batch_id}] API call attempt #{retry_count + 1}")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config
            )

            # Validate response structure
            if not hasattr(response, 'candidates') or not response.candidates:
                self._log(f"[Batch {batch_id}] Empty response: No candidates found")
                if retry_count < max_retries - 1:
                    self._log(f"[Batch {batch_id}] Retrying in 2 seconds... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2)  # Wait 2 seconds before retry
                    return self._call_gemini_api(client, model, contents, gen_config,
                                                 retry_count + 1, max_retries, batch_id)
                else:
                    self._log(f"[Batch {batch_id}] Maximum retries ({max_retries}) reached. Returning empty response.")
                    return None

            # Check if candidates[0].content exists
            if not hasattr(response.candidates[0], 'content') or response.candidates[0].content is None:
                self._log(f"[Batch {batch_id}] Invalid response: candidates[0].content is missing")
                if retry_count < max_retries - 1:
                    self._log(f"[Batch {batch_id}] Retrying in 2 seconds... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2)
                    return self._call_gemini_api(client, model, contents, gen_config,
                                                 retry_count + 1, max_retries, batch_id)
                else:
                    self._log(f"[Batch {batch_id}] Maximum retries ({max_retries}) reached. Returning empty response.")
                    return None

            # Check if content.parts exists
            if not hasattr(response.candidates[0].content, 'parts') or response.candidates[0].content.parts is None:
                self._log(f"[Batch {batch_id}] Invalid response: candidates[0].content.parts is missing")
                if retry_count < max_retries - 1:
                    self._log(f"[Batch {batch_id}] Retrying in 2 seconds... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2)
                    return self._call_gemini_api(client, model, contents, gen_config,
                                                 retry_count + 1, max_retries, batch_id)
                else:
                    self._log(f"[Batch {batch_id}] Maximum retries ({max_retries}) reached. Returning empty response.")
                    return None

            # Valid response, return it
            self._log(f"[Batch {batch_id}] Valid API response received")
            return response

        except Exception as e:
            self._log(f"[Batch {batch_id}] API call error: {str(e)}")
            if retry_count < max_retries - 1:
                wait_time = 2 * (retry_count + 1)  # Progressive backoff: 2s, 4s, 6s...
                self._log(
                    f"[Batch {batch_id}] Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
                time.sleep(wait_time)
                return self._call_gemini_api(client, model, contents, gen_config,
                                             retry_count + 1, max_retries, batch_id)
            else:
                self._log(f"[Batch {batch_id}] Maximum retries ({max_retries}) reached. Giving up.")
                return None

    def _process_api_response(self, response, batch_id=0, always_square=False):
        """Process API response and extract image tensor"""
        if response is None:
            self._log(f"[Batch {batch_id}] No valid response to process")
            error_msg = "API Failed to return an image"
            return self._create_error_image(error_msg), error_msg

        response_text = ""

        # Check if response contains valid data
        if not hasattr(response, 'candidates') or not response.candidates:
            self._log(f"[Batch {batch_id}] No candidates in API response")
            error_msg = "API returned an empty response"
            return self._create_error_image(error_msg), error_msg

        # Iterate through response parts
        for part in response.candidates[0].content.parts:
            # Check if it's a text part
            if hasattr(part, 'text') and part.text is not None:
                text_content = part.text
                response_text += text_content
                self._log(
                    f"[Batch {batch_id}] API returned text: {text_content[:100]}..." if len(
                        text_content) > 100 else text_content)

            # Check if it's an image part
            elif hasattr(part, 'inline_data') and part.inline_data is not None:
                self._log(f"[Batch {batch_id}] API returned image data")
                try:
                    # Get image data
                    image_data = part.inline_data.data
                    mime_type = part.inline_data.mime_type if hasattr(part.inline_data, 'mime_type') else "unknown"

                    # Confirm data is not empty
                    if not image_data or len(image_data) < 100:
                        self._log(f"[Batch {batch_id}] Warning: Image data is empty or too small")
                        continue

                    # Multiple methods to try opening the image
                    pil_image = None

                    # Method 1: Direct PIL open
                    try:
                        pil_image = Image.open(BytesIO(image_data))
                        self._log(
                            f"[Batch {batch_id}] Direct PIL open successful, size: {pil_image.width}x{pil_image.height}")
                    except Exception as e1:
                        self._log(f"[Batch {batch_id}] Direct PIL open failed: {str(e1)}")

                        # Method 2: Save to temp file and open
                        try:
                            temp_file = os.path.join(tempfile.gettempdir(),
                                                     f"gemini_image_{batch_id}_{int(time.time())}.png")
                            with open(temp_file, "wb") as f:
                                f.write(image_data)

                            pil_image = Image.open(temp_file)
                            self._log(f"[Batch {batch_id}] Opening via temp file successful")
                        except Exception as e2:
                            self._log(f"[Batch {batch_id}] Opening via temp file failed: {str(e2)}")

                            # Try more methods if needed
                            # Additional opening methods from original code could be added here if necessary

                    # Ensure image loaded successfully
                    if pil_image is None:
                        self._log(f"[Batch {batch_id}] Cannot open image, skipping")
                        continue

                    # Ensure image is RGB mode
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                        self._log(f"[Batch {batch_id}] Image converted to RGB mode")

                    # Store original dimensions for logging
                    width, height = pil_image.size
                    self._log(f"[Batch {batch_id}] Original image size: {width}x{height}")

                    # Apply padding if always_square is enabled and image needs it
                    if always_square and (width < self.min_size or height < self.min_size):
                        self._log(
                            f"[Batch {batch_id}] Image size {width}x{height} is smaller than minimum {self.min_size}x{self.min_size}, padding needed")
                        pil_image = self._pad_image_to_minimum_size(pil_image)
                    elif always_square:
                        self._log(f"[Batch {batch_id}] Always square enabled but image already meets minimum size")
                    else:
                        self._log(f"[Batch {batch_id}] Always square disabled, keeping original size: {width}x{height}")

                    # Convert to ComfyUI format
                    img_array = np.array(pil_image).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0)

                    self._log(f"[Batch {batch_id}] Image converted to tensor successfully, shape: {img_tensor.shape}")
                    return img_tensor, response_text
                except Exception as e:
                    self._log(f"[Batch {batch_id}] Image processing error: {e}")
                    traceback.print_exc()

        # If we got here, no image was found
        self._log(f"[Batch {batch_id}] No image data found in API response")
        error_msg = "API Failed to return an image"
        return self._create_error_image(error_msg), response_text if response_text else error_msg

    async def _generate_single_image_async(self, prompt, api_key, model, temperature, max_retries,
                                           batch_id, seed, reference_images, always_square=False,
                                           aspect_ratio="1:1", image_size="1K"):
        """Generate a single image asynchronously for batch processing"""
        try:
            # Create client instance - each batch gets its own client
            client = genai.Client(api_key=api_key)

            # Use provided seed or generate random one
            actual_seed = seed if seed != 0 else random.randint(1, 0xffffff)
            self._log(f"[Batch {batch_id}] Using seed: {actual_seed}")

            # Configure generation parameters
            gen_config_params = {
                "temperature": temperature,
                "seed": actual_seed,
                "response_modalities": ['Text', 'Image']
            }

            # Add image config if supported
            try:
                # Check if ImageConfig supports imageSize parameter (SDK >= 1.50)
                import inspect
                image_config_sig = inspect.signature(types.ImageConfig)
                supports_image_size = 'imageSize' in image_config_sig.parameters or 'image_size' in image_config_sig.parameters

                image_config_params = {"aspectRatio": aspect_ratio}
                # Only add imageSize for gemini-3-pro model if SDK supports it
                if "gemini-3-pro" in model and supports_image_size:
                    image_config_params["imageSize"] = image_size
                    self._log(f"[Batch {batch_id}] Using imageSize: {image_size}")
                elif "gemini-3-pro" in model:
                    self._log(f"[Batch {batch_id}] Warning: imageSize not supported in this SDK version. Upgrade google-genai to 1.50+ for 4K support")

                gen_config_params["image_config"] = types.ImageConfig(**image_config_params)
            except Exception as e:
                self._log(f"[Batch {batch_id}] ImageConfig not available: {e}, using basic config")

            gen_config = types.GenerateContentConfig(**gen_config_params)

            # Create content parts
            content_parts = []

            # Add prompt
            simple_prompt = f"Create a detailed image of: {prompt}"
            content_parts.append(simple_prompt)

            # Add reference images if provided
            for img in reference_images:
                if img is not None:
                    content_parts.append(img)

            # Make API call with synchronous method (will run in thread pool)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._call_gemini_api(
                    client=client,
                    model=model,
                    contents=content_parts,
                    gen_config=gen_config,
                    max_retries=max_retries,
                    batch_id=batch_id
                )
            )

            # Process the response and return the image tensor and text
            img_tensor, response_text = await loop.run_in_executor(
                None, lambda: self._process_api_response(response, batch_id, always_square)
            )

            # If processing failed, return the error image
            if img_tensor is None:
                error_msg = f"Batch {batch_id}: API Failed to return an image"
                return self._create_error_image(error_msg), error_msg, batch_id

            return img_tensor, response_text, batch_id

        except Exception as e:
            self._log(f"[Batch {batch_id}] Error in async image generation: {str(e)}")
            error_msg = f"Batch {batch_id}: Error: {str(e)}"
            return self._create_error_image(error_msg), error_msg, batch_id

    def generate_image(self, prompt, api_key, model, aspect_ratio, image_size, temperature, max_retries=3, batch_size=1,
                       seed=66666666, always_square=False, image1=None, image2=None, image3=None, image4=None):
        """Generate batch of images with parallel API calls"""
        # Reset log messages
        self.log_messages = []
        all_response_text = ""

        try:
            # Check if API key is provided
            if not api_key:
                error_message = "Error: No API key provided. Please enter Google API key in the node."
                self._log(error_message)
                error_img = self._create_error_image("API key required")
                full_text = "## Error\n" + error_message + "\n\n## Instructions\n1. Enter your Google API key in the node"

                # Create list of error images for all batch sizes
                error_imgs = [error_img] * batch_size
                return (error_imgs, full_text)

            self._log(f"Starting batch generation of {batch_size} images")

            # Process reference images once
            reference_pil_images = []
            image_tensors = [image1, image2, image3, image4]

            for i, img_tensor in enumerate(image_tensors):
                if img_tensor is not None:
                    pil_img = self._process_tensor_to_pil(img_tensor, f"Reference Image {i + 1}")
                    if pil_img:
                        reference_pil_images.append(pil_img)
                        self._log(f"Added reference image {i + 1} to batch processing")

            # Setup async tasks for each batch item
            async def run_batch():
                tasks = []

                # Create tasks for each batch item
                for i in range(batch_size):
                    # If seed is specified (non-zero), increment it for each batch item
                    # Otherwise each batch will use a random seed
                    batch_seed = seed + i if seed != 0 else 0

                    task = self._generate_single_image_async(
                        prompt=prompt,
                        api_key=api_key,
                        model=model,
                        temperature=temperature,
                        max_retries=max_retries,
                        batch_id=i + 1,
                        seed=batch_seed,
                        reference_images=reference_pil_images,
                        always_square=always_square,
                        aspect_ratio=aspect_ratio,
                        image_size=image_size
                    )
                    tasks.append(task)

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
            
            results = None  # Initialize results
            try:
                # Use thread pool executor to run async code in separate thread
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_sync_batch)
                    results = future.result(timeout=300)  # 5 minute timeout
            except concurrent.futures.TimeoutError:
                self._log("Async processing timed out after 5 minutes")
                error_imgs = [self._create_error_image("Processing timeout")] * batch_size
                return (error_imgs, "Processing timed out after 5 minutes")
            except Exception as e:
                self._log(f"Error in async processing: {str(e)}")
                traceback.print_exc()
                # Create list of error images
                error_imgs = [self._create_error_image(f"Async processing error: {str(e)}")] * batch_size
                return (error_imgs, f"Async processing error: {str(e)}")
            
            # Process results (ensure results is not None if an error occurred before assignment)
            if results is None:
                self._log("Async processing did not yield results, possibly due to an earlier error before gather.")
                error_imgs = [self._create_error_image("Async processing failed to produce results")] * batch_size
                return (error_imgs, "Async processing failed to produce results")

            # Process results
            all_tensors = []
            batch_texts = []

            for img_tensor, text, batch_id in results:
                # Always add tensor to list since we now always have a valid tensor
                # (either real image or error image)
                all_tensors.append(img_tensor)
                batch_texts.append(f"## Batch {batch_id} Response\n{text}")

            self._log(f"Successfully created list of {len(all_tensors)} images")

            # Combine all texts
            all_response_text = "## Batch Processing Results\n" + "\n".join(self.log_messages) + "\n\n" + "\n\n".join(
                batch_texts)

            return (all_tensors, all_response_text)

        except Exception as e:
            error_message = f"Error during batch processing: {str(e)}"
            self._log(error_message)
            traceback.print_exc()

            # Create list of error images
            error_imgs = [self._create_error_image(f"Error: {str(e)}")] * batch_size

            # Combine logs and error info
            full_text = "## Processing Log\n" + "\n".join(self.log_messages) + "\n\n## Error\n" + error_message
            return (error_imgs, full_text)