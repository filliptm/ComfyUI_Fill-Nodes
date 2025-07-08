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

class FL_GPT_Image1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "size": (["1024x1024", "1536x1024", "1024x1536"], {"default": "1024x1024"}),
                "quality": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            },
            "optional": {
                "output_compression": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                "moderation": (["auto", "low"], {"default": "auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Response")
    FUNCTION = "generate_image"
    CATEGORY = "🏵️Fill Nodes/GPT"

    def __init__(self):
        """Initialize logging system"""
        self.log_messages = []  # Global log message storage

    def _log(self, message):
        """Global logging function: record to log list"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[FL_GPT_Image1] {timestamp}: {message}"
        print(formatted_message)
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message

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

        # Handle multiline error messages by truncating or splitting
        if len(error_message) > 60:
            # Truncate long messages
            display_message = error_message[:57] + "..."
        else:
            display_message = error_message

        # Calculate text position (centered)
        try:
            text_width = draw.textlength(display_message, font=font) if hasattr(draw, 'textlength') else font.getsize(display_message)[0]
        except Exception:
            # If measuring fails, use a conservative estimate
            text_width = len(display_message) * 12  # Rough estimate of width
            
        text_x = (width - text_width) / 2
        text_y = height / 2 - 12  # Vertically centered

        # Draw text
        draw.text((text_x, text_y), display_message, fill=(255, 0, 0), font=font)

        # Convert to tensor format [1, H, W, 3]
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        self._log(f"Created error image with message: '{error_message}'")
        return img_tensor

    def _process_tensor_to_pil(self, tensor, name="Image"):
        """Convert a tensor (or a batch of tensors) to a list of PIL images."""
        try:
            if tensor is None:
                self._log(f"{name} is None, skipping")
                return []

            # Handle batch of images [B, H, W, 3] or single image [1, H, W, 3]
            if len(tensor.shape) == 4:
                pil_images = []
                for i in range(tensor.shape[0]):
                    image_np = tensor[i].cpu().numpy()
                    image_np = (image_np * 255).astype(np.uint8)
                    pil_image = Image.fromarray(image_np)
                    pil_images.append(pil_image)
                
                if not pil_images:
                    self._log(f"{name} processed but no images were created.")
                    return []

                self._log(f"{name} batch processed successfully, {len(pil_images)} images, size: {pil_images[0].width}x{pil_images[0].height}")
                return pil_images
            else:
                self._log(f"{name} format incorrect: {tensor.shape}")
                return []
        except Exception as e:
            self._log(f"Error processing {name}: {str(e)}")
            return []

    def _encode_image_to_base64(self, pil_image, format="PNG"):
        """Convert PIL image to base64 string"""
        try:
            buffered = BytesIO()
            pil_image.save(buffered, format=format)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str
        except Exception as e:
            self._log(f"Error encoding image to base64: {str(e)}")
            return None

    def _call_openai_api(self, api_key, payload, endpoint="generations", retry_count=0, max_retries=3):
        """Call OpenAI API with retry logic"""
        try:
            self._log(f"API call attempt #{retry_count + 1} to endpoint: {endpoint}")
            
            url = f"https://api.openai.com/v1/images/{endpoint}"
            
            # Different handling for edits endpoint which requires multipart/form-data
            if endpoint == "edits":
                self._log("Using multipart/form-data for edits endpoint")
                
                headers = {
                    "Authorization": f"Bearer {api_key}"
                }
                
                # Create a multipart form-data request.
                # We use a list of tuples for 'files' to support multiple images.
                form_data = []
                
                # Add all text fields to the multipart data
                for key, value in payload.items():
                    if key not in ["image", "mask"]:
                        form_data.append((key, (None, str(value))))

                # Add image file(s) if present
                if "image" in payload and payload["image"] is not None:
                    images = payload["image"]
                    if not isinstance(images, list):
                        images = [images]  # Ensure it's a list for consistency

                    for i, img_bytes in enumerate(images):
                        if isinstance(img_bytes, bytes):
                            form_data.append(('image[]', (f"image_{i}.png", img_bytes, "image/png")))
                    
                    self._log(f"Added {len(images)} image file(s) to multipart request")

                # Add mask file if present
                if "mask" in payload and payload["mask"] is not None:
                    if isinstance(payload["mask"], bytes):
                        form_data.append(('mask', ("mask.png", payload["mask"], "image/png")))
                        self._log("Added mask file to multipart request")
                
                self._log(f"Sending multipart request with {len(form_data)} parts")
                
                # Use requests to send the multipart form data
                response = requests.post(
                    url,
                    headers=headers,
                    files=form_data,  # Pass the list of tuples here
                    timeout=120
                )
            else:
                # Standard JSON request for other endpoints
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            # Check if the request was successful
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                self._log(error_msg)
                
                if retry_count < max_retries - 1:
                    wait_time = 2 * (retry_count + 1)  # Progressive backoff
                    self._log(f"Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(wait_time)
                    return self._call_openai_api(api_key, payload, endpoint, retry_count + 1, max_retries)
                else:
                    self._log(f"Maximum retries ({max_retries}) reached. Giving up.")
                    return {"error": error_msg}
                    
        except Exception as e:
            self._log(f"API call error: {str(e)}")
            if retry_count < max_retries - 1:
                wait_time = 2 * (retry_count + 1)  # Progressive backoff
                self._log(f"Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
                time.sleep(wait_time)
                return self._call_openai_api(api_key, payload, endpoint, retry_count + 1, max_retries)
            else:
                self._log(f"Maximum retries ({max_retries}) reached. Giving up.")
                return {"error": str(e)}

    def _process_api_response(self, response):
        """Process API response and extract image tensor"""
        try:
            if "error" in response:
                error_msg = response["error"]
                self._log(f"API returned an error: {error_msg}")
                
                # Check for organization verification error
                if isinstance(error_msg, str) and "organization verification" in error_msg.lower():
                    simple_error = "OpenAI organization verification required"
                    self._log("Organization verification required for GPT-image-1 access")
                    return self._create_error_image(simple_error), json.dumps(response, indent=2)
                
                # For other errors, create a simplified message
                simple_error = "API Error"
                if isinstance(error_msg, str) and len(error_msg) > 60:
                    simple_error = f"API Error: {error_msg[:57]}..."
                else:
                    simple_error = f"API Error: {str(error_msg)}"
                    
                return self._create_error_image(simple_error), json.dumps(response, indent=2)
                
            if "data" not in response or not response["data"]:
                self._log("No data in API response")
                return self._create_error_image("API returned no image data"), json.dumps(response, indent=2)
                
            # Process each image in the response
            image_tensors = []
            
            for i, img_data in enumerate(response["data"]):
                if "b64_json" in img_data:
                    # Decode base64 image
                    try:
                        img_bytes = base64.b64decode(img_data["b64_json"])
                        pil_image = Image.open(BytesIO(img_bytes))
                        
                        # Ensure image is RGB
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                            
                        # Convert to tensor
                        img_array = np.array(pil_image).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                        image_tensors.append(img_tensor)
                        
                        self._log(f"Successfully processed image {i+1}")
                    except Exception as e:
                        self._log(f"Error processing image {i+1}: {str(e)}")
                        image_tensors.append(self._create_error_image(f"Error processing image: {str(e)}"))
                        
                elif "url" in img_data:
                    # Download image from URL
                    try:
                        response = requests.get(img_data["url"], timeout=30)
                        if response.status_code == 200:
                            pil_image = Image.open(BytesIO(response.content))
                            
                            # Ensure image is RGB
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                                
                            # Convert to tensor
                            img_array = np.array(pil_image).astype(np.float32) / 255.0
                            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                            image_tensors.append(img_tensor)
                            
                            self._log(f"Successfully downloaded and processed image {i+1}")
                        else:
                            self._log(f"Failed to download image {i+1}: HTTP {response.status_code}")
                            image_tensors.append(self._create_error_image(f"Failed to download image: HTTP {response.status_code}"))
                    except Exception as e:
                        self._log(f"Error downloading image {i+1}: {str(e)}")
                        image_tensors.append(self._create_error_image(f"Error downloading image: {str(e)}"))
                else:
                    self._log(f"No image data found in response item {i+1}")
                    image_tensors.append(self._create_error_image("No image data found in response"))
            
            # Combine all tensors into a batch
            if not image_tensors:
                return self._create_error_image("No images could be processed"), json.dumps(response, indent=2)
            elif len(image_tensors) == 1:
                return image_tensors[0], json.dumps(response, indent=2)
            else:
                return torch.cat(image_tensors, dim=0), json.dumps(response, indent=2)
                
        except Exception as e:
            self._log(f"Error processing API response: {str(e)}")
            return self._create_error_image(f"Error processing API response: {str(e)}"), json.dumps(response, indent=2)

    def generate_image(self, prompt, api_key, batch_size=1, size="auto", quality="auto", background="auto",
                       output_format="png", output_compression=100, moderation="auto", seed=0,
                       image=None, mask=None):
        """Generate images using OpenAI's GPT-image-1 model"""
        # Reset log messages
        self.log_messages = []
        
        try:
            # Check if API key is provided
            if not api_key:
                error_message = "Error: No API key provided. Please enter OpenAI API key in the node."
                self._log(error_message)
                error_img = self._create_error_image("API key required")
                full_text = "## Error\n" + error_message + "\n\n## Instructions\n1. Enter your OpenAI API key in the node"
                return (error_img, full_text)
                
            # Add a note about organization verification
            self._log("Note: GPT-image-1 requires OpenAI organization verification. If you encounter a 403 error, please visit: https://help.openai.com/en/articles/10910291-api-organization-verification")
            
            # Determine which endpoint to use based on inputs
            endpoint = "generations"  # Default endpoint
            
            # Prepare the payload
            payload = {
                "model": "gpt-image-1",
                "prompt": prompt,
                "n": batch_size,  # OpenAI API uses 'n' parameter, but we call it 'batch_size' in the UI
                "size": size,
            }
            
            # Add optional parameters if they're not default values
            if quality != "auto":
                payload["quality"] = quality
                
            if background != "auto":
                payload["background"] = background
                
            if output_format != "png":
                payload["output_format"] = output_format
                
            if output_compression != 100 and output_format in ["webp", "jpeg"]:
                payload["output_compression"] = output_compression
                
            if moderation != "auto":
                payload["moderation"] = moderation
                
            # Check if we're doing image editing
            if image is not None:
                endpoint = "edits"
                
                # Process the input image batch
                pil_images = self._process_tensor_to_pil(image, "Input Image")
                if not pil_images:
                    return self._create_error_image("Failed to process input image(s)"), "Error: Failed to process input image(s)"

                if len(pil_images) > 16:
                    self._log("Error: A maximum of 16 images can be provided for editing.")
                    return self._create_error_image("Too many images (max 16)"), "Error: A maximum of 16 images can be provided for editing."

                self._log(f"Setting up image editing request for {len(pil_images)} image(s)")
                
                # Convert PIL images to a list of bytes
                image_bytes_list = []
                for i, pil_image in enumerate(pil_images):
                    img_byte_arr = BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    image_bytes_list.append(img_byte_arr.getvalue())
                
                self._log(f"Converted {len(image_bytes_list)} image(s) to bytes")
                
                # Add image bytes list to payload
                payload["image"] = image_bytes_list
                
                # Process mask if provided. A single mask is applied to all images.
                if mask is not None:
                    # The mask is still a single image tensor, so we expect a list with one item
                    pil_masks = self._process_tensor_to_pil(mask, "Mask Image")
                    if pil_masks:
                        pil_mask = pil_masks[0] # Get the first (and only) mask
                        # Convert mask to bytes
                        mask_byte_arr = BytesIO()
                        pil_mask.save(mask_byte_arr, format='PNG')
                        mask_bytes = mask_byte_arr.getvalue()
                        self._log(f"Converted mask to bytes, size: {len(mask_bytes)} bytes")
                        
                        # Add mask bytes to payload
                        payload["mask"] = mask_bytes
            
            # Make the API call
            self._log(f"Calling OpenAI API with endpoint: {endpoint}")
            response = self._call_openai_api(api_key, payload, endpoint)
            
            # Process the response
            img_tensor, response_text = self._process_api_response(response)
            
            # Add logs to the response text
            full_response = "## Processing Log\n" + "\n".join(self.log_messages) + "\n\n## API Response\n" + response_text
            
            return (img_tensor, full_response)
            
        except Exception as e:
            error_message = f"Error during processing: {str(e)}"
            self._log(error_message)
            traceback.print_exc()
            
            # Create error image with simplified message
            simple_error = "Processing error"
            if len(str(e)) < 60:
                simple_error = f"Error: {str(e)}"
            
            error_img = self._create_error_image(simple_error)
            
            # Combine logs and error info
            full_text = "## Processing Log\n" + "\n".join(self.log_messages) + "\n\n## Error\n" + error_message
            
            return (error_img, full_text)