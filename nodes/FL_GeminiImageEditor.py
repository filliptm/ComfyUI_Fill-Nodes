import os
import base64
import io
import json
import torch
import numpy as np
from PIL import Image
import requests
import tempfile
from io import BytesIO
from google import genai
from google.genai import types
import time
import traceback


class FL_GeminiImageEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["models/gemini-2.0-flash-exp"], {"default": "models/gemini-2.0-flash-exp"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
            },
            "optional": {
                "seed": ("INT", {"default": 66666666, "min": 0, "max": 2147483647}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Respond")
    FUNCTION = "generate_image"
    CATEGORY = "🏵️Fill Nodes/AI"

    def __init__(self):
        """Initialize logging system and API key storage"""
        self.log_messages = []  # Global log message storage
        self.key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_api_key.txt")

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

    def _get_api_key(self, user_input_key):
        """Get API key, prioritize user input key"""
        # If user entered a valid key, use and save it
        if user_input_key and len(user_input_key) > 10:
            self._log("Using user-provided API key")
            # Save to file
            try:
                with open(self.key_file, "w") as f:
                    f.write(user_input_key)
                self._log("API key saved to node directory")
            except Exception as e:
                self._log(f"Failed to save API key: {e}")
            return user_input_key

        # If user didn't input, try reading from file
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "r") as f:
                    saved_key = f.read().strip()
                if saved_key and len(saved_key) > 10:
                    self._log("Using saved API key")
                    return saved_key
            except Exception as e:
                self._log(f"Failed to read saved API key: {e}")

        # If neither, return empty string
        self._log("Warning: No valid API key provided")
        return ""

    def _generate_empty_image(self, width, height):
        """Generate standard format empty RGB image tensor - ensure ComfyUI compatible format [B,H,W,C]"""
        # Create a ComfyUI standard image tensor
        # ComfyUI expects [batch, height, width, channels] format!
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
        tensor = torch.from_numpy(empty_image).unsqueeze(0)  # [1, H, W, 3]

        self._log(f"Created ComfyUI compatible empty image: shape={tensor.shape}, type={tensor.dtype}")
        return tensor

    def _validate_and_fix_tensor(self, tensor, name="Image"):
        """Validate and fix tensor format, ensure fully ComfyUI compatible"""
        try:
            # Basic shape check
            if tensor is None:
                self._log(f"Warning: {name} is None")
                return None

            self._log(f"Validating {name}: shape={tensor.shape}, type={tensor.dtype}, device={tensor.device}")

            # Ensure shape is correct: [B, C, H, W]
            if len(tensor.shape) != 4:
                self._log(f"Error: {name} shape is incorrect: {tensor.shape}")
                return None

            if tensor.shape[1] != 3:
                self._log(f"Error: {name} channels count is not 3: {tensor.shape[1]}")
                return None

            # Ensure type is float32
            if tensor.dtype != torch.float32:
                self._log(f"Fixing {name} type: {tensor.dtype} -> torch.float32")
                tensor = tensor.to(dtype=torch.float32)

            # Ensure memory is contiguous
            if not tensor.is_contiguous():
                self._log(f"Fixing {name} memory layout: making contiguous")
                tensor = tensor.contiguous()

            # Ensure value range is 0-1
            min_val = tensor.min().item()
            max_val = tensor.max().item()

            if min_val < 0 or max_val > 1:
                self._log(f"Fixing {name} value range: [{min_val}, {max_val}] -> [0, 1]")
                tensor = torch.clamp(tensor, 0.0, 1.0)

            return tensor
        except Exception as e:
            self._log(f"Error validating tensor: {e}")
            traceback.print_exc()
            return None

    def _save_tensor_as_image(self, image_tensor, file_path):
        """Save image tensor to file"""
        try:
            # Convert to numpy array
            if torch.is_tensor(image_tensor):
                if len(image_tensor.shape) == 4:
                    image_tensor = image_tensor[0]  # Get first image in batch

                # [C, H, W] -> [H, W, C]
                image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image_tensor

            # Scale to 0-255
            image_np = (image_np * 255).astype(np.uint8)

            # Create PIL image
            pil_image = Image.fromarray(image_np)

            # Save to file
            pil_image.save(file_path, format="PNG")
            self._log(f"Image saved to: {file_path}")
            return True
        except Exception as e:
            self._log(f"Image save error: {str(e)}")
            return False

    def _process_image_data(self, image_data, width, height):
        """Process API returned image data, return ComfyUI format image tensor [B,H,W,C]"""
        try:
            # Print image data type and size for debugging
            self._log(f"Image data type: {type(image_data)}")
            self._log(f"Image data length: {len(image_data) if hasattr(image_data, '__len__') else 'unknown'}")

            # Try to directly convert to PIL image
            try:
                pil_image = Image.open(BytesIO(image_data))
                self._log(
                    f"Successfully opened image, size: {pil_image.width}x{pil_image.height}, mode: {pil_image.mode}")
            except Exception as e:
                self._log(f"Cannot directly open image data: {e}")

                # Try other parsing methods, e.g. base64 decoding
                try:
                    # Check if it's a base64 encoded string
                    if isinstance(image_data, str):
                        # Try removing base64 prefix
                        if "base64," in image_data:
                            image_data = image_data.split("base64,")[1]
                        decoded_data = base64.b64decode(image_data)
                        pil_image = Image.open(BytesIO(decoded_data))
                    else:
                        # If it's a vector or other format, generate a placeholder image
                        self._log("Cannot parse image data, creating an empty image")
                        return self._generate_empty_image(width, height)
                except Exception as e2:
                    self._log(f"Alternate parsing method also failed: {e2}")
                    return self._generate_empty_image(width, height)

            # Ensure image is RGB mode
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                self._log(f"Image converted to RGB mode")

            # Resize image
            if pil_image.width != width or pil_image.height != height:
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
                self._log(f"Image resized to target dimensions: {width}x{height}")

            # Key fix: Use ComfyUI compatible format [batch, height, width, channels]
            # instead of PyTorch standard [batch, channels, height, width]
            img_array = np.array(pil_image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            self._log(f"Generated image tensor format: shape={img_tensor.shape}, type={img_tensor.dtype}")
            return (img_tensor,)

        except Exception as e:
            self._log(f"Error processing image data: {e}")
            traceback.print_exc()
            return self._generate_empty_image(width, height)

    def _call_gemini_api(self, client, model, contents, gen_config, retry_count=0, max_retries=3):
        """Call Gemini API with retry logic"""
        try:
            self._log(f"API call attempt #{retry_count + 1}")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config
            )

            # Validate response structure
            if not hasattr(response, 'candidates') or not response.candidates:
                self._log("Empty response: No candidates found")
                if retry_count < max_retries - 1:
                    self._log(f"Retrying in 2 seconds... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2)  # Wait 2 seconds before retry
                    return self._call_gemini_api(client, model, contents, gen_config, retry_count + 1, max_retries)
                else:
                    self._log(f"Maximum retries ({max_retries}) reached. Returning empty response.")
                    return None

            # Check if candidates[0].content exists
            if not hasattr(response.candidates[0], 'content') or response.candidates[0].content is None:
                self._log("Invalid response: candidates[0].content is missing")
                if retry_count < max_retries - 1:
                    self._log(f"Retrying in 2 seconds... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2)
                    return self._call_gemini_api(client, model, contents, gen_config, retry_count + 1, max_retries)
                else:
                    self._log(f"Maximum retries ({max_retries}) reached. Returning empty response.")
                    return None

            # Check if content.parts exists
            if not hasattr(response.candidates[0].content, 'parts') or response.candidates[0].content.parts is None:
                self._log("Invalid response: candidates[0].content.parts is missing")
                if retry_count < max_retries - 1:
                    self._log(f"Retrying in 2 seconds... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2)
                    return self._call_gemini_api(client, model, contents, gen_config, retry_count + 1, max_retries)
                else:
                    self._log(f"Maximum retries ({max_retries}) reached. Returning empty response.")
                    return None

            # Valid response, return it
            self._log("Valid API response received")
            return response

        except Exception as e:
            self._log(f"API call error: {str(e)}")
            if retry_count < max_retries - 1:
                wait_time = 2 * (retry_count + 1)  # Progressive backoff: 2s, 4s, 6s...
                self._log(f"Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
                time.sleep(wait_time)
                return self._call_gemini_api(client, model, contents, gen_config, retry_count + 1, max_retries)
            else:
                self._log(f"Maximum retries ({max_retries}) reached. Giving up.")
                return None

    def generate_image(self, prompt, api_key, model, width, height, temperature, max_retries=3, seed=66666666,
                       image=None):
        """Generate image - using simplified API key management"""
        temp_img_path = None
        response_text = ""

        # Reset log messages
        self.log_messages = []

        try:
            # Get API key
            actual_api_key = self._get_api_key(api_key)

            if not actual_api_key:
                error_message = "Error: No valid API key provided. Please enter API key in the node or ensure saved key exists."
                self._log(error_message)
                full_text = "## Error\n" + error_message + "\n\n## Instructions\n1. Enter your Google API key in the node\n2. The key will be automatically saved to the node directory for future use"
                return (self._generate_empty_image(width, height), full_text)

            # Create client instance
            client = genai.Client(api_key=actual_api_key)

            # Handle seed value
            if seed == 0:
                import random
                seed = random.randint(1, 2 ** 31 - 1)
                self._log(f"Generated random seed value: {seed}")
            else:
                self._log(f"Using specified seed value: {seed}")

            # Build simple prompt
            simple_prompt = f"Create a detailed image of: {prompt}"

            # Configure generation parameters, using user-specified temperature
            gen_config = types.GenerateContentConfig(
                temperature=temperature,
                seed=seed,
                response_modalities=['Text', 'Image']
            )

            # Log temperature setting
            self._log(f"Using temperature: {temperature}, seed: {seed}")

            # Handle reference image
            contents = []
            has_reference = False

            if image is not None:
                try:
                    # Ensure image format is correct
                    if len(image.shape) == 4 and image.shape[0] == 1:  # [1, H, W, 3] format
                        # Get first frame image
                        input_image = image[0].cpu().numpy()

                        # Convert to PIL image
                        input_image = (input_image * 255).astype(np.uint8)
                        pil_image = Image.fromarray(input_image)

                        # Save to temp file
                        temp_img_path = os.path.join(tempfile.gettempdir(), f"reference_{int(time.time())}.png")
                        pil_image.save(temp_img_path)

                        self._log(f"Reference image processed successfully, size: {pil_image.width}x{pil_image.height}")

                        # Read image data
                        with open(temp_img_path, "rb") as f:
                            image_bytes = f.read()

                        # Add image part and text part
                        img_part = {"inline_data": {"mime_type": "image/png", "data": image_bytes}}
                        txt_part = {"text": simple_prompt + " Use this reference image as style guidance."}

                        # Combine content (image first, text second)
                        contents = [img_part, txt_part]
                        has_reference = True
                        self._log("Reference image added to request")
                    else:
                        self._log(f"Reference image format incorrect: {image.shape}")
                        contents = simple_prompt
                except Exception as img_error:
                    self._log(f"Reference image processing error: {str(img_error)}")
                    contents = simple_prompt
            else:
                # No reference image, use text only
                contents = simple_prompt

            # Print request info
            self._log(f"Requesting Gemini API image generation, seed: {seed}, includes reference: {has_reference}")

            # Call API with retry logic
            response = self._call_gemini_api(
                client=client,
                model="models/gemini-2.0-flash-exp",
                contents=contents,
                gen_config=gen_config,
                max_retries=max_retries
            )

            # Check if response is None (all retries failed)
            if response is None:
                error_message = f"Failed to get valid response after {max_retries} attempts"
                self._log(error_message)
                full_text = "## Processing Log\n" + "\n".join(self.log_messages) + "\n\n## Error\n" + error_message
                return (self._generate_empty_image(width, height), full_text)

            # Response processing
            self._log("API response received successfully, processing...")

            if not hasattr(response, 'candidates') or not response.candidates:
                self._log("No candidates in API response")
                # Combine logs and return values
                full_text = "\n".join(self.log_messages) + "\n\nAPI returned an empty response"
                return (self._generate_empty_image(width, height), full_text)

            # Check if response contains image
            image_found = False

            # Iterate through response parts
            for part in response.candidates[0].content.parts:
                # Check if it's a text part
                if hasattr(part, 'text') and part.text is not None:
                    text_content = part.text
                    response_text += text_content
                    self._log(
                        f"API returned text: {text_content[:100]}..." if len(text_content) > 100 else text_content)

                # Check if it's an image part
                elif hasattr(part, 'inline_data') and part.inline_data is not None:
                    self._log("API returned data processing")
                    try:
                        # Log image data info for debugging
                        image_data = part.inline_data.data
                        mime_type = part.inline_data.mime_type if hasattr(part.inline_data, 'mime_type') else "unknown"
                        self._log(
                            f"Image data type: {type(image_data)}, MIME type: {mime_type}, data length: {len(image_data) if image_data else 0}")

                        # Confirm data is not empty and sufficient length
                        if not image_data or len(image_data) < 100:
                            self._log("Warning: Image data is empty or too small")
                            continue

                        # Try to check first few bytes of data to confirm valid image format
                        is_valid_image = False
                        if len(image_data) > 8:
                            # Check common image format magic bytes
                            magic_bytes = image_data[:8]
                            self._log(f"Image magic bytes(hex): {magic_bytes.hex()[:16]}...")
                            # PNG header is \x89PNG\r\n\x1a\n
                            if magic_bytes.startswith(b'\x89PNG'):
                                self._log("Detected valid PNG image format")
                                is_valid_image = True
                            # JPEG header is \xff\xd8
                            elif magic_bytes.startswith(b'\xff\xd8'):
                                self._log("Detected valid JPEG image format")
                                is_valid_image = True
                            # GIF header is GIF87a or GIF89a
                            elif magic_bytes.startswith(b'GIF87a') or magic_bytes.startswith(b'GIF89a'):
                                self._log("Detected valid GIF image format")
                                is_valid_image = True

                        if not is_valid_image:
                            self._log("Warning: Data may not be valid image format")

                        # Multiple methods to try opening the image
                        pil_image = None

                        # Method 1: Direct PIL open
                        try:
                            pil_image = Image.open(BytesIO(image_data))
                            self._log(f"Method 1 success: Direct PIL open, size: {pil_image.width}x{pil_image.height}")
                        except Exception as e1:
                            self._log(f"Method 1 failed: {str(e1)}")

                            # Method 2: Save to temp file and open
                            try:
                                temp_file = os.path.join(tempfile.gettempdir(), f"gemini_image_{int(time.time())}.png")
                                with open(temp_file, "wb") as f:
                                    f.write(image_data)
                                self._log(f"Saved image data to temp file: {temp_file}")

                                pil_image = Image.open(temp_file)
                                self._log(f"Method 2 success: Opening via temp file")
                            except Exception as e2:
                                self._log(f"Method 2 failed: {str(e2)}")

                                # Method 3: Try fixing headers then open
                                try:
                                    # If MIME type is PNG but header incorrect, try adding correct PNG header
                                    if mime_type == "image/png" and not image_data.startswith(b'\x89PNG'):
                                        self._log("Trying to fix PNG header")
                                        fixed_data = b'\x89PNG\r\n\x1a\n' + image_data[8:] if len(
                                            image_data) > 8 else image_data
                                        pil_image = Image.open(BytesIO(fixed_data))
                                        self._log("Method 3 success: Fixed PNG header")
                                    # If MIME type is JPEG but header incorrect, try adding correct JPEG header
                                    elif mime_type == "image/jpeg" and not image_data.startswith(b'\xff\xd8'):
                                        self._log("Trying to fix JPEG header")
                                        fixed_data = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00\x48\x00\x48\x00\x00' + image_data[
                                                                                                                               20:] if len(
                                            image_data) > 20 else image_data
                                        pil_image = Image.open(BytesIO(fixed_data))
                                        self._log("Method 3 success: Fixed JPEG header")
                                except Exception as e3:
                                    self._log(f"Method 3 failed: {str(e3)}")

                                    # Method 4: Try base64 decode then open
                                    try:
                                        if isinstance(image_data, bytes):
                                            # Try converting bytes to string and base64 decode
                                            str_data = image_data.decode('utf-8', errors='ignore')
                                            if 'base64,' in str_data:
                                                base64_part = str_data.split('base64,')[1]
                                                decoded_data = base64.b64decode(base64_part)
                                                pil_image = Image.open(BytesIO(decoded_data))
                                                self._log("Method 4 success: Base64 decode")
                                    except Exception as e4:
                                        self._log(f"Method 4 failed: {str(e4)}")

                                        # All methods failed, skip this data
                                        self._log("All image processing methods failed, cannot process returned data")
                                        continue

                        # Ensure image loaded successfully
                        if pil_image is None:
                            self._log("Cannot open image, skipping")
                            continue

                        # Ensure image is RGB mode
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                            self._log(f"Image converted to RGB mode")

                        # Resize image
                        if pil_image.width != width or pil_image.height != height:
                            pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
                            self._log(f"Image resized to target dimensions: {width}x{height}")

                        # Convert to ComfyUI format
                        img_array = np.array(pil_image).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

                        self._log(f"Image converted to tensor successfully, shape: {img_tensor.shape}")
                        image_found = True

                        # Combine logs and API returned text
                        full_text = "## Processing Log\n" + "\n".join(
                            self.log_messages) + "\n\n## API Response\n" + response_text
                        return (img_tensor, full_text)
                    except Exception as e:
                        self._log(f"Image processing error: {e}")
                        traceback.print_exc()  # Add detailed error traceback

            # No image data found, but may have text
            if not image_found:
                self._log("No image data found in API response, returning text only")
                if not response_text:
                    response_text = "API did not return any image or text"

            # Combine logs and API returned text
            full_text = "## Processing Log\n" + "\n".join(self.log_messages) + "\n\n## API Response\n" + response_text
            return (self._generate_empty_image(width, height), full_text)

        except Exception as e:
            error_message = f"Error during processing: {str(e)}"
            self._log(f"Gemini image generation error: {str(e)}")

            # Combine logs and error info
            full_text = "## Processing Log\n" + "\n".join(self.log_messages) + "\n\n## Error\n" + error_message
            return (self._generate_empty_image(width, height), full_text)
