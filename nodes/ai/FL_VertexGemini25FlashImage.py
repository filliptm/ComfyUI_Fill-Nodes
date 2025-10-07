import os
import io
import json
import base64
import torch
import numpy as np
from PIL import Image
import time
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types
from google.oauth2 import service_account
from typing import Optional, Tuple


class FL_VertexGemini25FlashImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "service_account_json": ("STRING", {"default": "", "multiline": False}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 666666}),
            },
            "optional": {
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "API_Response")
    FUNCTION = "generate_image"
    CATEGORY = "🏵️Fill Nodes/AI"

    def __init__(self):
        """Initialize logging system"""
        self.log_messages = []

    def _log(self, message):
        """Global logging function: record to log list"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[FL_VertexGemini25FlashImage] {timestamp}: {message}"
        print(formatted_message)
        self.log_messages.append(message)
        return message

    def _process_tensor_to_pil(self, tensor, name="Image"):
        """Convert a tensor to a PIL image for API submission"""
        try:
            if tensor is None:
                self._log(f"{name} is None, skipping")
                return None

            # Ensure tensor is in correct format [1, H, W, 3]
            if len(tensor.shape) == 4 and tensor.shape[0] >= 1:
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

    def _prepare_image_for_api(self, pil_image):
        """Prepare PIL image for Vertex AI - just return the PIL image directly"""
        try:
            self._log(f"Preparing image for API: {pil_image.width}x{pil_image.height}")

            # For Vertex AI generate_content with Gemini models,
            # we can pass PIL images directly and the SDK handles conversion
            return pil_image

        except Exception as e:
            self._log(f"Error preparing image: {str(e)}")
            traceback.print_exc()
            return None

    def _create_error_frame(self, error_message="Image Generation Failed", width=512, height=512):
        """Create a black error frame with red text"""
        from PIL import ImageDraw, ImageFont

        image = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Draw error text
        text_bbox = draw.textbbox((0, 0), error_message, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (width - text_width) / 2
        text_y = (height - text_height) / 2

        draw.text((text_x, text_y), error_message, fill=(255, 0, 0), font=font)

        # Convert to tensor format [1, H, W, 3]
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        self._log(f"Created error frame with message: '{error_message}'")
        return img_tensor

    def generate_image(self, prompt: str, service_account_json: str,
                      batch_count: int = 1,
                      temperature: float = 1.0, seed: int = 0, reference_image_1=None,
                      reference_image_2=None, reference_image_3=None):
        """Generate image using Gemini 2.5 Flash Image API"""

        # Reset log messages
        self.log_messages = []

        # Log seed parameter (not sent to API, just for node tracking)
        self._log(f"Seed parameter: {seed} (not sent to API)")

        try:
            # Validate service account JSON file
            if not service_account_json:
                error_message = "Error: No service account JSON file provided."
                self._log(error_message)
                error_frame = self._create_error_frame("Service account required")
                return (error_frame, error_message)

            # Build full path to service account JSON
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            json_path = os.path.join(script_dir, service_account_json)

            if not os.path.exists(json_path):
                error_message = f"Error: Service account JSON file not found at {json_path}"
                self._log(error_message)
                error_frame = self._create_error_frame("JSON file not found")
                return (error_frame, error_message)

            # Validate prompt
            if not prompt or len(prompt.strip()) == 0:
                error_message = "Error: No prompt provided. Please enter an image generation prompt."
                self._log(error_message)
                error_frame = self._create_error_frame("Prompt required")
                return (error_frame, error_message)

            # Load credentials from service account JSON
            self._log(f"Loading service account credentials from: {json_path}")

            # Read the JSON file to get project_id
            with open(json_path, 'r') as f:
                service_account_info = json.load(f)
                project_id = service_account_info.get('project_id')

            if not project_id:
                error_message = "Error: No project_id found in service account JSON"
                self._log(error_message)
                error_frame = self._create_error_frame("Missing project_id")
                return (error_frame, error_message)

            credentials = service_account.Credentials.from_service_account_file(
                json_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )

            # Initialize client with Vertex AI
            client = genai.Client(
                vertexai=True,
                project=project_id,
                location='us-central1',
                credentials=credentials
            )
            self._log(f"Gemini client initialized with Vertex AI (project: {project_id})")

            # Build the contents for the API call
            contents = []

            # Add reference images if provided (up to 3)
            reference_images = [reference_image_1, reference_image_2, reference_image_3]
            image_count = 0

            for i, ref_image in enumerate(reference_images, 1):
                if ref_image is not None:
                    pil_reference = self._process_tensor_to_pil(ref_image, f"Reference Image {i}")
                    if pil_reference:
                        # Prepare image for Vertex AI
                        image_for_api = self._prepare_image_for_api(pil_reference)
                        if not image_for_api:
                            error_message = f"Failed to prepare reference image {i}"
                            self._log(error_message)
                            error_frame = self._create_error_frame(error_message)
                            return (error_frame, error_message)
                        contents.append(image_for_api)
                        image_count += 1
                        self._log(f"Reference image {i} added to request")

            if image_count > 0:
                self._log(f"Total reference images: {image_count}")

            # Add the prompt
            contents.append(prompt)

            # Build the configuration
            # Note: Vertex AI doesn't support aspect_ratio in current version
            # Users can specify desired aspect ratio in the prompt text itself
            # candidate_count is hardcoded to 1, use batch_count for multiple images
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                candidate_count=1,
                temperature=temperature
            )

            self._log(f"Generating {batch_count} image(s) with prompt: {prompt[:100]}...")
            self._log(f"Temperature: {temperature}")

            # Function to make a single API call
            def generate_single_image(batch_idx):
                try:
                    self._log(f"Starting batch {batch_idx + 1}/{batch_count}...")

                    response = client.models.generate_content(
                        model="gemini-2.5-flash-image",
                        contents=contents,
                        config=config
                    )

                    self._log(f"Batch {batch_idx + 1} API response received")

                    # Log response structure for debugging
                    self._log(f"Batch {batch_idx + 1} - Number of candidates: {len(response.candidates) if hasattr(response, 'candidates') else 'N/A'}")

                    # Check for prompt feedback (content filtering, safety blocks, etc.)
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                        self._log(f"Batch {batch_idx + 1} - Prompt feedback: {response.prompt_feedback}")

                    # Extract image from response
                    image_found = False
                    for candidate_idx, candidate in enumerate(response.candidates):
                        self._log(f"Batch {batch_idx + 1} - Candidate {candidate_idx}: {len(candidate.content.parts) if hasattr(candidate, 'content') else 'N/A'} parts")

                        # Check for finish reason (safety blocks, etc.)
                        if hasattr(candidate, 'finish_reason'):
                            self._log(f"Batch {batch_idx + 1} - Candidate {candidate_idx} finish_reason: {candidate.finish_reason}")

                        # Check for safety ratings
                        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                            self._log(f"Batch {batch_idx + 1} - Candidate {candidate_idx} safety_ratings: {candidate.safety_ratings}")

                        for part_idx, part in enumerate(candidate.content.parts):
                            self._log(f"Batch {batch_idx + 1} - Part {part_idx} type: {type(part).__name__}")
                            self._log(f"Batch {batch_idx + 1} - Part {part_idx} has inline_data: {hasattr(part, 'inline_data')}")

                            if hasattr(part, 'inline_data') and part.inline_data:
                                image_data = part.inline_data.data
                                self._log(f"Batch {batch_idx + 1} - inline_data size: {len(image_data)} bytes")

                                # Check mime type if available
                                if hasattr(part.inline_data, 'mime_type'):
                                    self._log(f"Batch {batch_idx + 1} - inline_data mime_type: {part.inline_data.mime_type}")

                                # Check if data is base64 encoded (starts with base64 characters)
                                # Raw binary images start with magic bytes (e.g., PNG: \x89PNG, JPEG: \xff\xd8)
                                # Base64 encoded data will be ASCII/UTF-8 text
                                is_base64 = False
                                try:
                                    # Check if the data looks like base64 (ASCII text)
                                    if isinstance(image_data, bytes):
                                        # Try to decode as ASCII - base64 is ASCII
                                        test_str = image_data[:20].decode('ascii')
                                        # Check if it starts with common base64 PNG prefix
                                        if test_str.startswith('iVBORw0KG') or test_str.startswith('/9j/'):
                                            is_base64 = True
                                            self._log(f"Batch {batch_idx + 1} - Detected base64-encoded data")
                                except:
                                    pass

                                # Decode base64 if needed
                                if is_base64:
                                    try:
                                        image_data = base64.b64decode(image_data)
                                        self._log(f"Batch {batch_idx + 1} - Decoded base64, new size: {len(image_data)} bytes")
                                    except Exception as e:
                                        self._log(f"Batch {batch_idx + 1} - Failed to decode base64: {str(e)}")

                                # Check image header to identify format (after potential base64 decode)
                                image_header = image_data[:16] if len(image_data) >= 16 else image_data
                                self._log(f"Batch {batch_idx + 1} - image header (hex): {image_header.hex()}")

                                pil_image = Image.open(io.BytesIO(image_data))
                                self._log(f"Batch {batch_idx + 1} image received: {pil_image.width}x{pil_image.height}")

                                # Convert to tensor format [H, W, 3]
                                img_array = np.array(pil_image).astype(np.float32) / 255.0

                                # Handle RGBA images
                                if img_array.shape[-1] == 4:
                                    img_array = img_array[:, :, :3]

                                image_found = True
                                return torch.from_numpy(img_array)
                            elif hasattr(part, 'text'):
                                self._log(f"Batch {batch_idx + 1} - Part {part_idx} contains text: {part.text[:100] if len(part.text) > 100 else part.text}")

                    if not image_found:
                        self._log(f"Batch {batch_idx + 1} - WARNING: No image data found in response")

                    return None
                except Exception as e:
                    self._log(f"Error in batch {batch_idx + 1}: {str(e)}")
                    self._log(f"Batch {batch_idx + 1} - Full traceback: {traceback.format_exc()}")
                    return None

            # Generate images in parallel using ThreadPoolExecutor
            generated_images = []
            with ThreadPoolExecutor(max_workers=batch_count) as executor:
                futures = [executor.submit(generate_single_image, i) for i in range(batch_count)]
                for future in futures:
                    result = future.result()
                    if result is not None:
                        generated_images.append(result)

            if not generated_images:
                error_message = "No images generated by the API"
                self._log(error_message)
                error_frame = self._create_error_frame(error_message)
                return (error_frame, error_message)

            # Stack images into a batch tensor [B, H, W, C]
            images_batch = torch.stack(generated_images, dim=0)

            # Prepare response text
            response_text = "## Image Generation Successful\n"
            response_text += f"Generated {len(generated_images)} image(s)\n"
            response_text += f"Batch shape: {images_batch.shape}\n"
            response_text += "\n## Processing Log\n" + "\n".join(self.log_messages)

            self._log(f"Successfully generated {len(generated_images)} image(s)")

            return (images_batch, response_text)

        except Exception as e:
            error_message = f"Error during image generation: {str(e)}"
            self._log(error_message)
            traceback.print_exc()

            error_frame = self._create_error_frame(f"Error: {str(e)}")
            full_text = "## Processing Log\n" + "\n".join(self.log_messages) + "\n\n## Error\n" + error_message

            return (error_frame, full_text)
