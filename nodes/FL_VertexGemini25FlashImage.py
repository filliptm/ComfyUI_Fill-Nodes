import os
import io
import json
import torch
import numpy as np
from PIL import Image
import time
import traceback
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
                "candidate_count": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "API_Response")
    FUNCTION = "generate_image"
    CATEGORY = "🏵️Fill Nodes/AI/Image"

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
                      candidate_count: int = 1,
                      temperature: float = 1.0, reference_image=None):
        """Generate image using Gemini 2.5 Flash Image API"""

        # Reset log messages
        self.log_messages = []

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

            # Add reference image if provided
            if reference_image is not None:
                pil_reference = self._process_tensor_to_pil(reference_image, "Reference Image")
                if pil_reference:
                    # Prepare image for Vertex AI
                    image_for_api = self._prepare_image_for_api(pil_reference)
                    if not image_for_api:
                        error_message = "Failed to prepare reference image"
                        self._log(error_message)
                        error_frame = self._create_error_frame(error_message)
                        return (error_frame, error_message)
                    contents.append(image_for_api)
                    self._log("Reference image added to request")

            # Add the prompt
            contents.append(prompt)

            # Build the configuration
            # Note: Vertex AI doesn't support aspect_ratio in current version
            # Users can specify desired aspect ratio in the prompt text itself
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                candidate_count=candidate_count,
                temperature=temperature
            )

            self._log(f"Generating image with prompt: {prompt[:100]}...")
            self._log(f"Candidates: {candidate_count}, Temperature: {temperature}")

            # Make the API call
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=contents,
                config=config
            )

            self._log("API response received")

            # Extract images from response
            generated_images = []
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # Convert inline data to PIL Image
                        image_data = part.inline_data.data
                        pil_image = Image.open(io.BytesIO(image_data))

                        self._log(f"Image received: {pil_image.width}x{pil_image.height}")

                        # Convert to tensor format [H, W, 3]
                        img_array = np.array(pil_image).astype(np.float32) / 255.0

                        # Handle RGBA images
                        if img_array.shape[-1] == 4:
                            img_array = img_array[:, :, :3]

                        img_tensor = torch.from_numpy(img_array)
                        generated_images.append(img_tensor)

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
