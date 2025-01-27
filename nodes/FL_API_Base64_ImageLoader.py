import torch
import base64
import numpy as np
from PIL import Image
import io
import re


class FL_API_Base64_ImageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base64_string": ("STRING", {"default": "", "multiline": True}),
                "job_id": ("STRING", {"default": ""}),
                "user_id": ("STRING", {"default": ""}),  # Added user_id field
                "category": ("STRING", {"default": ""}),
            },
            "optional": {
                "resize_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "resize_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "maintain_aspect_ratio": ("BOOLEAN", {"default": True}),
                "auto_clean_base64": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING", "STRING", "STRING")  # Added STRING for user_id
    RETURN_NAMES = ("image", "width", "height", "job_id", "user_id", "category")  # Added user_id
    FUNCTION = "load_base64_image"
    CATEGORY = "🏵️Fill Nodes/API Tools"

    def clean_base64_string(self, base64_string: str) -> str:
        """Remove whitespace, newlines, and data URL prefixes from base64 string"""
        # Remove whitespace and newlines
        base64_string = re.sub(r'\s+', '', base64_string)
        # Remove data URL prefix if present
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        return base64_string

    def calculate_new_dimensions(self, original_width, original_height, target_width, target_height, maintain_aspect):
        """Calculate new dimensions based on resize parameters"""
        if target_width == 0 and target_height == 0:
            return original_width, original_height

        if maintain_aspect:
            if target_width == 0:
                ratio = target_height / original_height
                return int(original_width * ratio), target_height
            elif target_height == 0:
                ratio = target_width / original_width
                return target_width, int(original_height * ratio)
            else:
                # Use the smaller ratio to ensure image fits within specified dimensions
                ratio = min(target_width / original_width, target_height / original_height)
                return int(original_width * ratio), int(original_height * ratio)
        else:
            return (target_width if target_width != 0 else original_width,
                    target_height if target_height != 0 else original_height)

    def load_base64_image(self,
                         base64_string: str,
                         job_id: str,
                         user_id: str,  # Added user_id parameter
                         category: str,
                         resize_width: int = 0,
                         resize_height: int = 0,
                         maintain_aspect_ratio: bool = True,
                         auto_clean_base64: bool = True) -> tuple:
        try:
            # Clean base64 string if enabled
            if auto_clean_base64:
                base64_string = self.clean_base64_string(base64_string)

            # Decode base64 string
            image_data = base64.b64decode(base64_string)

            # Open image
            image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Get original dimensions
            original_width, original_height = image.size

            # Calculate new dimensions if resize is requested
            new_width, new_height = self.calculate_new_dimensions(
                original_width, original_height,
                resize_width, resize_height,
                maintain_aspect_ratio
            )

            # Resize if dimensions changed
            if (new_width, new_height) != (original_width, original_height):
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to numpy array and normalize
            image_array = np.array(image).astype(np.float32) / 255.0

            # Reshape to ComfyUI's expected format [batch, height, width, channels]
            image_tensor = torch.from_numpy(image_array).unsqueeze(0)

            return (image_tensor, new_width, new_height, job_id, user_id, category)  # Added user_id to return tuple

        except Exception as e:
            raise ValueError(f"Error loading base64 image: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, base64_string, job_id, user_id, category, resize_width, resize_height, maintain_aspect_ratio,
                   auto_clean_base64):  # Added user_id to IS_CHANGED
        return float("NaN")