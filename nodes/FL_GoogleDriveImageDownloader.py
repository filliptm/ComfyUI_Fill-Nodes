import os
import re
import gdown
from PIL import Image
import torch
import numpy as np


class FL_GoogleDriveImageDownloader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "google_drive_link": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter Google Drive image link"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "download_and_process_image"
    CATEGORY = "🏵️Fill Nodes/Google Drive"

    def extract_file_id_from_link(self, share_link: str) -> str:
        """Extracts the file ID from a Google Drive share link."""
        if not share_link:
            raise ValueError("Google Drive link cannot be empty")
        if not share_link.startswith(('https://drive.google.com', 'http://drive.google.com')):
            raise ValueError("Invalid Google Drive URL format")

        match = re.search(r'(?:/d/|id=)([a-zA-Z0-9_-]+)', share_link)
        if match:
            return match.group(1)
        raise ValueError("Unable to extract file ID. Please ensure you're using a valid sharing link")

    def download_and_process_image(self, google_drive_link: str) -> tuple:
        try:
            # Create temporary directory for download
            temp_dir = os.path.join(os.getcwd(), 'temp_downloads')
            os.makedirs(temp_dir, exist_ok=True)

            # Extract file ID and create download URL
            file_id = self.extract_file_id_from_link(google_drive_link)
            url = f'https://drive.google.com/uc?id={file_id}'

            # Download the file
            print("Downloading image from Google Drive...")
            temp_path = os.path.join(temp_dir, f"temp_image_{file_id}")
            output = gdown.download(url=url, output=temp_path, quiet=False, fuzzy=True)

            if not output:
                raise ValueError("Failed to download image")

            # Open and verify it's an image
            try:
                image = Image.open(output)
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                raise ValueError(f"Downloaded file is not a valid image: {str(e)}")

            # Convert to the format expected by ComfyUI
            # ComfyUI expects images as: torch.Tensor [B, H, W, C] in float32
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

            # Cleanup
            os.remove(output)

            print(f"Successfully processed image: {image_tensor.shape}")
            return (image_tensor,)

        except Exception as e:
            raise ValueError(f"Error processing image from Google Drive: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, google_drive_link):
        return float("NaN")