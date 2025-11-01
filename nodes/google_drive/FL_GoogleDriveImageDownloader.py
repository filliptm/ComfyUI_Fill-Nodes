import os
import re
import io
import base64
import gdown
from PIL import Image
import torch
import numpy as np
import hashlib
import json
from server import PromptServer


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
                "use_cache": ("BOOLEAN", {
                    "default": True,
                    "label": "Use Cached Image"
                }),
                "show_preview": ("BOOLEAN", {
                    "default": False,
                    "label": "Show Preview on Node"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "download_and_process_image"
    OUTPUT_NODE = True
    CATEGORY = "🏵️Fill Nodes/Google Drive"

    def __init__(self):
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
        self.cache_index_file = os.path.join(self.cache_dir, "image_cache_index.json")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize the cache index if it doesn't exist
        if not os.path.exists(self.cache_index_file):
            with open(self.cache_index_file, "w") as f:
                json.dump({}, f)

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

    def get_cached_image(self, file_id):
        """Check if image is in cache and return the path if it exists"""
        try:
            with open(self.cache_index_file, "r") as f:
                cache_index = json.load(f)

            if file_id in cache_index:
                cached_path = cache_index[file_id]
                if os.path.exists(cached_path):
                    return cached_path
        except Exception as e:
            print(f"Cache lookup error: {str(e)}")

        return None

    def save_to_cache(self, file_id, image_path):
        """Save downloaded image to cache"""
        try:
            # Generate a unique filename
            cache_filename = os.path.join(self.cache_dir, f"{file_id}.png")

            # Copy the image to cache
            img = Image.open(image_path)
            img.save(cache_filename)

            # Update the cache index
            with open(self.cache_index_file, "r") as f:
                cache_index = json.load(f)

            cache_index[file_id] = cache_filename

            with open(self.cache_index_file, "w") as f:
                json.dump(cache_index, f)

            return cache_filename
        except Exception as e:
            print(f"Cache save error: {str(e)}")
            return None

    def prepare_image_for_display(self, pil_image):
        """Convert PIL image to base64 for frontend display"""
        # Create a copy to avoid modifying the original
        display_img = pil_image.copy()

        # Resize image if it's too large for preview
        max_size = (512, 512)
        display_img.thumbnail(max_size, Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        display_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def download_and_process_image(self, google_drive_link: str, use_cache: bool = True, show_preview: bool = True) -> tuple:
        try:
            # Extract file ID
            file_id = self.extract_file_id_from_link(google_drive_link)

            image_path = None
            temp_path = None  # Initialize temp_path to ensure it's always defined

            # Check cache first if caching is enabled
            if use_cache:
                cached_path = self.get_cached_image(file_id)
                if cached_path:
                    print(f"Using cached image for file ID: {file_id}")
                    image_path = cached_path

            # If not in cache or caching disabled, download the image
            if not image_path:
                # Create temporary directory for download
                temp_dir = os.path.join(os.getcwd(), 'temp_downloads')
                os.makedirs(temp_dir, exist_ok=True)

                # Create download URL
                url = f'https://drive.google.com/uc?id={file_id}'

                # Download the file
                print("Downloading image from Google Drive...")
                temp_path = os.path.join(temp_dir, f"temp_image_{file_id}")
                output = gdown.download(url=url, output=temp_path, quiet=False, fuzzy=True)

                if not output:
                    raise ValueError("Failed to download image")

                image_path = output

                # Save to cache if enabled
                if use_cache:
                    cached_path = self.save_to_cache(file_id, image_path)
                    if cached_path:
                        image_path = cached_path

            # Open and process the image
            try:
                image = Image.open(image_path)
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                raise ValueError(f"Invalid image file: {str(e)}")

            # Send image to frontend for preview if enabled
            if show_preview:
                display_image = self.prepare_image_for_display(image)
                PromptServer.instance.send_sync("fl_google_drive_image_downloader", {"image": display_image})

            # Convert to the format expected by ComfyUI
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

            # Cleanup temp file if it exists and isn't the cached version
            if temp_path and os.path.exists(temp_path) and temp_path != image_path:
                os.remove(temp_path)

            print(f"Successfully processed image: {image_tensor.shape}")
            return (image_tensor,)

        except Exception as e:
            raise ValueError(f"Error processing image from Google Drive: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, google_drive_link, use_cache, show_preview):
        # Only signal a change if use_cache is False
        # This ensures the node won't rerun when caching is enabled
        if not use_cache:
            return float("NaN")

        # Otherwise, we should check if the link has changed since last run
        # This is done by hashing the link
        if not google_drive_link:
            return 0

        # Simple hash of the link string
        link_hash = hashlib.md5(google_drive_link.encode()).hexdigest()
        return link_hash
