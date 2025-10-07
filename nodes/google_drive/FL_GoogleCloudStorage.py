import torch
import numpy as np
import cv2
import os
import io
import uuid
import tempfile
import datetime
from datetime import datetime as dt
from typing import List, Tuple, Dict, Any, Optional

# Check if Google Cloud Storage is installed
GOOGLE_CLOUD_AVAILABLE = False
try:
    from google.cloud import storage
    from google.oauth2 import service_account
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    print("[GoogleCloudStorage] Error: Google Cloud Storage library not installed.")
    print("[GoogleCloudStorage] Please install it with: pip install google-cloud-storage")

class FL_GoogleCloudStorage:
    """
    A ComfyUI node for uploading images and videos to Google Cloud Storage.
    Can handle single images, batches of images, and optionally compile batches into videos.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "bucket_name": ("STRING", {"default": "my-bucket"}),
                "folder_path": ("STRING", {"default": "uploads/", "placeholder": "Path within bucket (e.g., 'folder/subfolder/')"}),
                "file_prefix": ("STRING", {"default": "image_", "placeholder": "Prefix for filenames"}),
                "file_format": (["png", "jpg", "webp"], {"default": "png"}),
                "jpg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "compile_video": ("BOOLEAN", {"default": False}),
                "video_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "video_codec": (["mp4v", "avc1", "XVID"], {"default": "mp4v"}),
                "video_quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "credentials_json": ("STRING", {"default": "", "multiline": True, "placeholder": "Paste your service account JSON credentials here"}),
                "make_public": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "metadata": ("STRING", {"default": "{}", "multiline": True, "placeholder": "JSON metadata to attach to uploads"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("status", "urls", "error_message")
    FUNCTION = "upload_to_gcs"
    CATEGORY = "🏵️Fill Nodes/Google Drive"
    
    def upload_to_gcs(self, images, bucket_name, folder_path, file_prefix, file_format,
                     jpg_quality, compile_video, video_fps, video_codec, video_quality,
                     credentials_json, make_public, metadata=None):
        """
        Upload images or compiled video to Google Cloud Storage
        
        Args:
            images: Tensor of images to upload
            bucket_name: GCS bucket name
            folder_path: Path within the bucket
            file_prefix: Prefix for filenames
            file_format: Image format (png, jpg, webp)
            jpg_quality: Quality for JPG compression
            compile_video: Whether to compile images into a video
            video_fps: Frames per second for video
            video_codec: Video codec to use
            video_quality: Video quality (1-100)
            credentials_json: JSON string containing GCS service account credentials
            make_public: Whether to make uploaded files publicly accessible
            metadata: Optional JSON metadata to attach to uploads
            
        Returns:
            Tuple of (status, urls, error_message)
        """
        # Check if Google Cloud Storage is available
        if not GOOGLE_CLOUD_AVAILABLE:
            error_msg = (
                "Google Cloud Storage library not installed. "
                "Please install it with: pip install google-cloud-storage"
            )
            print(f"[GoogleCloudStorage] Error: {error_msg}")
            return "Error", "", error_msg
            
        try:
            # Normalize folder path to ensure it ends with a slash
            if folder_path and not folder_path.endswith('/'):
                folder_path += '/'
                
            # Parse metadata if provided
            metadata_dict = {}
            if metadata and metadata.strip() != "{}":
                try:
                    import json
                    metadata_dict = json.loads(metadata)
                    if not isinstance(metadata_dict, dict):
                        metadata_dict = {}
                except Exception as e:
                    print(f"[GoogleCloudStorage] Warning: Could not parse metadata JSON: {str(e)}")
                    metadata_dict = {}
            
            # Initialize GCS client
            if credentials_json and credentials_json.strip():
                try:
                    import json
                    import io
                    # Parse the credentials JSON string
                    credentials_info = json.loads(credentials_json)
                    # Create credentials from the parsed JSON
                    credentials = service_account.Credentials.from_service_account_info(credentials_info)
                    client = storage.Client(credentials=credentials)
                except Exception as e:
                    return "Error", "", f"Invalid credentials JSON: {str(e)}"
            else:
                # Use default credentials (environment variable or application default)
                client = storage.Client()
            
            # Get bucket
            bucket = client.bucket(bucket_name)
            
            # Check if bucket exists, if not return error
            if not bucket.exists():
                return "Error", "", f"Bucket '{bucket_name}' does not exist"
            
            # Generate timestamp for unique filenames
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            
            # Convert tensor to numpy images
            if len(images.shape) == 3:  # Single image
                images = images.unsqueeze(0)
            
            # Convert to numpy and ensure range 0-255
            np_images = (images * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
            
            # List to store uploaded URLs
            uploaded_urls = []
            
            # If compiling to video
            if compile_video and len(np_images) > 1:
                # Create a temporary file for the video
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                    temp_video_path = temp_video.name
                
                try:
                    # Get dimensions from the first image
                    height, width = np_images[0].shape[:2]
                    
                    # Initialize video writer
                    fourcc = cv2.VideoWriter_fourcc(*video_codec)
                    video_out = cv2.VideoWriter(
                        temp_video_path, 
                        fourcc, 
                        video_fps, 
                        (width, height)
                    )
                    
                    # Add each frame to the video
                    for img in np_images:
                        # Convert from RGB to BGR (OpenCV uses BGR)
                        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        video_out.write(bgr_img)
                    
                    # Release the video writer
                    video_out.release()
                    
                    # Upload the video to GCS
                    video_filename = f"{file_prefix}{timestamp}.mp4"
                    blob = bucket.blob(f"{folder_path}{video_filename}")
                    
                    # Add metadata if provided
                    if metadata_dict:
                        blob.metadata = metadata_dict
                    
                    # Upload the video
                    blob.upload_from_filename(temp_video_path)
                    
                    # Make public if requested
                    if make_public:
                        blob.make_public()
                        uploaded_urls.append(blob.public_url)
                    else:
                        # Generate a signed URL that expires in 1 hour
                        uploaded_urls.append(blob.generate_signed_url(
                            version="v4",
                            expiration=datetime.timedelta(hours=1),
                            method="GET"
                        ))
                    
                    print(f"[GoogleCloudStorage] Uploaded video: {video_filename}")
                    
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_video_path):
                        os.unlink(temp_video_path)
            
            # Upload individual images
            else:
                for i, img in enumerate(np_images):
                    # Convert to PIL Image
                    from PIL import Image
                    pil_img = Image.fromarray(img)
                    
                    # Create in-memory file
                    img_byte_arr = io.BytesIO()
                    
                    # Save with appropriate format and quality
                    if file_format == "jpg":
                        pil_img.save(img_byte_arr, format='JPEG', quality=jpg_quality)
                    elif file_format == "webp":
                        pil_img.save(img_byte_arr, format='WEBP', quality=jpg_quality)
                    else:  # png
                        pil_img.save(img_byte_arr, format='PNG')
                    
                    img_byte_arr.seek(0)
                    
                    # Generate unique filename
                    filename = f"{file_prefix}{timestamp}_{i}.{file_format}"
                    
                    # Create blob
                    blob = bucket.blob(f"{folder_path}{filename}")
                    
                    # Add metadata if provided
                    if metadata_dict:
                        blob.metadata = metadata_dict
                    
                    # Upload from memory
                    blob.upload_from_file(img_byte_arr)
                    
                    # Make public if requested
                    if make_public:
                        blob.make_public()
                        uploaded_urls.append(blob.public_url)
                    else:
                        # Generate a signed URL that expires in 1 hour
                        uploaded_urls.append(blob.generate_signed_url(
                            version="v4",
                            expiration=datetime.timedelta(hours=1),
                            method="GET"
                        ))
                    
                    print(f"[GoogleCloudStorage] Uploaded image: {filename}")
            
            # Return success status and URLs
            urls_text = "\n".join(uploaded_urls)
            if compile_video and len(np_images) > 1:
                status = f"Successfully uploaded video with {len(np_images)} frames"
            else:
                status = f"Successfully uploaded {len(np_images)} image(s)"
                
            return status, urls_text, ""
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[GoogleCloudStorage] Error: {str(e)}\n{error_details}")
            return "Error", "", f"Upload failed: {str(e)}"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Generate a unique value each time to ensure the node always processes
        return str(uuid.uuid4())