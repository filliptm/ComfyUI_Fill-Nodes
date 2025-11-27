# FL_Fal_Pixverse: Fal AI Image-to-Video API Node with frame decomposition
import os
import uuid
import json
import time
import io
import requests
import torch
import numpy as np
import tempfile
import cv2
import concurrent.futures
import fal_client
from typing import Tuple, List, Dict, Union, Optional
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class FL_Fal_Pixverse:
    """
    A ComfyUI node for the Fal AI Image-to-Video API.
    Takes an image and converts it to a video using Fal AI's pixverse/v4/image-to-video endpoint.
    Downloads the video, extracts frames, and returns them as image tensors.
    """

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("frames_1", "frames_2", "frames_3", "frames_4", "frames_5", "video_urls", "status_msg")
    FUNCTION = "generate_video"
    CATEGORY = "🏵️Fill Nodes/AI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False,
                                      "description": "Fal AI API key"}),
                "prompt": ("STRING", {"default": "A woman walking through a beautiful landscape",
                                    "multiline": True, "description": "Text prompt describing the video"}),
                "aspect_ratio": (["16:9", "4:3", "1:1", "3:4", "9:16"], {"default": "16:9",
                                                                        "description": "The aspect ratio of the generated video"}),
                "resolution": (["360p", "540p", "720p", "1080p"], {"default": "720p",
                                                                   "description": "The resolution of the generated video"}),
                "duration": (["5", "8"], {"default": "5",
                                        "description": "Duration in seconds (8s videos cost double, 1080p limited to 5s)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647,
                               "description": "Random seed for video generation (0 = random)"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 5,
                                      "description": "Number of videos to generate with different seeds"}),
                "nth_frame": ("INT", {"default": 1, "min": 1, "max": 4,
                                     "description": "Extract every Nth frame (1=all frames, 2=every 2nd frame, etc.)"})
            },
            "optional": {
                "image": ("IMAGE", {"description": "Input image to animate"}),
                "negative_prompt": ("STRING", {"default": "blurry, low quality, low resolution, pixelated, noisy, grainy, out of focus, poorly lit, poorly exposed, poorly composed, poorly framed, poorly cropped, poorly color corrected, poorly color graded",
                                             "multiline": True, "description": "Negative prompt"}),
                "style": (["none", "anime", "3d_animation", "clay", "comic", "cyberpunk"], {"default": "none",
                                                                                           "description": "Style of the generated video"}),
                "camera_movement": (["none", "horizontal_left", "horizontal_right", "vertical_up", "vertical_down",
                                   "zoom_in", "zoom_out", "crane_up", "quickly_zoom_in", "quickly_zoom_out",
                                   "smooth_zoom_in", "camera_rotation", "robo_arm", "super_dolly_out",
                                   "whip_pan", "hitchcock", "left_follow", "right_follow", "pan_left",
                                   "pan_right", "fix_bg"], {"default": "none",
                                                           "description": "Type of camera movement to apply"})
            }
        }

    def generate_video(self, api_key, prompt="A woman walking through a beautiful landscape", aspect_ratio="16:9",
                       resolution="720p", duration="5", seed=0, batch_size=1, nth_frame=1,
                       image=None, negative_prompt="", style="none", camera_movement="none"):
        # Clear any existing FAL_KEY environment variable to prevent caching issues
        if "FAL_KEY" in os.environ:
            del os.environ["FAL_KEY"]
            print("[Fal Pixverse] Cleared existing FAL_KEY environment variable")
        """
        Generate a video from an image, download it, and extract frames

        Args:
            api_key: Fal AI API key
            prompt: Text prompt describing the video
            negative_prompt: Negative prompt
            duration: Video duration in seconds
            quality: Video quality
            seed: Random seed for video generation (0 = random)
            batch_size: Number of videos to generate with different seeds
            nth_frame: Extract every Nth frame (1=all frames, 2=every 2nd frame, etc.)
            image: (Optional) Input image tensor

        Returns:
            Tuple of (frames_tensor_1, frames_tensor_2, frames_tensor_3, frames_tensor_4, frames_tensor_5,
                     video_urls, status_message)
            Note: If batch_size < 5, the unused frame tensors will be empty (1,1,1,3) tensors
        """
        try:
            # Helper function for error returns
            def error_return(error_msg):
                empty_tensor = torch.zeros((1, 1, 1, 3))
                return empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, "", error_msg
                
            # 1. Validate API key
            if not api_key or api_key.strip() == "":
                return error_return("Error: API Key is required")
                
            # 2. Validate image input
            if image is None:
                return error_return("Error: Input image is required")

            # Initialize return values
            frame_tensors = [torch.zeros((1, 1, 1, 3)) for _ in range(5)]  # 5 empty tensors by default
            video_urls = []
            status_messages = []
            
            # Limit batch size to maximum of 5
            batch_size = min(batch_size, 5)
            
            # Validate duration for 1080p (limited to 5 seconds)
            if resolution == "1080p" and duration == "8":
                print(f"[Fal Pixverse] Warning: 1080p videos are limited to 5 seconds, changing duration from 8s to 5s")
                duration = "5"
            
            # Convert image tensor to base64
            if image is not None:
                # Take first image if batch
                if len(image.shape) == 4:
                    image_tensor = image[0]
                else:
                    image_tensor = image
                
                # Convert to uint8
                if image_tensor.dtype != torch.uint8:
                    image_tensor = (image_tensor * 255).to(torch.uint8)
                
                # Convert to numpy for PIL
                np_img = image_tensor.cpu().numpy()
                
                try:
                    pil_image = Image.fromarray(np_img)
                    print(f"[Fal Pixverse] Successfully converted image tensor to PIL image")

                    # Set API key first - needed for CDN upload
                    clean_api_key = api_key.strip()
                    os.environ["FAL_KEY"] = clean_api_key

                    # Upload image to fal.media CDN to avoid 10MB request body limit
                    print(f"[Fal Pixverse] Uploading image to fal.media CDN...")
                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    image_bytes = buffered.getvalue()
                    img_url = fal_client.upload(image_bytes, content_type="image/png")
                    print(f"[Fal Pixverse] Uploaded image to CDN: {img_url[:80]}...")

                except Exception as e:
                    print(f"[Fal Pixverse] Error: Failed to upload image to CDN: {str(e)}")
                    return error_return(f"Error: Failed to upload image: {str(e)}")
            else:
                return error_return("Error: No image provided")
            
            # Process batches in parallel
            def process_batch(batch_idx):
                try:
                    # Calculate seed for this batch
                    batch_seed = np.random.randint(1, 2147483647) if seed == 0 else seed + batch_idx
                    
                    print(f"[Fal Pixverse] Batch {batch_idx+1}/{batch_size}: Generating video with seed {batch_seed}...")
                    
                    # Prepare the API request
                    # Ensure API key is properly formatted (trim any whitespace)
                    clean_api_key = api_key.strip()
                    
                    # Prepare the arguments for fal_client
                    arguments = {
                        "prompt": prompt,
                        "image_url": img_url,
                        "aspect_ratio": aspect_ratio,
                        "resolution": resolution,
                        "duration": duration,
                        "seed": batch_seed
                    }
                    
                    # Add optional parameters if provided and valid
                    if negative_prompt and negative_prompt.strip():
                        arguments["negative_prompt"] = negative_prompt.strip()
                    
                    if style and style != "none" and style != "":
                        arguments["style"] = style
                    
                    # Camera movement might have resolution restrictions - only add if not none and resolution is compatible
                    if camera_movement and camera_movement != "none" and camera_movement != "":
                        # Some camera movements might have resolution limits - let's be more conservative
                        if resolution in ["360p", "540p"]:
                            arguments["camera_movement"] = camera_movement
                            print(f"[Fal Pixverse] Adding camera_movement: {camera_movement} (resolution: {resolution})")
                        else:
                            print(f"[Fal Pixverse] Skipping camera_movement for {resolution} - may have restrictions")
                    
                    # Keep duration as string per API documentation
                    arguments["duration"] = str(duration)
                    
                    # Remove any None values from arguments
                    arguments = {k: v for k, v in arguments.items() if v is not None}
                    
                    # Print arguments without exposing potentially large base64 data
                    safe_arguments = {k: v if not (isinstance(v, str) and v.startswith('data:')) else f"<data_uri_{len(v)}_chars>" for k, v in arguments.items()}
                    print(f"[Fal Pixverse] API arguments: {safe_arguments}")
                    
                    # Set the API key as an environment variable for fal_client (using cleaned key)
                    # Print the first few characters of the key for debugging (don't print the whole key for security)
                    key_preview = clean_api_key[:8] + "..." if len(clean_api_key) > 8 else "invalid_key"
                    print(f"[Fal Pixverse] Using API key starting with: {key_preview}")
                    
                    # Clear and set the environment variable
                    if "FAL_KEY" in os.environ:
                        del os.environ["FAL_KEY"]
                    os.environ["FAL_KEY"] = clean_api_key
                    
                    print(f"[Fal Pixverse] Calling Fal AI API with fal_client...")
                    
                    # Define a callback for queue updates
                    def on_queue_update(update):
                        if isinstance(update, fal_client.InProgress):
                            for log in update.logs:
                                print(f"[Fal Pixverse] Log: {log['message']}")
                    
                    try:
                        # Use the new v5 endpoint
                        endpoint = "fal-ai/pixverse/v5/image-to-video"
                        print(f"[Fal Pixverse] Using v5 endpoint: {endpoint}")
                        
                        # Force reload the fal_client module to avoid caching issues
                        import sys
                        if 'fal_client' in sys.modules:
                            del sys.modules['fal_client']
                        import fal_client
                        
                        # Make the API call using fal_client.subscribe
                        print(f"[Fal Pixverse] Making API call with fal_client.subscribe...")
                        result = fal_client.subscribe(
                            endpoint,
                            arguments=arguments,
                            with_logs=True,
                            on_queue_update=on_queue_update,
                        )
                        
                        print(f"[Fal Pixverse] API call completed successfully")
                    except Exception as e:
                        error_msg = f"API Error: {str(e)}"
                        print(f"[Fal Pixverse] {error_msg}")
                        return {
                            "batch_idx": batch_idx,
                            "success": False,
                            "error": error_msg
                        }
                    
                    # Extract video URL from the result
                    if "video" in result and "url" in result["video"]:
                        video_url = result["video"]["url"]
                        print(f"[Fal Pixverse] Batch {batch_idx+1}: Video ready! URL: {video_url}")
                        
                        # Download and process the video
                        try:
                            print(f"[Fal Pixverse] Batch {batch_idx+1}: Downloading video...")
                            
                            # Create a temporary file
                            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                                temp_video_path = temp_video.name
                                
                                # Download video to temp file
                                dl_response = requests.get(video_url, stream=True)
                                dl_response.raise_for_status()
                                
                                # Get file size for progress bar
                                file_size = int(dl_response.headers.get('content-length', 0))
                                progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading Batch {batch_idx+1}")
                                
                                for chunk in dl_response.iter_content(chunk_size=8192):
                                    temp_video.write(chunk)
                                    progress_bar.update(len(chunk))
                                    
                                progress_bar.close()
                            
                            # Extract frames using OpenCV
                            print(f"[Fal Pixverse] Batch {batch_idx+1}: Extracting frames from video...")
                            cap = cv2.VideoCapture(temp_video_path)
                            
                            if not cap.isOpened():
                                os.unlink(temp_video_path)  # Clean up temp file
                                return {
                                    "batch_idx": batch_idx,
                                    "success": False,
                                    "error": "Could not open video file"
                                }
                            
                            # Get video properties
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            
                            print(f"[Fal Pixverse] Batch {batch_idx+1}: Video has {total_frames} frames at {fps} FPS")
                            
                            frames = []
                            frame_count = 0
                            
                            # Use nth_frame directly as the stride
                            stride = nth_frame
                            
                            # Calculate approximately how many frames we'll extract
                            frames_to_extract = total_frames // stride + (1 if total_frames % stride > 0 else 0)
                            
                            progress_bar = tqdm(total=frames_to_extract, desc=f"Extracting frames (Batch {batch_idx+1})")
                            
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                    
                                if frame_count % stride == 0 and len(frames) < frames_to_extract:
                                    # Convert BGR to RGB (OpenCV uses BGR by default)
                                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    
                                    # Normalize to 0-1 range for ComfyUI
                                    normalized_frame = rgb_frame.astype(np.float32) / 255.0
                                    
                                    frames.append(normalized_frame)
                                    progress_bar.update(1)
                                    
                                    # Break if we've extracted enough frames
                                    if len(frames) >= frames_to_extract:
                                        break
                                        
                                frame_count += 1
                                
                            progress_bar.close()
                            cap.release()
                            
                            # Clean up temp file
                            os.unlink(temp_video_path)
                            
                            # Convert frames to tensor
                            if frames:
                                frames_tensor = torch.from_numpy(np.stack(frames))
                                print(f"[Fal Pixverse] Batch {batch_idx+1}: Extracted {len(frames)} frames as tensor with shape {frames_tensor.shape}")
                                return {
                                    "batch_idx": batch_idx,
                                    "success": True,
                                    "frames_tensor": frames_tensor,
                                    "video_url": video_url
                                }
                            else:
                                return {
                                    "batch_idx": batch_idx,
                                    "success": False,
                                    "error": "No frames could be extracted"
                                }
                                
                        except Exception as e:
                            return {
                                "batch_idx": batch_idx,
                                "success": False,
                                "error": f"Processing Error: {str(e)}"
                            }
                    else:
                        return {
                            "batch_idx": batch_idx,
                            "success": False,
                            "error": "No video URL in API response"
                        }
                        
                except Exception as e:
                    return {
                        "batch_idx": batch_idx,
                        "success": False,
                        "error": f"Batch processing error: {str(e)}"
                    }
            
            # Process batches in parallel
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_batch = {
                    executor.submit(process_batch, idx): idx
                    for idx in range(batch_size)
                }
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "batch_idx": batch_idx,
                            "success": False,
                            "error": f"Thread Error: {str(e)}"
                        })
            
            # Collect results
            for result in results:
                batch_idx = result["batch_idx"]
                if result["success"]:
                    frame_tensors[batch_idx] = result["frames_tensor"]
                    video_urls.append(f"Batch {batch_idx+1}: {result['video_url']}")
                    status_messages.append(f"Success (Batch {batch_idx+1})")
                else:
                    video_urls.append(f"Batch {batch_idx+1}: Failed")
                    status_messages.append(f"Error (Batch {batch_idx+1}): {result['error']}")
            
            # Combine status messages
            combined_status = " | ".join(status_messages) if status_messages else "No videos processed"
            
            # Combine video URLs
            combined_urls = " | ".join(video_urls) if video_urls else "No videos generated"
            
            # Return the results
            return tuple(frame_tensors + [combined_urls, combined_status])
            
        except Exception as e:
            print(f"[Fal Pixverse] Error: {str(e)}")
            # Try to return proper empty tensors
            empty_tensor = torch.zeros((1, 1, 1, 3))
            return empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, "", f"Error: {str(e)}"