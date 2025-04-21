# FL_PixVerseImageToVideo: Enhanced PixVerse Image-to-Video API Node with frame decomposition
import os
import uuid
import json
import time
import io
import requests
import http.client
import torch
import numpy as np
import tempfile
import cv2
import concurrent.futures
from typing import Tuple, List, Dict
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class FL_PixVerseAPI:
    """
    A ComfyUI node for the PixVerse Image-to-Video API.
    Takes an image and converts it to a video using PixVerse's API.
    Downloads the video, extracts frames, and returns them as image tensors.
    """

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("frames_1", "frames_2", "frames_3", "frames_4", "frames_5", "video_urls", "status_msg", "credit_balance")
    FUNCTION = "generate_video"
    CATEGORY = "🏵️Fill Nodes/AI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"default": ""}),
                "duration": ("INT", {"default": 5, "min": 5, "max": 8}),
                "quality": (["360p", "540p", "720p", "1080p"], {"default": "540p"}),
                "motion_mode": (["normal", "fast"], {"default": "normal"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647,
                                "description": "Random seed for video generation (0 = random)"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 5,
                                      "description": "Number of videos to generate with different seeds"}),
                "nth_frame": ("INT", {"default": 1, "min": 1, "max": 4,
                                     "description": "Extract every Nth frame (1=all frames, 2=every 2nd frame, etc.)"}),
                "use_transition": ("BOOLEAN", {"default": False,
                                              "description": "Use transition API instead of standard image-to-video"})
            },
            "optional": {
                "image": ("IMAGE", {"description": "Main image for standard mode (required if not using transition)"}),
                "first_frame_img": ("IMAGE", {"description": "Start frame image for transition (requires use_transition=True)"}),
                "last_frame_img": ("IMAGE", {"description": "End frame image for transition (requires use_transition=True)"})
            }
        }

    def generate_video(self, api_key, prompt="", negative_prompt="", duration=5,
                       quality="540p", motion_mode="normal", seed=0, batch_size=1, nth_frame=1,
                       use_transition=False, image=None, first_frame_img=None, last_frame_img=None):
        """
        Generate a video from an image, download it, and extract frames

        Args:
            api_key: PixVerse API key
            prompt: Text prompt describing the video
            negative_prompt: Negative prompt
            duration: Video duration in seconds
            quality: Video quality
            motion_mode: Motion speed
            seed: Random seed for video generation (0 = random)
            batch_size: Number of videos to generate with different seeds
            nth_frame: Extract every Nth frame (1=all frames, 2=every 2nd frame, etc.)
            use_transition: Whether to use transition API instead of standard image-to-video
            image: (Optional) Main input image tensor
            first_frame_img: (Optional) Start frame image for transition
            last_frame_img: (Optional) End frame image for transition

        Returns:
            Tuple of (frames_tensor_1, frames_tensor_2, frames_tensor_3, frames_tensor_4, frames_tensor_5,
                     video_urls, status_message, credit_balance)
            Note: If batch_size < 5, the unused frame tensors will be empty (1,1,1,3) tensors
        """
        try:
            # Helper function for error returns
            def error_return(error_msg):
                empty_tensor = torch.zeros((1, 1, 1, 3))
                return empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, "", error_msg, "N/A"
                
            # 1. Validate API key
            if not api_key or api_key.strip() == "":
                return error_return("Error: API Key is required")
                
            # 2. Validate image inputs based on mode
            if use_transition:
                # Transition mode validation
                if first_frame_img is None and last_frame_img is None:
                    return error_return("Error: When using transition mode, at least one of first_frame_img or last_frame_img must be provided")
                
                # For transition mode, we need either:
                # 1. Main image (which can be used for missing frames), or
                # 2. Both first_frame_img and last_frame_img
                if image is None and (first_frame_img is None or last_frame_img is None):
                    return error_return("Error: For transition mode without main image, both first_frame_img AND last_frame_img must be provided")
            else:
                # Standard mode validation
                if image is None:
                    return error_return("Error: Main image is required when not using transition mode")

            # Initialize return values
            frame_tensors = [torch.zeros((1, 1, 1, 3)) for _ in range(5)]  # 5 empty tensors by default
            video_urls = []
            status_messages = []
            
            # Limit batch size to maximum of 5
            batch_size = min(batch_size, 5)
            
            # Generate trace ID for the main request
            main_trace_id = str(uuid.uuid4())
            
            # 3. Upload the main image if provided
            img_id = 0
            if image is not None:
                print(f"[PixVerse] Processing and uploading main image...")
                img_id = self.upload_image(api_key, image, main_trace_id, "main image")
                
                if img_id == 0:
                    return error_return("Error: Failed to upload main image")
            
            # 4. Process batches in parallel
            # Define helper functions for parallel processing
            def generate_video_request(batch_idx, batch_seed, img_id):
                """Make the initial API call to generate a video"""
                trace_id = str(uuid.uuid4())
                print(f"[PixVerse] Batch {batch_idx+1}/{batch_size}: Generating video with seed {batch_seed}...")
                
                conn = http.client.HTTPSConnection("app-api.pixverse.ai")
                
                # Determine which API endpoint to use based on use_transition flag
                if use_transition:
                    # Process first and last frame images
                    first_frame_id = 0
                    last_frame_id = 0
                    
                    # For transition mode, if main image is provided but first/last frame is not,
                    # use the main image for the missing frame(s)
                    
                    # Handle first frame
                    if first_frame_img is not None:
                        first_frame_id = self.upload_image(api_key, first_frame_img, trace_id, "first frame")
                        if first_frame_id == 0:  # Upload failed
                            return {
                                "batch_idx": batch_idx,
                                "success": False,
                                "error": "Failed to upload first frame image",
                                "trace_id": trace_id
                            }
                    elif img_id > 0:  # Use main image if first frame not provided
                        first_frame_id = img_id
                        print(f"[PixVerse] Using main image as first frame")
                    
                    # Handle last frame
                    if last_frame_img is not None:
                        last_frame_id = self.upload_image(api_key, last_frame_img, trace_id, "last frame")
                        if last_frame_id == 0:  # Upload failed
                            return {
                                "batch_idx": batch_idx,
                                "success": False,
                                "error": "Failed to upload last frame image",
                                "trace_id": trace_id
                            }
                    elif img_id > 0:  # Use main image if last frame not provided
                        last_frame_id = img_id
                        print(f"[PixVerse] Using main image as last frame")
                    
                    # Use transition API
                    endpoint = "/openapi/v2/video/transition/generate"
                    payload = json.dumps({
                        "prompt": prompt,
                        "model": "v3.5",
                        "duration": duration,
                        "quality": quality,
                        "motion_mode": motion_mode,
                        "seed": batch_seed,
                        "first_frame_img": first_frame_id,
                        "last_frame_img": last_frame_id,
                        "negative_prompt": negative_prompt,
                        "water_mark": False
                    })
                else:
                    # Use standard image-to-video API
                    # This should never happen due to our validation, but just in case
                    if img_id == 0:
                        return {
                            "batch_idx": batch_idx,
                            "success": False,
                            "error": "No valid image provided for standard mode",
                            "trace_id": trace_id
                        }
                        
                    endpoint = "/openapi/v2/video/img/generate"
                    payload = json.dumps({
                        "duration": duration,
                        "img_id": img_id,
                        "model": "v3.5",
                        "motion_mode": motion_mode,
                        "negative_prompt": negative_prompt,
                        "prompt": prompt,
                        "quality": quality,
                        "seed": batch_seed,
                        "template_id": 0,
                        "water_mark": False
                    })
                
                headers = {
                    'API-KEY': api_key,
                    'Ai-trace-id': trace_id,
                    'Content-Type': 'application/json'
                }
                
                try:
                    conn.request("POST", endpoint, payload, headers)
                    response = conn.getresponse()
                    data = response.read().decode("utf-8")
                    result = json.loads(data)
                    
                    if result.get("ErrCode", -1) != 0:
                        return {
                            "batch_idx": batch_idx,
                            "success": False,
                            "error": f"API Error: {result.get('ErrMsg', 'Unknown error')}",
                            "trace_id": trace_id
                        }
                    
                    video_id = result["Resp"]["video_id"]
                    print(f"[PixVerse] Batch {batch_idx+1}: Video generation initiated with ID: {video_id}")
                    
                    return {
                        "batch_idx": batch_idx,
                        "success": True,
                        "video_id": video_id,
                        "trace_id": trace_id
                    }
                except Exception as e:
                    return {
                        "batch_idx": batch_idx,
                        "success": False,
                        "error": f"Request Error: {str(e)}",
                        "trace_id": trace_id
                    }
            
            def poll_video_completion(batch_idx, video_id, trace_id):
                """Poll for video completion"""
                print(f"[PixVerse] Batch {batch_idx+1}: Polling for video completion...")
                max_polls = 60  # 5 minutes with 5-second intervals
                poll_count = 0
                
                while poll_count < max_polls:
                    time.sleep(5)
                    poll_count += 1
                    
                    poll_conn = http.client.HTTPSConnection("app-api.pixverse.ai")
                    poll_conn.request("GET", f"/openapi/v2/video/result/{video_id}", headers={
                        'API-KEY': api_key,
                        'Ai-trace-id': trace_id
                    })
                    
                    try:
                        poll_response = poll_conn.getresponse()
                        poll_data = poll_response.read().decode("utf-8")
                        poll_result = json.loads(poll_data)
                        
                        if poll_result.get("ErrCode", -1) != 0:
                            return {
                                "batch_idx": batch_idx,
                                "success": False,
                                "error": f"Polling Error: {poll_result.get('ErrMsg', 'Unknown error')}"
                            }
                        
                        status = poll_result["Resp"]["status"]
                        
                        if status == 1:  # Success
                            video_url = poll_result["Resp"]["url"]
                            print(f"[PixVerse] Batch {batch_idx+1}: Video ready! URL: {video_url}")
                            return {
                                "batch_idx": batch_idx,
                                "success": True,
                                "video_url": video_url
                            }
                        
                        elif status in [2, 3, 4]:  # Failed, timeout, rejected
                            status_messages = {
                                2: "Failed",
                                3: "Timeout",
                                4: "Rejected"
                            }
                            return {
                                "batch_idx": batch_idx,
                                "success": False,
                                "error": f"Video generation {status_messages.get(status)}"
                            }
                        
                        print(f"[PixVerse] Batch {batch_idx+1}: Video still processing... (poll {poll_count}/{max_polls})")
                    
                    except Exception as e:
                        return {
                            "batch_idx": batch_idx,
                            "success": False,
                            "error": f"Polling Error: {str(e)}"
                        }
                
                return {
                    "batch_idx": batch_idx,
                    "success": False,
                    "error": "Polling timed out - video may still be processing"
                }
            
            def process_video(batch_idx, video_url):
                """Download and process the video"""
                try:
                    print(f"[PixVerse] Batch {batch_idx+1}: Downloading video...")
                    
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                        temp_video_path = temp_video.name
                        
                        # Download video to temp file
                        response = requests.get(video_url, stream=True)
                        response.raise_for_status()
                        
                        # Get file size for progress bar
                        file_size = int(response.headers.get('content-length', 0))
                        progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading Batch {batch_idx+1}")
                        
                        for chunk in response.iter_content(chunk_size=8192):
                            temp_video.write(chunk)
                            progress_bar.update(len(chunk))
                            
                        progress_bar.close()
                    
                    # Extract frames using OpenCV
                    print(f"[PixVerse] Batch {batch_idx+1}: Extracting frames from video...")
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
                    
                    print(f"[PixVerse] Batch {batch_idx+1}: Video has {total_frames} frames at {fps} FPS")
                    
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
                        print(f"[PixVerse] Batch {batch_idx+1}: Extracted {len(frames)} frames as tensor with shape {frames_tensor.shape}")
                        return {
                            "batch_idx": batch_idx,
                            "success": True,
                            "frames_tensor": frames_tensor
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
            
            # Prepare batch parameters
            batch_params = []
            for batch_idx in range(batch_size):
                # Calculate seed for this batch
                batch_seed = np.random.randint(1, 2147483647) if seed == 0 else seed + batch_idx
                batch_params.append((batch_idx, batch_seed))
            
            # Step 1: Make all API calls in parallel
            print(f"[PixVerse] Making {batch_size} API calls in parallel...")
            api_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_batch = {
                    executor.submit(generate_video_request, idx, seed_val, img_id): (idx, seed_val)
                    for idx, seed_val in batch_params
                }
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_idx, batch_seed = future_to_batch[future]
                    try:
                        result = future.result()
                        api_results.append(result)
                    except Exception as e:
                        api_results.append({
                            "batch_idx": batch_idx,
                            "success": False,
                            "error": f"Thread Error: {str(e)}"
                        })
            
            # Step 2: Poll for completion in parallel
            print(f"[PixVerse] Polling for {len(api_results)} videos in parallel...")
            poll_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_batch = {}
                
                for result in api_results:
                    if result["success"]:
                        future = executor.submit(
                            poll_video_completion,
                            result["batch_idx"],
                            result["video_id"],
                            result["trace_id"]
                        )
                        future_to_batch[future] = result["batch_idx"]
                    else:
                        # If API call failed, add the error to poll results
                        poll_results.append({
                            "batch_idx": result["batch_idx"],
                            "success": False,
                            "error": result["error"]
                        })
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        result = future.result()
                        poll_results.append(result)
                    except Exception as e:
                        poll_results.append({
                            "batch_idx": batch_idx,
                            "success": False,
                            "error": f"Thread Error: {str(e)}"
                        })
            
            # Step 3: Process videos in parallel
            print(f"[PixVerse] Processing {len(poll_results)} videos in parallel...")
            process_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_batch = {}
                
                for result in poll_results:
                    if result["success"]:
                        future = executor.submit(
                            process_video,
                            result["batch_idx"],
                            result["video_url"]
                        )
                        future_to_batch[future] = result["batch_idx"]
                    else:
                        # If polling failed, add the error to process results
                        process_results.append({
                            "batch_idx": result["batch_idx"],
                            "success": False,
                            "error": result["error"]
                        })
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        result = future.result()
                        process_results.append(result)
                    except Exception as e:
                        process_results.append({
                            "batch_idx": batch_idx,
                            "success": False,
                            "error": f"Thread Error: {str(e)}"
                        })
            
            # Step 4: Collect results
            for result in process_results:
                batch_idx = result["batch_idx"]
                if result["success"]:
                    frame_tensors[batch_idx] = result["frames_tensor"]
                    video_urls.append(f"Batch {batch_idx+1}: Success")
                    status_messages.append(f"Success (Batch {batch_idx+1})")
                else:
                    video_urls.append(f"Batch {batch_idx+1}: Failed")
                    status_messages.append(f"Error (Batch {batch_idx+1}): {result['error']}")
            
            # 9. Get account balance
            credit_balance = self.get_account_balance(api_key, main_trace_id)
            
            # Combine status messages
            combined_status = " | ".join(status_messages) if status_messages else "No videos processed"
            
            # Combine video URLs
            combined_urls = " | ".join(video_urls) if video_urls else "No videos generated"
            
            # Return the results
            return tuple(frame_tensors + [combined_urls, combined_status, credit_balance])
            
        except Exception as e:
            print(f"[PixVerse] Error: {str(e)}")
            # Try to return proper empty tensors
            empty_tensor = torch.zeros((1, 1, 1, 3))
            return empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, "", f"Error: {str(e)}", "N/A"
            
    def upload_image(self, api_key, image_tensor, trace_id, image_type="image"):
        """
        Upload an image to PixVerse and return the image ID
        
        Args:
            api_key: PixVerse API key
            image_tensor: Image tensor to upload (can be None)
            trace_id: Trace ID for the request
            image_type: Type of image being uploaded (for logging)
            
        Returns:
            Image ID if successful, 0 if failed
        """
        try:
            # Check if image_tensor is None
            if image_tensor is None:
                print(f"[PixVerse] Error: {image_type} is None")
                return 0
                
            # Take first image if batch
            if len(image_tensor.shape) == 4:
                image_tensor = image_tensor[0]
                
            # Convert to uint8
            if image_tensor.dtype != torch.uint8:
                image_tensor = (image_tensor * 255).to(torch.uint8)
                
            # Convert to numpy for PIL
            np_img = image_tensor.cpu().numpy()
            
            try:
                pil_image = Image.fromarray(np_img)
                print(f"[PixVerse] Successfully converted {image_type} tensor to PIL image")
            except Exception as e:
                print(f"[PixVerse] Error: Failed to convert {image_type} tensor to PIL image: {str(e)}")
                return 0
                
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)  # Reset pointer to beginning of buffer
            
            upload_url = "https://app-api.pixverse.ai/openapi/v2/image/upload"
            upload_headers = {
                'API-KEY': api_key,
                'Ai-trace-id': trace_id
            }
            
            # Send bytes directly without saving to disk
            files = {'image': (f'{image_type}.png', img_byte_arr, 'image/png')}
            upload_response = requests.post(upload_url, headers=upload_headers, files=files)
            
            if upload_response.status_code != 200:
                print(f"[PixVerse] Error: Failed to upload {image_type}. HTTP Status: {upload_response.status_code}")
                return 0
                
            upload_result = upload_response.json()
            if upload_result.get("ErrCode", -1) != 0:
                print(f"[PixVerse] Error uploading {image_type}: {upload_result.get('ErrMsg', 'Unknown error')}")
                return 0
                
            img_id = upload_result["Resp"]["img_id"]
            print(f"[PixVerse] {image_type.capitalize()} uploaded successfully. Image ID: {img_id}")
            return img_id
            
        except Exception as e:
            print(f"[PixVerse] Error uploading {image_type}: {str(e)}")
            return 0
            
    def get_account_balance(self, api_key, trace_id):
        """
        Get the account balance from the PixVerse API
        
        Args:
            api_key: PixVerse API key
            trace_id: Trace ID for the request
            
        Returns:
            String representation of the credit balance or error message
        """
        try:
            # Create connection to PixVerse API
            conn = http.client.HTTPSConnection("app-api.pixverse.ai")
            
            # Set up headers
            headers = {
                'API-KEY': api_key,
                'Ai-trace-id': trace_id
            }
            
            # Make the request
            conn.request("GET", "/openapi/v2/account/balance", "", headers)
            
            # Get the response
            response = conn.getresponse()
            data = response.read().decode("utf-8")
            
            # Parse the response
            result = json.loads(data)
            
            if result.get("ErrCode", -1) == 0:
                # Extract balance information
                balance = result.get("Resp", {}).get("balance", "Unknown")
                print(f"[PixVerse] Account balance: {balance}")
                return f"Credits: {balance}"
            else:
                error_msg = result.get("ErrMsg", "Unknown error")
                print(f"[PixVerse] Error getting balance: {error_msg}")
                return f"Balance error: {error_msg}"
                
        except Exception as e:
            print(f"[PixVerse] Error getting balance: {str(e)}")
            return f"Balance error: {str(e)}"