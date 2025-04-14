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
from typing import Tuple
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class FL_PixVerseAPI:
    """
    A ComfyUI node for the PixVerse Image-to-Video API.
    Takes an image and converts it to a video using PixVerse's API.
    Downloads the video, extracts frames, and returns them as image tensors.
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("frames", "video_url", "status_msg")
    FUNCTION = "generate_video"
    CATEGORY = "🏵️Fill Nodes/AI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"default": ""}),
                "duration": ("INT", {"default": 5, "min": 5, "max": 8}),
                "quality": (["360p", "540p", "720p", "1080p"], {"default": "540p"}),
                "motion_mode": (["normal", "fast"], {"default": "normal"}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 300,
                                       "description": "Max frames to extract (0 = all frames)"})
            }
        }

    def generate_video(self, api_key, image, prompt="", negative_prompt="", duration=5,
                       quality="540p", motion_mode="normal", max_frames=0) -> Tuple[torch.Tensor, str, str]:
        """
        Generate a video from an image, download it, and extract frames

        Args:
            api_key: PixVerse API key
            image: Input image tensor
            prompt: Text prompt describing the video
            negative_prompt: Negative prompt
            duration: Video duration in seconds
            quality: Video quality
            motion_mode: Motion speed
            max_frames: Maximum number of frames to extract (0 = all frames)

        Returns:
            Tuple of (frames_tensor, video_url, status_message)
        """
        try:
            # 1. Validate API key
            if not api_key or api_key.strip() == "":
                return torch.zeros((1, 1, 1, 3)), "", "Error: API Key is required"

            # 2. Generate trace ID
            trace_id = str(uuid.uuid4())

            # 3. Convert tensor to PIL image
            print(f"[PixVerse] Processing image tensor with shape: {image.shape}")

            # Take first image if batch
            if len(image.shape) == 4:
                image = image[0]

            # Convert to uint8
            if image.dtype != torch.uint8:
                image = (image * 255).to(torch.uint8)

            # Convert to numpy for PIL
            np_img = image.cpu().numpy()

            try:
                pil_image = Image.fromarray(np_img)
                print("[PixVerse] Successfully converted tensor to PIL image")
            except Exception as e:
                return torch.zeros((1, 1, 1, 3)), "", f"Error: Failed to convert image tensor to PIL image: {str(e)}"

            # 4. Upload image to PixVerse using BytesIO instead of temp file
            print("[PixVerse] Uploading image to PixVerse...")

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
            files = {'image': ('image.png', img_byte_arr, 'image/png')}
            upload_response = requests.post(upload_url, headers=upload_headers, files=files)

            if upload_response.status_code != 200:
                return torch.zeros(
                    (1, 1, 1, 3)), "", f"Error: Failed to upload image. HTTP Status: {upload_response.status_code}"

            upload_result = upload_response.json()
            if upload_result.get("ErrCode", -1) != 0:
                return torch.zeros(
                    (1, 1, 1, 3)), "", f"Error uploading image: {upload_result.get('ErrMsg', 'Unknown error')}"

            img_id = upload_result["Resp"]["img_id"]
            print(f"[PixVerse] Image uploaded successfully. Image ID: {img_id}")

            # 5. Call the image-to-video API
            print("[PixVerse] Generating video from image...")

            # Create API request
            conn = http.client.HTTPSConnection("app-api.pixverse.ai")
            payload = json.dumps({
                "duration": duration,
                "img_id": img_id,
                "model": "v3.5",
                "motion_mode": motion_mode,
                "negative_prompt": negative_prompt,
                "prompt": prompt,
                "quality": quality,
                "seed": 0,
                "template_id": 0,
                "water_mark": False
            })

            headers = {
                'API-KEY': api_key,
                'Ai-trace-id': trace_id,
                'Content-Type': 'application/json'
            }

            conn.request("POST", "/openapi/v2/video/img/generate", payload, headers)
            response = conn.getresponse()
            data = response.read().decode("utf-8")
            result = json.loads(data)

            if result.get("ErrCode", -1) != 0:
                return torch.zeros((1, 1, 1, 3)), "", f"API Error: {result.get('ErrMsg', 'Unknown error')}"

            video_id = result["Resp"]["video_id"]
            print(f"[PixVerse] Video generation initiated with ID: {video_id}")

            # 6. Poll for completion
            print("[PixVerse] Polling for video completion...")
            max_polls = 60  # 5 minutes with 5-second intervals
            poll_count = 0
            video_url = ""

            while poll_count < max_polls:
                time.sleep(5)
                poll_count += 1

                poll_conn = http.client.HTTPSConnection("app-api.pixverse.ai")
                poll_conn.request("GET", f"/openapi/v2/video/result/{video_id}", headers={
                    'API-KEY': api_key,
                    'Ai-trace-id': trace_id
                })

                poll_response = poll_conn.getresponse()
                poll_data = poll_response.read().decode("utf-8")
                poll_result = json.loads(poll_data)

                if poll_result.get("ErrCode", -1) != 0:
                    return torch.zeros((1, 1, 1, 3)), "", f"Polling Error: {poll_result.get('ErrMsg', 'Unknown error')}"

                status = poll_result["Resp"]["status"]

                if status == 1:  # Success
                    video_url = poll_result["Resp"]["url"]
                    print(f"[PixVerse] Video ready! URL: {video_url}")
                    break

                elif status in [2, 3, 4]:  # Failed, timeout, rejected
                    status_messages = {
                        2: "Failed",
                        3: "Timeout",
                        4: "Rejected"
                    }
                    return torch.zeros((1, 1, 1, 3)), "", f"Video generation {status_messages.get(status)}"

                print(f"[PixVerse] Video still processing... (poll {poll_count}/{max_polls})")

            if not video_url:
                return torch.zeros((1, 1, 1, 3)), "", "Polling timed out - video may still be processing"

            # 7. Download the video to a temporary file
            print("[PixVerse] Downloading video...")

            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video_path = temp_video.name

                # Download video to temp file
                response = requests.get(video_url, stream=True)
                response.raise_for_status()

                # Get file size for progress bar
                file_size = int(response.headers.get('content-length', 0))
                progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading")

                for chunk in response.iter_content(chunk_size=8192):
                    temp_video.write(chunk)
                    progress_bar.update(len(chunk))

                progress_bar.close()

            # 8. Extract frames using OpenCV
            print("[PixVerse] Extracting frames from video...")
            cap = cv2.VideoCapture(temp_video_path)

            if not cap.isOpened():
                os.unlink(temp_video_path)  # Clean up temp file
                return torch.zeros((1, 1, 1, 3)), video_url, "Error: Could not open video file"

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"[PixVerse] Video has {total_frames} frames at {fps} FPS")

            frames = []
            frame_count = 0

            # Determine how many frames to extract
            frames_to_extract = total_frames if max_frames == 0 else min(max_frames, total_frames)

            # Calculate stride if we need to skip frames
            stride = max(1, total_frames // frames_to_extract) if frames_to_extract < total_frames else 1

            progress_bar = tqdm(total=frames_to_extract, desc="Extracting frames")

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

            # Convert frames to tensor in format expected by ComfyUI (B,H,W,C)
            if frames:
                frames_tensor = torch.from_numpy(np.stack(frames))
                print(f"[PixVerse] Extracted {len(frames)} frames as tensor with shape {frames_tensor.shape}")
                return frames_tensor, video_url, "Success"
            else:
                return torch.zeros((1, 1, 1, 3)), video_url, "Warning: No frames could be extracted"

        except Exception as e:
            print(f"[PixVerse] Error: {str(e)}")
            # Try to return a proper empty tensor
            return torch.zeros((1, 1, 1, 3)), "", f"Error: {str(e)}"