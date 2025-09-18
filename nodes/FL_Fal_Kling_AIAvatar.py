# FL_Fal_Kling_AIAvatar: Fal AI Kling Video AI Avatar API Node
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
import base64
import fal_client
import torchaudio
from typing import Tuple, List, Dict, Union, Optional
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class FL_Fal_Kling_AIAvatar:
    """
    A ComfyUI node for the Fal AI Kling Video AI Avatar API.
    Takes an image and audio to generate realistic avatar videos.
    """

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("frames", "audio", "video_url", "status_msg")
    FUNCTION = "generate_ai_avatar"
    CATEGORY = "🏵️Fill Nodes/AI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False,
                                      "description": "Fal AI API key"}),
                "image": ("IMAGE", {"description": "Input image to use as avatar"}),
                "audio": ("AUDIO", {"description": "Input audio tensor"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999,
                               "description": "Random seed (0 = random, max 6 digits)"}),
                "nth_frame": ("INT", {"default": 1, "min": 1, "max": 4,
                                     "description": "Extract every Nth frame (1=all frames, 2=every 2nd frame, etc.)"})
            },
            "optional": {
                "prompt": ("STRING", {"default": "", "multiline": True,
                                    "description": "Optional text prompt to refine video generation"})
            }
        }

    def sanitize_error_message(self, msg):
        """Remove potentially large base64 data from error messages"""
        if not isinstance(msg, str):
            msg = str(msg)
        # Replace data URIs with placeholders to avoid wall of text
        import re
        msg = re.sub(r'data:[^;]+;base64,[A-Za-z0-9+/=]{100,}', '<data_uri_removed>', msg)
        return msg

    def generate_ai_avatar(self, api_key, image, audio, seed=0, nth_frame=1, prompt=""):
        """
        Generate AI Avatar video using Fal AI Kling Video AI Avatar API

        Args:
            api_key: Fal AI API key
            image: Input image tensor to use as avatar
            audio: Input audio tensor
            seed: Random seed for generation (0 = random, max 6 digits)
            nth_frame: Extract every Nth frame
            prompt: Optional text prompt to refine video generation

        Returns:
            Tuple of (frames_tensor, audio, video_url, status_message)
        """
        try:
            # Helper function for error returns
            def error_return(error_msg):
                empty_tensor = torch.zeros((1, 1, 1, 3))
                empty_audio = {"waveform": torch.zeros((1, 1, 0)), "sample_rate": 44100}
                clean_msg = self.sanitize_error_message(error_msg)
                return empty_tensor, empty_audio, "", clean_msg

            # Clear any existing FAL_KEY environment variable to prevent caching issues
            if "FAL_KEY" in os.environ:
                del os.environ["FAL_KEY"]
                print("[Fal Kling AI Avatar] Cleared existing FAL_KEY environment variable")

            # 1. Validate API key
            if not api_key or api_key.strip() == "":
                return error_return("Error: API Key is required")

            # 2. Validate required inputs
            if image is None:
                return error_return("Error: Image input is required")

            if audio is None:
                return error_return("Error: Audio input is required")

            # 3. Process seed (ensure it's within 6-digit limit)
            if seed < 0 or seed > 999999:
                return error_return("Error: Seed must be between 0 and 999999 (6 digits max)")

            # 4. Set up tensors for processing after API key setup
            audio_tensor_to_process = audio
            image_tensor_to_process = image
            print(f"[Fal Kling AI Avatar] Image and audio tensors provided, will process after API setup")
            print(f"[Fal Kling AI Avatar] Using seed: {seed}")

            print(f"[Fal Kling AI Avatar] Starting AI Avatar generation...")

            # Prepare the API request
            clean_api_key = api_key.strip()

            # Prepare the arguments for fal_client (URLs will be set after upload)
            arguments = {
                "image_url": "",  # Will be set after image upload
                "audio_url": ""   # Will be set after audio upload
            }

            # Add seed if specified (0 means random)
            if seed > 0:
                arguments["seed"] = seed

            # Add optional prompt if provided
            if prompt and prompt.strip():
                arguments["prompt"] = prompt.strip()
                print(f"[Fal Kling AI Avatar] Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

            # Remove any None values from arguments
            arguments = {k: v for k, v in arguments.items() if v is not None and v != ""}

            # Print arguments without exposing potentially large base64 data
            safe_arguments = {k: v if not (isinstance(v, str) and v.startswith('data:')) else f"<data_uri_{len(v)}_chars>" for k, v in arguments.items()}
            print(f"[Fal Kling AI Avatar] API arguments: {safe_arguments}")

            # Set the API key as an environment variable for fal_client
            key_preview = clean_api_key[:8] + "..." if len(clean_api_key) > 8 else "invalid_key"
            print(f"[Fal Kling AI Avatar] Using API key starting with: {key_preview}")

            # Clear and set the environment variable
            if "FAL_KEY" in os.environ:
                del os.environ["FAL_KEY"]
            os.environ["FAL_KEY"] = clean_api_key

            print(f"[Fal Kling AI Avatar] Calling Fal AI API with fal_client...")

            # Define a callback for queue updates
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        print(f"[Fal Kling AI Avatar] Log: {log['message']}")

            try:
                # Use the Kling AI Avatar endpoint
                endpoint = "fal-ai/kling-video/v1/pro/ai-avatar"
                print(f"[Fal Kling AI Avatar] Using endpoint: {endpoint}")

                # Force reload the fal_client module to avoid caching issues
                import sys
                if 'fal_client' in sys.modules:
                    del sys.modules['fal_client']
                import fal_client

                # Process audio tensor upload if needed (after fal_client is properly loaded)
                if audio_tensor_to_process is not None:
                    try:
                        # Convert audio tensor to temporary file and upload to Fal
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                            temp_audio_path = temp_audio.name

                        # Save audio tensor to temporary file
                        waveform = audio_tensor_to_process['waveform']
                        sample_rate = audio_tensor_to_process['sample_rate']

                        # Ensure waveform is in correct format for torchaudio.save
                        if len(waveform.shape) == 3:  # [batch, channels, samples]
                            waveform = waveform.squeeze(0)  # Remove batch dimension

                        torchaudio.save(temp_audio_path, waveform, sample_rate)
                        print(f"[Fal Kling AI Avatar] Saved audio tensor to temporary file: {temp_audio_path}")

                        # Upload to Fal
                        final_audio_url = fal_client.upload_file(temp_audio_path)
                        print(f"[Fal Kling AI Avatar] Uploaded audio to Fal: {final_audio_url}")

                        # Update arguments with the uploaded audio URL
                        arguments["audio_url"] = final_audio_url

                        # Clean up temporary file
                        os.unlink(temp_audio_path)

                    except Exception as e:
                        print(f"[Fal Kling AI Avatar] Error processing audio tensor: {str(e)}")
                        return error_return(f"Error: Failed to process audio: {str(e)}")

                # Process image tensor upload if needed (after fal_client is properly loaded)
                if image_tensor_to_process is not None:
                    try:
                        # Convert image tensor to temporary file and upload to Fal
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_image:
                            temp_image_path = temp_image.name

                        # Convert image tensor to PIL Image
                        if len(image_tensor_to_process.shape) == 4:
                            image_tensor = image_tensor_to_process[0]
                        else:
                            image_tensor = image_tensor_to_process

                        # Convert to uint8
                        if image_tensor.dtype != torch.uint8:
                            image_tensor = (image_tensor * 255).to(torch.uint8)

                        # Convert to numpy for PIL
                        np_img = image_tensor.cpu().numpy()
                        pil_image = Image.fromarray(np_img)

                        # Save to temporary file
                        pil_image.save(temp_image_path, format="PNG")
                        print(f"[Fal Kling AI Avatar] Saved image tensor to temporary file: {temp_image_path}")

                        # Upload to Fal
                        final_image_url = fal_client.upload_file(temp_image_path)
                        print(f"[Fal Kling AI Avatar] Uploaded image to Fal: {final_image_url}")

                        # Update arguments with the uploaded image URL
                        arguments["image_url"] = final_image_url

                        # Clean up temporary file
                        os.unlink(temp_image_path)

                    except Exception as e:
                        print(f"[Fal Kling AI Avatar] Error processing image tensor: {str(e)}")
                        return error_return(f"Error: Failed to process image: {str(e)}")

                # Make the API call using fal_client.subscribe
                print(f"[Fal Kling AI Avatar] Making API call with fal_client.subscribe...")
                result = fal_client.subscribe(
                    endpoint,
                    arguments=arguments,
                    with_logs=True,
                    on_queue_update=on_queue_update,
                )

                print(f"[Fal Kling AI Avatar] API call completed successfully")
            except Exception as e:
                error_msg = f"API Error: {str(e)}"
                print(f"[Fal Kling AI Avatar] {error_msg}")
                return error_return(error_msg)

            # Extract video URL from the result
            if "video" in result and "url" in result["video"]:
                output_video_url = result["video"]["url"]
                print(f"[Fal Kling AI Avatar] Video ready! URL: {output_video_url}")

                # Download and process the video
                try:
                    print(f"[Fal Kling AI Avatar] Downloading video...")

                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                        temp_video_path = temp_video.name

                        # Download video to temp file
                        dl_response = requests.get(output_video_url, stream=True)
                        dl_response.raise_for_status()

                        # Get file size for progress bar
                        file_size = int(dl_response.headers.get('content-length', 0))
                        progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading AI Avatar Video")

                        for chunk in dl_response.iter_content(chunk_size=8192):
                            temp_video.write(chunk)
                            progress_bar.update(len(chunk))

                        progress_bar.close()

                    # Extract frames using OpenCV
                    print(f"[Fal Kling AI Avatar] Extracting frames from video...")
                    cap = cv2.VideoCapture(temp_video_path)

                    if not cap.isOpened():
                        os.unlink(temp_video_path)  # Clean up temp file
                        return error_return("Could not open video file")

                    # Get video properties
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    print(f"[Fal Kling AI Avatar] Video has {total_frames} frames at {fps} FPS")

                    frames = []
                    frame_count = 0

                    # Use nth_frame directly as the stride
                    stride = nth_frame

                    # Calculate approximately how many frames we'll extract
                    frames_to_extract = total_frames // stride + (1 if total_frames % stride > 0 else 0)

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

                    # Convert frames to tensor
                    if frames:
                        frames_tensor = torch.from_numpy(np.stack(frames))
                        print(f"[Fal Kling AI Avatar] Extracted {len(frames)} frames as tensor with shape {frames_tensor.shape}")

                        # Prepare audio output - use the original input audio if provided
                        output_audio = audio if audio is not None else {"waveform": torch.zeros((1, 1, 0)), "sample_rate": 44100}

                        return frames_tensor, output_audio, output_video_url, "Success: AI Avatar video generated and processed"
                    else:
                        return error_return("No frames could be extracted")

                except Exception as e:
                    return error_return(f"Processing Error: {str(e)}")
            else:
                return error_return("No video URL in API response")

        except Exception as e:
            print(f"[Fal Kling AI Avatar] Error: {str(e)}")
            # Try to return proper empty tensors
            empty_tensor = torch.zeros((1, 1, 1, 3))
            empty_audio = {"waveform": torch.zeros((1, 1, 0)), "sample_rate": 44100}
            # Sanitize error message to remove potential base64 data
            clean_error = self.sanitize_error_message(f"Error: {str(e)}")
            return empty_tensor, empty_audio, "", clean_error