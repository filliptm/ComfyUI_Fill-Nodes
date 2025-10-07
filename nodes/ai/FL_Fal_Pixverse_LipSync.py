# FL_Fal_Pixverse_LipSync: Fal AI Pixverse LipSync API Node
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


class FL_Fal_Pixverse_LipSync:
    """
    A ComfyUI node for the Fal AI Pixverse LipSync API.
    Takes a video and audio/text and generates realistic lipsync animations.
    """

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("frames", "audio", "video_url", "status_msg")
    FUNCTION = "generate_lipsync"
    CATEGORY = "🏵️Fill Nodes/AI"

    @classmethod
    def INPUT_TYPES(cls):
        voice_options = [
            "Auto", "Emily", "James", "Isabella", "Liam", "Sophia", "Alexander",
            "Ava", "Benjamin", "Charlotte", "Daniel", "Emma", "Gabriel", "Grace",
            "Henry", "Luna", "Jackson", "Mia", "Lucas", "Olivia", "Matthew",
            "Zoe", "Michael", "Aria", "Owen", "Chloe", "Samuel", "Lily"
        ]

        return {
            "required": {
                "api_key": ("STRING", {"multiline": False,
                                      "description": "Fal AI API key"}),
                "frames": ("IMAGE", {"description": "Input video frames as image sequence"}),
                "mode": (["audio_input", "text_to_speech"], {"default": "audio_input",
                                                           "description": "Input mode: use audio file or text-to-speech"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999,
                               "description": "Random seed (0 = random, max 6 digits)"}),
                "nth_frame": ("INT", {"default": 1, "min": 1, "max": 4,
                                     "description": "Extract every Nth frame (1=all frames, 2=every 2nd frame, etc.)"})
            },
            "optional": {
                "audio": ("AUDIO", {"description": "Input audio tensor (for audio_input mode)"}),
                "text": ("STRING", {"default": "Hello, this is a test message.",
                                 "multiline": True, "description": "Text for speech synthesis (for text_to_speech mode)"}),
                "voice_id": (voice_options, {"default": "Auto",
                                           "description": "Voice selection for text-to-speech"})
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

    def generate_lipsync(self, api_key, frames, mode="audio_input", seed=0, nth_frame=1,
                        audio=None, text="Hello, this is a test message.", voice_id="Auto"):
        """
        Generate lipsync video using Fal AI Pixverse LipSync API

        Args:
            api_key: Fal AI API key
            frames: Input video frames as image sequence
            mode: Input mode - "audio_input" or "text_to_speech"
            seed: Random seed for generation (0 = random, max 6 digits)
            nth_frame: Extract every Nth frame
            audio: Input audio tensor (for audio_input mode)
            text: Text for speech synthesis (for text_to_speech mode)
            voice_id: Voice selection for text-to-speech

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
                print("[Fal Pixverse LipSync] Cleared existing FAL_KEY environment variable")

            # 1. Validate API key
            if not api_key or api_key.strip() == "":
                return error_return("Error: API Key is required")

            # 2. Validate frames input
            if frames is None:
                return error_return("Error: Frames input is required")

            # 3. Process seed (ensure it's within 6-digit limit)
            if seed < 0 or seed > 999999:
                return error_return("Error: Seed must be between 0 and 999999 (6 digits max)")

            # 4. Check audio input and validate mode - defer upload until after API key setup
            audio_tensor_to_process = None
            if mode == "audio_input":
                if audio is not None:
                    audio_tensor_to_process = audio
                    print(f"[Fal Pixverse LipSync] Audio tensor provided, will process after API setup")
                else:
                    return error_return("Error: Audio tensor is required for audio_input mode")
            elif mode == "text_to_speech":
                if not text or text.strip() == "":
                    return error_return("Error: Text is required for text_to_speech mode")
            else:
                return error_return("Error: Invalid mode. Must be 'audio_input' or 'text_to_speech'")

            # 5. Set up frames for processing (convert to video after API key setup)
            frames_to_process = frames
            print(f"[Fal Pixverse LipSync] Frames provided, will convert to video after API setup")
            print(f"[Fal Pixverse LipSync] Using seed: {seed}")

            print(f"[Fal Pixverse LipSync] Starting lipsync generation...")
            print(f"[Fal Pixverse LipSync] Mode: {mode}")

            # Prepare the API request
            clean_api_key = api_key.strip()

            # Prepare the arguments for fal_client (video_url will be set after frames conversion)
            arguments = {
                "video_url": ""  # Will be set after frames to video conversion
            }

            # Add seed if specified (0 means random)
            if seed > 0:
                arguments["seed"] = seed

            # Add mode-specific parameters (URLs will be set after upload)
            if mode == "audio_input":
                arguments["audio_url"] = ""  # Will be set after audio upload
            elif mode == "text_to_speech":
                arguments["text"] = text.strip()
                arguments["voice_id"] = voice_id
                print(f"[Fal Pixverse LipSync] Text: {text[:50]}{'...' if len(text) > 50 else ''}")
                print(f"[Fal Pixverse LipSync] Voice ID: {voice_id}")

            # Remove any None values from arguments
            arguments = {k: v for k, v in arguments.items() if v is not None and v != ""}

            # Print arguments without exposing potentially large base64 data
            safe_arguments = {k: v if not (isinstance(v, str) and v.startswith('data:')) else f"<data_uri_{len(v)}_chars>" for k, v in arguments.items()}
            print(f"[Fal Pixverse LipSync] API arguments: {safe_arguments}")

            # Set the API key as an environment variable for fal_client
            key_preview = clean_api_key[:8] + "..." if len(clean_api_key) > 8 else "invalid_key"
            print(f"[Fal Pixverse LipSync] Using API key starting with: {key_preview}")

            # Clear and set the environment variable
            if "FAL_KEY" in os.environ:
                del os.environ["FAL_KEY"]
            os.environ["FAL_KEY"] = clean_api_key

            print(f"[Fal Pixverse LipSync] Calling Fal AI API with fal_client...")

            # Define a callback for queue updates
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        print(f"[Fal Pixverse LipSync] Log: {log['message']}")

            try:
                # Use the lipsync endpoint
                endpoint = "fal-ai/pixverse/lipsync"
                print(f"[Fal Pixverse LipSync] Using endpoint: {endpoint}")

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
                        print(f"[Fal Pixverse LipSync] Saved audio tensor to temporary file: {temp_audio_path}")

                        # Upload to Fal
                        final_audio_url = fal_client.upload_file(temp_audio_path)
                        print(f"[Fal Pixverse LipSync] Uploaded audio to Fal: {final_audio_url}")

                        # Update arguments with the uploaded audio URL
                        arguments["audio_url"] = final_audio_url

                        # Clean up temporary file
                        os.unlink(temp_audio_path)

                    except Exception as e:
                        print(f"[Fal Pixverse LipSync] Error processing audio tensor: {str(e)}")
                        return error_return(f"Error: Failed to process audio: {str(e)}")

                # Process frames to video conversion and upload (after fal_client is properly loaded)
                try:
                    # Convert frames to temporary video file and upload to Fal
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                        temp_video_path = temp_video.name

                    print(f"[Fal Pixverse LipSync] Converting frames to video...")

                    # Convert frames tensor to video using OpenCV
                    if len(frames_to_process.shape) == 4:  # [batch, height, width, channels]
                        frames_np = frames_to_process.cpu().numpy()
                    else:
                        frames_np = frames_to_process.unsqueeze(0).cpu().numpy()

                    # Convert to uint8 if needed
                    if frames_np.dtype != np.uint8:
                        frames_np = (frames_np * 255).astype(np.uint8)

                    # Get video properties
                    batch_size, height, width, channels = frames_np.shape
                    fps = 24  # Default FPS for video

                    # Create video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

                    # Write frames to video
                    for i in range(batch_size):
                        frame = frames_np[i]
                        # Convert RGB to BGR for OpenCV
                        if channels == 3:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        else:
                            frame_bgr = frame
                        out.write(frame_bgr)

                    out.release()
                    print(f"[Fal Pixverse LipSync] Saved frames to temporary video: {temp_video_path}")

                    # Upload video to Fal
                    video_url = fal_client.upload_file(temp_video_path)
                    print(f"[Fal Pixverse LipSync] Uploaded video to Fal: {video_url}")

                    # Update arguments with the uploaded video URL
                    arguments["video_url"] = video_url

                    # Clean up temporary file
                    os.unlink(temp_video_path)

                except Exception as e:
                    print(f"[Fal Pixverse LipSync] Error processing frames to video: {str(e)}")
                    return error_return(f"Error: Failed to convert frames to video: {str(e)}")

                # Make the API call using fal_client.subscribe
                print(f"[Fal Pixverse LipSync] Making API call with fal_client.subscribe...")
                result = fal_client.subscribe(
                    endpoint,
                    arguments=arguments,
                    with_logs=True,
                    on_queue_update=on_queue_update,
                )

                print(f"[Fal Pixverse LipSync] API call completed successfully")
            except Exception as e:
                error_msg = f"API Error: {str(e)}"
                print(f"[Fal Pixverse LipSync] {error_msg}")
                return error_return(error_msg)

            # Extract video URL from the result
            if "video" in result and "url" in result["video"]:
                output_video_url = result["video"]["url"]
                print(f"[Fal Pixverse LipSync] Video ready! URL: {output_video_url}")

                # Download and process the video
                try:
                    print(f"[Fal Pixverse LipSync] Downloading video...")

                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                        temp_video_path = temp_video.name

                        # Download video to temp file
                        dl_response = requests.get(output_video_url, stream=True)
                        dl_response.raise_for_status()

                        # Get file size for progress bar
                        file_size = int(dl_response.headers.get('content-length', 0))
                        progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading LipSync Video")

                        for chunk in dl_response.iter_content(chunk_size=8192):
                            temp_video.write(chunk)
                            progress_bar.update(len(chunk))

                        progress_bar.close()

                    # Extract frames using OpenCV
                    print(f"[Fal Pixverse LipSync] Extracting frames from video...")
                    cap = cv2.VideoCapture(temp_video_path)

                    if not cap.isOpened():
                        os.unlink(temp_video_path)  # Clean up temp file
                        return error_return("Could not open video file")

                    # Get video properties
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    print(f"[Fal Pixverse LipSync] Video has {total_frames} frames at {fps} FPS")

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
                        print(f"[Fal Pixverse LipSync] Extracted {len(frames)} frames as tensor with shape {frames_tensor.shape}")

                        # Prepare audio output - use the original input audio if provided, otherwise empty audio
                        output_audio = audio if audio is not None else {"waveform": torch.zeros((1, 1, 0)), "sample_rate": 44100}

                        return frames_tensor, output_audio, output_video_url, "Success: LipSync video generated and processed"
                    else:
                        return error_return("No frames could be extracted")

                except Exception as e:
                    return error_return(f"Processing Error: {str(e)}")
            else:
                return error_return("No video URL in API response")

        except Exception as e:
            print(f"[Fal Pixverse LipSync] Error: {str(e)}")
            # Try to return proper empty tensors
            empty_tensor = torch.zeros((1, 1, 1, 3))
            empty_audio = {"waveform": torch.zeros((1, 1, 0)), "sample_rate": 44100}
            # Sanitize error message to remove potential base64 data
            clean_error = self.sanitize_error_message(f"Error: {str(e)}")
            return empty_tensor, empty_audio, "", clean_error