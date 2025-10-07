import argparse
import json
import logging # Keep for now, can be replaced by self._log
import os
import time
import tempfile
import traceback # Import traceback
from typing import Dict, Optional, Union, Any

import requests
import torch
import numpy as np
from PIL import Image
import cv2 # For video frame extraction

# ComfyUI specific imports
from comfy.utils import ProgressBar

# Attempt to load dotenv for local dev, but API key primarily from input
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class FL_Hedra_API: # Renamed class
    BASE_URL = "https://api.hedra.com/web-app/public"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": os.getenv("HEDRA_API_KEY", ""), "multiline": False}),
                "image": ("IMAGE",),
                "audio_file_path": ("STRING", {"default": "assets/audio.mp3", "multiline": False}),
                "text_prompt": ("STRING", {"default": "character talking on a white background", "multiline": True}),
                "aspect_ratio": (["1:1", "16:9", "9:16"], {"default": "1:1"}),
                "resolution": (["720p", "540p"], {"default": "720p"}),
            },
            "optional": {
                "duration_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1, "precision": 1}), # 0.0 means API default
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}), # 0 means API default/random
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "api_log")
    FUNCTION = "execute_hedra_pipeline"
    CATEGORY = "🏵️Fill Nodes/AI"
    DESCRIPTION = "Generates a video using the Hedra API from an image, audio, and prompt, then outputs its frames."

    def __init__(self):
        self.log_messages = []

    def _log(self, message: str, level: str = "INFO"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[FL_Hedra_API] {timestamp} [{level}]: {message}" # Updated log prefix
        print(formatted_message)
        self.log_messages.append(formatted_message)

    class _HedraSession(requests.Session):
        def __init__(self, api_key: str, base_url: str):
            super().__init__()
            self.base_url = base_url
            self.headers["x-api-key"] = api_key
        
        def prepare_request(self, request: requests.Request) -> requests.PreparedRequest:
            if not request.url.startswith(("http://", "https://")):
                request.url = f"{self.base_url.rstrip('/')}/{request.url.lstrip('/')}"
            return super().prepare_request(request)

    def _convert_tensor_to_temp_image(self, image_tensor: torch.Tensor) -> Optional[str]:
        if image_tensor is None:
            return None
        try:
            img_np = image_tensor[0].cpu().numpy() # Assuming batch size 1 or taking the first image
            img_np = (img_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            
            fd, temp_image_path = tempfile.mkstemp(suffix=".png", dir=tempfile.gettempdir())
            os.close(fd) # Close the file descriptor as PIL will open/close it
            pil_image.save(temp_image_path, format='PNG')
            self._log(f"Saved input tensor to temporary image: {temp_image_path}")
            return temp_image_path
        except Exception as e:
            self._log(f"Error converting tensor to temp image: {e}", level="ERROR")
            return None

    def _upload_asset(self, session: _HedraSession, file_path: str, asset_type: str, asset_name: Optional[str] = None) -> Optional[str]:
        if not os.path.exists(file_path):
            self._log(f"Asset file not found: {file_path}", level="ERROR")
            return None
        
        asset_name = asset_name or os.path.basename(file_path)
        self._log(f"Uploading {asset_type}: {asset_name} from {file_path}")
        
        try:
            asset_response = session.post("/assets", json={"name": asset_name, "type": asset_type})
            asset_response.raise_for_status()
            asset_id = asset_response.json()["id"]
            
            with open(file_path, "rb") as f:
                upload_response = session.post(f"/assets/{asset_id}/upload", files={"file": f})
                upload_response.raise_for_status()
            
            self._log(f"Successfully uploaded {asset_type} with ID: {asset_id}")
            return asset_id
        except requests.exceptions.RequestException as e:
            self._log(f"Error during {asset_type} asset upload: {e}. Response: {e.response.text if e.response else 'N/A'}", level="ERROR")
        except Exception as e:
            self._log(f"Unexpected error during {asset_type} asset upload: {e}", level="ERROR")
        return None

    def _get_available_models(self, session: _HedraSession) -> list:
        self._log("Fetching available models...")
        try:
            response = session.get("/models")
            response.raise_for_status()
            models = response.json()
            self._log(f"Found {len(models)} available models.")
            return models
        except requests.exceptions.RequestException as e:
            self._log(f"Error fetching models: {e}", level="ERROR")
        return []

    def _generate_video_request(self, session: _HedraSession, model_id: str, image_id: str, audio_id: str,
                               text_prompt: str, resolution: str, aspect_ratio: str,
                               duration: Optional[float] = None, seed: Optional[int] = None) -> Optional[Dict[str, Any]]:
        self._log("Submitting video generation request...")
        payload = {
            "type": "video",
            "ai_model_id": model_id,
            "start_keyframe_id": image_id,
            "audio_id": audio_id,
            "generated_video_inputs": {
                "text_prompt": text_prompt,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
            },
        }
        if duration and duration > 0.0: # API expects duration_ms > 0
            payload["generated_video_inputs"]["duration_ms"] = int(duration * 1000)
        if seed and seed != 0: # API might treat 0 as random/unset
            payload["generated_video_inputs"]["seed"] = seed
        
        try:
            response = session.post("/generations", json=payload)
            response.raise_for_status()
            result = response.json()
            self._log(f"Generation request submitted with ID: {result['id']}")
            return result
        except requests.exceptions.RequestException as e:
            self._log(f"Error submitting generation request: {e}. Response: {e.response.text if e.response else 'N/A'}", level="ERROR")
        return None

    def _poll_generation_status(self, session: _HedraSession, generation_id: str, pbar: ProgressBar) -> Optional[Dict[str, Any]]:
        self._log(f"Polling generation status for ID: {generation_id}")
        poll_interval = 5 # seconds
        total_polls = 0 # For progress bar update
        max_polls_for_pbar = 60 # Arbitrary limit for pbar updates (5 mins)
        
        while True:
            try:
                status_response = session.get(f"/generations/{generation_id}/status")
                status_response.raise_for_status()
                status_data = status_response.json()
                status = status_data.get("status", "unknown")
                progress = status_data.get("progress") # Hedra might provide progress %

                log_msg = f"Current status: {status}"
                if progress is not None:
                    log_msg += f" (Progress: {progress}%)"
                    if isinstance(progress, (int, float)) and 0 <= progress <= 100:
                         # Use the total value the pbar was initialized with for scaling
                         # Assuming pbar here refers to the polling_pbar passed to this method
                         # which was initialized with polling_pbar_total
                         pbar_total_for_scaling = pbar.total # ProgressBar stores its total in pbar.total
                         pbar.update_absolute(int(progress * (pbar_total_for_scaling / 100.0)))
                else:
                    # Fallback pbar update if no explicit progress
                    pbar.update_absolute(min(total_polls, max_polls_for_pbar))

                self._log(log_msg)
                
                if status in ["complete", "error"]:
                    if status == "complete": pbar.update_absolute(pbar.total) # Ensure 100% on complete
                    return status_data
                
                time.sleep(poll_interval)
                total_polls +=1
            except requests.exceptions.RequestException as e:
                self._log(f"Error polling status: {e}", level="ERROR")
                return {"status": "error", "error_message": f"Polling failed: {e}"}
            except Exception as e:
                self._log(f"Unexpected error during polling: {e}", level="ERROR")
                return {"status": "error", "error_message": f"Unexpected polling error: {e}"}


    def _download_video_file(self, download_url: str, output_filename: str) -> Optional[str]:
        self._log(f"Downloading video from {download_url} to {output_filename}")
        try:
            with requests.get(download_url, stream=True, timeout=300) as r: # 5 min timeout for download
                r.raise_for_status()
                with open(output_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            self._log(f"Successfully downloaded video to {output_filename}")
            return output_filename
        except requests.exceptions.RequestException as e:
            self._log(f"Failed to download video: {e}", level="ERROR")
        except IOError as e:
            self._log(f"Failed to save video file: {e}", level="ERROR")
        return None

    def _extract_frames_from_video(self, video_path: str) -> Optional[torch.Tensor]:
        if not os.path.exists(video_path):
            self._log(f"Video file not found for frame extraction: {video_path}", level="ERROR")
            return None
        
        self._log(f"Extracting frames from {video_path}")
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            pbar_frames = ProgressBar(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                img_array = np.array(pil_image).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0) # Add batch dim
                frames.append(img_tensor)
                frame_count +=1
                pbar_frames.update_absolute(frame_count)
            cap.release()
            
            if not frames:
                self._log("No frames extracted from video.", level="WARNING")
                return None
            
            self._log(f"Extracted {len(frames)} frames.")
            return torch.cat(frames, dim=0)
        except Exception as e:
            self._log(f"Error extracting frames: {e}", level="ERROR")
        return None

    def execute_hedra_pipeline(self, api_key: str, image: torch.Tensor, audio_file_path: str,
                               text_prompt: str, aspect_ratio: str, resolution: str,
                               duration_seconds: float = 0.0, seed: int = 0):
        self.log_messages = [] # Reset logs for this run
        
        if not api_key:
            self._log("API key not provided.", level="ERROR")
            return (None, "API key not provided.")

        session = self._HedraSession(api_key=api_key, base_url=self.BASE_URL)
        
        temp_image_path = None
        temp_video_path = None
        final_frames = None
        
        # Progress bar for the whole operation (approximated steps)
        # 1: upload image, 2: upload audio, 3: get models, 4: submit generation, 5: poll (scaled), 6: download, 7: extract frames
        # Polling itself will have its own progress if API provides it, or timed progress.
        # Let's use a simple step-based pbar for overall, and another for polling/extraction.
        overall_pbar = ProgressBar(7) 

        try:
            # 1. Upload Image
            temp_image_path = self._convert_tensor_to_temp_image(image)
            if not temp_image_path:
                raise RuntimeError("Failed to convert input image tensor.")
            image_id = self._upload_asset(session, temp_image_path, "image", "input_image.png")
            if not image_id:
                raise RuntimeError("Failed to upload image asset.")
            overall_pbar.update_absolute(1)

            # 2. Upload Audio
            audio_id = self._upload_asset(session, audio_file_path, "audio")
            if not audio_id:
                raise RuntimeError(f"Failed to upload audio asset from {audio_file_path}.")
            overall_pbar.update_absolute(2)

            # 3. Get Model ID
            models = self._get_available_models(session)
            if not models:
                raise RuntimeError("No models available from Hedra API.")
            model_id = models[0]["id"] # Use the first available model as per script
            self._log(f"Using model ID: {model_id}")
            overall_pbar.update_absolute(3)

            # 4. Generate Video Request
            # For polling pbar, let's assume 100 steps for the polling phase itself
            polling_pbar_total = 100 
            polling_pbar = ProgressBar(polling_pbar_total) 
            
            generation_request = self._generate_video_request(
                session, model_id, image_id, audio_id, text_prompt,
                resolution, aspect_ratio, duration_seconds if duration_seconds > 0 else None,
                seed if seed != 0 else None
            )
            if not generation_request or "id" not in generation_request:
                raise RuntimeError("Failed to submit video generation request.")
            generation_id = generation_request["id"]
            overall_pbar.update_absolute(4)

            # 5. Poll Generation Status
            self._log("Starting to poll generation status...")
            final_status = self._poll_generation_status(session, generation_id, polling_pbar)
            if not final_status or final_status.get("status") != "complete":
                error_msg = final_status.get("error_message", "Video generation did not complete successfully or status unknown.")
                raise RuntimeError(f"Generation failed or status incomplete: {error_msg}")
            
            download_url = final_status.get("url")
            if not download_url:
                raise RuntimeError("Generation complete but no download URL provided.")
            overall_pbar.update_absolute(5)

            # 6. Download Video
            temp_video_dir = tempfile.gettempdir()
            temp_video_path = os.path.join(temp_video_dir, f"hedra_video_{generation_id}.mp4")
            downloaded_path = self._download_video_file(download_url, temp_video_path)
            if not downloaded_path:
                raise RuntimeError("Failed to download generated video.")
            overall_pbar.update_absolute(6)

            # 7. Extract Frames
            final_frames = self._extract_frames_from_video(downloaded_path)
            if final_frames is None:
                raise RuntimeError("Failed to extract frames from downloaded video.")
            overall_pbar.update_absolute(7)

        except Exception as e:
            self._log(f"Pipeline error: {e}", level="CRITICAL")
            traceback.print_exc() # For more detailed debug in console
            # Create a single black error frame if pipeline fails
            error_img_pil = Image.new('RGB', (256, 256), color='black') # Small error frame
            draw = ImageDraw.Draw(error_img_pil)
            try: font = ImageFont.load_default()
            except: font = None
            draw.text((10,10), f"Error: {str(e)[:100]}", fill="red", font=font)
            error_img_np = np.array(error_img_pil).astype(np.float32) / 255.0
            final_frames = torch.from_numpy(error_img_np).unsqueeze(0)
        finally:
            if temp_image_path and os.path.exists(temp_image_path):
                try: os.remove(temp_image_path)
                except Exception as e_rem: self._log(f"Could not remove temp image {temp_image_path}: {e_rem}", "WARNING")
            if temp_video_path and os.path.exists(temp_video_path):
                try: os.remove(temp_video_path)
                except Exception as e_rem: self._log(f"Could not remove temp video {temp_video_path}: {e_rem}", "WARNING")

        return (final_frames, "\n".join(self.log_messages))

# Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FL_Hedra_API": FL_Hedra_API # Renamed mapping key
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_Hedra_API": "FL Hedra API" # Renamed display name
}