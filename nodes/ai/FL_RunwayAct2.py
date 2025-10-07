import os
import time
import httpx
import base64
import io
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Literal, Union, List, Dict, Any

from runwayml import RunwayML
from runwayml._types import NotGiven, NOT_GIVEN
from runwayml.types import TaskRetrieveResponse
from runwayml._exceptions import APIError, APIStatusError, APIConnectionError, RateLimitError
from comfy.utils import ProgressBar


class FL_RunwayAct2:
    DEFAULT_RUNWAY_API_VERSION = "2024-11-06"
    MODEL_NAME: Literal["act_two"] = "act_two"
    RATIO_OPTIONS = ['1280:720', '720:1280', '960:960', '1104:832', '832:1104', '1584:672']

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": os.environ.get('RUNWAYML_API_SECRET', '')}),
                "character_type": (["video", "image"], {"default": "video"}),
                "ratio": (cls.RATIO_OPTIONS, {"default": "1280:720"}),
                "runway_api_version": ("STRING", {"multiline": False, "default": cls.DEFAULT_RUNWAY_API_VERSION}),
            },
            "optional": {
                "character_video": ("IMAGE",),  # Will be converted to video format
                "character_image": ("IMAGE",),
                "reference_video": ("IMAGE",),  # Will be converted to video format
                "body_control": ("BOOLEAN", {"default": True}),
                "expression_intensity": ("INT", {"default": 3, "min": 1, "max": 5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
                "public_figure_threshold": (["auto", "low", "NOT_GIVEN"], {"default": "NOT_GIVEN"}),
                "polling_interval": ("INT", {"default": 5, "min": 1, "max": 60}),
                "max_polling_attempts": ("INT", {"default": 60, "min": 1, "max": 120}),
                "timeout_sdk": ("FLOAT", {"default": 60.0, "min": 5.0, "max": 300.0, "step": 1.0}),
                "timeout_download": ("FLOAT", {"default": 120.0, "min": 5.0, "max": 600.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("video_frames", "status_text")
    FUNCTION = "generate_character_performance"
    CATEGORY = "🏵️Fill Nodes/AI"

    def __init__(self):
        self.log_messages = []

    def _log(self, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[FL_RunwayAct2] {timestamp}: {message}"
        print(full_message)
        self.log_messages.append(full_message)

    def _create_error_image(self, error_message="API Failed", width=1280, height=720) -> torch.Tensor:
        image = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((0,0), error_message, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (width - text_width) / 2
        text_y = (height - text_height) / 2
        draw.text((text_x, text_y), error_message, fill=(255, 0, 0), font=font)
        
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        self._log(f"Created error image: '{error_message}'")
        return img_tensor

    def _process_tensor_to_pil(self, tensor: torch.Tensor, name="Image") -> Union[Image.Image, None]:
        if tensor is None:
            return None
        
        # Handle single image: [1, height, width, 3]
        if len(tensor.shape) == 4 and tensor.shape[0] == 1:
            image_np = tensor[0].cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            return Image.fromarray(image_np)
        
        # Handle video batch: [frames, height, width, 3] - use first frame
        elif len(tensor.shape) == 4 and tensor.shape[0] > 1:
            self._log(f"{name} appears to be a video with {tensor.shape[0]} frames. Using first frame.")
            image_np = tensor[0].cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            return Image.fromarray(image_np)
        
        # Handle single frame without batch dimension: [height, width, 3]
        elif len(tensor.shape) == 3:
            image_np = tensor.cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            return Image.fromarray(image_np)
        
        self._log(f"Error: {name} tensor format not supported: {tensor.shape}")
        return None

    def _pil_to_data_uri(self, pil_image: Image.Image, image_format="PNG") -> str:
        buffered = io.BytesIO()
        pil_image.save(buffered, format=image_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{image_format.lower()};base64,{img_str}"

    def _create_video_data_uri_from_tensor(self, tensor: torch.Tensor, name="Video") -> str:
        """Convert video tensor to data URI - for now using first frame as image"""
        # Note: This is a simplified implementation. For full video support,
        # you would need to encode the entire tensor as a video file (MP4, etc.)
        # and create a proper video data URI. For now, we'll use the first frame.
        
        if len(tensor.shape) == 4 and tensor.shape[0] > 1:
            # Video tensor: [frames, height, width, 3]
            self._log(f"{name} has {tensor.shape[0]} frames. Using first frame for API compatibility.")
            first_frame = tensor[0].cpu().numpy()
            first_frame = (first_frame * 255).astype(np.uint8)
            pil_image = Image.fromarray(first_frame)
            return self._pil_to_data_uri(pil_image, "PNG")
        else:
            # Single image, convert normally
            pil_image = self._process_tensor_to_pil(tensor, name)
            if pil_image:
                return self._pil_to_data_uri(pil_image, "PNG")
            else:
                raise ValueError(f"Could not process {name} tensor")

    def _download_and_process_video(self, video_url: str, timeout: float) -> torch.Tensor:
        """Download video and convert to image tensor (first frame for now)"""
        try:
            with httpx.Client() as download_client:
                response = download_client.get(video_url, follow_redirects=True, timeout=timeout)
                response.raise_for_status()
                
                # For now, we'll treat the downloaded content as an image
                # In a full implementation, you'd use video processing libraries
                pil_image = Image.open(io.BytesIO(response.content)).convert("RGB")
                img_array = np.array(pil_image).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                self._log("Video downloaded and processed successfully.")
                return img_tensor
        except Exception as e:
            self._log(f"Error downloading video: {str(e)}")
            raise

    def generate_character_performance(
        self, api_key: str, character_type: str, ratio: str, runway_api_version: str,
        character_video: torch.Tensor = None, character_image: torch.Tensor = None,
        reference_video: torch.Tensor = None, body_control: bool = True,
        expression_intensity: int = 3, seed: int = 0, public_figure_threshold: str = "NOT_GIVEN",
        polling_interval: int = 5, max_polling_attempts: int = 60,
        timeout_sdk: float = 60.0, timeout_download: float = 120.0
    ):
        self.log_messages = []

        if not api_key or not api_key.strip():
            self._log("Error: RunwayML API key not provided.")
            return self._create_error_image("API key missing"), "\n".join(self.log_messages)

        # Validate inputs based on character type
        if character_type == "video" and character_video is None:
            self._log("Error: Character video is required when character_type is 'video'.")
            return self._create_error_image("Character video missing"), "\n".join(self.log_messages)
        
        if character_type == "image" and character_image is None:
            self._log("Error: Character image is required when character_type is 'image'.")
            return self._create_error_image("Character image missing"), "\n".join(self.log_messages)

        if reference_video is None:
            self._log("Error: Reference video is required.")
            return self._create_error_image("Reference video missing"), "\n".join(self.log_messages)

        current_runway_api_version = runway_api_version if runway_api_version and runway_api_version.strip() else self.DEFAULT_RUNWAY_API_VERSION

        try:
            client = RunwayML(api_key=api_key.strip(), runway_version=current_runway_api_version, timeout=timeout_sdk)
            self._log(f"RunwayML client initialized. API Version: {client.runway_version}, Timeout: {timeout_sdk}s")
        except Exception as e:
            self._log(f"Failed to initialize RunwayML client: {str(e)}")
            return self._create_error_image("SDK Client Init Failed"), "\n".join(self.log_messages)

        active_task_id = None

        try:
            # Process character input
            character_payload: Dict[str, Any] = {"type": character_type}
            
            if character_type == "video" and character_video is not None:
                try:
                    character_payload["uri"] = self._create_video_data_uri_from_tensor(character_video, "Character Video")
                except Exception as e:
                    self._log(f"Error processing character video: {str(e)}")
                    return self._create_error_image("Failed to process character video"), "\n".join(self.log_messages)
            
            elif character_type == "image" and character_image is not None:
                pil_img = self._process_tensor_to_pil(character_image, "Character Image")
                if pil_img:
                    character_payload["uri"] = self._pil_to_data_uri(pil_img)
                else:
                    self._log("Error processing character image")
                    return self._create_error_image("Failed to process character image"), "\n".join(self.log_messages)

            # Process reference video
            try:
                reference_payload = {
                    "type": "video",
                    "uri": self._create_video_data_uri_from_tensor(reference_video, "Reference Video")
                }
            except Exception as e:
                self._log(f"Error processing reference video: {str(e)}")
                return self._create_error_image("Failed to process reference video"), "\n".join(self.log_messages)

            # Build create arguments
            create_args: Dict[str, Any] = {
                "model": self.MODEL_NAME,
                "character": character_payload,
                "reference": reference_payload,
                "ratio": ratio,
                "bodyControl": body_control,
                "expressionIntensity": expression_intensity,
            }

            current_seed = NOT_GIVEN if seed == 0 else seed
            create_args["seed"] = current_seed

            if public_figure_threshold != "NOT_GIVEN":
                create_args["contentModeration"] = {"publicFigureThreshold": public_figure_threshold}

            self._log(f"Requesting character performance generation with model '{self.MODEL_NAME}'")
            self._log(f"Character type: {character_type}, Ratio: {ratio}, Body control: {body_control}")
            self._log(f"Expression intensity: {expression_intensity}")

            # Submit the task using the working SDK method
            try:
                self._log("Submitting character performance request...")
                initial_response = client.character_performance.create(**create_args)
                active_task_id = initial_response.id
                self._log(f"Task submitted successfully! Task ID: {active_task_id}")
            except AttributeError as e:
                self._log(f"SDK method not found: {str(e)}")
                self._log("The character_performance method is not available in this SDK version.")
                return self._create_error_image("SDK Method Not Available"), "\n".join(self.log_messages)
            except Exception as e:
                self._log(f"Error submitting task: {str(e)}")
                return self._create_error_image("Task Submission Failed"), "\n".join(self.log_messages)

            # Poll for completion
            self._log(f"Polling task status for Task ID: {active_task_id} (every {polling_interval}s)...")
            task_details: Union[TaskRetrieveResponse, None] = None
            pbar = ProgressBar(max_polling_attempts)
            
            for attempt in range(max_polling_attempts):
                task_details = client.tasks.retrieve(id=active_task_id)
                progress_percent = 0.0
                if task_details.status == "RUNNING" and task_details.progress is not None:
                    progress_percent = task_details.progress * 100
                
                status_log = f"Attempt {attempt + 1}/{max_polling_attempts} - Status: {task_details.status}"
                if progress_percent > 0:
                    status_log += f", Progress: {progress_percent:.2f}%"
                self._log(status_log)

                if task_details.status == "SUCCEEDED":
                    self._log("Task SUCCEEDED!")
                    if task_details.output and len(task_details.output) > 0 and isinstance(task_details.output[0], str):
                        video_url = task_details.output[0]
                        self._log(f"Video URL: {video_url}")
                        self._log(f"Downloading video...")
                        try:
                            video_tensor = self._download_and_process_video(video_url, timeout_download)
                            return video_tensor, "\n".join(self.log_messages)
                        except httpx.HTTPStatusError as e_http:
                            self._log(f"Error downloading video: HTTP {e_http.response.status_code} - {e_http.response.text}")
                            return self._create_error_image(f"Download Failed: {e_http.response.status_code}"), "\n".join(self.log_messages)
                        except Exception as e_dl:
                            self._log(f"An error occurred during video download: {str(e_dl)}")
                            return self._create_error_image("Download Error"), "\n".join(self.log_messages)
                    else:
                        self._log("Task succeeded but no valid output URL found.")
                        return self._create_error_image("No Output URL"), "\n".join(self.log_messages)
                
                elif task_details.status in ["FAILED", "CANCELLED"]:
                    self._log(f"Task {task_details.status}.")
                    failure_reason = task_details.failure if hasattr(task_details, 'failure') and task_details.failure else 'N/A'
                    failure_code = task_details.failure_code if hasattr(task_details, 'failure_code') and task_details.failure_code else 'N/A'
                    self._log(f"  Reason: {failure_reason}")
                    self._log(f"  Failure code: {failure_code}")
                    return self._create_error_image(f"Task {task_details.status}: {failure_reason[:60]}"), "\n".join(self.log_messages)
                
                time.sleep(polling_interval)
            
            # If loop finishes without returning
            if task_details:
                self._log(f"Max poll attempts ({max_polling_attempts}) reached. Last known task status: {task_details.status}.")
            else:
                self._log(f"Max poll attempts ({max_polling_attempts}) reached. Could not retrieve task details after submission.")
            return self._create_error_image("Polling Timeout"), "\n".join(self.log_messages)

        except APIConnectionError as e:
            self._log(f"RunwayML API Connection Error: {str(e)}")
            return self._create_error_image("API Connection Error"), "\n".join(self.log_messages)
        except RateLimitError as e:
            self._log(f"RunwayML Rate Limit Error: {str(e)}")
            return self._create_error_image("Rate Limit Error"), "\n".join(self.log_messages)
        except APIStatusError as e:
            self._log(f"RunwayML API Status Error: {str(e)}. Status Code: {e.status_code if hasattr(e, 'status_code') else 'N/A'}. Response: {e.response.text if hasattr(e, 'response') and hasattr(e.response, 'text') else 'N/A'}")
            return self._create_error_image(f"API Status Error: {e.status_code if hasattr(e, 'status_code') else ''}"), "\n".join(self.log_messages)
        except APIError as e:
            self._log(f"RunwayML API Error: {str(e)}")
            return self._create_error_image("Runway API Error"), "\n".join(self.log_messages)
        except Exception as e:
            self._log(f"An unexpected error occurred: {str(e)}")
            if active_task_id:
                self._log(f"This error occurred for Task ID: {active_task_id}")
            return self._create_error_image("Unexpected Node Error"), "\n".join(self.log_messages)
        finally:
            if 'client' in locals() and client is not None and hasattr(client, 'close'):
                try:
                    client.close()
                    self._log("RunwayML client closed.")
                except Exception as e_close:
                    self._log(f"Error closing RunwayML client: {str(e_close)}")