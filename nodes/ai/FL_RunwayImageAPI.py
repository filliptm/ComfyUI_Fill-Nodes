import os
import time
import httpx # For downloading the image
import base64
import io
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Literal, Union, List, Dict, Any

from runwayml import RunwayML
from runwayml._types import NotGiven, NOT_GIVEN # Import NOT_GIVEN sentinel
from runwayml.types import TaskRetrieveResponse # For type hinting polled task
from runwayml._exceptions import APIError, APIStatusError, APIConnectionError, RateLimitError
from comfy.utils import ProgressBar # Import ProgressBar


class FL_RunwayImageAPI:
    # Default API version used by the SDK if not specified in constructor
    # From your script, it seems the SDK handles this well if client is initialized with runway_version
    DEFAULT_RUNWAY_API_VERSION = "2024-11-06" 
    MODEL_NAME: Literal["gen4_image"] = "gen4_image"
    IMAGE_RATIO_OPTIONS = ['1920:1080', '1080:1920', '1024:1024', '1360:768', '1080:1080', '1168:880', '1440:1080', '1080:1440', '1808:768', '2112:912']

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": os.environ.get('RUNWAYML_API_SECRET', '')}),
                "prompt": ("STRING", {"multiline": True, "default": "A stunning photograph of a majestic wolf howling at a full moon."}),
                "ratio": (cls.IMAGE_RATIO_OPTIONS, {"default": "1024:1024"}),
                "runway_api_version": ("STRING", {"multiline": False, "default": cls.DEFAULT_RUNWAY_API_VERSION}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "tag1": ("STRING", {"multiline": False, "default": ""}), # Default to empty, let user decide
                "tag2": ("STRING", {"multiline": False, "default": ""}),
                "tag3": ("STRING", {"multiline": False, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}), # 0 will mean NOT_GIVEN
                "public_figure_threshold": (["auto", "low", "NOT_GIVEN"], {"default": "NOT_GIVEN"}),
                "polling_interval": ("INT", {"default": 5, "min": 1, "max": 60}),
                "max_polling_attempts": ("INT", {"default": 36, "min": 1, "max": 120}), # Max 10 mins for 5s interval
                "timeout_sdk": ("FLOAT", {"default": 60.0, "min": 5.0, "max": 300.0, "step": 1.0}), # For SDK client
                "timeout_download": ("FLOAT", {"default": 60.0, "min": 5.0, "max": 300.0, "step": 1.0}), # For image download
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status_text")
    FUNCTION = "generate_image_from_script_logic"
    CATEGORY = "🏵️Fill Nodes/AI"

    def __init__(self):
        self.log_messages = []

    def _log(self, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[FL_RunwayImageAPI] {timestamp}: {message}"
        print(full_message)
        self.log_messages.append(full_message) # Store full message with timestamp

    def _create_error_image(self, error_message="API Failed", width=1024, height=1024) -> torch.Tensor:
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
        if tensor is None: return None
        if len(tensor.shape) == 4 and tensor.shape[0] == 1:
            image_np = tensor[0].cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            return Image.fromarray(image_np)
        self._log(f"Error: {name} tensor format incorrect: {tensor.shape}")
        return None

    def _pil_to_data_uri(self, pil_image: Image.Image, image_format="PNG") -> str:
        buffered = io.BytesIO()
        pil_image.save(buffered, format=image_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{image_format.lower()};base64,{img_str}"

    def generate_image_from_script_logic(
        self, api_key: str, prompt: str, ratio: str, runway_api_version: str,
        image1: torch.Tensor = None, image2: torch.Tensor = None, image3: torch.Tensor = None,
        tag1: str = "", tag2: str = "", tag3: str = "",
        seed: int = 0, public_figure_threshold: str = "NOT_GIVEN",
        polling_interval: int = 5, max_polling_attempts: int = 36,
        timeout_sdk: float = 60.0, timeout_download: float = 60.0
    ):
        self.log_messages = [] # Reset logs for each run

        if not api_key or not api_key.strip():
            self._log("Error: RunwayML API key not provided.")
            return self._create_error_image("API key missing"), "\n".join(self.log_messages)

        # Use runway_api_version from input, or default if empty
        current_runway_api_version = runway_api_version if runway_api_version and runway_api_version.strip() else self.DEFAULT_RUNWAY_API_VERSION

        try:
            client = RunwayML(api_key=api_key.strip(), runway_version=current_runway_api_version, timeout=timeout_sdk)
            self._log(f"RunwayML client initialized. API Version: {client.runway_version}, Timeout: {timeout_sdk}s")
        except Exception as e:
            self._log(f"Failed to initialize RunwayML client: {str(e)}")
            return self._create_error_image("SDK Client Init Failed"), "\n".join(self.log_messages)

        active_task_id = None
        
        reference_images_payload: List[Dict[str, Any]] = []
        input_images_with_tags = [(image1, tag1), (image2, tag2), (image3, tag3)]
        for i, (img_tensor, tag_str) in enumerate(input_images_with_tags):
            if img_tensor is not None:
                pil_img = self._process_tensor_to_pil(img_tensor, f"Reference Image {i+1}")
                if pil_img:
                    data_uri = self._pil_to_data_uri(pil_img)
                    ref_img_obj = {"uri": data_uri}
                    clean_tag = tag_str.strip() if tag_str else ""
                    if clean_tag and 3 <= len(clean_tag) <= 16 and clean_tag[0].isalpha() and clean_tag.replace('_', '').isalnum():
                        ref_img_obj["tag"] = clean_tag
                    elif clean_tag:
                        self._log(f"Warning: Tag '{clean_tag}' for image {i+1} is invalid and will be ignored.")
                    reference_images_payload.append(ref_img_obj)
                else:
                    self._log(f"Error processing reference image {i+1}")
                    return self._create_error_image(f"Failed to process ref image {i+1}"), "\n".join(self.log_messages)
        
        create_args: Dict[str, Any] = {
            "model": self.MODEL_NAME,
            "prompt_text": prompt,
            "ratio": ratio,
        }
        if reference_images_payload: # Only add if there are valid reference images
            create_args["reference_images"] = reference_images_payload
        
        current_seed = NOT_GIVEN if seed == 0 else seed
        create_args["seed"] = current_seed

        if public_figure_threshold != "NOT_GIVEN":
            create_args["content_moderation"] = {"publicFigureThreshold": public_figure_threshold}

        try:
            self._log(f"Requesting image generation with prompt: '{prompt}' using model '{self.MODEL_NAME}'")
            self._log(f"Create args (excluding images for brevity if many): { {k:v for k,v in create_args.items() if k != 'reference_images'} }")
            if 'reference_images' in create_args:
                 self._log(f"Number of reference images: {len(create_args['reference_images'])}")


            initial_response = client.text_to_image.create(**create_args)
            active_task_id = initial_response.id
            self._log(f"Task submitted successfully! Task ID: {active_task_id}")

            self._log(f"Polling task status for Task ID: {active_task_id} (every {polling_interval}s)...")
            task_details: Union[TaskRetrieveResponse, None] = None
            pbar = ProgressBar(max_polling_attempts) # Initialize ProgressBar
            for attempt in range(max_polling_attempts):
                task_details = client.tasks.retrieve(id=active_task_id)
                progress_percent = 0.0
                if task_details.status == "RUNNING" and task_details.progress is not None:
                    progress_percent = task_details.progress * 100
                
                status_log = f"Attempt {attempt + 1}/{max_polling_attempts} - Status: {task_details.status}"
                if progress_percent > 0:
                    status_log += f", Progress: {progress_percent:.2f}%"
                self._log(status_log)
                
                # Update progress bar: current value is attempt number, max is max_polling_attempts
                # If RUNNING and progress is available, use that for a more granular bar update.
                # However, ProgressBar typically expects an iteration count.
                # We can simulate this by updating based on attempts, or if RUNNING, map task_details.progress to pbar's scale.
                # For simplicity, we'll update based on attempts.
                # If task_details.progress is available, we can use it to show a more fine-grained update
                # by mapping it to the total attempts.
                # For now, just update based on attempt count.
                # A more sophisticated approach might involve setting pbar total to 100 and updating with progress_percent
                # but that might look weird if the task finishes early or takes many polls without progress change.
                
                current_progress_for_bar = attempt + 1 # Default to attempt number
                if task_details.status == "RUNNING" and task_details.progress is not None:
                    # Map task progress (0-1) to polling attempts for a smoother bar
                    # This is an approximation. If progress jumps, bar will jump.
                    # If progress is slow, bar updates per poll.
                    current_progress_for_bar = int(task_details.progress * max_polling_attempts)
                    # Ensure it doesn't exceed max_polling_attempts due to rounding or jumps
                    current_progress_for_bar = min(current_progress_for_bar, max_polling_attempts)
                    # Ensure it's at least the current attempt, so it doesn't go backwards
                    current_progress_for_bar = max(current_progress_for_bar, attempt + 1)

                # Temporarily comment out to diagnose AttributeError
                # pbar.update_absolute(current_progress_for_bar, max_polling_attempts, status_log)


                if task_details.status == "SUCCEEDED":
                    self._log("Task SUCCEEDED!")
                    if task_details.output and len(task_details.output) > 0 and isinstance(task_details.output[0], str):
                        image_url = task_details.output[0]
                        self._log(f"Image URL: {image_url}")
                        self._log(f"Downloading image...")
                        try:
                            with httpx.Client() as download_client:
                                response = download_client.get(image_url, follow_redirects=True, timeout=timeout_download)
                                response.raise_for_status()
                                pil_image = Image.open(io.BytesIO(response.content)).convert("RGB")
                                img_array = np.array(pil_image).astype(np.float32) / 255.0
                                img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                                self._log("Image downloaded and processed successfully.")
                                return img_tensor, "\n".join(self.log_messages)
                        except httpx.HTTPStatusError as e_http:
                            self._log(f"Error downloading image: HTTP {e_http.response.status_code} - {e_http.response.text}")
                            return self._create_error_image(f"Download Failed: {e_http.response.status_code}"), "\n".join(self.log_messages)
                        except Exception as e_dl:
                            self._log(f"An error occurred during image download: {str(e_dl)}")
                            return self._create_error_image("Download Error"), "\n".join(self.log_messages)
                    else:
                        self._log("Task succeeded but no valid output URL found.")
                        return self._create_error_image("No Output URL"), "\n".join(self.log_messages)
                
                elif task_details.status in ["FAILED", "CANCELLED"]: # Your script uses CANCELLED, SDK might use CANCELED
                    self._log(f"Task {task_details.status}.")
                    failure_reason = task_details.failure if hasattr(task_details, 'failure') and task_details.failure else 'N/A'
                    failure_code = task_details.failure_code if hasattr(task_details, 'failure_code') and task_details.failure_code else 'N/A'
                    self._log(f"  Reason: {failure_reason}")
                    self._log(f"  Failure code: {failure_code}")
                    # ... (additional suggestions from your script can be added here if desired)
                    return self._create_error_image(f"Task {task_details.status}: {failure_reason[:60]}"), "\n".join(self.log_messages)
                
                time.sleep(polling_interval)
            
            # If loop finishes without returning
            if task_details:
                self._log(f"Max poll attempts ({max_polling_attempts}) reached. Last known task status: {task_details.status}.")
            else: # Should not happen if initial submission was successful
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
        except APIError as e: # General SDK error
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