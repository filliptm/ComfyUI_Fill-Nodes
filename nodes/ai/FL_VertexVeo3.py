import os
import base64
import io
import json
import torch
import numpy as np
from PIL import Image
import requests
import tempfile
import time
import traceback
from google import genai
from google.genai import types
from google.auth import default
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import urllib.request
from typing import Optional, Tuple


class FL_Veo3VideoGen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "service_account_json": ("STRING", {"default": "", "multiline": False}),
                "model": (
                    [
                        "veo-3.0-generate-001",
                        "veo-3.0-fast-generate-001",
                        "veo-2.0-generate-001"
                    ],
                    {"default": "veo-3.0-generate-001"}
                ),
                "aspect_ratio": (
                    ["16:9", "9:16"],
                    {"default": "16:9"}
                ),
                "resolution": (
                    ["720p", "1080p"],
                    {"default": "720p"}
                ),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "polling_interval": ("INT", {"default": 10, "min": 5, "max": 60, "step": 5, "description": "Seconds between status checks"}),
                "max_wait_time": ("INT", {"default": 360, "min": 60, "max": 600, "step": 30, "description": "Maximum seconds to wait for video generation"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "reference_image": ("IMAGE",),
                "enable_person_generation": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("frames", "video_path", "API_Response")
    FUNCTION = "generate_video"
    CATEGORY = "🏵️Fill Nodes/AI"

    def __init__(self):
        """Initialize logging system"""
        self.log_messages = []

    def _log(self, message):
        """Global logging function: record to log list"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[FL_Veo3VideoGen] {timestamp}: {message}"
        print(formatted_message)
        self.log_messages.append(message)
        return message

    def _process_tensor_to_pil(self, tensor, name="Image"):
        """Convert a tensor to a PIL image for API submission"""
        try:
            if tensor is None:
                self._log(f"{name} is None, skipping")
                return None

            # Ensure tensor is in correct format [1, H, W, 3]
            if len(tensor.shape) == 4 and tensor.shape[0] >= 1:
                # Get first frame image
                image_np = tensor[0].cpu().numpy()

                # Convert to uint8 format for PIL
                image_np = (image_np * 255).astype(np.uint8)

                # Create PIL image
                pil_image = Image.fromarray(image_np)

                self._log(f"{name} processed successfully, size: {pil_image.width}x{pil_image.height}")
                return pil_image
            else:
                self._log(f"{name} format incorrect: {tensor.shape}")
                return None
        except Exception as e:
            self._log(f"Error processing {name}: {str(e)}")
            return None

    def _prepare_image_for_api(self, pil_image):
        """Prepare PIL image for Vertex AI"""
        try:
            from google.genai.types import Image as GenAIImage

            self._log(f"Preparing image for API: {pil_image.width}x{pil_image.height}")

            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            self._log(f"Image converted to bytes: {len(img_bytes)} bytes")

            # Create Image object with bytes
            image_obj = GenAIImage(
                image_bytes=img_bytes,
                mime_type="image/png"
            )

            return image_obj

        except Exception as e:
            self._log(f"Error preparing image: {str(e)}")
            traceback.print_exc()
            return None

    def _create_error_frame(self, error_message="Video Generation Failed", width=1280, height=720):
        """Create a black error frame with red text"""
        from PIL import ImageDraw, ImageFont

        image = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Draw error text
        text_bbox = draw.textbbox((0, 0), error_message, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (width - text_width) / 2
        text_y = (height - text_height) / 2

        draw.text((text_x, text_y), error_message, fill=(255, 0, 0), font=font)

        # Convert to tensor format [1, H, W, 3]
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        self._log(f"Created error frame with message: '{error_message}'")
        return img_tensor

    def _download_video(self, video_url: str) -> Optional[str]:
        """Download video from URL to temporary file"""
        try:
            self._log(f"Downloading video from: {video_url}")

            temp_file = os.path.join(
                tempfile.gettempdir(),
                f"veo3_video_{int(time.time())}.mp4"
            )

            urllib.request.urlretrieve(video_url, temp_file)

            file_size = os.path.getsize(temp_file)
            self._log(f"Video downloaded successfully: {file_size} bytes")

            return temp_file

        except Exception as e:
            self._log(f"Error downloading video: {str(e)}")
            return None

    def _extract_frames_from_video(self, video_path: str) -> list:
        """Extract all frames from video file"""
        try:
            import cv2

            self._log(f"Extracting frames from: {video_path}")

            cap = cv2.VideoCapture(video_path)
            frames = []

            if not cap.isOpened():
                self._log("Error: Could not open video file")
                return []

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to tensor format [H, W, 3] with values 0-1
                frame_array = frame_rgb.astype(np.float32) / 255.0
                frame_tensor = torch.from_numpy(frame_array)

                frames.append(frame_tensor)
                frame_count += 1

            cap.release()

            self._log(f"Extracted {frame_count} frames from video")

            return frames

        except Exception as e:
            self._log(f"Error extracting frames: {str(e)}")
            traceback.print_exc()
            return []

    def _call_veo_api(self, client, model: str, prompt: str, config,
                      reference_image = None,
                      retry_count: int = 0, max_retries: int = 3) -> Optional[any]:
        """Call VEO API with retry logic"""
        try:
            self._log(f"API call attempt #{retry_count + 1}")
            self._log(f"Model: {model}")
            self._log(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")

            # Make the API call using generate_videos
            if reference_image is not None:
                self._log(f"Using reference image")
                if config is not None:
                    operation = client.models.generate_videos(
                        model=model,
                        prompt=prompt,
                        image=reference_image,
                        config=config
                    )
                else:
                    operation = client.models.generate_videos(
                        model=model,
                        prompt=prompt,
                        image=reference_image
                    )
            else:
                if config is not None:
                    operation = client.models.generate_videos(
                        model=model,
                        prompt=prompt,
                        config=config
                    )
                else:
                    operation = client.models.generate_videos(
                        model=model,
                        prompt=prompt
                    )

            self._log("Initial API response received")
            return operation

        except Exception as e:
            self._log(f"API call error: {str(e)}")
            if retry_count < max_retries - 1:
                wait_time = 5 * (retry_count + 1)
                self._log(f"Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
                time.sleep(wait_time)
                return self._call_veo_api(client, model, prompt, config, reference_image,
                                         retry_count + 1, max_retries)
            else:
                self._log(f"Maximum retries ({max_retries}) reached. Giving up.")
                return None

    def _poll_video_generation(self, client, operation,
                              polling_interval: int = 10,
                              max_wait_time: int = 360) -> Optional[dict]:
        """Poll the API until video generation is complete"""
        try:
            start_time = time.time()
            self._log(f"Starting to poll for video generation")

            while not operation.done:
                elapsed_time = time.time() - start_time

                if elapsed_time > max_wait_time:
                    self._log(f"Maximum wait time ({max_wait_time}s) exceeded")
                    return None

                self._log(f"Still processing... ({elapsed_time:.1f}s elapsed)")
                time.sleep(polling_interval)

                # Refresh operation status
                try:
                    operation = client.operations.get(operation)
                except Exception as e:
                    self._log(f"Error checking operation status: {str(e)}")
                    time.sleep(polling_interval)

            elapsed_time = time.time() - start_time
            self._log(f"Video generation completed after {elapsed_time:.1f} seconds")
            return operation

        except Exception as e:
            self._log(f"Error during polling: {str(e)}")
            return None

    def generate_video(self, prompt: str, service_account_json: str, model: str,
                      aspect_ratio: str = "16:9", resolution: str = "720p",
                      max_retries: int = 3, polling_interval: int = 10,
                      max_wait_time: int = 360, negative_prompt: str = "",
                      seed: int = 0, reference_image=None,
                      enable_person_generation: bool = True):
        """Generate video using VEO 3 API"""

        # Reset log messages
        self.log_messages = []

        try:
            # Validate service account JSON file
            if not service_account_json:
                error_message = "Error: No service account JSON file provided."
                self._log(error_message)
                error_frame = self._create_error_frame("Service account required")
                return (error_frame, "", error_message)

            # Build full path to service account JSON
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            json_path = os.path.join(script_dir, service_account_json)

            if not os.path.exists(json_path):
                error_message = f"Error: Service account JSON file not found at {json_path}"
                self._log(error_message)
                error_frame = self._create_error_frame("JSON file not found")
                return (error_frame, "", error_message)

            # Validate prompt
            if not prompt or len(prompt.strip()) == 0:
                error_message = "Error: No prompt provided. Please enter a video generation prompt."
                self._log(error_message)
                error_frame = self._create_error_frame("Prompt required")
                return (error_frame, "", error_message)

            # Validate resolution for aspect ratio
            if resolution == "1080p" and aspect_ratio != "16:9":
                self._log("Warning: 1080p is only available for 16:9 aspect ratio. Defaulting to 720p.")
                resolution = "720p"

            # Load credentials from service account JSON
            self._log(f"Loading service account credentials from: {json_path}")

            # Read the JSON file to get project_id
            with open(json_path, 'r') as f:
                service_account_info = json.load(f)
                project_id = service_account_info.get('project_id')

            if not project_id:
                error_message = "Error: No project_id found in service account JSON"
                self._log(error_message)
                error_frame = self._create_error_frame("Missing project_id")
                return (error_frame, "", error_message)

            credentials = service_account.Credentials.from_service_account_file(
                json_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )

            # Initialize client with Vertex AI
            client = genai.Client(
                vertexai=True,
                project=project_id,
                location='us-central1',
                credentials=credentials
            )
            self._log(f"Gemini client initialized with Vertex AI (project: {project_id})")

            # Store credentials for later use in downloads
            self._credentials = credentials

            # Process reference image if provided
            image_for_api = None
            if reference_image is not None:
                pil_reference = self._process_tensor_to_pil(reference_image, "Reference Image")
                if pil_reference:
                    # Prepare image for Vertex AI
                    image_for_api = self._prepare_image_for_api(pil_reference)
                    if not image_for_api:
                        error_message = "Failed to prepare reference image"
                        self._log(error_message)
                        error_frame = self._create_error_frame(error_message)
                        return (error_frame, "", error_message)

            # Build generation config
            # Start with None and only add if basic call fails
            gen_config = None

            self._log(f"Requested aspect ratio: {aspect_ratio}")
            self._log(f"Requested resolution: {resolution}")
            if negative_prompt:
                self._log(f"Negative prompt: {negative_prompt[:100]}...")
            if seed > 0:
                self._log(f"Note: Seed {seed} provided")
            self._log(f"Person generation: {'enabled' if enable_person_generation else 'disabled'}")

            # Make initial API call
            self._log("Initiating video generation...")
            operation = self._call_veo_api(
                client=client,
                model=model,
                prompt=prompt,
                config=gen_config,
                reference_image=image_for_api,
                max_retries=max_retries
            )

            if operation is None:
                error_message = "Failed to initiate video generation"
                self._log(error_message)
                error_frame = self._create_error_frame(error_message)
                return (error_frame, "", error_message)

            self._log(f"Operation started, waiting for completion...")

            # Poll for completion
            completed_operation = self._poll_video_generation(
                client=client,
                operation=operation,
                polling_interval=polling_interval,
                max_wait_time=max_wait_time
            )

            if completed_operation is None:
                error_message = "Video generation timed out or failed"
                self._log(error_message)
                error_frame = self._create_error_frame(error_message)
                return (error_frame, "", error_message)

            # Extract video from completed operation
            video_file = self._extract_video_from_operation(completed_operation)

            if not video_file:
                error_message = "No video found in completed operation"
                self._log(error_message)
                error_frame = self._create_error_frame(error_message)
                return (error_frame, "", error_message)

            return self._process_video_result(client, video_file)

        except Exception as e:
            error_message = f"Error during video generation: {str(e)}"
            self._log(error_message)
            traceback.print_exc()

            error_frame = self._create_error_frame(f"Error: {str(e)}")
            full_text = "## Processing Log\n" + "\n".join(self.log_messages) + "\n\n## Error\n" + error_message

            return (error_frame, "", full_text)

    def _extract_video_from_operation(self, operation) -> Optional[any]:
        """Extract video file object from completed operation"""
        try:
            # Try multiple possible structures

            # Try operation.result first
            if hasattr(operation, 'result') and operation.result:
                self._log(f"Found result attribute")
                self._log(f"Result type: {type(operation.result)}")
                self._log(f"Result attributes: {dir(operation.result)}")

                # Check for generated_videos in result
                if hasattr(operation.result, 'generated_videos') and operation.result.generated_videos:
                    if len(operation.result.generated_videos) > 0:
                        generated_video = operation.result.generated_videos[0]
                        self._log(f"Generated video type: {type(generated_video)}")
                        self._log(f"Generated video attributes: {dir(generated_video)}")
                        if hasattr(generated_video, 'video'):
                            self._log(f"Found video file object")
                            return generated_video.video

                # Maybe the result IS the video
                return operation.result

            # Try operation.response
            if hasattr(operation, 'response') and operation.response:
                if hasattr(operation.response, 'generated_videos') and operation.response.generated_videos:
                    if len(operation.response.generated_videos) > 0:
                        generated_video = operation.response.generated_videos[0]
                        if hasattr(generated_video, 'video'):
                            self._log(f"Found video file object in response")
                            return generated_video.video

            self._log("Could not find video in operation structure")
            self._log(f"Operation type: {type(operation)}")

            return None

        except Exception as e:
            self._log(f"Error extracting video from operation: {str(e)}")
            traceback.print_exc()
            return None

    def _process_video_result(self, client, video_file) -> Tuple:
        """Download and process the generated video"""
        try:
            self._log(f"Processing video result")
            self._log(f"Video file type: {type(video_file)}")

            # Download video file using the client
            temp_file = os.path.join(
                tempfile.gettempdir(),
                f"veo3_video_{int(time.time())}.mp4"
            )

            self._log(f"Attempting to save video to: {temp_file}")

            # Check if video_bytes is available (Vertex AI)
            if hasattr(video_file, 'video_bytes') and video_file.video_bytes:
                self._log(f"Using video_bytes from Video object")

                video_data = video_file.video_bytes
                self._log(f"video_bytes size: {len(video_data)} bytes")

                # Check if data is base64 encoded
                # Raw MP4 files start with specific magic bytes
                # MP4/M4V: 0x66747970 ('ftyp') at offset 4
                # Base64 encoded data will be ASCII/UTF-8 text
                is_base64 = False
                try:
                    if isinstance(video_data, bytes):
                        # Check first 20 bytes
                        test_header = video_data[:20]
                        self._log(f"Video data header (hex): {test_header.hex()}")

                        # Try to decode as ASCII - base64 is ASCII
                        try:
                            test_str = test_header.decode('ascii')
                            # Check if it looks like base64 (no control chars, valid base64 chars)
                            # Common base64 patterns for video might start with various chars
                            # But we can check if it's printable ASCII which binary MP4 is not
                            if all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r' for c in test_str):
                                is_base64 = True
                                self._log(f"Detected base64-encoded video data")
                        except UnicodeDecodeError:
                            # Binary data, not base64
                            self._log(f"Video data is binary (not base64)")
                            pass
                except Exception as e:
                    self._log(f"Error checking video data format: {str(e)}")

                # Decode base64 if needed
                if is_base64:
                    try:
                        original_size = len(video_data)
                        video_data = base64.b64decode(video_data)
                        self._log(f"Decoded base64 video data: {original_size} -> {len(video_data)} bytes")

                        # Verify it's now a valid video file
                        video_header = video_data[:20] if len(video_data) >= 20 else video_data
                        self._log(f"Decoded video header (hex): {video_header.hex()}")
                    except Exception as e:
                        self._log(f"Failed to decode base64 video data: {str(e)}")
                        traceback.print_exc()
                        return self._create_error_frame(f"Base64 decode error: {str(e)}"), "", str(e)

                # Write the video data to file
                with open(temp_file, 'wb') as f:
                    f.write(video_data)

                file_size = os.path.getsize(temp_file)
                self._log(f"Video saved: {file_size} bytes")

                # Verify the file is a valid MP4
                with open(temp_file, 'rb') as f:
                    file_header = f.read(20)
                    self._log(f"Saved file header (hex): {file_header.hex()}")
                    # Check for MP4 signature (ftyp at offset 4-7)
                    if len(file_header) >= 8:
                        ftyp_check = file_header[4:8]
                        if ftyp_check == b'ftyp':
                            self._log(f"Verified valid MP4 file signature")
                        else:
                            self._log(f"WARNING: File may not be a valid MP4 (ftyp signature not found)")
                    else:
                        self._log(f"WARNING: File too small to verify MP4 signature")

            # Otherwise try using the URI (Google AI API)
            elif hasattr(video_file, 'uri') and video_file.uri:
                self._log(f"Attempting to download from URI")
                # Extract file ID from URI
                # URI format: https://generativelanguage.googleapis.com/v1beta/files/{file_id}:download?alt=media
                try:
                    file_id_match = video_file.uri.split('/files/')
                    if len(file_id_match) > 1:
                        file_id = file_id_match[1].split(':')[0]
                        self._log(f"Extracted file ID: {file_id}")

                        # Try to download using the file ID with authenticated request
                        try:
                            # Refresh credentials if needed
                            if not self._credentials.valid:
                                self._credentials.refresh(Request())

                            # Get the access token from service account credentials
                            download_url = f"https://generativelanguage.googleapis.com/v1beta/files/{file_id}"

                            headers = {
                                'Authorization': f'Bearer {self._credentials.token}'
                            }

                            self._log(f"Downloading from: {download_url}")

                            response = requests.get(download_url, headers=headers)
                            response.raise_for_status()

                            # Get the actual download URL from the response
                            file_info = response.json()
                            self._log(f"File info: {file_info}")

                            # Check if file is still processing
                            if file_info.get('state') == 'PROCESSING':
                                self._log(f"File is still processing, waiting for it to be ready...")
                                max_file_wait = 60  # Wait up to 60 seconds for file processing
                                file_wait_start = time.time()

                                while file_info.get('state') == 'PROCESSING':
                                    if time.time() - file_wait_start > max_file_wait:
                                        self._log("File processing timeout")
                                        return self._create_error_frame("File processing timeout"), "", "Timeout"

                                    time.sleep(2)
                                    response = requests.get(download_url, headers=headers)
                                    response.raise_for_status()
                                    file_info = response.json()
                                    self._log(f"File state: {file_info.get('state')}")

                            self._log(f"File is ready, state: {file_info.get('state')}")

                            # Now download the actual video
                            if 'downloadUri' in file_info:
                                video_uri = file_info['downloadUri']
                                self._log(f"Downloading video from downloadUri: {video_uri}")

                                video_response = requests.get(video_uri, headers=headers, stream=True)
                                video_response.raise_for_status()

                                with open(temp_file, 'wb') as f:
                                    for chunk in video_response.iter_content(chunk_size=8192):
                                        f.write(chunk)

                                file_size = os.path.getsize(temp_file)
                                self._log(f"Video saved: {file_size} bytes")
                            elif 'uri' in file_info:
                                video_uri = file_info['uri']
                                self._log(f"Downloading video from: {video_uri}")

                                video_response = requests.get(video_uri, headers=headers, stream=True)
                                video_response.raise_for_status()

                                with open(temp_file, 'wb') as f:
                                    for chunk in video_response.iter_content(chunk_size=8192):
                                        f.write(chunk)

                                file_size = os.path.getsize(temp_file)
                                self._log(f"Video saved: {file_size} bytes")
                            else:
                                self._log("No URI in file info")
                                return self._create_error_frame("No download URI"), "", "No URI"

                        except Exception as e:
                            self._log(f"Authenticated download failed: {e}")
                            traceback.print_exc()
                            raise
                except Exception as e:
                    self._log(f"URI download failed: {e}")
                    traceback.print_exc()
                    return self._create_error_frame(f"Download error: {str(e)}"), "", str(e)
            else:
                self._log("No video_bytes or URI found in video object")
                return self._create_error_frame("No video data found"), "", "No video data"

            video_path = temp_file

            # Extract frames
            frames = self._extract_frames_from_video(video_path)

            if not frames or len(frames) == 0:
                error_message = "Failed to extract frames from video"
                self._log(error_message)
                error_frame = self._create_error_frame(error_message)
                return (error_frame, "", error_message)

            # Clean up temp file
            try:
                os.remove(video_path)
                self._log("Temporary video file cleaned up")
            except:
                pass

            # Stack frames into a batch tensor [B, H, W, C]
            frames_batch = torch.stack(frames, dim=0)

            # Prepare response text
            response_text = "## Video Generation Successful\n"
            response_text += f"Video saved to: {video_path}\n"
            response_text += f"Total frames extracted: {len(frames)}\n"
            response_text += f"Batch shape: {frames_batch.shape}\n"
            response_text += "\n## Processing Log\n" + "\n".join(self.log_messages)

            self._log(f"Successfully processed video with {len(frames)} frames")

            return (frames_batch, video_path, response_text)

        except Exception as e:
            error_message = f"Error processing video result: {str(e)}"
            self._log(error_message)
            traceback.print_exc()
            error_frame = self._create_error_frame(error_message)
            return (error_frame, "", error_message)
