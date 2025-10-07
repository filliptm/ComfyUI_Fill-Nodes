import base64
import io
import json
import numpy as np
import os
import tempfile
import requests
from PIL import Image
import torch
import cv2
from tqdm import tqdm
import fal_client

class FL_Fal_Seedance_i2v:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Enter your Fal AI API key"
                }),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the video animation you want to create"
                }),
                "cut_prompt_1": ("STRING",),
                "cut_prompt_2": ("STRING",),
                "cut_prompt_3": ("STRING",),
                "cut_prompt_4": ("STRING",),
                "resolution": (["480p", "720p", "1080p"], {
                    "default": "720p"
                }),
                "duration": ("INT", {
                    "default": 6,
                    "min": 3,
                    "max": 12,
                    "step": 1
                }),
                "camera_fixed": ("BOOLEAN", {
                    "default": True
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 999999999
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("frames", "video_url", "status_message")
    FUNCTION = "generate_video"
    CATEGORY = "🏵️Fill Nodes/AI"
    
    def tensor_to_base64(self, tensor_image):
        # Convert tensor to PIL Image
        if tensor_image.dim() == 4:
            tensor_image = tensor_image.squeeze(0)
        if tensor_image.shape[0] == 3:  # CHW to HWC
            tensor_image = tensor_image.permute(1, 2, 0)
        
        # Ensure values are in [0, 1] range and convert to [0, 255]
        if tensor_image.max() <= 1.0:
            tensor_image = (tensor_image * 255).clamp(0, 255)
        
        # Convert to uint8 and then to PIL Image
        np_image = tensor_image.cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(np_image)
        
        # Convert to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_base64}"

    def download_video_frames(self, video_url):
        """Download video and extract all frames"""
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path = os.path.join(temp_dir, "video.mp4")
                
                # Download video with progress bar
                response = requests.get(video_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(video_path, 'wb') as f:
                    with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading video") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                # Extract frames using OpenCV
                cap = cv2.VideoCapture(video_path)
                frames = []
                frame_count = 0
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                with tqdm(total=total_frames, desc="Extracting frames") as pbar:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Extract every frame
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Convert to PIL Image
                        pil_image = Image.fromarray(frame_rgb)
                        
                        # Convert to tensor
                        tensor_frame = torch.from_numpy(np.array(pil_image)).float() / 255.0
                        tensor_frame = tensor_frame.unsqueeze(0)  # Add batch dimension
                        
                        frames.append(tensor_frame)
                        
                        frame_count += 1
                        pbar.update(1)
                
                cap.release()
                
                if frames:
                    # Stack all frames
                    all_frames = torch.cat(frames, dim=0)
                    return all_frames
                else:
                    # Return a dummy black frame if no frames extracted
                    dummy_frame = torch.zeros(1, 512, 512, 3)
                    return dummy_frame
                    
        except Exception as e:
            print(f"Error downloading/processing video: {str(e)}")
            # Return a dummy black frame
            dummy_frame = torch.zeros(1, 512, 512, 3)
            return dummy_frame

    def generate_video(self, api_key, image, prompt, resolution, duration, camera_fixed, seed, 
                       cut_prompt_1="", cut_prompt_2="", cut_prompt_3="", cut_prompt_4=""):
        if not api_key or api_key.strip() == "":
            # Create dummy outputs
            dummy_frames = torch.zeros(1, 512, 512, 3)
            return (dummy_frames, "", "Error: API key is required")
        
        try:
            # Convert the first image from the batch to base64
            if image.dim() == 4 and image.shape[0] > 0:
                first_image = image[0]
            else:
                first_image = image
            
            image_b64 = self.tensor_to_base64(first_image)
            
            # Clean and set API key
            clean_api_key = api_key.strip()
            key_preview = clean_api_key[:8] + "..." if len(clean_api_key) > 8 else clean_api_key
            print(f"[Fal Seedance] Using API key starting with: {key_preview}")
            
            # Clear and set the environment variable
            if "FAL_KEY" in os.environ:
                del os.environ["FAL_KEY"]
            os.environ["FAL_KEY"] = clean_api_key
            
            # Build the complete prompt with cut prompts
            complete_prompt = prompt.strip()
            
            # Add cut prompts if provided
            cut_prompts = [cut_prompt_1, cut_prompt_2, cut_prompt_3, cut_prompt_4]
            for i, cut_prompt in enumerate(cut_prompts):
                if cut_prompt and cut_prompt.strip():
                    complete_prompt += f" [cut] {cut_prompt.strip()}"
            
            print(f"[Fal Seedance] Complete prompt: {complete_prompt}")
            
            # Prepare arguments
            arguments = {
                "image_url": image_b64,
                "prompt": complete_prompt,
                "resolution": resolution,
                "duration": duration,
                "camera_fixed": camera_fixed
            }
            
            # Add seed if specified
            if seed >= 0:
                arguments["seed"] = seed
            
            # Print arguments without exposing potentially large base64 data
            safe_arguments = {k: v if not (isinstance(v, str) and v.startswith('data:')) else f"<data_uri_{len(v)}_chars>" for k, v in arguments.items()}
            print(f"[Fal Seedance] Making API call with arguments: {safe_arguments}")
            
            # Force reload the fal_client module to avoid caching issues
            import sys
            if 'fal_client' in sys.modules:
                del sys.modules['fal_client']
            import fal_client
            
            # Submit the request synchronously
            result = fal_client.subscribe(
                "fal-ai/bytedance/seedance/v1/pro/image-to-video",
                arguments=arguments,
                with_logs=True
            )
            
            print(f"[Fal Seedance] API call completed successfully")
            
            if result and "video" in result and "url" in result["video"]:
                video_url = result["video"]["url"]
                
                # Download and extract frames
                all_frames = self.download_video_frames(video_url)
                
                total_frames = all_frames.shape[0]
                status_message = f"Successfully generated video using Fal AI Bytedance Seedance. Extracted {total_frames} frames."
                
                return (all_frames, video_url, status_message)
                
            else:
                error_msg = "No video URL in API response"
                print(f"[Fal Seedance] {error_msg}")
                # Return dummy outputs
                dummy_frames = torch.zeros(1, 512, 512, 3)
                return (dummy_frames, "", error_msg)
            
        except Exception as e:
            error_msg = f"Error in video generation: {str(e)}"
            print(f"[Fal Seedance] {error_msg}")
            
            # Return dummy outputs
            dummy_frames = torch.zeros(1, 512, 512, 3)
            return (dummy_frames, "", error_msg)

# Node registration
NODE_CLASS_MAPPINGS = {
    "FL_Fal_Seedance_i2v": FL_Fal_Seedance_i2v
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_Fal_Seedance_i2v": "FL Fal Seedance i2v"
}