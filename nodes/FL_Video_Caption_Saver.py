import os
import re
import numpy as np
import torch
from moviepy.editor import ImageSequenceClip
from comfy.utils import ProgressBar

class FL_VideoCaptionSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "folder_name": ("STRING", {"default": "output_videos"}),
                "caption_text": ("STRING", {"default": "Your caption here"}),
                "video_name": ("STRING", {"default": "video"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "format": (["mp4", "webm"], {"default": "mp4"}),
                "quality": ("INT", {"default": 8, "min": 1, "max": 10, "step": 1}),
                "overwrite": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_video_with_caption"
    CATEGORY = "🏵️Fill Nodes/Captioning"
    OUTPUT_NODE = True

    def sanitize_text(self, text):
        return re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)

    def save_video_with_caption(self, images, folder_name, caption_text, video_name, fps, format, quality, overwrite):
        os.makedirs(folder_name, exist_ok=True)
        sanitized_caption = self.sanitize_text(caption_text)
        
        # Sanitize the video name
        video_name = re.sub(r'[^a-zA-Z0-9_-]', '_', video_name)
        
        # Create file paths
        video_file_name = f"{folder_name}/{video_name}.{format}"
        text_file_name = f"{folder_name}/{video_name}.txt"
        
        # Handle overwrite
        if not overwrite:
            counter = 1
            while os.path.exists(video_file_name) or os.path.exists(text_file_name):
                video_file_name = f"{folder_name}/{video_name}_{counter}.{format}"
                text_file_name = f"{folder_name}/{video_name}_{counter}.txt"
                counter += 1
        
        # Convert tensor batch to numpy arrays
        frames = []
        print(f"[FL_VideoCaptionSaver] Processing {images.shape[0]} frames for video")
        
        pbar = ProgressBar(images.shape[0])
        for i in range(images.shape[0]):
            # Convert tensor to numpy array (0-255 range)
            img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            
            # If the image is grayscale (2D), convert to RGB
            if len(img_np.shape) == 2:
                img_np = np.stack((img_np,) * 3, axis=-1)
            elif img_np.shape[2] == 1:  # If it's (height, width, 1)
                img_np = np.repeat(img_np, 3, axis=2)
                
            frames.append(img_np)
            pbar.update_absolute(i)
        
        # Calculate video bitrate based on quality (1-10)
        # Higher quality means higher bitrate
        base_bitrate = 1000000  # 1 Mbps as base
        bitrate = base_bitrate * (quality / 5)  # Scale by quality
        
        print(f"[FL_VideoCaptionSaver] Creating {format} video at {fps} fps with quality {quality}")
        
        try:
            # Create clip from frames
            clip = ImageSequenceClip(frames, fps=fps)
            
            # Save as video
            if format == "mp4":
                clip.write_videofile(
                    video_file_name,
                    codec='libx264',
                    bitrate=f"{int(bitrate)}",
                    audio=False,
                    logger=None  # Suppress MoviePy progress bars
                )
            elif format == "webm":
                clip.write_videofile(
                    video_file_name,
                    codec='libvpx',
                    bitrate=f"{int(bitrate)}",
                    audio=False,
                    logger=None  # Suppress MoviePy progress bars
                )
            
            # Save caption text
            with open(text_file_name, "w") as text_file:
                text_file.write(sanitized_caption)
                
            print(f"[FL_VideoCaptionSaver] Successfully saved video to {video_file_name}")
            print(f"[FL_VideoCaptionSaver] Successfully saved caption to {text_file_name}")
            
            return (f"Saved video with {images.shape[0]} frames and caption in '{folder_name}'",)
            
        except Exception as e:
            error_msg = f"Error saving video: {str(e)}"
            print(f"[FL_VideoCaptionSaver] {error_msg}")
            return (f"Error: {error_msg}",)