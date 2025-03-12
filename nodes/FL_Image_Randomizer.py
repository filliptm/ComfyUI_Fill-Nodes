import os
import numpy as np
import torch
from PIL import Image, ImageOps
import cv2
import random

class FL_ImageRandomizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["Image", "Video"], {"default": "Image"}),
                "directory_path": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "search_subdirectories": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "PATH")
    FUNCTION = "select_media"
    CATEGORY = "🏵️Fill Nodes/Image"

    def select_media(self, mode, directory_path, seed, search_subdirectories=False):
        if not directory_path:
            raise ValueError("Directory path is not provided.")
        
        if mode == "Image":
            return self.select_image(directory_path, seed, search_subdirectories)
        else:  # Video mode
            return self.select_video_frames(directory_path, seed, search_subdirectories)
    
    def select_image(self, directory_path, seed, search_subdirectories=False):
        images = self.load_files(directory_path, search_subdirectories, file_type="image")
        if not images:
            raise ValueError("No images found in the specified directory.")

        num_images = len(images)
        selected_index = seed % num_images

        selected_image_path = images[selected_index]

        image = Image.open(selected_image_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        return (image_tensor, selected_image_path)
    
    def select_video_frames(self, directory_path, seed, search_subdirectories=False):
        videos = self.load_files(directory_path, search_subdirectories, file_type="video")
        if not videos:
            raise ValueError("No videos found in the specified directory.")

        num_videos = len(videos)
        selected_index = seed % num_videos

        selected_video_path = videos[selected_index]

        # Open the video file
        cap = cv2.VideoCapture(selected_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {selected_video_path}")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count <= 0:
            raise ValueError(f"No frames found in video: {selected_video_path}")
        
        # Extract all frames from the video
        frames = []
        success = True
        while success:
            success, frame = cap.read()
            if success:
                # Convert BGR to RGB (OpenCV uses BGR by default)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Normalize
                frame_np = np.array(frame).astype(np.float32) / 255.0
                frames.append(frame_np)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"Failed to extract frames from video: {selected_video_path}")
            
        # Convert list of frames to tensor with batch dimension
        frames_tensor = torch.from_numpy(np.stack(frames))
        
        return (frames_tensor, selected_video_path)

    def load_files(self, directory, search_subdirectories=False, file_type="image"):
        if file_type == "image":
            supported_formats = ["jpg", "jpeg", "png", "bmp", "gif", "webp"]
        else:  # video
            supported_formats = ["mp4", "avi", "mov", "mkv", "wmv", "webm"]
            
        file_paths = []

        if search_subdirectories:
            for root, _, files in os.walk(directory):
                for f in files:
                    if f.split('.')[-1].lower() in supported_formats:
                        file_paths.append(os.path.join(root, f))
        else:
            file_paths = sorted([os.path.join(directory, f) for f in os.listdir(directory)
                                if os.path.isfile(os.path.join(directory, f)) and f.split('.')[-1].lower() in supported_formats])

        return sorted(file_paths)