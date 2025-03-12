import os
import torch
import numpy as np
import cv2
import subprocess
from PIL import Image


class FL_ProResVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "step": 1
                }),
                "filename": ("STRING", {
                    "default": "prores_output",
                    "multiline": False
                }),
                "save_directory": ("STRING", {
                    "default": "",  # Empty string will use ComfyUI/output
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "create_video"
    CATEGORY = "🏵️Fill Nodes/Video"
    OUTPUT_NODE = True

    def create_video(self, images, fps, filename, save_directory):
        # Clean filename (remove any extension if user added one)
        clean_filename = os.path.splitext(filename)[0]

        # Generate unique timestamp
        timestamp = torch.randint(0, 100000, (1,)).item()

        # If save_directory is empty, use ComfyUI output directory
        if not save_directory.strip():
            save_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

        # Create output directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Create full output path
        output_path = os.path.join(save_directory, f"{clean_filename}.mov")

        # If file exists, append timestamp
        if os.path.exists(output_path):
            output_path = os.path.join(save_directory, f"{clean_filename}_{timestamp}.mov")

        # Create temporary path in the same directory
        temp_path = os.path.join(save_directory, f"temp_{timestamp}.mp4")

        # Get dimensions from first frame
        height, width = images[0].shape[0], images[0].shape[1]

        # Initialize video writer for temporary file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        # Write frames
        for image in images:
            # Convert from torch tensor to numpy array and scale to 0-255
            frame = (255.0 * image.cpu().numpy()).astype(np.uint8)
            # OpenCV expects BGR format
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        # Release the video writer
        out.release()

        # Convert to ProRes using ffmpeg
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', temp_path,  # Input file
            '-c:v', 'prores_ks',  # ProRes codec
            '-profile:v', '4444',  # ProRes 4444
            '-pix_fmt', 'yuva444p10le',  # Pixel format
            '-vendor', 'apl0',  # Apple compatibility
            '-bits_per_mb', '8000',  # High quality
            output_path  # Output file
        ]

        try:
            # Run ffmpeg command
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            message = f"Video saved to {output_path}"
        except subprocess.CalledProcessError as e:
            message = f"Error creating ProRes video: {e.stderr.decode()}"
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return (message,)