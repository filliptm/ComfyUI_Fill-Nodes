import os
import torch
from moviepy.editor import ImageSequenceClip
import numpy as np
import tempfile
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
                "output_path": ("STRING", {
                    "default": "output.mov"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "create_video"
    CATEGORY = "🏵️Fill Nodes/video"
    OUTPUT_NODE = True

    def create_video(self, images, fps, output_path):
        # Ensure output path ends with .mov
        if not output_path.lower().endswith('.mov'):
            output_path += '.mov'

        # Convert tensor images to numpy arrays
        image_list = []
        for i in range(images.shape[0]):
            img = images[i].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            image_list.append(img)

        # Create a temporary directory to store PNG files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save images as PNG files to preserve alpha
            temp_image_files = []
            for i, img in enumerate(image_list):
                img_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                Image.fromarray(img).save(img_path)
                temp_image_files.append(img_path)

            # Create video clip
            clip = ImageSequenceClip(temp_image_files, fps=fps)

            # Write video file
            clip.write_videofile(
                output_path,
                codec='prores_ks',
                preset='4444',
                ffmpeg_params=["-pix_fmt", "yuva444p10le"],
                verbose=False,
                logger=None
            )

        return (f"Video saved to {output_path}",)