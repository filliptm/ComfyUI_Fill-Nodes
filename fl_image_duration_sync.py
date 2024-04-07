import numpy as np
import torch
from PIL import Image, ImageOps

class FL_ImageDurationSync:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "frame_count": ("INT", {"default": 30}),
                "bpm": ("INT", {"default": 120}),
                "fps": ("INT", {"default": 30}),
                "bars": ("FLOAT", {"default": 4.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("output_images", "hold_frames")
    FUNCTION = "sync_image_to_duration"
    CATEGORY = "🏵️Fill Nodes"

    def sync_image_to_duration(self, images, bpm, frame_count, bars, fps):
        # Calculate the duration of each bar in seconds
        bar_duration = 60 / bpm * 4

        # Calculate the total duration in seconds
        total_duration = bar_duration * bars

        # Calculate the number of frames to hold the image
        hold_frames = int(total_duration * fps)

        # Repeat the image for the calculated number of frames
        output_images = images.repeat(hold_frames, 1, 1, 1)

        return (output_images, hold_frames)