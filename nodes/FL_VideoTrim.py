import torch

class FL_VideoTrim:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "trim_start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "trim_end": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "trim_video"
    CATEGORY = "🏵️Fill Nodes/Video"

    def trim_video(self, images, trim_start, trim_end):
        # Get the number of frames in the batch
        batch_size = images.shape[0]
        
        # Check if trimming is possible
        if trim_start + trim_end >= batch_size:
            print(f"[FL_VideoTrim] Warning: Cannot trim {trim_start} frames from start and {trim_end} frames from end as batch only has {batch_size} frames.")
            print(f"[FL_VideoTrim] Returning original batch.")
            return (images,)
        
        # Calculate the start and end indices for the trimmed batch
        start_idx = trim_start
        end_idx = batch_size - trim_end
        
        # Extract the trimmed frames
        trimmed_images = images[start_idx:end_idx]
        
        print(f"[FL_VideoTrim] Trimmed {trim_start} frames from start and {trim_end} frames from end.")
        print(f"[FL_VideoTrim] Original batch size: {batch_size}, New batch size: {trimmed_images.shape[0]}")
        
        return (trimmed_images,)