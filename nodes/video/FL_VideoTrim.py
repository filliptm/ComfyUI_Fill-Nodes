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

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("trimmed_video", "start_frames", "end_frames")
    FUNCTION = "trim_video"
    CATEGORY = "🏵️Fill Nodes/Video"

    def trim_video(self, images, trim_start, trim_end):
        # Get the number of frames in the batch
        batch_size = images.shape[0]
        
        # Check if trimming is possible
        if trim_start + trim_end >= batch_size:
            print(f"[FL_VideoTrim] Warning: Cannot trim {trim_start} frames from start and {trim_end} frames from end as batch only has {batch_size} frames.")
            print(f"[FL_VideoTrim] Returning original batch with empty start/end frames.")
            # Return original batch and empty tensors for start/end frames
            empty_tensor = torch.empty(0, *images.shape[1:], dtype=images.dtype, device=images.device)
            return (images, empty_tensor, empty_tensor)
        
        # Calculate the start and end indices for the trimmed batch
        start_idx = trim_start
        end_idx = batch_size - trim_end
        
        # Extract the trimmed frames
        trimmed_images = images[start_idx:end_idx]
        
        # Extract start frames (if any)
        if trim_start > 0:
            start_frames = images[:trim_start]
        else:
            start_frames = torch.empty(0, *images.shape[1:], dtype=images.dtype, device=images.device)
        
        # Extract end frames (if any)
        if trim_end > 0:
            end_frames = images[end_idx:]
        else:
            end_frames = torch.empty(0, *images.shape[1:], dtype=images.dtype, device=images.device)
        
        print(f"[FL_VideoTrim] Trimmed {trim_start} frames from start and {trim_end} frames from end.")
        print(f"[FL_VideoTrim] Original batch size: {batch_size}, New batch size: {trimmed_images.shape[0]}")
        print(f"[FL_VideoTrim] Start frames: {start_frames.shape[0]}, End frames: {end_frames.shape[0]}")
        
        return (trimmed_images, start_frames, end_frames)