import os
import torch
import numpy as np
from PIL import Image
import imageio # For WebM creation
import shutil

class FL_SaveWebM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "directory": ("STRING", {"default": "./output/webm"}),
                "filename_prefix": ("STRING", {"default": "animation"}),
                "fps": ("INT", {"default": 15, "min": 1, "max": 60}),
                "preserve_alpha": (["disable", "enable"], {"default": "disable"}),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "description": "0 for infinite loop (Note: may not affect WebM playback directly via imageio)"}),
                "quality": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step":0.1, "description": "Quality (0-10, lower is better for VP9, but imageio might map it, typically 5-8 is good)"}),
            },
            "optional": {
                "metadata_comment": ("STRING", {"multiline": True, "default": "Created with ComfyUI FL_SaveWebM"}),
            }
        }

    RETURN_TYPES = ("STRING",) # Returns the path to the saved WebM file
    FUNCTION = "save_webm_animation"
    OUTPUT_NODE = True
    CATEGORY = "🏵️Fill Nodes/Image"

    def save_webm_animation(self, images: torch.Tensor, directory: str, filename_prefix: str, fps: int, preserve_alpha: str, loop_count: int, quality: float, metadata_comment: str = ""):
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Determine unique filename
        i = 0
        while True:
            filename = f"{filename_prefix}_{i:04d}.webm"
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                break
            i += 1
        
        batch_size, height, width, channels = images.shape
        frames = []

        has_alpha = preserve_alpha == "enable" and channels == 4
        
        for i in range(batch_size):
            img_tensor = images[i] # HWC
            # Convert to numpy array, scale to 0-255, and change to uint8
            np_frame = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            
            if has_alpha:
                # Ensure it's RGBA
                if np_frame.shape[-1] == 3: # If input is RGB but alpha is requested, add opaque alpha
                    alpha_channel = np.full((height, width, 1), 255, dtype=np.uint8)
                    np_frame = np.concatenate((np_frame, alpha_channel), axis=-1)
                # imageio expects RGBA for formats supporting alpha
            else:
                # Ensure it's RGB if alpha is not preserved or not present
                if np_frame.shape[-1] == 4:
                    np_frame = np_frame[..., :3] # Slice off alpha
            
            frames.append(np_frame)

        # imageio-ffmpeg is typically needed for WebM (VP9)
        # For VP9 with alpha, the pixel format 'yuva420p' is often used.
        # imageio's mimsave handles this if the input frames have an alpha channel.
        
        writer_kwargs = {
            'fps': fps,
            'quality': quality, # For VP9, lower is better. imageio might map this.
            # 'loop': loop_count, # Loop is not directly supported for WebM writer like this
            'format': 'WEBM', # Explicitly specify WEBM
            'codec': 'libvpx-vp9', # VP9 is standard for WebM
        }
        
        # Handle metadata separately
        ffmpeg_metadata_params = []
        if metadata_comment:
            ffmpeg_metadata_params.extend(['-metadata', f'comment={metadata_comment}'])
        
        if ffmpeg_metadata_params:
            writer_kwargs['ffmpeg_params'] = ffmpeg_metadata_params

        if has_alpha:
            print("Attempting to save WebM with alpha channel. Frames are RGBA. Relying on imageio and libvpx-vp9 for alpha handling.")
        else:
            print("Saving WebM without alpha channel (RGB).")
            
        try:
            # When codec is 'libvpx-vp9' and frames are RGBA, imageio-ffmpeg should handle alpha.
            imageio.mimsave(filepath, frames, **writer_kwargs)
            print(f"Saved WebM to: {filepath}")
        except Exception as e:
            # Fallback if ffmpeg is not found or there's an issue
            if 'ffmpeg' in str(e).lower() and shutil.which('ffmpeg') is None:
                msg = "ffmpeg not found. Please install ffmpeg and ensure it's in your system PATH. Cannot save WebM."
                print(f"Error: {msg} - {e}")
                return (f"Error: {msg}",)

            print(f"Error saving WebM: {e}")
            # Attempt fallback without specific codec or alpha handling if primary fails
            try:
                print("Attempting fallback WebM save...")
                fallback_kwargs = {'fps': fps, 'format': 'WEBM'} # Removed loop from fallback as well
                if not has_alpha: # If original attempt was with alpha, try without for fallback
                     rgb_frames = [(frame[..., :3] if frame.shape[-1] == 4 else frame) for frame in frames]
                     imageio.mimsave(filepath, rgb_frames, **fallback_kwargs)
                else: # If original was already RGB or alpha is critical, this might not help much
                     imageio.mimsave(filepath, frames, **fallback_kwargs)
                print(f"Fallback WebM saved to: {filepath}")
            except Exception as e2:
                print(f"Fallback WebM save also failed: {e2}")
                return (f"Error: Could not save WebM. {e} / {e2}",)

        return (filepath,)