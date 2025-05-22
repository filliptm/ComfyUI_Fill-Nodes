import torch
import numpy as np
from PIL import Image

class FL_ImageBlank:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "red": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "green": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_blank_image"
    CATEGORY = "🏵️Fill Nodes/Image"

    def create_blank_image(self, width, height, red, green, blue):
        # Create a 3-channel image (H, W, C)
        image_np = np.full((height, width, 3), [red, green, blue], dtype=np.uint8)
        
        # Convert to PIL Image first to handle potential mode issues if needed, then to numpy float32
        # Though for a simple color, direct numpy to tensor is fine.
        # image_pil = Image.fromarray(image_np, 'RGB')
        # image_np_float = np.array(image_pil).astype(np.float32) / 255.0
        
        image_np_float = image_np.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension (B, H, W, C)
        image_tensor = torch.from_numpy(image_np_float)[None,]
        
        return (image_tensor,)