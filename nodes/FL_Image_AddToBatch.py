import torch
import torch.nn.functional as F

class FL_ImageAddToBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images1": ("IMAGE", {}),
                "images2": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_to_batch"
    CATEGORY = "🏵️Fill Nodes/Image"

    def add_to_batch(self, images1, images2):
        # Convert both inputs to the same format
        # ComfyUI typically uses [batch, height, width, channels] format
        # Make sure both have batch dimension
        if images1.dim() == 3:  # [height, width, channels]
            images1 = images1.unsqueeze(0)  # Add batch dimension
        
        if images2.dim() == 3:  # [height, width, channels]
            images2 = images2.unsqueeze(0)  # Add batch dimension
        
        # Get target dimensions from first image
        _, h1, w1, c1 = images1.shape
        _, h2, w2, c2 = images2.shape
        
        # Ensure channels match
        if c1 != c2:
            raise ValueError(f"Channel dimensions must match. Got {c1} and {c2}")
        
        # Resize second image if dimensions don't match
        if h1 != h2 or w1 != w2:
            # Convert to [batch, channels, height, width] for interpolate
            images2_chw = images2.permute(0, 3, 1, 2)
            
            # Choose algorithm based on scaling direction
            is_upscaling = (h1 > h2) or (w1 > w2)
            mode = 'bicubic' if is_upscaling else 'area'
            
            # Resize
            images2_chw_resized = F.interpolate(
                images2_chw,
                size=(h1, w1),
                mode=mode,
                align_corners=False if mode in ['bicubic', 'bilinear'] else None
            )
            
            # Convert back to [batch, height, width, channels]
            images2 = images2_chw_resized.permute(0, 2, 3, 1)
        
        # Now concatenate along batch dimension
        combined_images = torch.cat([images1, images2], dim=0)
        
        return (combined_images,)