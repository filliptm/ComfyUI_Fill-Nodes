import torch
from .utils import tensor_to_pil, pil_to_tensor

class FL_ImageSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "indices": ("STRING", {"default": "0,1,2"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select_images"
    CATEGORY = "🏵️Fill Nodes/Image"

    def select_images(self, images, indices):
        # Parse the indices string
        try:
            # Split by comma and convert to integers
            index_list = [int(idx.strip()) for idx in indices.split(',') if idx.strip()]
        except ValueError:
            print("Error: Indices must be comma-separated integers")
            # Return the original batch if parsing fails
            return (images,)

        # Validate indices
        batch_size = images.shape[0]
        valid_indices = [idx for idx in index_list if 0 <= idx < batch_size]
        
        if not valid_indices:
            print(f"Warning: No valid indices provided. Batch size is {batch_size}")
            # Return empty batch with same dimensions as input
            return (images[0:0],)
        
        # Select the specified images
        selected_images = images[valid_indices]
        
        return (selected_images,)