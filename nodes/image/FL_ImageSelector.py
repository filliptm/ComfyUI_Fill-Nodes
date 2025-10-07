import torch
from ..utils import tensor_to_pil, pil_to_tensor

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
        # Get batch size for "last" keyword processing
        batch_size = images.shape[0]
        
        # Parse the indices string
        try:
            # Split by comma and process each index
            index_parts = [idx.strip() for idx in indices.split(',') if idx.strip()]
            index_list = []
            
            for idx_str in index_parts:
                if idx_str.lower() == "last":
                    # Replace "last" with the last index (batch_size - 1)
                    if batch_size > 0:
                        index_list.append(batch_size - 1)
                else:
                    # Convert to integer
                    index_list.append(int(idx_str))
                    
        except ValueError:
            print("Error: Indices must be comma-separated integers or 'last'")
            # Return the original batch if parsing fails
            return (images,)

        # Validate indices
        valid_indices = [idx for idx in index_list if 0 <= idx < batch_size]
        
        if not valid_indices:
            print(f"Warning: No valid indices provided. Batch size is {batch_size}")
            # Return empty batch with same dimensions as input
            return (images[0:0],)
        
        # Select the specified images
        selected_images = images[valid_indices]
        
        return (selected_images,)