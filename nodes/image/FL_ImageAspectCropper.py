import torch
from ..utils import tensor_to_pil, pil_to_tensor

class FL_ImageAspectCropper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "aspect_ratio_w": ("INT", {"default": 16, "min": 1, "max": 100, "step": 1}),
                "aspect_ratio_h": ("INT", {"default": 9, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_to_aspect"
    CATEGORY = "🏵️Fill Nodes/Image"

    def crop_to_aspect(self, images, aspect_ratio_w, aspect_ratio_h):
        batch_size = images.shape[0]
        height = images.shape[1]
        width = images.shape[2]
        channels = images.shape[3]
        
        # Calculate target aspect ratio (convert to float for division)
        target_ratio = float(aspect_ratio_w) / float(aspect_ratio_h)
        current_ratio = width / height
        
        result_tensors = []
        
        for i in range(batch_size):
            img = images[i]
            
            if current_ratio > target_ratio:
                # Image is wider than target ratio, crop width
                new_width = int(height * target_ratio)
                new_width = new_width - (new_width % 2) # Ensure even number
                # Calculate left position to center the crop
                left = (width - new_width) // 2
                # Crop the image
                cropped_img = img[:, left:left+new_width, :]
            else:
                # Image is taller than target ratio, crop height
                new_height = int(width / target_ratio)
                new_height = new_height - (new_height % 2) # Ensure even number
                # Calculate top position to center the crop
                top = (height - new_height) // 2
                # Crop the image
                cropped_img = img[top:top+new_height, :, :]
            
            result_tensors.append(cropped_img.unsqueeze(0))
        
        # Concatenate all cropped images back into a batch
        if result_tensors:
            return (torch.cat(result_tensors, dim=0),)
        else:
            # Return empty batch with same dimensions as input if no images were processed
            return (images[0:0],)