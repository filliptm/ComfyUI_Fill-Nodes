import torch
from nodes import ImageBatch # Assuming ImageBatch is a ComfyUI core node or accessible

class FL_ImageBatch: # Renamed class
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 100, "step": 1}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine_images"
    CATEGORY = "🏵️Fill Nodes/Image" # Kept category, can be changed if needed
    DESCRIPTION = """
FL Image Batch allows combining multiple images into a batch.
The number of image inputs can be dynamically adjusted.
""" # Updated description

    def combine_images(self, inputcount, **kwargs):
        image_batch_node = ImageBatch()
        
        if "image_1" not in kwargs:
             return (torch.empty(0),)

        image = kwargs["image_1"]
        
        if inputcount == 1:
            return (image,)

        for c in range(1, inputcount):
            image_key = f"image_{c + 1}"
            if image_key in kwargs:
                new_image = kwargs[image_key]
                if image.ndim == 3: 
                    image = image.unsqueeze(0) 
                if new_image.ndim == 3: 
                    new_image = new_image.unsqueeze(0)

                image, = image_batch_node.batch(image, new_image)
            else:
                print(f"Warning: {image_key} not found in inputs for FL_ImageBatch.")
                pass
        return (image,)