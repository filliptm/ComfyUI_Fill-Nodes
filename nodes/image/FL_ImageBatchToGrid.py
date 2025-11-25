import torch
import math

class FL_ImageBatchToGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "images_per_row": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_grid"
    CATEGORY = "🏵️Fill Nodes/Image"

    def create_grid(self, images, images_per_row):
        batch_size, height, width, channels = images.shape

        # Calculate the number of rows needed
        num_rows = math.ceil(batch_size / images_per_row)

        # Create a blank canvas for the grid (filled with black)
        grid = torch.zeros((num_rows * height, images_per_row * width, channels), dtype=images.dtype, device=images.device)

        for i in range(batch_size):
            row = i // images_per_row
            col = i % images_per_row

            y_start = row * height
            y_end = (row + 1) * height
            x_start = col * width
            x_end = (col + 1) * width

            grid[y_start:y_end, x_start:x_end, :] = images[i]

        # Add batch dimension
        return (grid.unsqueeze(0),)