import torch
import numpy as np
from PIL import Image
from scipy import ndimage


class FL_ReplaceColor:
    DESCRIPTION = """
FL_ReplaceColor allows you to replace a specific color in an image with another color. 
You can use the interactive color pickers to select the source color to replace and the target color to replace it with.
The tolerance parameter controls how closely colors need to match the source color to be replaced.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_color": ("STRING", {"default": "#FF0000"}),
                "target_color": ("STRING", {"default": "#00FF00"}),
                "tolerance": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "Color matching tolerance (0.0 = exact match, 1.0 = very loose)"
                }),
                "expand_pixels": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "description": "Expand replacement area by N pixels (dilation)"
                }),
                "contract_pixels": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "description": "Contract replacement area by N pixels (erosion)"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "replace_color"
    CATEGORY = "🏵️Fill Nodes/Image"

    def replace_color(self, image: torch.Tensor, source_color: str, target_color: str, tolerance: float, expand_pixels: int, contract_pixels: int):
        batch_size = image.shape[0]
        result = torch.zeros_like(image)

        # Convert hex colors to RGB
        source_rgb = self.hex_to_rgb(source_color)
        target_rgb = self.hex_to_rgb(target_color)

        for b in range(batch_size):
            img_b = image[b] * 255.0
            img_array = img_b.numpy().astype('uint8')
            
            # Convert to PIL Image for processing
            pil_image = Image.fromarray(img_array, 'RGB')
            result_image = self.replace_color_in_image(pil_image, source_rgb, target_rgb, tolerance, expand_pixels, contract_pixels)
            
            # Convert back to tensor
            result_array = np.array(result_image).astype(np.float32) / 255.0
            result[b] = torch.from_numpy(result_array)

        return (result,)

    def hex_to_rgb(self, hex_color: str):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def replace_color_in_image(self, image: Image.Image, source_rgb, target_rgb, tolerance, expand_pixels, contract_pixels):
        """Replace colors in PIL Image with expand/contract controls"""
        img_array = np.array(image)
        
        # Calculate color distance for each pixel
        source_color = np.array(source_rgb)
        
        # Compute Euclidean distance in RGB space
        distances = np.sqrt(np.sum((img_array - source_color) ** 2, axis=2))
        
        # Normalize distance to 0-1 range (max distance in RGB is sqrt(3*255^2))
        max_distance = np.sqrt(3 * 255 ** 2)
        normalized_distances = distances / max_distance
        
        # Create initial mask for pixels within tolerance
        mask = normalized_distances <= tolerance
        
        # Apply morphological operations for expand/contract
        mask = self.apply_morphological_operations(mask, expand_pixels, contract_pixels)
        
        # Replace colors
        result_array = img_array.copy()
        result_array[mask] = target_rgb
        
        return Image.fromarray(result_array.astype(np.uint8))
    
    def apply_morphological_operations(self, mask, expand_pixels, contract_pixels):
        """Apply morphological operations to expand or contract the mask"""
        processed_mask = mask.copy()
        
        # Apply contraction first (erosion) if specified
        if contract_pixels > 0:
            # Create circular structuring element for more natural results
            structure = self.create_circular_kernel(contract_pixels)
            processed_mask = ndimage.binary_erosion(processed_mask, structure=structure)
        
        # Apply expansion (dilation) if specified
        if expand_pixels > 0:
            # Create circular structuring element for more natural results
            structure = self.create_circular_kernel(expand_pixels)
            processed_mask = ndimage.binary_dilation(processed_mask, structure=structure)
        
        return processed_mask
    
    def create_circular_kernel(self, radius):
        """Create a circular structuring element for morphological operations"""
        if radius == 0:
            return None
        
        # Create a circular kernel
        size = 2 * radius + 1
        y, x = np.ogrid[:size, :size]
        center = radius
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        
        return mask.astype(np.uint8)