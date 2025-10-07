import comfy.utils
import math
import nodes
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, grey_dilation, binary_fill_holes, binary_closing

class FL_InpaintCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Mode selection (determines which parameters are used)
                "mode": (["free", "long side", "short side", "range", "forced"], {"default": "free"}),
                
                # Basic parameters (used by all modes)
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "fill_mask_holes": ("BOOLEAN", {"default": True}),
                "use_gpu": ("BOOLEAN", {"default": False}),
                
                # Context expansion parameters (used by all modes)
                "context_pixels": ("INT", {"default": 10, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "context_factor": ("FLOAT", {"default": 1.01, "min": 1.0, "max": 100.0, "step": 0.01}),
                
                # Square ratio option (for all modes)
                "force_square": ("BOOLEAN", {"default": False, "display": "Force 1:1 Square Ratio"}),
                
                # Mode-specific parameters
                "forced_size": ("INT", {"default": 1024, "min": 32, "max": 2048, "step": 32, "display": "Size (for forced mode)"}),
                "minimum_size": ("INT", {"default": 512, "min": 32, "max": 2048, "step": 32, "display": "Minimum Size (for range mode)"}),
                "maximum_size": ("INT", {"default": 1024, "min": 32, "max": 2048, "step": 32, "display": "Maximum Size (for range mode)"}),
                "target_size": ("INT", {"default": 1024, "min": 32, "max": 2048, "step": 32, "display": "Target Size (for long/short side modes)"}),
                "free_rescale": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 100.0, "step": 0.01, "display": "Rescale Factor (for free mode)"}),
                "free_padding": ("INT", {"default": 32, "min": 8, "max": 512, "step": 8, "display": "Padding (for free mode)"}),
           },
           "optional": {
                "optional_context_mask": ("MASK",),
           }
        }

    CATEGORY = "🏵️Fill Nodes/Utility"

    RETURN_TYPES = ("STITCH", "IMAGE", "MASK")
    RETURN_NAMES = ("stitch", "cropped_image", "cropped_mask")

    FUNCTION = "inpaint_crop"

    def adjust_to_square(self, x_min, x_max, y_min, y_max, width, height, target_size = None):
        if target_size is None:
            x_size = x_max - x_min + 1
            y_size = y_max - y_min + 1
            target_size = max(x_size, y_size)

        # Calculate the midpoint of the current x and y ranges
        x_mid = (x_min + x_max) // 2
        y_mid = (y_min + y_max) // 2

        # Adjust x_min, x_max, y_min, y_max to make the range square centered around the midpoints
        x_min = max(x_mid - target_size // 2, 0)
        x_max = x_min + target_size - 1
        y_min = max(y_mid - target_size // 2, 0)
        y_max = y_min + target_size - 1

        # Ensure the ranges do not exceed the image boundaries
        if x_max >= width:
            x_max = width - 1
            x_min = x_max - target_size + 1
        if y_max >= height:
            y_max = height - 1
            y_min = y_max - target_size + 1

        # Additional checks to make sure all coordinates are within bounds
        if x_min < 0:
            x_min = 0
            x_max = target_size - 1
        if y_min < 0:
            y_min = 0
            y_max = target_size - 1

        return x_min, x_max, y_min, y_max

    def apply_padding(self, min_val, max_val, max_boundary, padding):
        # Calculate the midpoint and the original range size
        original_range_size = max_val - min_val + 1
        midpoint = (min_val + max_val) // 2

        # Determine the smallest multiple of padding that is >= original_range_size
        if original_range_size % padding == 0:
            new_range_size = original_range_size
        else:
            new_range_size = (original_range_size // padding + 1) * padding

        # Calculate the new min and max values centered on the midpoint
        new_min_val = max(midpoint - new_range_size // 2, 0)
        new_max_val = new_min_val + new_range_size - 1

        # Ensure the new max doesn't exceed the boundary
        if new_max_val >= max_boundary:
            new_max_val = max_boundary - 1
            new_min_val = max(new_max_val - new_range_size + 1, 0)

        # Ensure the range still ends on a multiple of padding
        # Adjust if the calculated range isn't feasible within the given constraints
        if (new_max_val - new_min_val + 1) != new_range_size:
            new_min_val = max(new_max_val - new_range_size + 1, 0)

        return new_min_val, new_max_val

    # GPU-accelerated version of binary_fill_holes
    def binary_fill_holes_gpu(self, binary_mask_tensor):
        # Create a kernel for morphological operations
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=binary_mask_tensor.device)
        
        # Perform closing operation (dilation followed by erosion)
        # First pad the tensor to avoid border issues
        padding = kernel_size // 2
        padded = F.pad(binary_mask_tensor.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='constant', value=1)
        
        # Dilate
        dilated = F.conv2d(padded, kernel, padding=0)
        dilated = (dilated > 0).float()
        
        # Erode
        inverted = 1 - dilated
        eroded_inv = F.conv2d(inverted, kernel, padding=0)
        closed = 1 - (eroded_inv > 0).float()
        
        # Remove padding
        closed = closed[0, 0, padding:-padding, padding:-padding]
        
        # Fill holes (find isolated regions of 0s surrounded by 1s)
        # Start with the border
        h, w = closed.shape
        filled = closed.clone()
        
        # Create a mask of border pixels
        border = torch.zeros_like(filled)
        border[0, :] = 1
        border[-1, :] = 1
        border[:, 0] = 1
        border[:, -1] = 1
        
        # Set border pixels to 0 in our working mask if they're 0 in the original
        seed = torch.ones_like(filled)
        seed[border == 1] = closed[border == 1]
        
        # Iteratively propagate the seed inward
        # This is a simplified approach - for complex images, more iterations might be needed
        for _ in range(max(h, w)):
            # Pad for convolution
            padded_seed = F.pad(seed.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='constant', value=0)
            
            # Use convolution to check neighbors
            neighbors = F.conv2d(padded_seed, torch.ones(1, 1, 3, 3, device=seed.device), padding=0)
            neighbors = neighbors[0, 0]
            
            # Propagate: if any neighbor is marked and the current pixel is not a foreground pixel in the original
            new_seed = (neighbors > 0).float() * (1 - closed)
            
            # Update seed with newly marked pixels
            seed = torch.clamp(seed + new_seed, 0, 1)
            
            # Check if we've converged
            if not new_seed.any():
                break
        
        # The holes are the areas that weren't reached
        holes = (1 - seed) * (1 - closed)
        
        # Add the holes to the original mask
        filled = torch.clamp(closed + holes, 0, 1)
        
        return filled

    # Parts of this function are from KJNodes: https://github.com/kijai/ComfyUI-KJNodes
    def inpaint_crop(self, mode, image, mask, invert_mask, fill_mask_holes, use_gpu, context_pixels, context_factor, force_square, forced_size, minimum_size, maximum_size, target_size, free_rescale, free_padding, optional_context_mask = None):
        # Check if GPU is available when requested
        use_gpu = use_gpu and torch.cuda.is_available()
        if use_gpu:
            print("Using GPU for inpaint crop operations")
        else:
            print("Using CPU for inpaint crop operations")
        original_image = image
        original_mask = mask
        original_width = image.shape[2]
        original_height = image.shape[1]

        #Validate or initialize mask
        if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
            non_zero_indices = torch.nonzero(mask[0], as_tuple=True)
            if not non_zero_indices[0].size(0):
                mask = torch.zeros_like(image[:, :, :, 0])
            else:
                assert False, "mask size must match image size"

        # Invert mask if requested
        if invert_mask:
            mask = 1.0 - mask

        # Fill holes if requested
        if fill_mask_holes:
            if use_gpu:
                # GPU implementation
                device = mask.device
                holemask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
                out = []
                for m in holemask:
                    # Convert to binary mask
                    binary_mask = m > 0
                    # Apply GPU-accelerated hole filling
                    filled_mask = self.binary_fill_holes_gpu(binary_mask)
                    # Scale to match the expected output range
                    output = filled_mask * 255
                    out.append(output)
                mask = torch.stack(out, dim=0)
                mask = torch.clamp(mask, 0.0, 1.0)
            else:
                # Original CPU implementation
                holemask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
                out = []
                for m in holemask:
                    mask_np = m.numpy()
                    binary_mask = mask_np > 0
                    struct = np.ones((5, 5))
                    closed_mask = binary_closing(binary_mask, structure=struct, border_value=1)
                    filled_mask = binary_fill_holes(closed_mask)
                    output = filled_mask.astype(np.float32) * 255
                    output = torch.from_numpy(output)
                    out.append(output)
                mask = torch.stack(out, dim=0)
                mask = torch.clamp(mask, 0.0, 1.0)

        # Validate or initialize context mask
        if optional_context_mask is None:
            context_mask = mask
        elif optional_context_mask.shape[1] != image.shape[1] or optional_context_mask.shape[2] != image.shape[2]:
            non_zero_indices = torch.nonzero(optional_context_mask[0], as_tuple=True)
            if not non_zero_indices[0].size(0):
                context_mask = mask
            else:
                assert False, "context_mask size must match image size"
        else:
            context_mask = optional_context_mask + mask 
            context_mask = torch.clamp(context_mask, 0.0, 1.0)

        # If there are no non-zero indices in the context_mask, return the original image and original mask
        non_zero_indices = torch.nonzero(context_mask[0], as_tuple=True)
        if not non_zero_indices[0].size(0):
            stitch = {'x': 0, 'y': 0, 'original_image': original_image, 'cropped_mask': mask, 'rescale_x': 1.0, 'rescale_y': 1.0}
            return (stitch, original_image, original_mask)

        # Compute context area from context mask
        y_min = torch.min(non_zero_indices[0]).item()
        y_max = torch.max(non_zero_indices[0]).item()
        x_min = torch.min(non_zero_indices[1]).item()
        x_max = torch.max(non_zero_indices[1]).item()
        height = context_mask.shape[1]
        width = context_mask.shape[2]
        
        # Grow context area if requested
        y_size = y_max - y_min + 1
        x_size = x_max - x_min + 1
        y_grow = round(max(y_size*(context_factor-1), context_pixels))
        x_grow = round(max(x_size*(context_factor-1), context_pixels))
        y_min = max(y_min - y_grow // 2, 0)
        y_max = min(y_max + y_grow // 2, height - 1)
        x_min = max(x_min - x_grow // 2, 0)
        x_max = min(x_max + x_grow // 2, width - 1)

        effective_upscale_factor_x = 1.0
        effective_upscale_factor_y = 1.0
        
        # Apply square adjustment if requested
        if force_square:
            x_min, x_max, y_min, y_max = self.adjust_to_square(x_min, x_max, y_min, y_max, width, height)
        
        # Adjust to preferred size based on mode
        if mode == 'forced':
            # Force to exact size (square if force_square is True)
            current_width = x_max - x_min + 1
            current_height = y_max - y_min + 1
            
            if force_square:
                current_size = current_width  # Should be equal to current_height after adjust_to_square
                if current_size != forced_size:
                    # Upscale to fit in the forced_size, will be downsized at stitch phase
                    upscale_factor = forced_size / current_size
            else:
                # For non-square, we'll scale to make the larger dimension match forced_size
                larger_dimension = max(current_width, current_height)
                if larger_dimension != forced_size:
                    upscale_factor = forced_size / larger_dimension

                samples = image
                samples = samples.movedim(-1, 1)

                width = math.floor(samples.shape[3] * upscale_factor)
                height = math.floor(samples.shape[2] * upscale_factor)
                samples = comfy.utils.bislerp(samples, width, height)
                effective_upscale_factor_x = float(width)/float(original_width)
                effective_upscale_factor_y = float(height)/float(original_height)
                samples = samples.movedim(1, -1)
                image = samples

                samples = mask
                samples = samples.unsqueeze(1)
                samples = comfy.utils.bislerp(samples, width, height)
                samples = samples.squeeze(1)
                mask = samples

                x_min = math.floor(x_min * effective_upscale_factor_x)
                x_max = math.floor(x_max * effective_upscale_factor_x)
                y_min = math.floor(y_min * effective_upscale_factor_y)
                y_max = math.floor(y_max * effective_upscale_factor_y)

                # Readjust to forced size because the upscale math may not round well
                x_min, x_max, y_min, y_max = self.adjust_to_square(x_min, x_max, y_min, y_max, width, height, target_size=forced_size)

        elif mode == 'range':
            current_width = x_max - x_min + 1
            current_height = y_max - y_min + 1
            
            if force_square:
                current_size = current_width  # Should be equal to current_height after adjust_to_square
                
                # Only upscale if the current size is smaller than the minimum size
                if current_size < minimum_size:
                    # Upscale to fit the minimum size, will be downsized at stitch phase
                    upscale_factor = minimum_size / current_size
                    print(f"Upscaling from {current_size} to minimum size {minimum_size}")
                    
                    samples = image
                    samples = samples.movedim(-1, 1)

                    width = math.floor(samples.shape[3] * upscale_factor)
                    height = math.floor(samples.shape[2] * upscale_factor)
                    samples = comfy.utils.bislerp(samples, width, height)
                    effective_upscale_factor_x = float(width)/float(original_width)
                    effective_upscale_factor_y = float(height)/float(original_height)
                    samples = samples.movedim(1, -1)
                    image = samples

                    samples = mask
                    samples = samples.unsqueeze(1)
                    samples = comfy.utils.bislerp(samples, width, height)
                    samples = samples.squeeze(1)
                    mask = samples

                    x_min = math.floor(x_min * effective_upscale_factor_x)
                    x_max = math.floor(x_max * effective_upscale_factor_x)
                    y_min = math.floor(y_min * effective_upscale_factor_y)
                    y_max = math.floor(y_max * effective_upscale_factor_y)

                    # Readjust to minimum size because the upscale math may not round well
                    x_min, x_max, y_min, y_max = self.adjust_to_square(x_min, x_max, y_min, y_max, width, height, target_size=minimum_size)
                    
                # Only downscale if the current size is larger than the maximum size
                elif current_size > maximum_size:
                    # Downscale to fit the maximum size
                    upscale_factor = maximum_size / current_size
                    print(f"Downscaling from {current_size} to maximum size {maximum_size}")
                    
                    samples = image
                    samples = samples.movedim(-1, 1)

                    width = math.floor(samples.shape[3] * upscale_factor)
                    height = math.floor(samples.shape[2] * upscale_factor)
                    samples = comfy.utils.bislerp(samples, width, height)
                    effective_upscale_factor_x = float(width)/float(original_width)
                    effective_upscale_factor_y = float(height)/float(original_height)
                    samples = samples.movedim(1, -1)
                    image = samples

                    samples = mask
                    samples = samples.unsqueeze(1)
                    samples = comfy.utils.bislerp(samples, width, height)
                    samples = samples.squeeze(1)
                    mask = samples

                    x_min = math.floor(x_min * effective_upscale_factor_x)
                    x_max = math.floor(x_max * effective_upscale_factor_x)
                    y_min = math.floor(y_min * effective_upscale_factor_y)
                    y_max = math.floor(y_max * effective_upscale_factor_y)

                    # Readjust to maximum size because the upscale math may not round well
                    x_min, x_max, y_min, y_max = self.adjust_to_square(x_min, x_max, y_min, y_max, width, height, target_size=maximum_size)
                    
                else:
                    print(f"No scaling needed. Current size {current_size} is within range {minimum_size}-{maximum_size}")
            else:
                # For non-square, we'll check both dimensions
                larger_dimension = max(current_width, current_height)
                
                if larger_dimension < minimum_size:
                    upscale_factor = minimum_size / larger_dimension
                    print(f"Upscaling from {larger_dimension} to minimum size {minimum_size}")
                    
                    samples = image
                    samples = samples.movedim(-1, 1)

                    width = math.floor(samples.shape[3] * upscale_factor)
                    height = math.floor(samples.shape[2] * upscale_factor)
                    samples = comfy.utils.bislerp(samples, width, height)
                    effective_upscale_factor_x = float(width)/float(original_width)
                    effective_upscale_factor_y = float(height)/float(original_height)
                    samples = samples.movedim(1, -1)
                    image = samples

                    samples = mask
                    samples = samples.unsqueeze(1)
                    samples = comfy.utils.bislerp(samples, width, height)
                    samples = samples.squeeze(1)
                    mask = samples

                    x_min = math.floor(x_min * effective_upscale_factor_x)
                    x_max = math.floor(x_max * effective_upscale_factor_x)
                    y_min = math.floor(y_min * effective_upscale_factor_y)
                    y_max = math.floor(y_max * effective_upscale_factor_y)
                    
                elif larger_dimension > maximum_size:
                    upscale_factor = maximum_size / larger_dimension
                    print(f"Downscaling from {larger_dimension} to maximum size {maximum_size}")
                    
                    samples = image
                    samples = samples.movedim(-1, 1)

                    width = math.floor(samples.shape[3] * upscale_factor)
                    height = math.floor(samples.shape[2] * upscale_factor)
                    samples = comfy.utils.bislerp(samples, width, height)
                    effective_upscale_factor_x = float(width)/float(original_width)
                    effective_upscale_factor_y = float(height)/float(original_height)
                    samples = samples.movedim(1, -1)
                    image = samples

                    samples = mask
                    samples = samples.unsqueeze(1)
                    samples = comfy.utils.bislerp(samples, width, height)
                    samples = samples.squeeze(1)
                    mask = samples

                    x_min = math.floor(x_min * effective_upscale_factor_x)
                    x_max = math.floor(x_max * effective_upscale_factor_x)
                    y_min = math.floor(y_min * effective_upscale_factor_y)
                    y_max = math.floor(y_max * effective_upscale_factor_y)
                    
                else:
                    print(f"No scaling needed. Larger dimension {larger_dimension} is within range {minimum_size}-{maximum_size}")
        
        elif mode == 'long side':
            current_width = x_max - x_min + 1
            current_height = y_max - y_min + 1
            longer_side = max(current_width, current_height)
            
            if longer_side != target_size:
                upscale_factor = target_size / longer_side
                print(f"Scaling to make longer side ({longer_side}) match target size {target_size}")
                
                samples = image
                samples = samples.movedim(-1, 1)
                
                width = math.floor(samples.shape[3] * upscale_factor)
                height = math.floor(samples.shape[2] * upscale_factor)
                samples = comfy.utils.bislerp(samples, width, height)
                effective_upscale_factor_x = float(width)/float(original_width)
                effective_upscale_factor_y = float(height)/float(original_height)
                samples = samples.movedim(1, -1)
                image = samples
                
                samples = mask
                samples = samples.unsqueeze(1)
                samples = comfy.utils.bislerp(samples, width, height)
                samples = samples.squeeze(1)
                mask = samples
                
                x_min = math.floor(x_min * effective_upscale_factor_x)
                x_max = math.floor(x_max * effective_upscale_factor_x)
                y_min = math.floor(y_min * effective_upscale_factor_y)
                y_max = math.floor(y_max * effective_upscale_factor_y)
                
                # If force_square is true, readjust to square after scaling
                if force_square:
                    x_min, x_max, y_min, y_max = self.adjust_to_square(x_min, x_max, y_min, y_max, width, height)
        
        elif mode == 'short side':
            current_width = x_max - x_min + 1
            current_height = y_max - y_min + 1
            shorter_side = min(current_width, current_height)
            
            if shorter_side != target_size:
                upscale_factor = target_size / shorter_side
                print(f"Scaling to make shorter side ({shorter_side}) match target size {target_size}")
                
                samples = image
                samples = samples.movedim(-1, 1)
                
                width = math.floor(samples.shape[3] * upscale_factor)
                height = math.floor(samples.shape[2] * upscale_factor)
                samples = comfy.utils.bislerp(samples, width, height)
                effective_upscale_factor_x = float(width)/float(original_width)
                effective_upscale_factor_y = float(height)/float(original_height)
                samples = samples.movedim(1, -1)
                image = samples
                
                samples = mask
                samples = samples.unsqueeze(1)
                samples = comfy.utils.bislerp(samples, width, height)
                samples = samples.squeeze(1)
                mask = samples
                
                x_min = math.floor(x_min * effective_upscale_factor_x)
                x_max = math.floor(x_max * effective_upscale_factor_x)
                y_min = math.floor(y_min * effective_upscale_factor_y)
                y_max = math.floor(y_max * effective_upscale_factor_y)
                
                # If force_square is true, readjust to square after scaling
                if force_square:
                    x_min, x_max, y_min, y_max = self.adjust_to_square(x_min, x_max, y_min, y_max, width, height)
        
        elif mode == 'free':
            # Upscale image and masks if requested, they will be downsized at stitch phase
            if free_rescale < 0.999 or free_rescale > 1.001:
                samples = image            
                samples = samples.movedim(-1, 1)

                width = math.floor(samples.shape[3] * free_rescale)
                height = math.floor(samples.shape[2] * free_rescale)
                samples = comfy.utils.bislerp(samples, width, height)
                effective_upscale_factor_x = float(width)/float(original_width)
                effective_upscale_factor_y = float(height)/float(original_height)
                samples = samples.movedim(1, -1)
                image = samples

                samples = mask
                samples = samples.unsqueeze(1)
                samples = comfy.utils.bislerp(samples, width, height)
                samples = samples.squeeze(1)
                mask = samples

                x_min = math.floor(x_min * effective_upscale_factor_x)
                x_max = math.floor(x_max * effective_upscale_factor_x)
                y_min = math.floor(y_min * effective_upscale_factor_y)
                y_max = math.floor(y_max * effective_upscale_factor_y)

                # Ensure that context area doesn't go outside of the image
                x_min = max(x_min, 0)
                x_max = min(x_max, width - 1)
                y_min = max(y_min, 0)
                y_max = min(y_max, height - 1)

            # Pad area (if possible, i.e. if pad is smaller than width/height) to avoid the sampler returning smaller results
            if free_padding > 1:
                x_min, x_max = self.apply_padding(x_min, x_max, width, free_padding)
                y_min, y_max = self.apply_padding(y_min, y_max, height, free_padding)


        # Crop the image and the mask, sized context area
        cropped_image = image[:, y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask[:, y_min:y_max+1, x_min:x_max+1]

        # Return stitch (to be consumed by the class below), image, and mask
        stitch = {'x': x_min, 'y': y_min, 'original_image': original_image, 'cropped_mask': cropped_mask, 'rescale_x': effective_upscale_factor_x, 'rescale_y': effective_upscale_factor_y}
        return (stitch, cropped_image, cropped_mask)

class FL_Inpaint_Stitch:
    """
    ComfyUI-InpaintCropAndStitch
    https://github.com/lquesada/ComfyUI-InpaintCropAndStitch

    This node stitches the inpainted image without altering unmasked areas.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitch": ("STITCH",),
                "inpainted_image": ("IMAGE",),
            }
        }

    CATEGORY = "🏵️Fill Nodes/Utility"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "inpaint_stitch"

    # This function is from comfy_extras: https://github.com/comfyanonymous/ComfyUI
    def composite(self, destination, source, x, y, mask = None, multiplier = 8, resize_source = False):
        source = source.to(destination.device)
        if resize_source:
            source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

        source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

        x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
        y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

        left, top = (x // multiplier, y // multiplier)
        right, bottom = (left + source.shape[3], top + source.shape[2],)

        if mask is None:
            mask = torch.ones_like(source)
        else:
            mask = mask.to(destination.device, copy=True)
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
            mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

        # calculate the bounds of the source that will be overlapping the destination
        # this prevents the source trying to overwrite latent pixels that are out of bounds
        # of the destination
        visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

        mask = mask[:, :, :visible_height, :visible_width]
        inverse_mask = torch.ones_like(mask) - mask
            
        source_portion = mask * source[:, :, :visible_height, :visible_width]
        destination_portion = inverse_mask  * destination[:, :, top:bottom, left:right]

        destination[:, :, top:bottom, left:right] = source_portion + destination_portion
        return destination

    def inpaint_stitch(self, stitch, inpainted_image):
        original_image = stitch['original_image']
        cropped_mask = stitch['cropped_mask']
        x = stitch['x']
        y = stitch['y']
        stitched_image = original_image.clone().movedim(-1, 1)

        inpaint_width = inpainted_image.shape[2]
        inpaint_height = inpainted_image.shape[1]

        # Downscale inpainted before stitching if we upscaled it before
        if stitch['rescale_x'] < 0.999 or stitch['rescale_x'] > 1.001 or stitch['rescale_y'] < 0.999 or stitch['rescale_y'] > 1.001:
            samples = inpainted_image.movedim(-1, 1)
            width = round(float(inpaint_width)/stitch['rescale_x'])
            height = round(float(inpaint_height)/stitch['rescale_y'])
            x = round(float(x)/stitch['rescale_x'])
            y = round(float(y)/stitch['rescale_y'])
            samples = comfy.utils.bislerp(samples, width, height)
            inpainted_image = samples.movedim(1, -1)
            
            samples = cropped_mask.movedim(-1, 1)
            samples = samples.unsqueeze(0)
            samples = comfy.utils.bislerp(samples, width, height)
            samples = samples.squeeze(0)
            cropped_mask = samples.movedim(1, -1)

        output = self.composite(stitched_image, inpainted_image.movedim(-1, 1), x, y, cropped_mask, 1).movedim(1, -1)

        return (output,)
