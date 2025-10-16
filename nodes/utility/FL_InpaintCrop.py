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
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "context_factor": ("FLOAT", {
                    "default": 1.2,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.05,
                    "description": "Multiply mask bounds by this factor (1.2 = 20% larger)"
                }),
                "target_size": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 4096,
                    "step": 64,
                    "description": "Target size (0 = no resize, uses bucketing to maintain aspect ratio)"
                }),
                "resize_mode": (["none", "max_dimension", "bucket"], {
                    "default": "bucket",
                    "description": "none: no resize | max_dimension: scale longest side | bucket: snap to nearest size"
                }),
                "force_square": ("BOOLEAN", {
                    "default": False,
                    "description": "Force output to be square (uses larger of width/height)"
                }),
                "divisible_by": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "description": "Ensure dimensions are divisible by this (expands symmetrically)"
                }),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "fill_mask_holes": ("BOOLEAN", {"default": True}),
                "use_gpu": ("BOOLEAN", {"default": False}),
           },
           "optional": {
                "optional_context_mask": ("MASK",),
           }
        }

    CATEGORY = "🏵️Fill Nodes/Utility"

    RETURN_TYPES = ("STITCH", "IMAGE", "MASK")
    RETURN_NAMES = ("stitch", "cropped_image", "cropped_mask")

    FUNCTION = "inpaint_crop"

    def expand_to_divisible(self, min_val, max_val, max_boundary, divisible_by):
        """
        Expand range symmetrically to make size divisible by divisible_by.
        This ensures perfect alignment when stitching back.
        """
        current_size = max_val - min_val + 1

        if divisible_by <= 1:
            return min_val, max_val

        # Calculate the next size that's divisible
        if current_size % divisible_by == 0:
            new_size = current_size
        else:
            new_size = ((current_size // divisible_by) + 1) * divisible_by

        # Expand symmetrically around the center
        center = (min_val + max_val) / 2
        new_min = int(center - new_size / 2)
        new_max = new_min + new_size - 1

        # Clamp to boundaries
        if new_min < 0:
            new_min = 0
            new_max = new_size - 1
        if new_max >= max_boundary:
            new_max = max_boundary - 1
            new_min = max(0, new_max - new_size + 1)

        return new_min, new_max

    def find_bucket_size(self, width, height, target_size, divisible_by):
        """
        Find the best bucket size that maintains aspect ratio.
        Returns dimensions close to target_size without stretching.
        """
        if target_size <= 0:
            return width, height

        aspect_ratio = width / height

        # Generate common bucket sizes around target_size
        # Use steps of divisible_by for efficiency
        step = max(divisible_by, 64)
        min_size = max(64, target_size - 512)
        max_size = target_size + 512

        best_width = width
        best_height = height
        best_diff = float('inf')

        # Try different sizes
        for size in range(min_size, max_size + 1, step):
            # Make sure it's divisible
            size = (size // divisible_by) * divisible_by

            if aspect_ratio >= 1.0:
                # Wider than tall
                test_width = size
                test_height = int(size / aspect_ratio)
                test_height = (test_height // divisible_by) * divisible_by
            else:
                # Taller than wide
                test_height = size
                test_width = int(size * aspect_ratio)
                test_width = (test_width // divisible_by) * divisible_by

            # Calculate how close we are to target
            avg_size = (test_width + test_height) / 2
            diff = abs(avg_size - target_size)

            if diff < best_diff:
                best_diff = diff
                best_width = test_width
                best_height = test_height

        return best_width, best_height

    # GPU-accelerated version of binary_fill_holes
    def binary_fill_holes_gpu(self, binary_mask_tensor):
        # Create a kernel for morphological operations
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=binary_mask_tensor.device)

        # Perform closing operation (dilation followed by erosion)
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

    # Simplified inpaint crop - clean and predictable
    def inpaint_crop(self, image, mask, context_factor, target_size, resize_mode, force_square, divisible_by, invert_mask, fill_mask_holes, use_gpu, optional_context_mask=None):
        """
        Simplified inpaint crop:
        1. Find mask bounds
        2. Expand by context_factor
        3. Make square if requested
        4. Expand to divisible_by (symmetrically)
        5. Resize using bucketing (maintains aspect ratio)
        6. Crop and return
        """
        use_gpu = use_gpu and torch.cuda.is_available()
        print(f"\n[FL Inpaint Crop] {'GPU' if use_gpu else 'CPU'} mode")

        original_image = image
        original_mask = mask
        original_height, original_width = image.shape[1], image.shape[2]

        # Validate mask size
        if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
            non_zero_indices = torch.nonzero(mask[0], as_tuple=True)
            if not non_zero_indices[0].size(0):
                mask = torch.zeros_like(image[:, :, :, 0])
            else:
                raise ValueError("mask size must match image size")

        # Invert mask if requested
        if invert_mask:
            mask = 1.0 - mask

        # Fill holes if requested
        if fill_mask_holes:
            if use_gpu:
                holemask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
                out = []
                for m in holemask:
                    binary_mask = m > 0
                    filled_mask = self.binary_fill_holes_gpu(binary_mask)
                    out.append(filled_mask * 255)
                mask = torch.stack(out, dim=0)
                mask = torch.clamp(mask, 0.0, 1.0)
            else:
                holemask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
                out = []
                for m in holemask:
                    mask_np = m.numpy()
                    binary_mask = mask_np > 0
                    struct = np.ones((5, 5))
                    closed_mask = binary_closing(binary_mask, structure=struct, border_value=1)
                    filled_mask = binary_fill_holes(closed_mask)
                    out.append(torch.from_numpy(filled_mask.astype(np.float32) * 255))
                mask = torch.stack(out, dim=0)
                mask = torch.clamp(mask, 0.0, 1.0)

        # Determine context mask
        if optional_context_mask is None:
            context_mask = mask
        elif optional_context_mask.shape[1] != image.shape[1] or optional_context_mask.shape[2] != image.shape[2]:
            non_zero_indices = torch.nonzero(optional_context_mask[0], as_tuple=True)
            if not non_zero_indices[0].size(0):
                context_mask = mask
            else:
                raise ValueError("context_mask size must match image size")
        else:
            context_mask = torch.clamp(optional_context_mask + mask, 0.0, 1.0)

        # Find mask bounds
        non_zero_indices = torch.nonzero(context_mask[0], as_tuple=True)
        if not non_zero_indices[0].size(0):
            # Empty mask - return original
            stitch = {'x': 0, 'y': 0, 'original_image': original_image, 'rescale_x': 1.0, 'rescale_y': 1.0}
            return (stitch, original_image, original_mask)

        y_min = torch.min(non_zero_indices[0]).item()
        y_max = torch.max(non_zero_indices[0]).item()
        x_min = torch.min(non_zero_indices[1]).item()
        x_max = torch.max(non_zero_indices[1]).item()

        print(f"[FL Inpaint Crop] Initial bounds: x[{x_min}, {x_max}] y[{y_min}, {y_max}]")

        # Step 1: Expand by context_factor
        y_size = y_max - y_min + 1
        x_size = x_max - x_min + 1

        y_grow = int((y_size * (context_factor - 1.0)) / 2)
        x_grow = int((x_size * (context_factor - 1.0)) / 2)

        y_min = max(y_min - y_grow, 0)
        y_max = min(y_max + y_grow, original_height - 1)
        x_min = max(x_min - x_grow, 0)
        x_max = min(x_max + x_grow, original_width - 1)

        print(f"[FL Inpaint Crop] After context expansion ({context_factor}x): x[{x_min}, {x_max}] y[{y_min}, {y_max}]")

        # Step 2: Make square if requested
        if force_square:
            x_size = x_max - x_min + 1
            y_size = y_max - y_min + 1
            target_size = max(x_size, y_size)

            # Center the square
            x_mid = (x_min + x_max) // 2
            y_mid = (y_min + y_max) // 2

            x_min = max(x_mid - target_size // 2, 0)
            x_max = min(x_min + target_size - 1, original_width - 1)
            y_min = max(y_mid - target_size // 2, 0)
            y_max = min(y_min + target_size - 1, original_height - 1)

            # Adjust if we hit boundaries
            if x_max == original_width - 1:
                x_min = max(0, x_max - target_size + 1)
            if y_max == original_height - 1:
                y_min = max(0, y_max - target_size + 1)

            print(f"[FL Inpaint Crop] After square adjustment: x[{x_min}, {x_max}] y[{y_min}, {y_max}]")

        # Step 3: Expand to divisible_by (symmetrically - critical for alignment!)
        if divisible_by > 1:
            x_min, x_max = self.expand_to_divisible(x_min, x_max, original_width, divisible_by)
            y_min, y_max = self.expand_to_divisible(y_min, y_max, original_height, divisible_by)

            print(f"[FL Inpaint Crop] After divisible-by-{divisible_by} expansion: x[{x_min}, {x_max}] y[{y_min}, {y_max}]")

        # Crop first
        cropped_image = image[:, y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask[:, y_min:y_max+1, x_min:x_max+1]

        crop_height = y_max - y_min + 1
        crop_width = x_max - x_min + 1

        print(f"[FL Inpaint Crop] Cropped size: {crop_width}x{crop_height}")

        # Step 4: Resize based on mode
        rescale_x = 1.0
        rescale_y = 1.0

        if resize_mode != "none" and target_size > 0:
            if resize_mode == "bucket":
                # Bucketing: find nearest size that maintains aspect ratio
                final_width, final_height = self.find_bucket_size(crop_width, crop_height, target_size, divisible_by)
                print(f"[FL Inpaint Crop] Bucketing to: {final_width}x{final_height} (target: {target_size}, maintains aspect ratio)")
            elif resize_mode == "max_dimension":
                # Scale longest side to target_size, maintain aspect ratio
                if crop_width >= crop_height:
                    final_width = target_size
                    final_height = int(crop_height * (target_size / crop_width))
                else:
                    final_height = target_size
                    final_width = int(crop_width * (target_size / crop_height))

                # Ensure dimensions are divisible_by
                if divisible_by > 1:
                    final_width = ((final_width + divisible_by - 1) // divisible_by) * divisible_by
                    final_height = ((final_height + divisible_by - 1) // divisible_by) * divisible_by

                print(f"[FL Inpaint Crop] Max dimension resize to: {final_width}x{final_height}")

            # Resize image
            samples = cropped_image.movedim(-1, 1)
            samples = comfy.utils.bislerp(samples, final_width, final_height)
            cropped_image = samples.movedim(1, -1)

            # Resize mask
            samples = cropped_mask.unsqueeze(1)
            samples = comfy.utils.bislerp(samples, final_width, final_height)
            cropped_mask = samples.squeeze(1)

            # Calculate rescale factors (needed for stitching back)
            rescale_x = final_width / crop_width
            rescale_y = final_height / crop_height
        else:
            print(f"[FL Inpaint Crop] No resize (mode: {resize_mode})")

        # Return stitch data
        stitch = {
            'x': x_min,
            'y': y_min,
            'original_image': original_image,
            'rescale_x': rescale_x,
            'rescale_y': rescale_y
        }

        print(f"[FL Inpaint Crop] Final output size: {cropped_image.shape[2]}x{cropped_image.shape[1]}")
        print(f"[FL Inpaint Crop] Rescale factors: x={rescale_x:.3f}, y={rescale_y:.3f}\n")

        return (stitch, cropped_image, cropped_mask)


class FL_Inpaint_Stitch:
    """
    Stitches the inpainted image back into the original image.
    Pastes the entire cropped rectangle (no mask blending).
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

    def composite(self, destination, source, x, y, mask=None, multiplier=8, resize_source=False):
        """Simple compositing function"""
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

        # Calculate visible bounds
        visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

        mask = mask[:, :, :visible_height, :visible_width]
        inverse_mask = torch.ones_like(mask) - mask

        source_portion = mask * source[:, :, :visible_height, :visible_width]
        destination_portion = inverse_mask * destination[:, :, top:bottom, left:right]

        destination[:, :, top:bottom, left:right] = source_portion + destination_portion
        return destination

    def inpaint_stitch(self, stitch, inpainted_image):
        """Stitch inpainted image back into original"""
        original_image = stitch['original_image']
        x = stitch['x']  # These are in ORIGINAL image coordinates - do NOT rescale!
        y = stitch['y']
        stitched_image = original_image.clone().movedim(-1, 1)

        inpaint_width = inpainted_image.shape[2]
        inpaint_height = inpainted_image.shape[1]

        print(f"\n[FL Inpaint Stitch] Input inpainted image: {inpaint_width}x{inpaint_height}")
        print(f"[FL Inpaint Stitch] Original crop position: ({x}, {y})")
        print(f"[FL Inpaint Stitch] Rescale factors: x={stitch['rescale_x']:.3f}, y={stitch['rescale_y']:.3f}")

        # Downscale inpainted image if we upscaled it before cropping
        if stitch['rescale_x'] < 0.999 or stitch['rescale_x'] > 1.001 or stitch['rescale_y'] < 0.999 or stitch['rescale_y'] > 1.001:
            samples = inpainted_image.movedim(-1, 1)
            # Calculate target size (reverse the rescaling)
            width = round(float(inpaint_width) / stitch['rescale_x'])
            height = round(float(inpaint_height) / stitch['rescale_y'])

            # CRITICAL: x,y are already in original image coordinates! Don't rescale them!
            # They were saved BEFORE resizing, so they're already correct

            samples = comfy.utils.bislerp(samples, width, height)
            inpainted_image = samples.movedim(1, -1)

            print(f"[FL Inpaint Stitch] Rescaled inpainted image to: {width}x{height}")
        else:
            print(f"[FL Inpaint Stitch] No rescaling needed")

        print(f"[FL Inpaint Stitch] Pasting at position: ({x}, {y})")

        # Paste the entire cropped rectangle back (no mask blending)
        output = self.composite(stitched_image, inpainted_image.movedim(-1, 1), x, y, mask=None, multiplier=1).movedim(1, -1)

        print(f"[FL Inpaint Stitch] Stitch complete!\n")

        return (output,)
