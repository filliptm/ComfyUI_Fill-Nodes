import torch
import numpy as np
from PIL import Image, ImageDraw

class FL_PaddingRemover:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tolerance": ("INT", {"default": 30, "min": 0, "max": 255, "step": 1}),
                "min_content_size": ("INT", {"default": 20, "min": 1, "max": 1000, "step": 1}),
                "sides_trim": ("INT", {"default": 1, "min": 0, "max": 256, "step": 1}),
                "top_bottom_trim": ("INT", {"default": 1, "min": 0, "max": 256, "step": 1}),
                "debug_view": ("BOOLEAN", {"default": False}),
                "use_gpu": ("BOOLEAN", {"default": False}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_padding"
    CATEGORY = "🏵️Fill Nodes/Image"

    def remove_padding(self, image, tolerance=30, min_content_size=20, sides_trim=1, top_bottom_trim=1, debug_view=False, use_gpu=False):
        # Check if GPU is available when requested
        use_gpu = use_gpu and torch.cuda.is_available()
        if use_gpu:
            print("Using GPU for padding removal")
        else:
            print("Using CPU for padding removal")
        # Process each image in the batch
        batch_size = image.shape[0]
        result_tensors = []
        
        for b in range(batch_size):
            # Get the current image tensor
            img_tensor = image[b]
            
            # Print the tensor shape for debugging
            print(f"Input tensor shape: {img_tensor.shape}")
            
            if use_gpu:
                # GPU processing path
                # Store original format for later conversion
                is_chw_format = len(img_tensor.shape) == 3 and img_tensor.shape[0] in [1, 3, 4]
                
                # Find the bounding box directly using GPU tensors
                # Scale to 0-255 range for consistency with CPU version
                img_tensor_255 = img_tensor * 255
                
                # Find the bounding box of the content using GPU
                bbox = self.find_content_bbox_gpu(img_tensor_255, tolerance, min_content_size)
                
                # Get dimensions for validation
                if is_chw_format:
                    height, width = img_tensor.shape[1], img_tensor.shape[2]
                else:
                    height, width = img_tensor.shape[0], img_tensor.shape[1]
            else:
                # CPU processing path (original implementation)
                # Check if the tensor is already in [H, W, C] format
                if len(img_tensor.shape) == 3 and img_tensor.shape[2] in [1, 3, 4]:
                    # Format is [H, W, C]
                    img_np = img_tensor.cpu().numpy()
                    channels = img_np.shape[2]
                elif len(img_tensor.shape) == 3 and img_tensor.shape[0] in [1, 3, 4]:
                    # Format is [C, H, W], convert to [H, W, C]
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                    channels = img_np.shape[2]
                else:
                    # Unsupported format
                    print(f"Unsupported tensor shape: {img_tensor.shape}")
                    result_tensors.append(img_tensor)
                    continue
                
                # Scale to 0-255 range for PIL
                img_np_uint8 = (img_np * 255).astype(np.uint8)
                
                # Create PIL image
                img_pil = Image.fromarray(img_np_uint8)
                
                # Find the bounding box of the content
                bbox = self.find_content_bbox(img_np_uint8, tolerance, min_content_size)
                
                # Get dimensions for validation
                height, width = img_pil.size[1], img_pil.size[0]
            
            # Apply additional trimming to the bounding box
            left, top, right, bottom = bbox
            
            # Apply sides trimming
            left += sides_trim
            right -= sides_trim
            
            # Apply top/bottom trimming
            top += top_bottom_trim
            bottom -= top_bottom_trim
            
            # Ensure the bounding box is valid
            left = max(0, left)
            top = max(0, top)
            right = min(width, right)
            bottom = min(height, bottom)
            
            # Ensure we have a valid box (width and height > 0)
            if right <= left or bottom <= top:
                print("Warning: After trimming, the bounding box is invalid. Using original bbox.")
                left, top, right, bottom = bbox
            
            # Update the bounding box
            bbox = (left, top, right, bottom)
            
            if use_gpu:
                # GPU processing path
                if debug_view:
                    # For debug view, we need to move to CPU to use PIL for drawing
                    if is_chw_format:
                        # Convert [C,H,W] to [H,W,C] for PIL
                        cpu_tensor = img_tensor.permute(1, 2, 0).cpu()
                    else:
                        cpu_tensor = img_tensor.cpu()
                    
                    # Convert to numpy and scale to 0-255
                    img_np = cpu_tensor.numpy() * 255
                    img_np_uint8 = img_np.astype(np.uint8)
                    img_pil = Image.fromarray(img_np_uint8)
                    
                    # Create debug view with rectangle
                    debug_img = img_pil.copy()
                    draw = ImageDraw.Draw(debug_img)
                    draw.rectangle(bbox, outline=(255, 0, 0), width=2)
                    
                    # Convert back to tensor
                    debug_np = np.array(debug_img).astype(np.float32) / 255.0
                    if is_chw_format:
                        debug_tensor = torch.from_numpy(debug_np).permute(2, 0, 1)
                    else:
                        debug_tensor = torch.from_numpy(debug_np)
                    
                    result_tensors.append(debug_tensor)
                else:
                    # Crop directly on GPU tensor
                    if is_chw_format:
                        # For [C,H,W] format
                        cropped_tensor = img_tensor[:, top:bottom, left:right]
                    else:
                        # For [H,W,C] format
                        cropped_tensor = img_tensor[top:bottom, left:right, :]
                    
                    print(f"Output tensor shape: {cropped_tensor.shape}")
                    result_tensors.append(cropped_tensor)
            else:
                # CPU processing path (original implementation)
                if debug_view:
                    # Create a debug view with the detected content area highlighted
                    debug_img = img_pil.copy()
                    draw = ImageDraw.Draw(debug_img)
                    draw.rectangle(bbox, outline=(255, 0, 0), width=2)
                    
                    # Convert back to numpy array
                    debug_np = np.array(debug_img).astype(np.float32) / 255.0
                    
                    # Ensure the output has the same shape as the input
                    if len(img_tensor.shape) == 3 and img_tensor.shape[0] in [1, 3, 4]:
                        # If input was [C, H, W], convert back to that format
                        debug_tensor = torch.from_numpy(debug_np).permute(2, 0, 1)
                    else:
                        # Otherwise keep as [H, W, C]
                        debug_tensor = torch.from_numpy(debug_np)
                    
                    result_tensors.append(debug_tensor)
                else:
                    # Crop the image to the content bounding box
                    cropped_img = img_pil.crop(bbox)
                    
                    # Convert back to numpy array
                    cropped_np = np.array(cropped_img).astype(np.float32) / 255.0
                    
                    # Ensure the output has the same shape as the input
                    if len(img_tensor.shape) == 3 and img_tensor.shape[0] in [1, 3, 4]:
                        # If input was [C, H, W], convert back to that format
                        cropped_tensor = torch.from_numpy(cropped_np).permute(2, 0, 1)
                    else:
                        # Otherwise keep as [H, W, C]
                        cropped_tensor = torch.from_numpy(cropped_np)
                    
                    # Print the output tensor shape for debugging
                    print(f"Output tensor shape: {cropped_tensor.shape}")
                    
                    result_tensors.append(cropped_tensor)
        
        # Stack the processed images back into a batch
        return (torch.stack(result_tensors),)
    
    def find_content_bbox(self, img_array, tolerance, min_content_size):
        """Find the bounding box of the content in the image by detecting padding."""
        height, width = img_array.shape[:2]
        
        # Improved padding detection using edge color sampling and histogram analysis
        def is_padding_row(y):
            # Sample colors from the row
            row = img_array[y, :, :]
            
            # Get the edge color (first pixel in the row)
            edge_color = row[0]
            
            # Count pixels that are different from the edge color by more than the tolerance
            different_pixels = 0
            for x in range(width):
                pixel = row[x]
                diff = np.sum(np.abs(pixel.astype(int) - edge_color.astype(int)))
                if diff > tolerance:
                    different_pixels += 1
            
            # If less than 5% of pixels are different, consider it padding
            return different_pixels < (width * 0.05)
        
        def is_padding_col(x):
            # Sample colors from the column
            col = img_array[:, x, :]
            
            # Get the edge color (first pixel in the column)
            edge_color = col[0]
            
            # Count pixels that are different from the edge color by more than the tolerance
            different_pixels = 0
            for y in range(height):
                pixel = col[y]
                diff = np.sum(np.abs(pixel.astype(int) - edge_color.astype(int)))
                if diff > tolerance:
                    different_pixels += 1
            
            # If less than 5% of pixels are different, consider it padding
            return different_pixels < (height * 0.05)
        
        # Find top padding
        top = 0
        while top < height - min_content_size and is_padding_row(top):
            top += 1
        
        # Find bottom padding
        bottom = height - 1
        while bottom >= top + min_content_size and is_padding_row(bottom):
            bottom -= 1
        
        # Find left padding
        left = 0
        while left < width - min_content_size and is_padding_col(left):
            left += 1
        
        # Find right padding
        right = width - 1
        while right >= left + min_content_size and is_padding_col(right):
            right -= 1
        
        # Ensure we have at least min_content_size x min_content_size content
        if right - left < min_content_size or bottom - top < min_content_size:
            return (0, 0, width, height)  # Return the full image if content is too small
        
        # Return the bounding box (left, top, right, bottom)
        return (left, top, right + 1, bottom + 1)  # +1 because PIL crop is exclusive on right/bottom
    
    def find_content_bbox_gpu(self, img_tensor, tolerance, min_content_size):
        """GPU version of find_content_bbox that works directly with tensors."""
        # Ensure the tensor is in the right format [H, W, C]
        if img_tensor.shape[0] in [1, 3, 4] and len(img_tensor.shape) == 3:
            # Format is [C, H, W], convert to [H, W, C]
            img_tensor = img_tensor.permute(1, 2, 0)
        
        height, width = img_tensor.shape[:2]
        
        # Helper functions for GPU tensor operations
        def is_padding_row_gpu(y):
            # Get the row
            row = img_tensor[y:y+1, :, :]
            
            # Get the edge color (first pixel)
            edge_color = row[:, 0:1, :]
            
            # Calculate differences from edge color for all pixels in the row
            diff = torch.sum(torch.abs(row - edge_color), dim=2)
            
            # Count pixels that differ by more than the tolerance
            different_pixels = torch.sum(diff > tolerance).item()
            
            # If less than 5% of pixels are different, consider it padding
            return different_pixels < (width * 0.05)
        
        def is_padding_col_gpu(x):
            # Get the column
            col = img_tensor[:, x:x+1, :]
            
            # Get the edge color (first pixel)
            edge_color = col[0:1, :, :]
            
            # Calculate differences from edge color for all pixels in the column
            diff = torch.sum(torch.abs(col - edge_color), dim=2)
            
            # Count pixels that differ by more than the tolerance
            different_pixels = torch.sum(diff > tolerance).item()
            
            # If less than 5% of pixels are different, consider it padding
            return different_pixels < (height * 0.05)
        
        # Find top padding
        top = 0
        while top < height - min_content_size and is_padding_row_gpu(top):
            top += 1
        
        # Find bottom padding
        bottom = height - 1
        while bottom >= top + min_content_size and is_padding_row_gpu(bottom):
            bottom -= 1
        
        # Find left padding
        left = 0
        while left < width - min_content_size and is_padding_col_gpu(left):
            left += 1
        
        # Find right padding
        right = width - 1
        while right >= left + min_content_size and is_padding_col_gpu(right):
            right -= 1
        
        # Ensure we have at least min_content_size x min_content_size content
        if right - left < min_content_size or bottom - top < min_content_size:
            return (0, 0, width, height)  # Return the full image if content is too small
        
        # Return the bounding box (left, top, right, bottom)
        return (left, top, right + 1, bottom + 1)  # +1 because crop is exclusive on right/bottom