import torch
import math
import torchvision.transforms.functional as TF
from torchvision.ops import masks_to_boxes
from comfy.utils import ProgressBar
import torch.nn.functional as F

def tensor2mask(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t
    if size[3] == 1:
        return t[:,:,:,0]
    elif size[3] == 4:
        # Not sure what the right thing to do here is. Going to try to be a little smart and use alpha unless all alpha is 1 in case we'll fallback to RGB behavior
        if torch.min(t[:, :, :, 3]).item() != 1.:
            return t[:,:,:,3]

    return TF.rgb_to_grayscale(tensor2rgb(t).permute(0,3,1,2), num_output_channels=1)[:,0,:,:]

def tensor2rgb(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 3)
    if size[3] == 1:
        return t.repeat(1, 1, 1, 3)
    elif size[3] == 4:
        return t[:, :, :, :3]
    else:
        return t

def gaussian_blur(img, kernel_size, sigma):
    """Apply gaussian blur to a tensor image"""
    # Create 1D kernels
    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    x_grid = x.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    
    # Calculate 2D gaussian kernel
    gaussian_kernel = torch.exp(-(x_grid.pow(2) + y_grid.pow(2)) / (2 * sigma * sigma))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    
    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(img.size(1), 1, 1, 1)
    
    # Create padding based on kernel size
    padding = kernel_size // 2
    
    # Apply gaussian filter
    return F.conv2d(img, gaussian_kernel, padding=padding, groups=img.size(1))

def tensor2rgba(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 4)
    elif size[3] == 1:
        return t.repeat(1, 1, 1, 4)
    elif size[3] == 3:
        alpha_tensor = torch.ones((size[0], size[1], size[2], 1))
        return torch.cat((t, alpha_tensor), dim=3)
    else:
        return t

class FL_PasteByMask:
    """
    Pastes `image_to_paste` onto `image_base` using `mask` to determine the location. The `resize_behavior` parameter determines how the image to paste is resized to fit the mask. If `mask_mapping_optional` obtained from a 'Separate Mask Components' node is used, it will control which image gets pasted onto which base image.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_base": ("IMAGE",),
                "image_to_paste": ("IMAGE",),
                "mask": ("IMAGE",),
                "resize_behavior": (["resize", "keep_ratio_fill", "keep_ratio_fit", "source_size", "source_size_unmasked"],),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "hard_light", "darken", "lighten", "difference"], {"default": "normal"}),
                "feather_amount": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "mask_mapping_optional": ("MASK_MAPPING",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste"

    CATEGORY = "🏵️Fill Nodes/Utility"

    def paste(self, image_base, image_to_paste, mask, resize_behavior, blend_mode="normal", feather_amount=0, mask_mapping_optional=None):
        image_base = tensor2rgba(image_base)
        image_to_paste = tensor2rgba(image_to_paste)
        mask = tensor2mask(mask)

        # Handle batch sizes more flexibly by repeating tensors as needed
        B, H, W, C = image_base.shape
        MB = mask.shape[0]
        PB = image_to_paste.shape[0]
        
        if mask_mapping_optional is None:
            # Find the maximum batch size needed
            max_batch = max(B, MB, PB)
            
            # Repeat base images to match max_batch size
            if B < max_batch:
                # Calculate how many times to repeat the tensor (ceiling division)
                repeat_count = (max_batch + B - 1) // B
                image_base = image_base.repeat(repeat_count, 1, 1, 1)
                # Trim to exact size needed
                image_base = image_base[:max_batch]
            
            # Update dimensions after possible resize
            B, H, W, C = image_base.shape
            
            # Repeat mask to match max_batch size
            if MB < max_batch:
                repeat_count = (max_batch + MB - 1) // MB
                mask = mask.repeat(repeat_count, 1, 1)
                mask = mask[:max_batch]
            
            # Repeat images to paste to match max_batch size
            if PB < max_batch:
                repeat_count = (max_batch + PB - 1) // PB
                image_to_paste = image_to_paste.repeat(repeat_count, 1, 1, 1)
                image_to_paste = image_to_paste[:max_batch]
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:,0,:,:]
        MB, MH, MW = mask.shape

        # masks_to_boxes errors if the tensor is all zeros, so we'll add a single pixel and zero it out at the end
        is_empty = ~torch.gt(torch.max(torch.reshape(mask,[MB, MH * MW]), dim=1).values, 0.)
        mask[is_empty,0,0] = 1.
        boxes = masks_to_boxes(mask)
        mask[is_empty,0,0] = 0.

        min_x = boxes[:,0]
        min_y = boxes[:,1]
        max_x = boxes[:,2]
        max_y = boxes[:,3]
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2

        target_width = max_x - min_x + 1
        target_height = max_y - min_y + 1

        result = image_base.detach().clone()
        
        # Create progress bar
        pbar = ProgressBar(MB)
        print(f"[FL_PasteByMask] Processing {MB} masks")
        
        for i in range(0, MB):
            if is_empty[i]:
                pbar.update_absolute(i)
                continue
            else:
                # Use modulo to cycle through available base images if needed
                image_index = i % B
                if mask_mapping_optional is not None:
                    image_index = mask_mapping_optional[i].item() % B
                source_size = image_to_paste.size()
                SB, SH, SW, _ = image_to_paste.shape

                # Figure out the desired size
                width = int(target_width[i].item())
                height = int(target_height[i].item())
                if resize_behavior == "keep_ratio_fill":
                    target_ratio = width / height
                    actual_ratio = SW / SH
                    if actual_ratio > target_ratio:
                        width = int(height * actual_ratio)
                    elif actual_ratio < target_ratio:
                        height = int(width / actual_ratio)
                elif resize_behavior == "keep_ratio_fit":
                    target_ratio = width / height
                    actual_ratio = SW / SH
                    if actual_ratio > target_ratio:
                        height = int(width / actual_ratio)
                    elif actual_ratio < target_ratio:
                        width = int(height * actual_ratio)
                elif resize_behavior == "source_size" or resize_behavior == "source_size_unmasked":
                    width = SW
                    height = SH

                # Resize the image we're pasting if needed
                # Use modulo to cycle through available images if needed
                paste_index = i % PB
                resized_image = image_to_paste[paste_index].unsqueeze(0)
                if SH != height or SW != width:
                    resized_image = torch.nn.functional.interpolate(resized_image.permute(0, 3, 1, 2), size=(height,width), mode='bicubic').permute(0, 2, 3, 1)

                pasting = torch.ones([H, W, C])
                ymid = float(mid_y[i].item())
                ymin = int(math.floor(ymid - height / 2)) + 1
                ymax = int(math.floor(ymid + height / 2)) + 1
                xmid = float(mid_x[i].item())
                xmin = int(math.floor(xmid - width / 2)) + 1
                xmax = int(math.floor(xmid + width / 2)) + 1

                _, source_ymax, source_xmax, _ = resized_image.shape
                source_ymin, source_xmin = 0, 0

                if xmin < 0:
                    source_xmin = abs(xmin)
                    xmin = 0
                if ymin < 0:
                    source_ymin = abs(ymin)
                    ymin = 0
                if xmax > W:
                    source_xmax -= (xmax - W)
                    xmax = W
                if ymax > H:
                    source_ymax -= (ymax - H)
                    ymax = H

                pasting[ymin:ymax, xmin:xmax, :] = resized_image[0, source_ymin:source_ymax, source_xmin:source_xmax, :]
                pasting[:, :, 3] = 1.

                pasting_alpha = torch.zeros([H, W])
                pasting_alpha[ymin:ymax, xmin:xmax] = resized_image[0, source_ymin:source_ymax, source_xmin:source_xmax, 3]

                # Apply feathering to the mask if requested
                if feather_amount > 0:
                    # Create a copy of the mask for feathering
                    feather_mask = pasting_alpha.clone()
                    if resize_behavior != "keep_ratio_fill" and resize_behavior != "source_size_unmasked":
                        feather_mask = torch.min(feather_mask, mask[i])
                    
                    # Apply gaussian blur to create feathered edges
                    # Convert to proper format for gaussian blur
                    feather_mask = feather_mask.unsqueeze(0).unsqueeze(0)
                    # Calculate sigma based on feather amount (higher feather = more blur)
                    sigma = feather_amount / 10.0
                    kernel_size = max(3, int(feather_amount / 5) * 2 + 1)  # Ensure odd kernel size
                    # Apply custom gaussian blur
                    feather_mask = gaussian_blur(feather_mask, kernel_size, sigma)
                    feather_mask = feather_mask.squeeze(0).squeeze(0)
                    
                    # Create the final paste mask
                    paste_mask = feather_mask.unsqueeze(2).repeat(1, 1, 4)
                else:
                    # Use original mask without feathering
                    if resize_behavior == "keep_ratio_fill" or resize_behavior == "source_size_unmasked":
                        # If we explicitly want to fill the area, we are ok with extending outside
                        paste_mask = pasting_alpha.unsqueeze(2).repeat(1, 1, 4)
                    else:
                        paste_mask = torch.min(pasting_alpha, mask[i]).unsqueeze(2).repeat(1, 1, 4)
                
                # Apply the selected blend mode
                if blend_mode == "normal":
                    # Standard alpha compositing (current behavior)
                    result[image_index] = pasting * paste_mask + result[image_index] * (1. - paste_mask)
                elif blend_mode == "multiply":
                    # Multiply blend mode
                    blended = pasting * result[image_index]
                    result[image_index] = blended * paste_mask + result[image_index] * (1. - paste_mask)
                elif blend_mode == "screen":
                    # Screen blend mode
                    blended = 1.0 - (1.0 - pasting) * (1.0 - result[image_index])
                    result[image_index] = blended * paste_mask + result[image_index] * (1. - paste_mask)
                elif blend_mode == "overlay":
                    # Overlay blend mode
                    mask_2x = result[image_index] * 2.0
                    blended = torch.where(
                        mask_2x <= 1.0,
                        pasting * mask_2x,
                        1.0 - (1.0 - pasting) * (1.0 - (mask_2x - 1.0))
                    )
                    result[image_index] = blended * paste_mask + result[image_index] * (1. - paste_mask)
                elif blend_mode == "soft_light":
                    # Soft Light blend mode
                    blended = torch.where(
                        pasting <= 0.5,
                        result[image_index] - (1.0 - 2.0 * pasting) * result[image_index] * (1.0 - result[image_index]),
                        result[image_index] + (2.0 * pasting - 1.0) * (torch.sqrt(result[image_index]) - result[image_index])
                    )
                    result[image_index] = blended * paste_mask + result[image_index] * (1. - paste_mask)
                elif blend_mode == "hard_light":
                    # Hard Light blend mode
                    blended = torch.where(
                        pasting <= 0.5,
                        2.0 * pasting * result[image_index],
                        1.0 - 2.0 * (1.0 - pasting) * (1.0 - result[image_index])
                    )
                    result[image_index] = blended * paste_mask + result[image_index] * (1. - paste_mask)
                elif blend_mode == "darken":
                    # Darken blend mode
                    blended = torch.min(pasting, result[image_index])
                    result[image_index] = blended * paste_mask + result[image_index] * (1. - paste_mask)
                elif blend_mode == "lighten":
                    # Lighten blend mode
                    blended = torch.max(pasting, result[image_index])
                    result[image_index] = blended * paste_mask + result[image_index] * (1. - paste_mask)
                elif blend_mode == "difference":
                    # Difference blend mode
                    blended = torch.abs(pasting - result[image_index])
                    result[image_index] = blended * paste_mask + result[image_index] * (1. - paste_mask)
                
                # Update progress bar
                pbar.update_absolute(i)
        return (result,)