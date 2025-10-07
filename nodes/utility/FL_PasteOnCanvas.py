import torch
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import resize as tv_resize, InterpolationMode
import numpy as np
from PIL import Image

class FL_PasteOnCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "canvas_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "canvas_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "background_red": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "background_green": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "background_blue": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "resize_algorithm": (["bilinear", "nearest", "bicubic", "lanczos"],),
                "include_alpha": ("BOOLEAN", {"default": False}),
                "use_full_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("IMAGE",),
                "bg_image_optional": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "cut_and_paste"
    CATEGORY = "🏵️Fill Nodes/Utility"

    def cut_and_paste(self, image, canvas_width, canvas_height, background_red, background_green, background_blue,
                      padding, resize_algorithm, include_alpha, use_full_mask, mask=None, bg_image_optional=None):
        # Ensure inputs are in the correct format
        image = self.tensor_to_rgba(image)
        B, H, W, C = image.shape
        
        # Validate canvas dimensions
        if canvas_width <= 0 or canvas_height <= 0:
            print("Error: Canvas dimensions must be positive")
            return (image,)  # Return original image as fallback
        
        # Handle mask input
        if mask is None:
            if use_full_mask:
                print("Creating full white mask as requested")
                # Create a full white mask matching the image dimensions
                mask = torch.ones((B, H, W), device=image.device)
            else:
                # Raise an error when no mask is provided and use_full_mask is False
                raise ValueError("No mask provided and use_full_mask is False. Either provide a mask or enable use_full_mask.")
        else:
            # Process the provided mask
            mask = self.tensor_to_mask(mask)
            
        # Resize mask to match image dimensions
        mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:, 0, :, :]
        MB, MH, MW = mask.shape

        if MB < B:
            # Handle batch size mismatch more gracefully
            if B % MB == 0:
                mask = mask.repeat(B // MB, 1, 1)
            else:
                print(f"Warning: Batch size mismatch between image ({B}) and mask ({MB})")
                # Repeat the first mask to match batch size
                mask = mask[0:1].repeat(B, 1, 1)

        # Prepare the background canvas
        if bg_image_optional is not None:
            canvas = self.prepare_background_image(bg_image_optional, canvas_width, canvas_height, B)
        else:
            background_color = torch.tensor([background_red, background_green, background_blue, 255],
                                            dtype=torch.float32, device=image.device) / 255.0
            canvas = background_color.expand(B, canvas_height, canvas_width, 4).clone()

        # Handle empty masks
        is_empty = ~torch.gt(mask.view(B, -1).max(dim=1).values, 0)
        mask[is_empty, 0, 0] = 1
        boxes = masks_to_boxes(mask)
        mask[is_empty, 0, 0] = 0

        # Create alpha mask
        alpha_mask = torch.ones((B, H, W, 4), device=image.device)
        alpha_mask[..., 3] = mask

        masked_image = image * alpha_mask

        for i in range(B):
            if not is_empty[i]:
                box = boxes[i].long()
                y1, x1, y2, x2 = box[1], box[0], box[3], box[2]
                cropped = masked_image[i, y1:y2 + 1, x1:x2 + 1, :]

                # Calculate scaling factor to fit within canvas, considering padding
                available_width = canvas_width - 2 * padding
                available_height = canvas_height - 2 * padding
                scale = min(available_width / cropped.shape[1], available_height / cropped.shape[0])
                new_h, new_w = int(cropped.shape[0] * scale), int(cropped.shape[1] * scale)

                # Resize cropped image using the specified algorithm
                resized = self.resize_image(cropped, (new_h, new_w), resize_algorithm)

                # Calculate position to center the image on canvas, including padding
                start_y = padding + (available_height - new_h) // 2
                start_x = padding + (available_width - new_w) // 2

                # Prepare the region of the canvas where we'll paste the image
                canvas_region = canvas[i, start_y:start_y + new_h, start_x:start_x + new_w].clone()

                # Blend the resized image with the canvas region
                alpha = resized[..., 3:4]
                blended = resized[..., :3] * alpha + canvas_region[..., :3] * (1 - alpha)

                # Update the alpha channel
                new_alpha = torch.maximum(canvas_region[..., 3:], resized[..., 3:])

                # Combine the blended color channels with the new alpha
                result = torch.cat([blended, new_alpha], dim=-1)

                # Update the canvas with the result
                canvas[i, start_y:start_y + new_h, start_x:start_x + new_w] = result

        # Remove alpha channel if not included
        if not include_alpha:
            canvas = canvas[..., :3]

        return (canvas,)

    def prepare_background_image(self, bg_image_optional, canvas_width, canvas_height, batch_size):
        bg_image_optional = self.tensor_to_rgba(bg_image_optional)

        # Resize background image to match canvas size
        resized_bg = F.interpolate(bg_image_optional.permute(0, 3, 1, 2),
                                   size=(canvas_height, canvas_width),
                                   mode='bilinear',
                                   align_corners=False).permute(0, 2, 3, 1)

        # If the background image batch size is 1, repeat it to match the main batch size
        if resized_bg.shape[0] == 1 and batch_size > 1:
            resized_bg = resized_bg.repeat(batch_size, 1, 1, 1)

        return resized_bg

    def resize_image(self, image, size, algorithm):
        if algorithm == "lanczos":
            # Convert to PIL Image for Lanczos resampling
            pil_image = Image.fromarray((image.cpu().numpy() * 255).astype('uint8'))
            resized_pil = pil_image.resize(size[::-1], Image.LANCZOS)  # PIL uses (width, height)
            return torch.from_numpy(np.array(resized_pil)).float().to(image.device) / 255.0
        else:
            # Use torchvision's resize for other algorithms
            interpolation_mode = {
                "bilinear": InterpolationMode.BILINEAR,
                "nearest": InterpolationMode.NEAREST,
                "bicubic": InterpolationMode.BICUBIC,
            }[algorithm]
            return tv_resize(image.permute(2, 0, 1), size, interpolation=interpolation_mode).permute(1, 2, 0)

    @staticmethod
    def tensor_to_rgba(tensor):
        if len(tensor.shape) == 3:
            return tensor.unsqueeze(-1).expand(-1, -1, -1, 4)
        elif tensor.shape[-1] == 1:
            return tensor.expand(-1, -1, -1, 4)
        elif tensor.shape[-1] == 3:
            return torch.cat([tensor, torch.ones_like(tensor[:, :, :, :1])], dim=-1)
        return tensor

    @staticmethod
    def tensor_to_mask(tensor):
        if tensor is None:
            return None
        if len(tensor.shape) == 4:
            return tensor.mean(dim=-1)
        return tensor