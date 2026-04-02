import torch
import torch.nn.functional as F
import math
import numpy as np
import io
import base64
from PIL import Image, ImageDraw
from server import PromptServer


class FL_ImageReplace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "replacement": ("IMAGE",),
                "index": ("INT", {"default": 0, "min": -100, "max": 9999, "step": 1}),
                "show_preview": ("BOOLEAN", {"default": False, "label": "Show Preview on Node"}),
            },
            "optional": {
                "masks": ("MASK",),
                "replacement_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "replace_in_batch"
    OUTPUT_NODE = True
    CATEGORY = "🏵️Fill Nodes/Image"

    def replace_in_batch(self, images, replacement, index, show_preview=False,
                         masks=None, replacement_mask=None):
        # Ensure batch dimensions exist
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if replacement.dim() == 3:
            replacement = replacement.unsqueeze(0)

        batch_size = images.shape[0]

        if batch_size == 0:
            result_masks = masks if masks is not None else torch.ones(0, device=images.device)
            return (images, result_masks)

        # Resolve negative index and clamp
        if index < 0:
            index = batch_size + index
        index = max(0, min(index, batch_size - 1))

        B, H, W, C = images.shape
        N, rH, rW, rC = replacement.shape

        # Ensure replacement is on the same device
        replacement = replacement.to(images.device)

        # Handle channel mismatch
        if rC != C:
            if rC < C:
                padding = torch.ones(N, rH, rW, C - rC, device=replacement.device)
                replacement = torch.cat([replacement, padding], dim=-1)
            else:
                replacement = replacement[..., :C]

        # Resize replacement if spatial dimensions don't match
        if rH != H or rW != W:
            replacement_chw = replacement.permute(0, 3, 1, 2)
            is_upscaling = (H > rH) or (W > rW)
            mode = 'bicubic' if is_upscaling else 'area'
            replacement_chw = F.interpolate(
                replacement_chw,
                size=(H, W),
                mode=mode,
                align_corners=False if mode in ['bicubic', 'bilinear'] else None
            )
            replacement = replacement_chw.permute(0, 2, 3, 1)
            replacement = replacement.clamp(0.0, 1.0)

        # Truncate replacement if it exceeds remaining space
        available = batch_size - index
        if N > available:
            replacement = replacement[:available]
            N = available

        # Perform the replacement
        result = images.clone()
        result[index:index + N] = replacement

        # Track which indices were replaced
        replaced_indices = set(range(index, index + N))

        # Handle masks
        result_masks = self._handle_masks(
            masks, replacement_mask, batch_size, index, N, H, W, images.device
        )

        # Send preview if enabled
        if show_preview:
            preview_image = self._create_preview_grid(result, replaced_indices)
            display_data = self._prepare_image_for_display(preview_image)
            PromptServer.instance.send_sync("fl_image_replace", {"image": display_data})

        return (result, result_masks)

    def _handle_masks(self, masks, replacement_mask, batch_size, index, N, H, W, device):
        if masks is not None:
            if masks.dim() == 2:
                masks = masks.unsqueeze(0)
            result_masks = masks.clone()
            tH, tW = masks.shape[1], masks.shape[2]
        elif replacement_mask is not None:
            result_masks = torch.ones(batch_size, H, W, device=device)
            tH, tW = H, W
        else:
            return torch.ones(batch_size, H, W, device=device)

        if replacement_mask is not None:
            if replacement_mask.dim() == 2:
                replacement_mask = replacement_mask.unsqueeze(0)

            replacement_mask = replacement_mask.to(device)
            mN, mH, mW = replacement_mask.shape

            # Resize if needed
            if mH != tH or mW != tW:
                replacement_mask = F.interpolate(
                    replacement_mask.unsqueeze(1), size=(tH, tW), mode='nearest'
                ).squeeze(1)

            # Truncate if needed
            available = batch_size - index
            if mN > available:
                replacement_mask = replacement_mask[:available]
                mN = available

            result_masks[index:index + mN] = replacement_mask

        return result_masks

    def _create_preview_grid(self, images, replaced_indices, max_preview_size=512, border_width=3):
        B, H, W, C = images.shape

        # Calculate grid layout
        images_per_row = math.ceil(math.sqrt(B))
        num_rows = math.ceil(B / images_per_row)

        # Calculate thumbnail size so grid fits within max_preview_size
        cell_w = max_preview_size // images_per_row
        cell_h = max_preview_size // num_rows
        scale = min(cell_w / W, cell_h / H)
        thumb_w = max(1, int(W * scale))
        thumb_h = max(1, int(H * scale))

        # Create grid canvas
        grid_w = thumb_w * images_per_row
        grid_h = thumb_h * num_rows
        grid_image = Image.new('RGB', (grid_w, grid_h), (0, 0, 0))
        draw = ImageDraw.Draw(grid_image)

        for i in range(B):
            row = i // images_per_row
            col = i % images_per_row

            # Convert tensor to PIL thumbnail
            img_np = (images[i].cpu().numpy() * 255).astype('uint8')
            if img_np.shape[-1] == 4:
                img_np = img_np[..., :3]
            pil_img = Image.fromarray(img_np)
            pil_img = pil_img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)

            x = col * thumb_w
            y = row * thumb_h
            grid_image.paste(pil_img, (x, y))

            # Draw green border on replaced images
            if i in replaced_indices:
                for b in range(border_width):
                    draw.rectangle(
                        [x + b, y + b, x + thumb_w - 1 - b, y + thumb_h - 1 - b],
                        outline=(0, 255, 0)
                    )

        return grid_image

    def _prepare_image_for_display(self, pil_image):
        """Convert PIL image to base64 for frontend display."""
        display_img = pil_image.copy()
        max_size = (512, 512)
        display_img.thumbnail(max_size, Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        display_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
