import torch
import numpy as np
import io
import base64
from PIL import Image, ImageFilter
from ..utils import tensor_to_pil, pil_to_tensor
from server import PromptServer


class FL_ImageOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "mask": ("MASK",),
                "x_offset": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "alignment": ([
                    "custom",
                    "center",
                    "top-left",
                    "top-center",
                    "top-right",
                    "center-left",
                    "center-right",
                    "bottom-left",
                    "bottom-center",
                    "bottom-right"
                ], {"default": "custom"}),
                "resize_overlay": ([
                    "none",
                    "fit_to_base",
                    "scale_50%",
                    "scale_75%",
                    "scale_125%",
                    "scale_150%",
                    "scale_200%"
                ], {"default": "none"}),
                "blend_mode": ([
                    "normal",
                    "multiply",
                    "screen",
                    "overlay",
                    "add"
                ], {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "mask_feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "boundary_behavior": (["clip", "extend_canvas"], {"default": "clip"}),
                "show_preview": ("BOOLEAN", {"default": False, "label": "Show Preview on Node"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "overlay_images"
    OUTPUT_NODE = True
    CATEGORY = "🏵️Fill Nodes/Image"

    def overlay_images(self, base_image, overlay_image, mask, x_offset, y_offset,
                      alignment, resize_overlay, blend_mode, opacity, invert_mask,
                      mask_feather, boundary_behavior, show_preview=False):
        # Process batch - use first image from each batch
        base_pil = tensor_to_pil(base_image, batch_index=0)
        overlay_pil = tensor_to_pil(overlay_image, batch_index=0)

        # Convert mask tensor to PIL (masks are typically [B, H, W])
        if len(mask.shape) == 3:
            mask_np = mask[0].cpu().numpy()
        elif len(mask.shape) == 2:
            mask_np = mask.cpu().numpy()
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        # Convert to 0-255 range and create PIL image
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')

        # Resize mask to match overlay dimensions
        if mask_pil.size != overlay_pil.size:
            mask_pil = mask_pil.resize(overlay_pil.size, Image.Resampling.LANCZOS)

        # Invert mask if requested
        if invert_mask:
            mask_pil = Image.eval(mask_pil, lambda x: 255 - x)

        # Apply feathering to mask
        if mask_feather > 0:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=mask_feather))

        # Resize overlay if requested
        overlay_pil = self.resize_overlay_image(overlay_pil, base_pil, resize_overlay, mask_pil)

        # Recalculate mask size after resize
        if mask_pil.size != overlay_pil.size:
            mask_pil = mask_pil.resize(overlay_pil.size, Image.Resampling.LANCZOS)

        # Calculate position based on alignment
        x_pos, y_pos = self.calculate_position(
            base_pil.size, overlay_pil.size, x_offset, y_offset, alignment
        )

        # Perform the compositing
        result_pil = self.composite_images(
            base_pil, overlay_pil, mask_pil, x_pos, y_pos,
            blend_mode, opacity, boundary_behavior
        )

        # Send preview to frontend if enabled
        if show_preview:
            display_image = self.prepare_image_for_display(result_pil)
            PromptServer.instance.send_sync("fl_image_overlay", {"image": display_image})

        # Convert back to tensor
        result_tensor = pil_to_tensor(result_pil)

        return (result_tensor,)

    def resize_overlay_image(self, overlay, base, resize_mode, mask):
        """Resize the overlay image based on the selected mode"""
        if resize_mode == "none":
            return overlay

        base_w, base_h = base.size
        overlay_w, overlay_h = overlay.size

        if resize_mode == "fit_to_base":
            # Resize overlay to match base dimensions
            new_size = (base_w, base_h)
        else:
            # Extract scale percentage
            scale_map = {
                "scale_50%": 0.5,
                "scale_75%": 0.75,
                "scale_125%": 1.25,
                "scale_150%": 1.5,
                "scale_200%": 2.0
            }
            scale = scale_map.get(resize_mode, 1.0)
            new_size = (int(overlay_w * scale), int(overlay_h * scale))

        resized_overlay = overlay.resize(new_size, Image.Resampling.LANCZOS)
        return resized_overlay

    def calculate_position(self, base_size, overlay_size, x_offset, y_offset, alignment):
        """Calculate the position to place the overlay based on alignment"""
        base_w, base_h = base_size
        overlay_w, overlay_h = overlay_size

        # Alignment presets
        if alignment == "center":
            x = (base_w - overlay_w) // 2
            y = (base_h - overlay_h) // 2
        elif alignment == "top-left":
            x, y = 0, 0
        elif alignment == "top-center":
            x = (base_w - overlay_w) // 2
            y = 0
        elif alignment == "top-right":
            x = base_w - overlay_w
            y = 0
        elif alignment == "center-left":
            x = 0
            y = (base_h - overlay_h) // 2
        elif alignment == "center-right":
            x = base_w - overlay_w
            y = (base_h - overlay_h) // 2
        elif alignment == "bottom-left":
            x = 0
            y = base_h - overlay_h
        elif alignment == "bottom-center":
            x = (base_w - overlay_w) // 2
            y = base_h - overlay_h
        elif alignment == "bottom-right":
            x = base_w - overlay_w
            y = base_h - overlay_h
        else:  # custom
            x, y = 0, 0

        # Apply offsets
        x += x_offset
        y += y_offset

        return x, y

    def composite_images(self, base, overlay, mask, x_pos, y_pos, blend_mode, opacity, boundary_behavior):
        """Composite the overlay onto the base image"""
        base_w, base_h = base.size
        overlay_w, overlay_h = overlay.size

        # Handle boundary behavior
        if boundary_behavior == "extend_canvas":
            # Calculate required canvas size
            canvas_w = max(base_w, x_pos + overlay_w, abs(min(0, x_pos)) + base_w)
            canvas_h = max(base_h, y_pos + overlay_h, abs(min(0, y_pos)) + base_h)

            # Create extended canvas
            canvas = Image.new('RGB', (canvas_w, canvas_h), (0, 0, 0))

            # Paste base image at appropriate position
            base_x = abs(min(0, x_pos))
            base_y = abs(min(0, y_pos))
            canvas.paste(base, (base_x, base_y))

            # Adjust overlay position for extended canvas
            overlay_x = x_pos if x_pos >= 0 else 0
            overlay_y = y_pos if y_pos >= 0 else 0

            result = canvas.copy()
        else:  # clip
            result = base.copy()
            overlay_x = x_pos
            overlay_y = y_pos

            # Calculate visible region of overlay
            src_x = max(0, -x_pos)
            src_y = max(0, -y_pos)
            dst_x = max(0, x_pos)
            dst_y = max(0, y_pos)

            # Calculate dimensions of visible region
            visible_w = min(overlay_w - src_x, base_w - dst_x)
            visible_h = min(overlay_h - src_y, base_h - dst_y)

            # If overlay is completely outside bounds, return base image
            if visible_w <= 0 or visible_h <= 0:
                return base

            # Crop overlay and mask to visible region
            overlay = overlay.crop((src_x, src_y, src_x + visible_w, src_y + visible_h))
            mask = mask.crop((src_x, src_y, src_x + visible_w, src_y + visible_h))
            overlay_x = dst_x
            overlay_y = dst_y

        # Apply blend mode
        if blend_mode != "normal":
            overlay = self.apply_blend_mode(result, overlay, blend_mode, overlay_x, overlay_y)

        # Apply global opacity to mask
        if opacity < 1.0:
            mask_np = np.array(mask).astype(np.float32)
            mask_np = (mask_np * opacity).astype(np.uint8)
            mask = Image.fromarray(mask_np, mode='L')

        # Composite using the mask
        result.paste(overlay, (overlay_x, overlay_y), mask)

        return result

    def apply_blend_mode(self, base, overlay, mode, x_pos, y_pos):
        """Apply blend mode to overlay based on the underlying base image region"""
        # Extract the region from base that overlay will cover
        overlay_w, overlay_h = overlay.size
        base_region = base.crop((x_pos, y_pos, x_pos + overlay_w, y_pos + overlay_h))

        # Convert to numpy arrays for blending
        base_np = np.array(base_region).astype(np.float32) / 255.0
        overlay_np = np.array(overlay).astype(np.float32) / 255.0

        # Apply blend mode
        if mode == "multiply":
            result_np = base_np * overlay_np
        elif mode == "screen":
            result_np = 1 - (1 - base_np) * (1 - overlay_np)
        elif mode == "overlay":
            # Overlay blend mode
            mask = base_np < 0.5
            result_np = np.where(mask,
                                 2 * base_np * overlay_np,
                                 1 - 2 * (1 - base_np) * (1 - overlay_np))
        elif mode == "add":
            result_np = np.clip(base_np + overlay_np, 0, 1)
        else:  # normal
            result_np = overlay_np

        # Convert back to PIL
        result_np = (result_np * 255).astype(np.uint8)
        return Image.fromarray(result_np, mode='RGB')

    def prepare_image_for_display(self, pil_image):
        """Convert PIL image to base64 for frontend display"""
        # Create a copy to avoid modifying the original
        display_img = pil_image.copy()

        # Resize image if it's too large for preview
        max_size = (512, 512)
        display_img.thumbnail(max_size, Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        display_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
