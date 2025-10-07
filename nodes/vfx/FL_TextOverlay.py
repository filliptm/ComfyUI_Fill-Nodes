import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from ..sup import ROOT_FONTS
from comfy.utils import ProgressBar

def parse_fonts() -> dict:
    mgr = font_manager.FontManager()
    return {f"{font.name[0].upper()}/{font.name}": font.fname for font in mgr.ttflist}

class FL_TextOverlayNode:
    env_var_value = os.getenv("FL_USE_SYSTEM_FONTS", 'false').strip().lower()
    if env_var_value.strip() in ('true', '1', 't'):
        FONTS = parse_fonts()
    else:
        FONTS = {f"{str(font.stem)}": str(font) for font in ROOT_FONTS.glob("*.[to][tf][f]")}
    
    FONT_NAMES = sorted(FONTS.keys())
    if not FONT_NAMES: # Add a default if no fonts are found
        FONT_NAMES.append("Default")
        FONTS["Default"] = "default" # Placeholder, PIL will use its default

    DESCRIPTION = """
FL_TextOverlayNode applies a text overlay to an image.
You can specify the text, its position (as a percentage of image dimensions),
font size, font color, and choose from available system or local fonts.
"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"default": "Hello, ComfyUI!", "multiline": True}),
                "font": (s.FONT_NAMES, {"default": s.FONT_NAMES[0] if s.FONT_NAMES else "Default"}),
                "font_size": ("INT", {"default": 51, "min": 1, "step": 1}),
                "font_color_r": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "description": "Red color value"}),
                "font_color_g": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "description": "Green color value"}),
                "font_color_b": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "description": "Blue color value"}),
                "x_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.1, "description": "X position as percentage from left"}),
                "y_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.1, "description": "Y position as percentage from top"}),
                "anchor": (["left-top", "center-top", "right-top",
                            "left-center", "center-center", "right-center",
                            "left-bottom", "center-bottom", "right-bottom"], 
                           {"default": "center-center", "description": "Text anchor point relative to X,Y coordinates"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_text_overlay"
    CATEGORY = "🏵️Fill Nodes/VFX"

    def _get_pil_anchor(self, anchor_str):
        # PIL anchor uses 'l' for left, 'm' for middle (horizontal), 'r' for right
        # 't' for top, 'm' for middle (vertical), 'b' for bottom
        # Example: "la" for left-ascent, "mm" for middle-middle
        # We simplify to 9 points. PIL's text_anchor is more nuanced with baselines.
        # We'll use the xy as the top-left of the text box and adjust based on text size for other anchors.
        # For direct PIL anchor mapping (approximations):
        mapping = {
            "left-top": "la", "center-top": "ma", "right-top": "ra",
            "left-center": "lm", "center-center": "mm", "right-center": "rm",
            "left-bottom": "lb", "center-bottom": "mb", "right-bottom": "rb"
        }
        return mapping.get(anchor_str, "mm") # default to middle-center

    def apply_text_overlay(self, image: torch.Tensor, text: str, font: str, font_size: int, font_color_r: int, font_color_g: int, font_color_b: int, x_percent: float, y_percent: float, anchor: str):
        batch_size = image.shape[0]
        result_images = []
        
        font_color_rgb = (font_color_r, font_color_g, font_color_b)
        
        try:
            if font == "Default" or self.FONTS[font] == "default":
                 font_path = ImageFont.load_default() # Use PIL's internal default
            else:
                font_path = ImageFont.truetype(self.FONTS[font], font_size)
        except Exception as e:
            print(f"Error loading font '{font}'. Using default. Error: {e}")
            try:
                font_path = ImageFont.load_default(size=font_size) # Try to get default with size
            except AttributeError: # Older PIL might not support size for load_default
                font_path = ImageFont.load_default()


        pbar = ProgressBar(batch_size)
        for i in range(batch_size):
            img_tensor = image[i]
            pil_image = Image.fromarray((img_tensor.cpu().numpy() * 255).astype(np.uint8))
            pil_image = pil_image.convert("RGB") # Ensure it's RGB
            
            draw = ImageDraw.Draw(pil_image)
            
            img_width, img_height = pil_image.size
            x_abs = int(img_width * (x_percent / 100.0))
            y_abs = int(img_height * (y_percent / 100.0))

            # For PIL versions that support anchor directly in text()
            # For older versions, we might need to calculate bounding box and adjust xy
            # Modern PIL versions (>=9.2.0) support `anchor` in `text()`
            try:
                pil_anchor_val = self._get_pil_anchor(anchor)
                draw.text((x_abs, y_abs), text, font=font_path, fill=font_color_rgb, anchor=pil_anchor_val)
            except TypeError: # Older PIL might not have 'anchor'
                print("Warning: PIL version might not support 'anchor' directly. Text will be drawn with top-left at X,Y.")
                # Manual anchor adjustment (simplified)
                # This requires textbbox which itself might need a newer PIL or more complex handling
                try:
                    # Get text bounding box (left, top, right, bottom)
                    bbox = draw.textbbox((x_abs, y_abs), text, font=font_path)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1] # Note: this is full box height, not just ascent/descent

                    if 'center' in anchor.split('-')[0]: # horizontal center
                        x_abs -= text_width // 2
                    elif 'right' in anchor.split('-')[0]: # horizontal right
                        x_abs -= text_width
                    
                    if 'center' in anchor.split('-')[1]: # vertical center
                        y_abs -= text_height // 2
                    elif 'bottom' in anchor.split('-')[1]: # vertical bottom
                        y_abs -= text_height
                    draw.text((x_abs, y_abs), text, font=font_path, fill=font_color_rgb)

                except Exception as e_bbox:
                    print(f"Fallback for anchor failed: {e_bbox}. Drawing at top-left.")
                    draw.text((x_abs, y_abs), text, font=font_path, fill=font_color_rgb)


            img_array = np.array(pil_image).astype(np.float32) / 255.0
            result_images.append(torch.from_numpy(img_array).unsqueeze(0))
            pbar.update_absolute(i + 1, batch_size)
            
        final_tensor = torch.cat(result_images, dim=0)
        return (final_tensor,)