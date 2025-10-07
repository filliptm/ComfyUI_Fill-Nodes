import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from comfy.utils import ProgressBar

class FL_ImageNotes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"default": "Text Here", "multiline": True}),
                "bar_height": ("INT", {"default": 50, "min": 10, "max": 200, "step": 2, "description": "Initial bar height (auto-adjusted if word wrap is enabled)"}),
                "text_size": ("INT", {"default": 24, "min": 10, "max": 100, "step": 1}),
                "border": ("INT", {"default": 0, "min": 0, "max": 200, "step": 5, "description": "Border size in pixels (left, right, bottom)"}),
                "word_wrap": (["disable", "enable"], {"default": "disable", "description": "Enable text wrapping and auto-height adjustment"}),
                "bar_position": (["top", "bottom"], {"default": "top", "description": "Position of the text bar"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_notes"
    CATEGORY = "🏵️Fill Nodes/Image"

    def add_notes(self, images, text, bar_height, text_size, border, word_wrap, bar_position):
        result = []
        total_images = len(images)
        pbar = ProgressBar(total_images)
        for i, image in enumerate(images, start=1):
            img = self.t2p(image)
            result_img = self.add_text_bar(img, text, bar_height, text_size, border, word_wrap, bar_position)
            result_img = self.p2t(result_img)
            result.append(result_img)

            pbar.update_absolute(i)

        return (torch.cat(result, dim=0),)

    def add_text_bar(self, image, text, bar_height, text_size, border, word_wrap, bar_position):
        width, height = image.size
        
        # Calculate available width for text (accounting for borders)
        available_width = width + (border * 2) - 20  # 10px padding on each side
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(current_dir, "..", "fonts", "arial.ttf")
        font = ImageFont.truetype(font_path, text_size)
        
        # Handle word wrapping and calculate required bar height
        if word_wrap == "enable":
            wrapped_lines = self.wrap_text(text, font, available_width)
            line_height = text_size * 1.2  # Add some line spacing
            required_height = int(len(wrapped_lines) * line_height) + 20  # Add padding
            actual_bar_height = max(bar_height, required_height)  # Use the larger of specified or required height
        else:
            wrapped_lines = [text]  # No wrapping, just use the text as is
            actual_bar_height = bar_height
        
        # Calculate new dimensions with border
        new_width = width + (border * 2)  # Left and right borders
        new_height = height + actual_bar_height + border  # Bar and bottom border
        
        # Create new image with black background
        new_image = Image.new("RGB", (new_width, new_height), color="black")
        
        # Determine image paste position based on bar position
        if bar_position == "top":
            image_y = actual_bar_height
            text_y_start = 10  # Padding from top
        else:  # bottom
            image_y = 0
            text_y_start = height + border  # Start text below the image
        
        # Paste original image with offset for borders
        new_image.paste(image, (border, image_y))
        
        draw = ImageDraw.Draw(new_image)
        
        # Draw each line of text
        line_height = text_size * 1.2
        for i, line in enumerate(wrapped_lines):
            text_width, text_height = self.get_text_size(line, font)
            x = (new_width - text_width) // 2  # Center text in the new width
            y = text_y_start + (i * line_height)
            draw.text((x, y), line, font=font, fill="white")

        return new_image

    def get_text_size(self, text, font):
        ascent, descent = font.getmetrics()
        text_width = font.getmask(text).getbbox()[2]
        text_height = font.getmask(text).getbbox()[3] + descent
        return text_width, text_height
    
    def wrap_text(self, text, font, max_width):
        """Split text into lines that fit within max_width."""
        words = text.split()
        if not words:
            return [""]
            
        lines = []
        current_line = words[0]
        
        for word in words[1:]:
            # Check if adding this word would exceed the max width
            test_line = current_line + " " + word
            test_width, _ = self.get_text_size(test_line, font)
            
            if test_width <= max_width:
                current_line = test_line
            else:
                # Line is full, start a new line
                lines.append(current_line)
                current_line = word
        
        # Add the last line
        lines.append(current_line)
        return lines

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def p2t(self, p):
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            t = torch.from_numpy(i).unsqueeze(0)
        return t
