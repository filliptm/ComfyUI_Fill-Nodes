import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap

class FL_ImageCaptionLayout:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_directory": ("STRING", {"default": ""}),
                "images_per_row": ("INT", {"default": 3, "min": 1, "max": 10}),
                "image_size": ("INT", {"default": 256, "min": 64, "max": 1024}),
                "caption_height": ("INT", {"default": 64, "min": 32, "max": 256}),
                "font_size": ("INT", {"default": 12, "min": 8, "max": 32}),
                "padding": ("INT", {"default": 10, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_layout"
    CATEGORY = "🏵️Fill Nodes/Captioning"
    OUTPUT_NODE = True

    def create_layout(self, image_directory, images_per_row, image_size, caption_height, font_size, padding):
        # Colors
        background_color = (255, 255, 255)  # White
        text_color = (0, 0, 0)  # Black
        caption_background_color = (255, 255, 255)  # White

        # Get all image files and their corresponding caption files
        image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()  # Sort files to ensure consistent order

        # Calculate layout dimensions with padding
        total_width = images_per_row * (image_size + padding) + padding
        rows = (len(image_files) + images_per_row - 1) // images_per_row
        total_height = rows * (image_size + caption_height + padding) + padding

        # Create the layout with padding
        layout = Image.new('RGB', (total_width, total_height), color=background_color)

        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        for i, image_file in enumerate(image_files):
            # Load and resize image
            img_path = os.path.join(image_directory, image_file)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((image_size, image_size), Image.LANCZOS)

            # Load caption
            caption_file = os.path.splitext(image_file)[0] + '.txt'
            caption_path = os.path.join(image_directory, caption_file)
            try:
                with open(caption_path, 'r') as f:
                    caption = f.read().strip()
            except FileNotFoundError:
                caption = "No caption found"

            # Calculate position with padding
            row = i // images_per_row
            col = i % images_per_row
            x = padding + col * (image_size + padding)
            y = padding + row * (image_size + caption_height + padding)

            # Paste image
            layout.paste(img, (x, y))

            # Create caption box
            caption_box = Image.new('RGB', (image_size, caption_height), color=caption_background_color)
            draw = ImageDraw.Draw(caption_box)

            # Wrap text
            wrapped_text = textwrap.fill(caption, width=(image_size - 10) // (font_size // 2))

            # Draw wrapped text
            draw.text((5, 5), wrapped_text, font=font, fill=text_color)

            # Paste caption box
            layout.paste(caption_box, (x, y + image_size))

        # Convert to tensor
        layout_tensor = torch.from_numpy(np.array(layout).astype(np.float32) / 255.0).unsqueeze(0)
        return (layout_tensor,)