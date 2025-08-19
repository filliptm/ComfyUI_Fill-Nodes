import os
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
from reportlab.lib.pagesizes import letter, portrait, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO


class FL_ImageCaptionLayoutPDF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_directory": ("STRING", {"default": ""}),
                "images_per_row": ("INT", {"default": 6, "min": 1, "max": 10}),
                "display_size": ("INT", {"default": 100, "min": 64, "max": 512}),
                "caption_height": ("INT", {"default": 80, "min": 32, "max": 256}),
                "font_size": ("INT", {"default": 4, "min": 4, "max": 32}),
                "padding": ("INT", {"default": 10, "min": 0, "max": 100}),
                "output_directory": ("STRING", {"default": ""}),
                "output_filename": ("STRING", {"default": "output"}),
                "orientation": (["horizontal", "vertical"], {"default": "horizontal"}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "PDF")
    FUNCTION = "create_layout"
    CATEGORY = "🏵️Fill Nodes/Captioning"
    OUTPUT_NODE = True

    def create_layout(self, image_directory, images_per_row, display_size, caption_height, font_size, padding,
                      output_directory, output_filename, orientation):
        output_path, pdf_bytes = self.create_pdf_layout(image_directory, images_per_row, display_size, caption_height,
                                                        font_size, padding, output_directory, output_filename,
                                                        orientation)
        preview_image = self.create_image_preview(image_directory, images_per_row, display_size, caption_height,
                                                  font_size, padding, orientation)
        return (output_path, preview_image, pdf_bytes)

    def create_pdf_layout(self, image_directory, images_per_row, display_size, caption_height, font_size, padding,
                          output_directory, output_filename, orientation):
        # Get the path to the fonts directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fonts_dir = os.path.join(os.path.dirname(current_dir), "fonts")
        font_path = os.path.join(fonts_dir, "arial.ttf")

        # Check if the font file exists
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Font file not found: {font_path}")

        # Register the font
        pdfmetrics.registerFont(TTFont('Arial', font_path))

        # Get all image files and their corresponding caption files
        image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()  # Sort files to ensure consistent order

        # PDF setup
        page_size = landscape(letter) if orientation == "horizontal" else portrait(letter)
        width, height = page_size
        output_path = os.path.join(output_directory, f"{output_filename}.pdf")
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=page_size)

        # Calculate layout dimensions
        display_size_pt = display_size
        caption_height_pt = caption_height
        padding_pt = padding
        item_width = display_size_pt + padding_pt
        item_height = display_size_pt + caption_height_pt + padding_pt

        # Calculate how many items can fit on a page
        items_per_row = min(images_per_row, math.floor((width - padding_pt) / item_width))
        rows_per_page = math.floor((height - padding_pt) / item_height)
        items_per_page = items_per_row * rows_per_page

        # Validate that at least one item can fit on the page
        if items_per_page == 0:
            # Calculate page dimensions for error message
            page_width_usable = width - padding_pt
            page_height_usable = height - padding_pt
            
            error_msg = (
                f"Layout Error: Items are too large to fit on the page.\n"
                f"Current settings:\n"
                f"  - Page size: {width:.0f} x {height:.0f} pts ({orientation} orientation)\n"
                f"  - Usable area: {page_width_usable:.0f} x {page_height_usable:.0f} pts (after padding)\n"
                f"  - Item size needed: {item_width:.0f} x {item_height:.0f} pts\n"
                f"  - Display size: {display_size_pt} pts\n"
                f"  - Caption height: {caption_height_pt} pts\n"
                f"  - Padding: {padding_pt} pts\n\n"
                f"Solutions:\n"
                f"  1. Reduce 'display_size' (currently {display_size_pt})\n"
                f"  2. Reduce 'caption_height' (currently {caption_height_pt})\n"
                f"  3. Reduce 'padding' (currently {padding_pt})\n"
                f"  4. Try switching orientation (currently '{orientation}')\n"
                f"  5. Use fewer 'images_per_row' (currently {images_per_row})"
            )
            raise ValueError(error_msg)

        for i in range(0, len(image_files), items_per_page):
            page_images = image_files[i:i + items_per_page]

            for j, image_file in enumerate(page_images):
                # Calculate position
                row = j // items_per_row
                col = j % items_per_row
                x = padding_pt + col * item_width
                y = height - padding_pt - (row + 1) * item_height

                # Load and draw image
                img_path = os.path.join(image_directory, image_file)
                img = Image.open(img_path)
                aspect_ratio = img.width / img.height
                display_height = display_size_pt / aspect_ratio

                c.drawImage(img_path, x, y + caption_height_pt, width=display_size_pt, height=display_height,
                            preserveAspectRatio=True, anchor='sw')

                # Load caption
                caption_file = os.path.splitext(image_file)[0] + '.txt'
                caption_path = os.path.join(image_directory, caption_file)
                try:
                    with open(caption_path, 'r') as f:
                        caption = f.read().strip()
                except FileNotFoundError:
                    caption = "No caption found"

                # Draw caption
                c.setFont("Arial", font_size)
                text_object = c.beginText(x, y + caption_height_pt - font_size)
                wrapped_text = textwrap.fill(caption, width=int(display_size_pt / (font_size * 0.6)))
                for line in wrapped_text.split('\n'):
                    text_object.textLine(line)
                c.drawText(text_object)

            c.showPage()  # Start a new page after each set of items

        c.save()
        pdf_bytes = pdf_buffer.getvalue()

        # Save the PDF to file
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)

        print(f"PDF saved as {output_path}")
        return output_path, pdf_bytes

    def create_image_preview(self, image_directory, images_per_row, display_size, caption_height, font_size, padding,
                             orientation):
        # Get all image files and their corresponding caption files
        image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()  # Sort files to ensure consistent order

        # Calculate layout dimensions
        total_width = images_per_row * (display_size + padding) + padding
        rows = (len(image_files) + images_per_row - 1) // images_per_row
        total_height = rows * (display_size + caption_height + padding) + padding

        # Create the layout
        layout = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))

        # Load font
        font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts", "arial.ttf")
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

        for i, image_file in enumerate(image_files):
            # Load and resize image
            img_path = os.path.join(image_directory, image_file)
            img = Image.open(img_path).convert('RGB')
            img.thumbnail((display_size, display_size), Image.LANCZOS)

            # Calculate position
            row = i // images_per_row
            col = i % images_per_row
            x = padding + col * (display_size + padding)
            y = padding + row * (display_size + caption_height + padding)

            # Paste image
            layout.paste(img, (x, y))

            # Load caption
            caption_file = os.path.splitext(image_file)[0] + '.txt'
            caption_path = os.path.join(image_directory, caption_file)
            try:
                with open(caption_path, 'r') as f:
                    caption = f.read().strip()
            except FileNotFoundError:
                caption = "No caption found"

            # Draw caption
            draw = ImageDraw.Draw(layout)
            wrapped_text = textwrap.fill(caption, width=int(display_size / (font_size * 0.6)))
            draw.text((x, y + display_size + 5), wrapped_text, font=font, fill=(0, 0, 0))

        # Convert to tensor for preview
        preview_tensor = torch.from_numpy(np.array(layout).astype(np.float32) / 255.0).unsqueeze(0)

        return preview_tensor