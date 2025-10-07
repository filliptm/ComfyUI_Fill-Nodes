import torch
from PIL import Image
import io
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

class FL_ImagesToPDF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "dpi": ("INT", {"default": 300, "min": 72, "max": 600}),
                "page_size": (["A4", "Letter", "Legal"], {"default": "A4"}),
            },
        }

    RETURN_TYPES = ("PDF",)
    FUNCTION = "create_pdf_from_images"
    CATEGORY = "🏵️Fill Nodes/PDF"

    def create_pdf_from_images(self, images, dpi, page_size):
        # Convert tensor images to PIL images
        pil_images = [Image.fromarray((img.squeeze().cpu().numpy() * 255).astype('uint8')) for img in images]

        # Create a BytesIO object to store the PDF
        pdf_buffer = io.BytesIO()

        # Set up the PDF canvas
        if page_size == "A4":
            page_width, page_height = 8.27 * inch, 11.69 * inch
        elif page_size == "Letter":
            page_width, page_height = 8.5 * inch, 11 * inch
        elif page_size == "Legal":
            page_width, page_height = 8.5 * inch, 14 * inch

        c = canvas.Canvas(pdf_buffer, pagesize=(page_width, page_height))

        for img in pil_images:
            # Calculate image dimensions to fit the page
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            if aspect > (page_height / page_width):
                img_height = page_height
                img_width = page_height / aspect
            else:
                img_width = page_width
                img_height = page_width * aspect

            # Center the image on the page
            x = (page_width - img_width) / 2
            y = (page_height - img_height) / 2

            # Convert PIL Image to ImageReader
            img_reader = ImageReader(img)

            # Add the image to the PDF
            c.drawImage(img_reader, x, y, width=img_width, height=img_height)
            c.showPage()

        c.save()

        # Get the PDF content
        pdf_content = pdf_buffer.getvalue()

        # Create a dictionary to store PDF information
        pdf_data = {
            'content': pdf_content,
            'num_pages': len(pil_images),
            'path': 'memory'  # Since it's not saved to disk yet
        }

        return (pdf_data,)

    @classmethod
    def IS_CHANGED(cls, images, dpi, page_size):
        # This method should return a unique value every time the node's output would be different
        return (hash(images.cpu().numpy().tobytes()), dpi, page_size)