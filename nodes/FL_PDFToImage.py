import fitz  # PyMuPDF
import torch
from PIL import Image
from io import BytesIO
from .utils import pil_to_tensor


class FL_PDFToImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf": ("PDF",),
                "dpi": ("INT", {"default": 200, "min": 72, "max": 600}),
                "output_format": (["PNG", "JPEG"], {"default": "PNG"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "convert_pdf_to_images"
    CATEGORY = "🏵️Fill Nodes/PDF"

    def convert_pdf_to_images(self, pdf, dpi, output_format):
        pdf_content = pdf['content']

        images = []
        image_data = []

        try:
            # Open the PDF
            doc = fitz.open(stream=pdf_content, filetype="pdf")

            for i, page in enumerate(doc):
                # Set the resolution
                zoom = dpi / 72  # 72 is the default DPI for PDF
                mat = fitz.Matrix(zoom, zoom)

                # Render page to an image
                pix = page.get_pixmap(matrix=mat)

                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Store image in memory
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format=output_format)
                img_byte_arr = img_byte_arr.getvalue()
                image_data.append(img_byte_arr)

                # Convert PIL image to tensor
                img_tensor = pil_to_tensor(img)
                images.append(img_tensor)

            # Stack all image tensors
            images_tensor = torch.cat(images, dim=0)

            # Join image data into a single string (base64 encoded)
            import base64
            image_data_str = ",".join([base64.b64encode(img).decode('utf-8') for img in image_data])

            return (images_tensor, image_data_str)

        except Exception as e:
            raise RuntimeError(f"Error converting PDF to images: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, pdf, dpi, output_format):
        # This method should return a unique value every time the node's output would be different
        return (pdf['path'], dpi, output_format)