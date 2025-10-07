import io
import torch
from PIL import Image
import numpy as np
from PyPDF2 import PdfReader
import fitz  # PyMuPDF


class FL_PDFToImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf": ("PDF",),
                "dpi": ("INT", {"default": 200, "min": 72, "max": 600}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_pdf_to_images"
    CATEGORY = "🏵️Fill Nodes/PDF"

    def convert_pdf_to_images(self, pdf, dpi):
        try:
            if isinstance(pdf, list):
                all_images = []
                for single_pdf in pdf:
                    images = self._process_single_pdf(single_pdf, dpi)
                    all_images.extend(images)
                return (torch.cat(all_images, dim=0),)
            else:
                images = self._process_single_pdf(pdf, dpi)
                return (torch.cat(images, dim=0),)

        except Exception as e:
            print(f"Error converting PDF: {str(e)}")
            # Return a blank image in case of error
            blank = torch.ones((1, 512, 512, 3), dtype=torch.float32)
            return (blank,)

    def _process_single_pdf(self, pdf, dpi):
        pdf_content = pdf['content']
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        images = []

        for page in doc:
            # Calculate zoom factor based on DPI (72 DPI is the base)
            zoom = dpi / 72
            matrix = fitz.Matrix(zoom, zoom)

            # Convert page to pixmap
            pix = page.get_pixmap(matrix=matrix)

            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Convert to numpy array and normalize
            img_np = np.array(img).astype(np.float32) / 255.0

            # Convert to tensor and add batch dimension
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            images.append(img_tensor)

        doc.close()
        return images

    @classmethod
    def IS_CHANGED(cls, pdf, dpi):
        if isinstance(pdf, list):
            return hash(tuple((p.get('path', ''), dpi) for p in pdf))
        return hash((pdf.get('path', ''), dpi))