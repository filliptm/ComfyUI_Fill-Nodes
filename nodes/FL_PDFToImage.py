import io
import torch
from PIL import Image
import numpy as np
from pdf2image import convert_from_bytes  # pdf2image is used instead of fitz

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
        if isinstance(pdf, list):
            # Handle list of PDFs
            all_images = []
            for single_pdf in pdf:
                images = self._process_single_pdf(single_pdf, dpi)
                all_images.extend(images)
            return (torch.cat(all_images, dim=0),)
        else:
            # Handle single PDF
            images = self._process_single_pdf(pdf, dpi)
            return (torch.cat(images, dim=0),)

    def _process_single_pdf(self, pdf, dpi):
        try:
            pdf_content = pdf['content']

            # Convert the PDF to images using pdf2image
            pil_images = convert_from_bytes(pdf_content, dpi=dpi)

            images = []
            for img in pil_images:
                # Convert PIL Image to numpy array and then to tensor in the format [B, H, W, C]
                img_np = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # Add batch dimension

                images.append(img_tensor)

            return images

        except Exception as e:
            raise RuntimeError(f"Error converting PDF to images: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, pdf, dpi):
        # This method should return a unique value every time the node's output would be different
        if isinstance(pdf, list):
            return hash(tuple((p['path'], dpi) for p in pdf))
        return hash((pdf['path'], dpi))
