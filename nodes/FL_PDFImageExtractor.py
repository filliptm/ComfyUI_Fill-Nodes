import io
import torch
from PIL import Image
import numpy as np
from pdf2image import convert_from_bytes

class FL_PDFImageExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf": ("PDF",),
                "min_width": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "min_height": ("INT", {"default": 100, "min": 1, "max": 1000}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "extract_images"
    CATEGORY = "🏵️Fill Nodes/PDF"
    OUTPUT_NODE = True

    def extract_images(self, pdf, min_width, min_height):
        try:
            pdf_content = pdf['content']

            # Convert the PDF to images using pdf2image
            pil_images = convert_from_bytes(pdf_content)

            extracted_images = []
            for img in pil_images:
                # Filter out images that don't meet the size criteria
                if img.width >= min_width and img.height >= min_height:
                    # Convert PIL Image to numpy array
                    img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0

                    # Convert to tensor in the format [1, H, W, C]
                    img_tensor = torch.from_numpy(img_np).unsqueeze(0)

                    extracted_images.append(img_tensor)

            if not extracted_images:
                # Return a dummy tensor if no images were extracted
                dummy_tensor = torch.zeros((1, min_height, min_width, 3))
                return (dummy_tensor,)

            # Combine all images into a single tensor with shape [B, H, W, C]
            combined_images = torch.cat(extracted_images, dim=0)
            return (combined_images,)

        except Exception as e:
            raise RuntimeError(f"Error extracting images from PDF: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, pdf, min_width, min_height):
        return hash(pdf['content']) + hash((min_width, min_height))
