import io
import fitz  # PyMuPDF
import torch
from PIL import Image
import numpy as np


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
            doc = fitz.open(stream=pdf['content'], filetype="pdf")
            extracted_images = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)

                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    image = Image.open(io.BytesIO(image_bytes))

                    if image.width >= min_width and image.height >= min_height:
                        # Convert PIL Image to numpy array
                        img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0

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