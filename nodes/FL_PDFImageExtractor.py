import io
import torch
from PIL import Image
import numpy as np
from PyPDF2 import PdfReader


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
            pdf_reader = PdfReader(io.BytesIO(pdf['content']))
            extracted_images = []

            for page_num, page in enumerate(pdf_reader.pages):
                if '/XObject' in page['/Resources']:
                    xObject = page['/Resources']['/XObject'].get_object()
                    for obj in xObject:
                        if xObject[obj]['/Subtype'] == '/Image':
                            base_image = xObject[obj]
                            filter_type = base_image.get('/Filter')

                            # Get decoded image data directly from PyPDF2
                            if filter_type == ['/ASCII85Decode', '/FlateDecode']:
                                decoded_data = base_image.get_data()  # PyPDF2 automatically decodes this

                                # Recreate the image using the decompressed data
                                img = Image.frombytes(
                                    "RGB",
                                    (base_image.get('/Width'), base_image.get('/Height')),
                                    decoded_data
                                )
                            elif filter_type == '/DCTDecode':
                                # JPEG format
                                img = Image.open(io.BytesIO(base_image.get_data()))
                            elif filter_type == '/FlateDecode':
                                # PNG-like format
                                img = Image.frombytes(
                                    "RGB",
                                    (base_image.get('/Width'), base_image.get('/Height')),
                                    base_image.get_data()
                                )
                            elif filter_type == '/JPXDecode':
                                # JPEG2000 format (rare)
                                img = Image.open(io.BytesIO(base_image.get_data()))
                            elif filter_type == '/LZWDecode':
                                # LZWDecode compression (used for certain images)
                                img = Image.open(io.BytesIO(base_image.get_data()))
                            else:
                                print(f"Encountered unsupported image filter: {filter_type}")
                                raise RuntimeError(f"Unsupported image format or filter: {filter_type}")

                            # Check image size and convert to tensor
                            if img.width >= min_width and img.height >= min_height:
                                img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
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
