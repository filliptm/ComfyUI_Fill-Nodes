import io
import zlib
import torch
from PIL import Image
import numpy as np
from PyPDF2 import PdfReader


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
            pdf_reader = PdfReader(io.BytesIO(pdf_content))
            images = []

            for page_num, page in enumerate(pdf_reader.pages):
                if '/XObject' in page['/Resources']:
                    xObject = page['/Resources']['/XObject'].get_object()
                    for obj in xObject:
                        if xObject[obj]['/Subtype'] == '/Image':
                            base_image = xObject[obj]
                            filter_type = base_image.get('/Filter')

                            # Handle combined filters like ASCII85Decode and FlateDecode
                            if filter_type == ['/ASCII85Decode', '/FlateDecode']:
                                # First decode ASCII85
                                decoded_data = base_image.get_data()  # PyPDF2 handles this

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

                            # Convert image to tensor
                            img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
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
