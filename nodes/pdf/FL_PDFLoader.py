import os
from PyPDF2 import PdfReader
import torch
from io import BytesIO

class FL_PDFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("PDF",)
    FUNCTION = "load_pdf"
    CATEGORY = "🏵️Fill Nodes/PDF"

    def load_pdf(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError("The specified file is not a PDF")

        try:
            with open(pdf_path, 'rb') as file:
                pdf_content = file.read()
                pdf_reader = PdfReader(BytesIO(pdf_content))
                pdf_data = {
                    'path': pdf_path,
                    'num_pages': len(pdf_reader.pages),
                    'content': pdf_content
                }
            return (pdf_data,)
        except Exception as e:
            raise RuntimeError(f"Error loading PDF: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, pdf_path):
        if os.path.exists(pdf_path):
            return os.path.getmtime(pdf_path)
        return float("nan")