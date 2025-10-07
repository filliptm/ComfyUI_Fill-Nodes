import os
from PyPDF2 import PdfReader
import torch
from io import BytesIO


class FL_BulkPDFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("PDF",)
    RETURN_NAMES = ("pdfs",)
    FUNCTION = "load_pdfs"
    CATEGORY = "🏵️Fill Nodes/PDF"
    OUTPUT_NODE = True

    def load_pdfs(self, directory_path):
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]

        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {directory_path}")

        print(f"Found {len(pdf_files)} PDF files in {directory_path}")

        loaded_pdfs = []
        errors = []

        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            try:
                print(f"Attempting to load: {pdf_path}")
                with open(pdf_path, 'rb') as file:
                    pdf_content = file.read()
                    pdf_reader = PdfReader(BytesIO(pdf_content))
                    pdf_data = {
                        'path': pdf_path,
                        'filename': pdf_file,
                        'num_pages': len(pdf_reader.pages),
                        'content': pdf_content
                    }
                    loaded_pdfs.append(pdf_data)
                    print(f"Successfully loaded: {pdf_file}")
            except Exception as e:
                error_msg = f"Error loading PDF {pdf_file}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)

        if not loaded_pdfs:
            error_summary = "\n".join(errors)
            raise RuntimeError(f"No PDFs were successfully loaded. Errors encountered:\n{error_summary}")

        print(f"Successfully loaded {len(loaded_pdfs)} out of {len(pdf_files)} PDFs")
        return (loaded_pdfs,)

    @classmethod
    def IS_CHANGED(cls, directory_path):
        if os.path.exists(directory_path):
            pdf_count = len([f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')])
            return (os.path.getmtime(directory_path), pdf_count)
        return float("nan")