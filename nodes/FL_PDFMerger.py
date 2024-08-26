import io
from PyPDF2 import PdfMerger, PdfReader

class FL_PDFMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf1": ("PDF",),
                "pdf2": ("PDF",),
            },
        }

    RETURN_TYPES = ("PDF",)
    FUNCTION = "merge_pdfs"
    CATEGORY = "🏵️Fill Nodes/PDF"

    def merge_pdfs(self, pdf1, pdf2):
        try:
            # Create a PdfMerger object
            merger = PdfMerger()

            # Add the first PDF
            pdf1_reader = PdfReader(io.BytesIO(pdf1['content']))
            merger.append(pdf1_reader)

            # Add the second PDF
            pdf2_reader = PdfReader(io.BytesIO(pdf2['content']))
            merger.append(pdf2_reader)

            # Create a BytesIO object to store the merged PDF
            merged_pdf = io.BytesIO()

            # Write the merged PDF to the BytesIO object
            merger.write(merged_pdf)
            merger.close()

            # Get the content of the merged PDF
            merged_content = merged_pdf.getvalue()

            # Create a dictionary to store the merged PDF information
            merged_pdf_data = {
                'content': merged_content,
                'num_pages': len(pdf1_reader.pages) + len(pdf2_reader.pages),
                'path': 'memory'  # Since it's not saved to disk yet
            }

            return (merged_pdf_data,)

        except Exception as e:
            raise RuntimeError(f"Error merging PDFs: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, pdf1, pdf2):
        # This method should return a unique value every time the node's output would be different
        return (hash(pdf1['content']) + hash(pdf2['content']))