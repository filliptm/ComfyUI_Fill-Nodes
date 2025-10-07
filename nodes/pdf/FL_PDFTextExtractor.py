import io
from PyPDF2 import PdfReader

class FL_PDFTextExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf": ("PDF",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "extract_text"
    CATEGORY = "🏵️Fill Nodes/PDF"

    def extract_text(self, pdf):
        try:
            # Create a PdfReader object
            pdf_reader = PdfReader(io.BytesIO(pdf['content']))

            # Initialize an empty string to store all the text
            all_text = ""

            # Iterate through all pages and extract text
            for page in pdf_reader.pages:
                all_text += page.extract_text() + "\n\n"  # Add newlines between pages

            # Trim any leading or trailing whitespace
            all_text = all_text.strip()

            return (all_text,)

        except Exception as e:
            raise RuntimeError(f"Error extracting text from PDF: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, pdf):
        # This method should return a unique value every time the node's output would be different
        return hash(pdf['content'])