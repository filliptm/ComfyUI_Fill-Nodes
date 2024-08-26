import os
import torch

class FL_PDFSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf": ("PDF",),
                "output_directory": ("STRING", {"default": ""}),
                "filename": ("STRING", {"default": "output"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_pdf"
    CATEGORY = "🏵️Fill Nodes/PDF"
    OUTPUT_NODE = True

    def save_pdf(self, pdf, output_directory, filename):
        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Ensure the filename ends with .pdf
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'

        # Construct the full output path
        output_path = os.path.join(output_directory, filename)

        try:
            # Write the PDF content to the file
            with open(output_path, 'wb') as file:
                file.write(pdf['content'])

            print(f"PDF saved successfully to: {output_path}")
            return (output_path,)

        except Exception as e:
            raise RuntimeError(f"Error saving PDF: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, pdf, output_directory, filename):
        # Ensure consistent handling of the filename extension
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        # This method should return a unique value every time the node's output would be different
        return (pdf['path'], output_directory, filename)