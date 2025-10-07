import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Spacer

class FL_TextToPDF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "page_size": (["A4", "Letter"], {"default": "A4"}),
                "font_size": ("INT", {"default": 12, "min": 6, "max": 36}),
                "margin": ("INT", {"default": 72, "min": 36, "max": 144, "step": 6}),
                "title": ("STRING", {"default": "Generated Document"}),
            },
        }

    RETURN_TYPES = ("PDF",)
    FUNCTION = "create_pdf_from_text"
    CATEGORY = "🏵️Fill Nodes/PDF"

    def create_pdf_from_text(self, text, page_size, font_size, margin, title):
        try:
            # Create a BytesIO object to store the PDF
            pdf_buffer = io.BytesIO()

            # Set up the page size
            page_format = A4 if page_size == "A4" else letter
            width, height = page_format

            # Use direct canvas drawing instead of Platypus for better handling of large text
            c = canvas.Canvas(pdf_buffer, pagesize=page_format)

            # Set fonts
            title_font = "Helvetica-Bold"
            body_font = "Helvetica"

            # Add title if provided
            y_position = height - margin
            if title:
                c.setFont(title_font, font_size + 4)
                c.drawString(margin, y_position, title)
                y_position -= (font_size + 4) * 1.5  # Space after title

            # Set font for main text
            c.setFont(body_font, font_size)

            # Calculate line height based on font size
            line_height = font_size * 1.2

            # Split the text into lines
            import re

            # Function to wrap text to fit within margins
            def wrap_text(text, width, font_name, font_size):
                from reportlab.pdfbase.pdfmetrics import stringWidth
                words = text.split()
                lines = []
                line = []
                line_width = 0

                for word in words:
                    word_width = stringWidth(word + ' ', font_name, font_size)
                    if line_width + word_width <= width:
                        line.append(word)
                        line_width += word_width
                    else:
                        if line:
                            lines.append(' '.join(line))
                        line = [word]
                        line_width = word_width

                if line:
                    lines.append(' '.join(line))

                return lines

            # Process paragraphs (treat each double newline as paragraph break)
            paragraphs = text.split('\n\n')
            available_width = width - 2 * margin

            for paragraph in paragraphs:
                if not paragraph.strip():
                    y_position -= line_height  # Empty paragraph = extra space
                    continue

                # Convert single newlines to spaces within each paragraph
                paragraph = paragraph.replace('\n', ' ')

                # Convert any HTML tags to plain text (replace < with [ and > with ])
                paragraph = paragraph.replace('<', '[').replace('>', ']')

                # Wrap the paragraph text to fit the page width
                lines = wrap_text(paragraph, available_width, body_font, font_size)

                for line in lines:
                    # Check if we need a new page
                    if y_position < margin + line_height:
                        c.showPage()
                        c.setFont(body_font, font_size)
                        y_position = height - margin

                    # Draw the line of text
                    c.drawString(margin, y_position, line)
                    y_position -= line_height

                # Add space after paragraph
                y_position -= line_height * 0.5

            # Save the PDF
            c.save()

            # Get the PDF content
            pdf_content = pdf_buffer.getvalue()

            # Create a dictionary to store PDF information
            pdf_data = {
                'content': pdf_content,
                'num_pages': self._count_pages(pdf_content),
                'path': 'memory'
            }

            return (pdf_data,)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            raise RuntimeError(f"Error creating PDF from text: {str(e)}\n{error_details}")
    
    def _count_pages(self, pdf_content):
        """Count the number of pages in the PDF content."""
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(pdf_content))
        return len(reader.pages)

    @classmethod
    def IS_CHANGED(cls, text, page_size, font_size, margin, title):
        # This method should return a unique value every time the node's output would be different
        return hash((text, page_size, font_size, margin, title))