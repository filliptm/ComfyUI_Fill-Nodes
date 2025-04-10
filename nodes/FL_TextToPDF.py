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
            
            # Create a SimpleDocTemplate for more complex document formatting
            doc = SimpleDocTemplate(
                pdf_buffer,
                pagesize=page_format,
                leftMargin=margin,
                rightMargin=margin,
                topMargin=margin,
                bottomMargin=margin,
                title=title
            )
            
            # Get sample stylesheet and set up styles
            styles = getSampleStyleSheet()
            title_style = styles['Heading1']
            normal_style = styles['Normal']
            normal_style.fontSize = font_size
            
            # Create the story (content) with title and text
            story = []
            
            # Add title
            if title:
                story.append(Paragraph(title, title_style))
                story.append(Spacer(1, 0.25 * inch))
            
            # Process text paragraphs
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    story.append(Paragraph(para.replace('\n', '<br/>'), normal_style))
                    story.append(Spacer(1, 0.1 * inch))
            
            # Build the document
            doc.build(story)
            
            # Get the PDF content
            pdf_content = pdf_buffer.getvalue()
            
            # Create a dictionary to store PDF information (same format as other nodes)
            pdf_data = {
                'content': pdf_content,
                'num_pages': self._count_pages(pdf_content),
                'path': 'memory'  # Since it's not saved to disk yet
            }
            
            return (pdf_data,)
            
        except Exception as e:
            raise RuntimeError(f"Error creating PDF from text: {str(e)}")
    
    def _count_pages(self, pdf_content):
        """Count the number of pages in the PDF content."""
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(pdf_content))
        return len(reader.pages)

    @classmethod
    def IS_CHANGED(cls, text, page_size, font_size, margin, title):
        # This method should return a unique value every time the node's output would be different
        return hash((text, page_size, font_size, margin, title))