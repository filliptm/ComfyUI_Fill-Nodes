import io
from PyPDF2 import PdfReader, PdfWriter

class FL_PDFEncryptor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf": ("PDF",),
                "user_password": ("STRING", {"default": "", "multiline": False}),
                "owner_password": ("STRING", {"default": "", "multiline": False}),
                "encryption_level": (["40-bit", "128-bit"], {"default": "128-bit"}),
            },
            "optional": {
                "allow_printing": ("BOOLEAN", {"default": True}),
                "allow_commenting": ("BOOLEAN", {"default": True}),
                "allow_copying": ("BOOLEAN", {"default": True}),
                "allow_content_extraction": ("BOOLEAN", {"default": True}),
                "allow_form_filling": ("BOOLEAN", {"default": True}),
                "allow_document_assembly": ("BOOLEAN", {"default": True}),
                "allow_page_extraction": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("PDF",)
    FUNCTION = "encrypt_pdf"
    CATEGORY = "🏵️Fill Nodes/PDF"

    def encrypt_pdf(self, pdf, user_password, owner_password, encryption_level,
                    allow_printing=True, allow_commenting=True, allow_copying=True,
                    allow_content_extraction=True, allow_form_filling=True,
                    allow_document_assembly=True, allow_page_extraction=True):
        try:
            # Read the input PDF
            reader = PdfReader(io.BytesIO(pdf['content']))
            
            # Create a PDF writer
            writer = PdfWriter()
            
            # Copy all pages from the reader to the writer
            for page in reader.pages:
                writer.add_page(page)
            
            # Set up permissions based on inputs
            # Using correct constant values for PyPDF2
            permissions = 0
            if allow_printing:
                # PyPDF2 permission for printing
                permissions |= 4  # Print permission
            if allow_commenting:
                # PyPDF2 permission for annotations
                permissions |= 8  # Annotation permission
            if allow_copying or allow_content_extraction:
                # PyPDF2 permission for content copying/extraction
                permissions |= 16  # Content extraction permission
            if allow_form_filling:
                # PyPDF2 permission for filling forms
                permissions |= 256  # Fill form permission
            if allow_document_assembly:
                # PyPDF2 permission for document assembly
                permissions |= 1024  # Assembly permission
            if allow_page_extraction:
                # Included in content extraction permission
                permissions |= 16
            
            # Determine encryption strength
            encryption_strength = 128 if encryption_level == "128-bit" else 40
            
            # Encrypt the PDF
            writer.encrypt(
                user_password=user_password if user_password else None,
                owner_password=owner_password if owner_password else (user_password if user_password else ""),
                use_128bit=encryption_strength == 128,
                permissions_flag=permissions
            )
            
            # Create a BytesIO object to hold the encrypted PDF
            output_pdf = io.BytesIO()
            
            # Write the encrypted PDF to the BytesIO object
            writer.write(output_pdf)
            output_pdf.seek(0)
            
            # Get the content of the encrypted PDF
            encrypted_content = output_pdf.getvalue()
            
            # Create a dictionary to store PDF information
            encrypted_pdf_data = {
                'content': encrypted_content,
                'num_pages': len(reader.pages),
                'path': 'memory'  # Since it's not saved to disk yet
            }
            
            return (encrypted_pdf_data,)
            
        except Exception as e:
            raise RuntimeError(f"Error encrypting PDF: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, pdf, user_password, owner_password, encryption_level, **kwargs):
        # This method should return a unique value when the node's output would be different
        permission_hash = 0
        for k, v in kwargs.items():
            if v:
                permission_hash |= hash(k)
        
        return (hash(pdf['content']), hash(user_password), hash(owner_password),
                hash(encryption_level), permission_hash)