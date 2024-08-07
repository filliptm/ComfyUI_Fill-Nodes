import os
import zipfile
import tempfile

class FL_ZipSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_directory": ("STRING", {"default": ""}),
                "output_directory": ("STRING", {"default": ""}),
                "zip_filename": ("STRING", {"default": "archive.zip"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("zip_path",)
    FUNCTION = "zip_and_save"
    CATEGORY = "🏵️Fill Nodes/File Operations"
    OUTPUT_NODE = True

    def zip_and_save(self, input_directory: str, output_directory: str, zip_filename: str) -> tuple[str]:
        if not os.path.exists(input_directory):
            raise ValueError(f"Input directory not found: {input_directory}")

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Ensure the zip filename ends with .zip
        if not zip_filename.lower().endswith('.zip'):
            zip_filename += '.zip'

        zip_path = os.path.join(output_directory, zip_filename)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(input_directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, input_directory)
                    zipf.write(file_path, arcname)

        print(f"Zip file created: {zip_path}")
        return (zip_path,)

    @classmethod
    def IS_CHANGED(cls, input_directory, output_directory, zip_filename):
        return float("NaN")