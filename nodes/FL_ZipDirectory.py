import os
import zipfile
import tempfile


class FL_ZipDirectory:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": ""}),
                "zip_filename": ("STRING", {"default": "archive.zip"}),
            },
        }

    RETURN_TYPES = ("ZIP",)
    FUNCTION = "zip_directory"
    CATEGORY = "🏵️Fill Nodes/File Operations"

    def zip_directory(self, directory_path: str, zip_filename: str) -> tuple[str]:
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")

        # Create a temporary directory to store the zip file
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, zip_filename)

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(directory_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, directory_path)
                        zipf.write(file_path, arcname)

            # Read the zip file into memory
            with open(zip_path, 'rb') as f:
                zip_data = f.read()

        return (zip_data,)

    @classmethod
    def IS_CHANGED(cls, directory_path, zip_filename):
        return float("NaN")