import os
import re
import gdown
import zipfile
import shutil
from pathlib import Path


class FL_GoogleDriveDownloader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "google_drive_link": ("STRING", {"default": ""}),
                "output_folder_name": ("STRING", {"default": "gdrive_download"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "download_and_extract"
    CATEGORY = "🏵️Fill Nodes/Google Drive"
    OUTPUT_NODE=True

    def extract_file_id_from_link(self, share_link: str) -> str:
        """Extracts the file ID from a Google Drive share link."""
        match = re.search(r'(?:/d/|id=)([a-zA-Z0-9_-]+)', share_link)
        if match:
            return match.group(1)
        raise ValueError("Invalid Google Drive share link. Unable to extract file ID.")

    def ensure_output_directory(self, base_path: str, folder_name: str) -> str:
        """Creates and returns the path to the output directory."""
        output_dir = os.path.join(base_path, 'output')
        gdrive_dir = os.path.join(output_dir, 'google_drive_downloads')
        if not os.path.exists(gdrive_dir):
            os.makedirs(gdrive_dir)

        download_dir = os.path.join(gdrive_dir, folder_name)
        if os.path.exists(download_dir):
            shutil.rmtree(download_dir)

        os.makedirs(download_dir)
        return download_dir

    def get_filename_from_cd(self, cd):
        """Get filename from content-disposition."""
        if not cd:
            return None
        fname = re.findall('filename="(.+)"', cd)
        if len(fname) == 0:
            return None
        return fname[0]

    def process_downloaded_file(self, file_path: str, output_dir: str) -> None:
        """Process the downloaded file based on its type."""
        try:
            if zipfile.is_zipfile(file_path):
                print(f"Processing ZIP file: {os.path.basename(file_path)}")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    zip_ref.extractall(output_dir)
                print(f"Extracted files: {', '.join(file_list)}")
                os.remove(file_path)  # Remove zip file after extraction
            else:
                print(f"Keeping file as is: {os.path.basename(file_path)}")
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise ValueError(f"Error processing file: {str(e)}")

    def download_and_extract(self, google_drive_link: str, output_folder_name: str) -> tuple[str]:
        try:
            base_path = os.getcwd()
            output_dir = self.ensure_output_directory(base_path, output_folder_name)
            file_id = self.extract_file_id_from_link(google_drive_link)

            # First, get the file metadata to get the real filename
            url = f'https://drive.google.com/uc?id={file_id}'

            # Use gdown's download_folder functionality
            print(f"Downloading from Google Drive to {output_dir}...")

            # Download with specific naming
            output = gdown.download(
                url=url,
                output=None,  # Let gdown determine the filename
                quiet=False,
                fuzzy=True
            )

            if not output:
                raise ValueError("Download failed")

            # Move the file to our desired location with the correct name
            original_filename = os.path.basename(output)
            new_file_path = os.path.join(output_dir, original_filename)

            # If the downloaded file exists in a different location, move it
            if os.path.exists(output) and output != new_file_path:
                shutil.move(output, new_file_path)

            if not os.path.exists(new_file_path):
                raise ValueError("Download failed - file not created")

            print(f"Downloaded file: {original_filename}")

            # Process the downloaded file
            self.process_downloaded_file(new_file_path, output_dir)

            print(f"Files available in: {output_dir}")
            return (output_dir,)

        except Exception as e:
            raise ValueError(f"Error processing Google Drive file: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, google_drive_link, output_folder_name):
        return float("NaN")