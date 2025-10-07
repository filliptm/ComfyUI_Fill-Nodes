import sys
import subprocess
import importlib.util
import os
import time
import threading
import io

# Check if huggingface_hub is installed, if not, install it
if importlib.util.find_spec("huggingface_hub") is None:
    print("huggingface_hub is not installed. Installing it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    print("huggingface_hub has been installed.")

import torch
from PIL import Image
from huggingface_hub import HfApi, create_repo, repo_exists
from tqdm import tqdm

class FL_HF_Character:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "owner": ("STRING", {"default": ""}),
                "repo_name": ("STRING", {"default": "my-awesome-model"}),
                "studio_name": ("STRING", {"default": ""}),
                "project_name": ("STRING", {"default": ""}),
                "character_name": ("STRING", {"default": ""}),
                "create_new_repo": (["True", "False"],),
                "repo_type": (["model", "dataset", "space"],),
            },
            "optional": {
                "lora_file": ("STRING", {"default": ""}),
                "dataset_zip": ("ZIP",),
                "caption_layout": ("IMAGE",),
                "caption_PDF_layout": ("PDF",),
                "csv_file": ("CSV",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "upload_to_hub"
    CATEGORY = "🏵️Fill Nodes/Hugging Face"

    def upload_to_hub(self, api_key: str, owner: str, repo_name: str, studio_name: str, project_name: str,
                      character_name: str, create_new_repo: str, repo_type: str,
                      lora_file: str = "", dataset_zip: bytes = None,
                      caption_layout: torch.Tensor = None, caption_PDF_layout: bytes = None,
                      csv_file: bytes = None) -> tuple[str]:
        # Initialize Hugging Face API
        api = HfApi(token=api_key)

        try:
            # Construct full repo_id
            full_repo_id = f"{owner}/{repo_name}"

            # Step 1: Create a new repository or check if it exists
            create_new_repo = create_new_repo == "True"
            if create_new_repo:
                repo_url = create_repo(repo_id=full_repo_id, token=api_key, exist_ok=True, repo_type=repo_type)
                print(f"Repository created or already exists: {repo_url}")
            else:
                if not repo_exists(repo_id=full_repo_id, token=api_key):
                    return (f"Error: Repository {full_repo_id} does not exist. Please create it first or use the 'Create New Repo' option.",)
                repo_url = f"https://huggingface.co/{full_repo_id}"
                print(f"Using existing repository: {repo_url}")

            # Step 2: Create directory structure
            base_path = f"{studio_name}/{project_name}/{character_name}"

            # Step 3: Upload files
            if lora_file:
                self.upload_file_with_progress(api, lora_file, f"{base_path}/lora", full_repo_id, api_key, "LoRA")
            if dataset_zip is not None:
                self.upload_zip(api, dataset_zip, f"{base_path}/dataset", full_repo_id, api_key, "Dataset")
            if caption_layout is not None:
                self.upload_image(api, caption_layout, base_path, full_repo_id, api_key, "caption_layout")
            if caption_PDF_layout is not None:
                self.upload_pdf(api, caption_PDF_layout, base_path, full_repo_id, api_key, "caption_PDF_layout")
            if csv_file is not None:
                self.upload_csv(api, csv_file, base_path, full_repo_id, api_key)

            return (f"Successfully uploaded to {repo_url}/{base_path}",)

        except Exception as e:
            return (f"Error: {str(e)}",)

    def upload_file_with_progress(self, api, file_path, repo_dir, full_repo_id, api_key, file_type):
        if file_path and os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            repo_path = f"{repo_dir}/{file_name}"

            pbar = tqdm(total=100, unit='%', desc=f"Uploading {file_type} file")

            def update_progress():
                progress = 0
                while progress < 95:
                    time.sleep(0.5)
                    increment = min(5, 95 - progress)
                    progress += increment
                    pbar.update(increment)

            progress_thread = threading.Thread(target=update_progress)
            progress_thread.start()

            with open(file_path, 'rb') as file:
                api.upload_file(
                    path_or_fileobj=file,
                    path_in_repo=repo_path,
                    repo_id=full_repo_id,
                    token=api_key
                )

            progress_thread.join()
            pbar.update(100 - pbar.n)
            pbar.close()

            print(f"{file_type} file uploaded successfully to {repo_path}")
        elif file_path:
            print(f"Error: {file_type} file not found at {file_path}")

    def upload_zip(self, api, zip_data, repo_dir, full_repo_id, api_key, file_type):
        repo_path = f"{repo_dir}/dataset.zip"

        pbar = tqdm(total=100, unit='%', desc=f"Uploading {file_type} ZIP")

        def update_progress():
            progress = 0
            while progress < 95:
                time.sleep(0.5)
                increment = min(5, 95 - progress)
                progress += increment
                pbar.update(increment)

        progress_thread = threading.Thread(target=update_progress)
        progress_thread.start()

        api.upload_file(
            path_or_fileobj=zip_data,
            path_in_repo=repo_path,
            repo_id=full_repo_id,
            token=api_key
        )

        progress_thread.join()
        pbar.update(100 - pbar.n)
        pbar.close()

        print(f"{file_type} ZIP uploaded successfully to {repo_path}")

    def upload_image(self, api, image, repo_dir, full_repo_id, api_key, image_type):
        img = Image.fromarray((image.squeeze().cpu().numpy() * 255).astype('uint8'))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        repo_path = f"{repo_dir}/{image_type}.png"
        api.upload_file(
            path_or_fileobj=img_byte_arr,
            path_in_repo=repo_path,
            repo_id=full_repo_id,
            token=api_key
        )
        print(f"{image_type} uploaded successfully")

    def upload_pdf(self, api, pdf_data, repo_dir, full_repo_id, api_key, pdf_type):
        repo_path = f"{repo_dir}/{pdf_type}.pdf"

        pbar = tqdm(total=100, unit='%', desc=f"Uploading {pdf_type} PDF")

        def update_progress():
            progress = 0
            while progress < 95:
                time.sleep(0.5)
                increment = min(5, 95 - progress)
                progress += increment
                pbar.update(increment)

        progress_thread = threading.Thread(target=update_progress)
        progress_thread.start()

        api.upload_file(
            path_or_fileobj=pdf_data,
            path_in_repo=repo_path,
            repo_id=full_repo_id,
            token=api_key
        )

        progress_thread.join()
        pbar.update(100 - pbar.n)
        pbar.close()

        print(f"{pdf_type} PDF uploaded successfully to {repo_path}")

    def upload_csv(self, api, csv_data, repo_dir, full_repo_id, api_key):
        repo_path = f"{repo_dir}/metadata.csv"

        pbar = tqdm(total=100, unit='%', desc="Uploading CSV file")

        def update_progress():
            progress = 0
            while progress < 95:
                time.sleep(0.5)
                increment = min(5, 95 - progress)
                progress += increment
                pbar.update(increment)

        progress_thread = threading.Thread(target=update_progress)
        progress_thread.start()

        api.upload_file(
            path_or_fileobj=csv_data,
            path_in_repo=repo_path,
            repo_id=full_repo_id,
            token=api_key
        )

        progress_thread.join()
        pbar.update(100 - pbar.n)
        pbar.close()

        print(f"CSV file uploaded successfully to {repo_path}")

    @classmethod
    def IS_CHANGED(cls, api_key, owner, repo_name, studio_name, project_name, character_name,
                   create_new_repo, repo_type, lora_file, dataset_zip, caption_layout, caption_PDF_layout, csv_file):
        return float("NaN")