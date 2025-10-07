import sys
import subprocess
import importlib.util
import os
import time
import threading

# Check if huggingface_hub is installed, if not, install it
if importlib.util.find_spec("huggingface_hub") is None:
    print("huggingface_hub is not installed. Installing it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    print("huggingface_hub has been installed.")

import torch
from PIL import Image
import io
from huggingface_hub import HfApi, create_repo, repo_exists
from tqdm import tqdm


class FL_HFHubModelUploader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "owner": ("STRING", {"default": ""}),
                "repo_name": ("STRING", {"default": "my-awesome-model"}),
                "readme_content": (
                "STRING", {"multiline": True, "default": "# My Awesome Model\n\nThis is a great model!"}),
                "create_new_repo": (["True", "False"],),
                "image_folder_path": ("STRING", {"default": "images"}),
                "repo_type": (["model", "dataset", "space"],),
            },
            "optional": {
                "image": ("IMAGE",),
                "model_card_header": ("IMAGE",),
                "zip_file": ("ZIP",),
                "zip_filename": ("STRING", {"default": "archive"}),
                "zip_folder_path": ("STRING", {"default": "zipped_content"}),
                "model_file_path": ("STRING", {"default": ""}),
                "model_repo_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "upload_to_hub"
    CATEGORY = "🏵️Fill Nodes/Hugging Face"

    def upload_to_hub(self, api_key: str, owner: str, repo_name: str, readme_content: str, create_new_repo: str,
                      image_folder_path: str, repo_type: str, image: torch.Tensor = None,
                      model_card_header: torch.Tensor = None, zip_file: bytes = None,
                      zip_filename: str = "archive", zip_folder_path: str = "zipped_content",
                      model_file_path: str = "", model_repo_path: str = "") -> tuple[str]:
        # Initialize Hugging Face API
        api = HfApi(token=api_key)

        # Ensure zip_filename ends with .zip
        if not zip_filename.lower().endswith('.zip'):
            zip_filename += '.zip'

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
                    return (
                    f"Error: Repository {full_repo_id} does not exist. Please create it first or use the 'Create New Repo' option.",)
                repo_url = f"https://huggingface.co/{full_repo_id}"
                print(f"Using existing repository: {repo_url}")

            # Step 2: Prepare and upload files
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Upload the main image if provided
                    if image is not None:
                        main_image = Image.fromarray((image.squeeze().cpu().numpy() * 255).astype('uint8'))
                        main_img_byte_arr = io.BytesIO()
                        main_image.save(main_img_byte_arr, format='PNG')
                        main_img_byte_arr = main_img_byte_arr.getvalue()

                        api.upload_file(
                            path_or_fileobj=main_img_byte_arr,
                            path_in_repo=f"{image_folder_path}/model_image.png",
                            repo_id=full_repo_id,
                            token=api_key
                        )
                        print("Main image uploaded successfully")

                    # Upload the model card header image if provided
                    if model_card_header is not None:
                        header_image = Image.fromarray(
                            (model_card_header.squeeze().cpu().numpy() * 255).astype('uint8'))
                        header_img_byte_arr = io.BytesIO()
                        header_image.save(header_img_byte_arr, format='PNG')
                        header_img_byte_arr = header_img_byte_arr.getvalue()

                        api.upload_file(
                            path_or_fileobj=header_img_byte_arr,
                            path_in_repo="model_card_header.png",
                            repo_id=full_repo_id,
                            token=api_key
                        )
                        print("Model card header image uploaded successfully")
                        # Add the header image to the README content
                        readme_content = f"![Model Card Header](model_card_header.png)\n\n{readme_content}"

                    # Upload ZIP file if provided
                    if zip_file is not None:
                        api.upload_file(
                            path_or_fileobj=zip_file,
                            path_in_repo=f"{zip_folder_path}/{zip_filename}",
                            repo_id=full_repo_id,
                            token=api_key
                        )
                        print(f"ZIP file uploaded successfully as {zip_filename}")

                    # Upload model file from absolute path if provided
                    if model_file_path and model_repo_path:
                        if os.path.exists(model_file_path):
                            file_size = os.path.getsize(model_file_path)

                            # Create a progress bar
                            pbar = tqdm(total=100, unit='%', desc="Uploading model file")

                            # Function to update progress bar
                            def update_progress():
                                progress = 0
                                while progress < 95:
                                    time.sleep(0.5)
                                    increment = min(5, 95 - progress)
                                    progress += increment
                                    pbar.update(increment)

                            # Start progress update in a separate thread
                            progress_thread = threading.Thread(target=update_progress)
                            progress_thread.start()

                            # Perform the actual upload
                            with open(model_file_path, 'rb') as file:
                                api.upload_file(
                                    path_or_fileobj=file,
                                    path_in_repo=model_repo_path,
                                    repo_id=full_repo_id,
                                    token=api_key
                                )

                            # Ensure progress reaches 100%
                            progress_thread.join()
                            pbar.update(100 - pbar.n)
                            pbar.close()

                            print(f"Model file uploaded successfully to {model_repo_path}")
                        else:
                            print(f"Error: Model file not found at {model_file_path}")

                    # Upload README
                    api.upload_file(
                        path_or_fileobj=readme_content.encode('utf-8'),
                        path_in_repo="README.md",
                        repo_id=full_repo_id,
                        token=api_key
                    )
                    print("README uploaded successfully")

                    break  # If successful, break out of the retry loop
                except Exception as e:
                    if "Repository Not Found" in str(e) and attempt < max_retries - 1:
                        print(f"Repository not found. Retrying in 5 seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(5)
                    else:
                        raise

            return (f"Successfully uploaded to {repo_url}",)

        except Exception as e:
            return (f"Error: {str(e)}",)

    @classmethod
    def IS_CHANGED(cls, api_key, owner, repo_name, readme_content, create_new_repo, image_folder_path, repo_type,
                   image, model_card_header, zip_file, zip_filename, zip_folder_path, model_file_path, model_repo_path):
        return float("NaN")