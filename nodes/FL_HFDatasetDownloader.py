import os
from huggingface_hub import snapshot_download

class FL_HFDatasetDownloader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {
                    "default": "jjuik2014/FaceVid-1K-Part",
                    "multiline": False
                }),
                "repo_type": (["dataset", "model", "space"], {
                    "default": "dataset"
                }),
                "local_dir": ("STRING", {
                    "default": "./output/HF-Downloads",
                    "multiline": False
                }),
                "max_workers": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 20,
                    "step": 1
                }),
                "download_trigger": ("BOOLEAN", {
                    "default": False,
                    "label": "Start Download"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("download_path",)
    FUNCTION = "download_repo"
    CATEGORY = "🏵️Fill Nodes/Hugging Face"

    def download_repo(self, repo_id, repo_type, local_dir, max_workers, download_trigger):
        if not download_trigger:
            return (local_dir,)

        try:
            # Ensure the directory exists
            os.makedirs(local_dir, exist_ok=True)
            
            # Download the repository
            download_path = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=local_dir,
                max_workers=max_workers
            )
            
            return (download_path,)
            
        except Exception as e:
            print(f"Error downloading repository: {str(e)}")
            return (local_dir,)