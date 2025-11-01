import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from comfy.utils import ProgressBar
import folder_paths


class FL_FILM:
    """
    FILM (Frame Interpolation for Large Motion) frame interpolation node.
    Generates intermediate frames between input frames, especially good for large motion.
    Downloads model to cache folder on first use.
    """

    MODEL_CONFIG = {
        "film_net": {
            "url": "https://huggingface.co/lividtm/RIFE/resolve/main/film_net_fp32.pt",
            "file": "film_net_fp32.pt"
        }
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "multiplier": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of frames to generate between each pair (2 = 2x frames)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "interpolate_frames"
    CATEGORY = "🏵️Fill Nodes/Video"

    def __init__(self):
        self.cache_dir = Path(__file__).parent.parent / "cache" / "film_models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

        # Device selection: CUDA or CPU only
        # MPS not supported - TorchScript model uses grid_sample with border padding
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def download_model(self):
        """Download FILM model weights to cache from HuggingFace"""
        import requests

        config = self.MODEL_CONFIG["film_net"]
        model_path = self.cache_dir / config["file"]

        # Check if already downloaded
        if model_path.exists():
            print(f"✓ FILM model already downloaded")
            return model_path

        print(f"📥 Downloading FILM model from HuggingFace...")
        print(f"   URL: {config['url']}")

        try:
            response = requests.get(config['url'], stream=True, timeout=120)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    downloaded += len(chunk)
                    f.write(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"   Progress: {progress:.1f}%", end='\r')

            print(f"\n✅ FILM model downloaded successfully to cache!")
            print(f"   Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
            return model_path

        except Exception as e:
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(f"Failed to download FILM model: {e}")

    def load_model(self):
        """Load FILM model"""
        if self.model is not None:
            return self.model

        model_path = self.download_model()

        print(f"🔄 Loading FILM model on {self.device}...")

        try:
            # Load TorchScript model directly
            self.model = torch.jit.load(str(model_path), map_location=self.device)
            self.model.eval()

            print(f"✅ FILM model loaded successfully!")
            return self.model

        except Exception as e:
            self.model = None
            raise RuntimeError(f"Failed to load FILM model: {e}")

    def make_inference(self, model, img0, img1, timestep):
        """Run FILM inference"""
        with torch.no_grad():
            # FILM expects timestep as a batch tensor
            batch_size = img0.shape[0]
            batch_dt = torch.full((batch_size, 1), timestep, dtype=img0.dtype, device=self.device)

            # Run FILM inference
            output = model(img0, img1, batch_dt)

        return output

    def interpolate_frames(self, images, multiplier):
        """Interpolate frames using FILM"""

        batch_size = images.shape[0]

        if batch_size < 2:
            print("⚠️  Warning: Need at least 2 frames for interpolation, returning input")
            return (images,)

        # Load model
        try:
            model = self.load_model()
        except Exception as e:
            print(f"❌ Error loading FILM model: {e}")
            print("   Returning input images without interpolation")
            return (images,)

        print(f"🎬 Interpolating {batch_size} frames with {multiplier}x multiplier...")
        print(f"   Using FILM (Frame Interpolation for Large Motion)")

        output_frames = []
        pbar = ProgressBar(batch_size - 1)

        # Process each pair of frames
        for i in range(batch_size - 1):
            # Get frame pair
            frame0 = images[i:i+1]  # [1, H, W, C]
            frame1 = images[i+1:i+2]  # [1, H, W, C]

            # Add first frame
            output_frames.append(frame0)

            # Convert to FILM format (B, C, H, W)
            img0 = frame0.permute(0, 3, 1, 2).to(self.device)
            img1 = frame1.permute(0, 3, 1, 2).to(self.device)

            # Generate intermediate frames
            for j in range(1, multiplier):
                timestep = j / multiplier

                try:
                    # Run FILM inference
                    pred = self.make_inference(model, img0, img1, timestep)

                    # Convert back to ComfyUI format (B, H, W, C)
                    pred = pred.permute(0, 2, 3, 1).cpu()
                    pred = torch.clamp(pred, 0, 1)

                    output_frames.append(pred)

                except Exception as e:
                    print(f"❌ Error during interpolation at frame {i}, step {j}: {e}")
                    # Fallback to linear interpolation
                    pred = frame0 * (1 - timestep) + frame1 * timestep
                    output_frames.append(pred)

            pbar.update(1)

        # Add last frame
        output_frames.append(images[-1:])

        # Concatenate all frames
        result = torch.cat(output_frames, dim=0)

        print(f"✅ Interpolation complete: {batch_size} → {result.shape[0]} frames")
        return (result,)
