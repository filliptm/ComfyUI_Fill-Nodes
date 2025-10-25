import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from comfy.utils import ProgressBar
import folder_paths


class FL_RIFE:
    """
    RIFE (Real-Time Intermediate Flow Estimation) frame interpolation node.
    Generates intermediate frames between input frames for smooth slow-motion effects.
    Downloads models to cache folder on first use.
    """

    # Model version to architecture version mapping
    # Using direct .pth downloads from HuggingFace (faster, no extraction needed)
    CKPT_CONFIGS = {
        "rife47": {"arch": "4.7", "url": "https://huggingface.co/lividtm/RIFE/resolve/main/rife47.pth", "file": "rife47.pth"},
        "rife49": {"arch": "4.7", "url": "https://huggingface.co/lividtm/RIFE/resolve/main/rife49.pth", "file": "rife49.pth"},
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "ckpt_name": (list(cls.CKPT_CONFIGS.keys()), {"default": "rife47"}),
                "multiplier": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of frames to generate between each pair (2 = 2x frames)"
                }),
                "ensemble": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use ensemble for better quality (slower, runs model twice and averages results)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "interpolate_frames"
    CATEGORY = "🏵️Fill Nodes/Video"

    def __init__(self):
        self.cache_dir = Path(__file__).parent.parent / "cache" / "rife_models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.current_ckpt = None

        # Device selection: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def download_model(self, ckpt_name):
        """Download RIFE model weights to cache from HuggingFace"""
        import requests

        config = self.CKPT_CONFIGS[ckpt_name]
        model_path = self.cache_dir / config["file"]

        # Check if already downloaded
        if model_path.exists():
            print(f"✓ RIFE model {ckpt_name} already downloaded")
            return model_path

        print(f"📥 Downloading RIFE model {ckpt_name} from HuggingFace...")
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

            print(f"\n✅ RIFE model {ckpt_name} downloaded successfully to cache!")
            print(f"   Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
            return model_path

        except Exception as e:
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(f"Failed to download RIFE model: {e}")

    def load_model(self, ckpt_name):
        """Load RIFE model"""
        if self.model is not None and self.current_ckpt == ckpt_name:
            return self.model

        config = self.CKPT_CONFIGS[ckpt_name]
        model_path = self.download_model(ckpt_name)

        print(f"🔄 Loading RIFE model {ckpt_name} (arch {config['arch']}) on {self.device}...")

        try:
            # Lazy import to avoid loading architecture at module import time
            from ..rife_arch import IFNet

            # Initialize model with correct architecture
            self.model = IFNet(arch_ver=config["arch"])

            # Load state dict
            state_dict = torch.load(str(model_path), map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)

            self.current_ckpt = ckpt_name
            print(f"✅ Model loaded successfully!")
            return self.model

        except Exception as e:
            self.model = None
            raise RuntimeError(f"Failed to load RIFE model: {e}")

    def make_inference(self, model, img0, img1, timestep, ensemble):
        """Run RIFE inference with hardcoded optimal settings for v4.7"""
        # Hardcoded scale_factor=1.0 for full quality
        scale_list = [8.0, 4.0, 2.0, 1.0]

        with torch.no_grad():
            output = model(
                img0,
                img1,
                timestep=timestep,
                scale_list=scale_list,
                training=False,
                fastmode=True,  # Hardcoded (no effect on v4.7)
                ensemble=ensemble
            )

        return output

    def interpolate_frames(self, images, ckpt_name, multiplier, ensemble):
        """Interpolate frames using RIFE"""

        batch_size = images.shape[0]

        if batch_size < 2:
            print("⚠️  Warning: Need at least 2 frames for interpolation, returning input")
            return (images,)

        # Load model
        try:
            model = self.load_model(ckpt_name)
        except Exception as e:
            print(f"❌ Error loading RIFE model: {e}")
            print("   Returning input images without interpolation")
            return (images,)

        print(f"🎬 Interpolating {batch_size} frames with {multiplier}x multiplier...")
        print(f"   Settings: ensemble={ensemble}, scale=1.0 (full quality)")

        output_frames = []
        pbar = ProgressBar(batch_size - 1)

        # Process each pair of frames
        for i in range(batch_size - 1):
            # Get frame pair
            frame0 = images[i:i+1]  # [1, H, W, C]
            frame1 = images[i+1:i+2]  # [1, H, W, C]

            # Add first frame
            output_frames.append(frame0)

            # Convert to RIFE format (B, C, H, W)
            img0 = frame0.permute(0, 3, 1, 2).to(self.device)
            img1 = frame1.permute(0, 3, 1, 2).to(self.device)

            # Generate intermediate frames
            for j in range(1, multiplier):
                timestep = j / multiplier

                try:
                    # Run RIFE inference
                    pred = self.make_inference(model, img0, img1, timestep, ensemble)

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
