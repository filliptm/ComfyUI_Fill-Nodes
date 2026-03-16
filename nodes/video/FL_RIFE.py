import torch
from pathlib import Path
from comfy.utils import ProgressBar
import folder_paths


class FL_RIFE:
    """
    RIFE (Real-Time Intermediate Flow Estimation) frame interpolation node.
    Generates intermediate frames between input frames for smooth slow-motion effects.
    Loads checkpoint weights from the ComfyUI models directory.
    """

    # Model version to architecture version mapping
    # Checkpoints are resolved from the ComfyUI models directory.
    CKPT_CONFIGS = {
        "rife47": {"arch": "4.7", "sub_dir": "ComfyUI_Fill-Nodes/rife", "file": "rife47.pth"},
        "rife49": {"arch": "4.7", "sub_dir": "ComfyUI_Fill-Nodes/rife", "file": "rife49.pth"},
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
        self.model = None
        self.current_ckpt = None

        # Device selection: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def get_model_path(self, ckpt_name):
        """Resolve a RIFE checkpoint from the ComfyUI models directory."""
        config = self.CKPT_CONFIGS[ckpt_name]
        return Path(folder_paths.models_dir) / config["sub_dir"] / config["file"]

    def load_model(self, ckpt_name):
        """Load RIFE model"""
        if self.model is not None and self.current_ckpt == ckpt_name:
            return self.model

        config = self.CKPT_CONFIGS[ckpt_name]
        model_path = self.get_model_path(ckpt_name)

        if not model_path.exists():
            raise RuntimeError(
                f"RIFE model {ckpt_name} not found. "
                f"Expected model at: {model_path}. "
                "Please ensure comfyagent downloaded the checkpoint to "
                "ComfyUI_Fill-Nodes/rife before running this node."
            )

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
