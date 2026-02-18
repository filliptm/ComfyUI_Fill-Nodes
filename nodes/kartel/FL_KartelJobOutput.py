import io
import json
import numpy as np
from PIL import Image
import requests


class FL_KartelJobOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "job_id": ("STRING", {"default": ""}),
                "callback_url": ("STRING", {"default": ""}),
            },
            "optional": {
                "user_id": ("STRING", {"default": ""}),
                "user_email": ("STRING", {"default": ""}),
                "app_name": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_results"
    CATEGORY = "🏵️Fill Nodes/Kartel"
    OUTPUT_NODE = True

    def send_results(self, images, job_id, callback_url,
                     user_id="", user_email="", app_name=""):
        if images is None:
            print("[FL_KartelJobOutput] No images received (None), skipping POST.")
            return ()

        print(f"[FL_KartelJobOutput] Received images with shape: {images.shape}")

        if images.shape[1] <= 1 and images.shape[2] <= 1:
            print("[FL_KartelJobOutput] Images are 1x1 placeholder, skipping POST.")
            return ()

        if not callback_url:
            print("[FL_KartelJobOutput] No callback_url provided, skipping POST.")
            return ()

        # Convert each batch image to PNG bytes in-memory
        png_buffers = []
        for i in range(images.shape[0]):
            img_np = 255.0 * images[i].cpu().numpy()
            img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            buf.seek(0)
            png_buffers.append((f"image_{i}.png", buf))

        # Build metadata form field
        metadata = {
            "job_id": job_id,
            "user_id": user_id,
            "user_email": user_email,
            "app_name": app_name,
            "image_count": len(png_buffers),
            "status": "completed",
        }
        data = {"metadata": json.dumps(metadata)}

        # POST with 1 retry on connection error
        max_attempts = 2
        last_error = None

        for attempt in range(max_attempts):
            # Build multipart files list (rebuild each attempt since streams are consumed)
            files = [("images", (fn, buf, "image/png")) for fn, buf in png_buffers]

            try:
                response = requests.post(
                    callback_url,
                    files=files,
                    data=data,
                    timeout=30,
                )
                response.raise_for_status()
                print(
                    f"[FL_KartelJobOutput] Successfully posted {len(png_buffers)} image(s) "
                    f"for job {job_id} -> {response.status_code}"
                )
                return ()
            except requests.ConnectionError as e:
                last_error = e
                print(
                    f"[FL_KartelJobOutput] Connection error on attempt "
                    f"{attempt + 1}/{max_attempts}: {e}"
                )
                # Reset buffer positions for retry
                for _, buf in png_buffers:
                    buf.seek(0)
            except requests.RequestException as e:
                print(f"[FL_KartelJobOutput] POST failed for job {job_id}: {e}")
                return ()

        print(
            f"[FL_KartelJobOutput] All {max_attempts} attempts failed "
            f"for job {job_id}: {last_error}"
        )
        return ()

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")
