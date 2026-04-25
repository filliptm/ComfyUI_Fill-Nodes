# FL_Fal_GPTImage2_Edit: Fal AI GPT-Image-2 Edit endpoint
# Takes N reference images + a single prompt, returns 1-4 edited images.
import os
import io
import time
import asyncio
import concurrent.futures
from typing import List, Optional

import requests
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import fal_client
from comfy.utils import ProgressBar


IMAGE_SIZE_PRESETS = [
    "auto",
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
    "custom",
]
QUALITY_OPTIONS = ["low", "medium", "high"]
OUTPUT_FORMATS = ["png", "jpeg", "webp"]


class FL_Fal_GPTImage2_Edit:
    """
    ComfyUI node for fal.ai's `openai/gpt-image-2/edit` endpoint.

    Takes a variable number of reference images (image_ref_1, image_ref_2, ...),
    uploads them to fal.media, then edits/composes against a shared prompt.
    Dynamic sockets are driven by the `image_count` widget + companion JS file.
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "image_urls", "status_msg")
    FUNCTION = "generate"
    CATEGORY = "🏵️Fill Nodes/AI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "image_count": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "prompt": ("STRING", {"default": "A cozy scene with all these elements combined.",
                                      "multiline": True, "forceInput": False}),
                "image_size": (IMAGE_SIZE_PRESETS, {"default": "auto"}),
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "custom_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "quality": (QUALITY_OPTIONS, {"default": "high"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "output_format": (OUTPUT_FORMATS, {"default": "png"}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
                "image_ref_1": ("IMAGE",),
            },
            "optional": {
                "mask": ("IMAGE",),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "retry_indefinitely": ("BOOLEAN", {"default": False}),
            },
        }

    def __init__(self):
        self.log_messages: List[str] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, message: str) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[FL_Fal_GPTImage2_Edit] {timestamp}: {message}"
        print(formatted)
        self.log_messages.append(message)
        return message

    @staticmethod
    def _tensor_to_pil(tensor: Optional[torch.Tensor]) -> Optional[Image.Image]:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return None
        if tensor.ndim == 4:
            if tensor.shape[0] == 0:
                return None
            arr = tensor[0].cpu().numpy()
        elif tensor.ndim == 3:
            arr = tensor.cpu().numpy()
        else:
            return None
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _create_error_image(self, error_message: str = "API Error",
                            width: int = 1024, height: int = 1024) -> torch.Tensor:
        image = Image.new("RGB", (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        font = None
        for font_name in ("arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf", "NotoSans-Regular.ttf"):
            try:
                font = ImageFont.truetype(font_name, 24)
                break
            except IOError:
                continue
        if font is None:
            font = ImageFont.load_default()

        try:
            bbox = draw.textbbox((0, 0), error_message, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(error_message, font=font)
        draw.text(((width - tw) / 2, (height - th) / 2), error_message,
                  fill=(255, 0, 0), font=font)
        arr = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def _upload_pil(self, pil: Image.Image, label: str) -> Optional[str]:
        try:
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            url = fal_client.upload(buf.getvalue(), content_type="image/png")
            self._log(f"Uploaded {label} to CDN: {url[:80]}...")
            return url
        except Exception as e:
            self._log(f"Error uploading {label}: {e}")
            return None

    # ------------------------------------------------------------------
    # Async core
    # ------------------------------------------------------------------

    async def _generate_async(
        self, api_key, prompt, ref_pils, mask_pil,
        image_size, custom_width, custom_height,
        quality, num_images, output_format,
        max_retries, retry_indefinitely, sync_mode,
    ):
        os.environ["FAL_KEY"] = api_key.strip()

        image_urls = []
        for idx, pil in enumerate(ref_pils, start=1):
            url = self._upload_pil(pil, f"image_ref_{idx}")
            if url is None:
                msg = f"Upload failed for image_ref_{idx}"
                return self._create_error_image(msg), "", msg
            image_urls.append(url)

        mask_url = None
        if mask_pil is not None:
            mask_url = self._upload_pil(mask_pil, "mask")
            if mask_url is None:
                msg = "Upload failed for mask"
                return self._create_error_image(msg), "", msg

        resolved_image_size: object
        if image_size == "custom":
            resolved_image_size = {"width": int(custom_width), "height": int(custom_height)}
        else:
            resolved_image_size = image_size

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "image_size": resolved_image_size,
            "quality": quality,
            "num_images": num_images,
            "output_format": output_format,
            "sync_mode": sync_mode,
        }
        if mask_url is not None:
            arguments["mask_url"] = mask_url

        self._log(f"Calling openai/gpt-image-2/edit with {len(image_urls)} ref image(s), "
                  f"quality={quality}, size={resolved_image_size}, n={num_images}")

        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for entry in update.logs:
                    self._log(f"API Log: {entry.get('message', '')}")

        loop = asyncio.get_event_loop()
        attempt = 0
        result = None
        last_err = ""
        while True:
            attempt += 1

            def call():
                try:
                    return fal_client.subscribe(
                        "openai/gpt-image-2/edit",
                        arguments=arguments,
                        with_logs=True,
                        on_queue_update=on_queue_update,
                    )
                except Exception as e:
                    return {"__error__": str(e)}

            raw = await loop.run_in_executor(None, call)
            if isinstance(raw, dict) and "__error__" in raw:
                last_err = raw["__error__"]
                self._log(f"Attempt {attempt} failed: {last_err}")
                if retry_indefinitely or attempt < max_retries:
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue
                msg = f"API error after {attempt} attempts: {last_err}"
                return self._create_error_image(msg), "", msg
            result = raw
            break

        if not result or "images" not in result or not result["images"]:
            msg = "API returned no images"
            return self._create_error_image(msg), "", msg

        out_urls = [img.get("url") for img in result["images"] if img.get("url")]
        if not out_urls:
            msg = "API response missing image URLs"
            return self._create_error_image(msg), "", msg

        tensors = []
        for i, url in enumerate(out_urls):
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                pil = Image.open(io.BytesIO(resp.content))
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")
                arr = np.array(pil).astype(np.float32) / 255.0
                tensors.append(torch.from_numpy(arr).unsqueeze(0))
                self._log(f"Downloaded result {i + 1}/{len(out_urls)} ({pil.size[0]}x{pil.size[1]})")
            except Exception as e:
                self._log(f"Failed to download {url}: {e}")

        if not tensors:
            msg = "All result downloads failed"
            return self._create_error_image(msg), "", msg

        combined = _pad_and_cat(tensors)
        joined_urls = " | ".join(out_urls)
        return combined, joined_urls, f"Success: {len(tensors)} image(s) generated"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def generate(self, api_key, image_count, prompt, image_size,
                 custom_width, custom_height, quality, num_images, output_format,
                 max_retries, image_ref_1,
                 mask=None, sync_mode=False, retry_indefinitely=False, **kwargs):
        self.log_messages = []
        if not api_key:
            msg = "API key not provided."
            self._log(msg)
            return (self._create_error_image(msg), "", msg)

        ref_tensors = [image_ref_1]
        for i in range(2, int(image_count) + 1):
            tensor = kwargs.get(f"image_ref_{i}")
            if tensor is not None:
                ref_tensors.append(tensor)

        ref_pils: List[Image.Image] = []
        for i, t in enumerate(ref_tensors, start=1):
            pil = self._tensor_to_pil(t)
            if pil is None:
                self._log(f"image_ref_{i} is empty or invalid — skipping")
                continue
            ref_pils.append(pil)

        if not ref_pils:
            msg = "At least one reference image is required."
            self._log(msg)
            return (self._create_error_image(msg), "", msg)

        mask_pil = self._tensor_to_pil(mask) if mask is not None else None

        pbar = ProgressBar(1)

        async def runner():
            return await self._generate_async(
                api_key, prompt, ref_pils, mask_pil,
                image_size, custom_width, custom_height,
                quality, num_images, output_format,
                max_retries, retry_indefinitely, sync_mode,
            )

        def run_sync():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(runner())
            finally:
                loop.close()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_sync)
                tensor, urls, status = future.result(timeout=600)
        except concurrent.futures.TimeoutError:
            msg = "Processing timed out after 10 minutes"
            self._log(msg)
            return (self._create_error_image(msg), "", msg)
        except Exception as e:
            msg = f"Processing error: {e}"
            self._log(msg)
            return (self._create_error_image(msg), "", msg)
        finally:
            pbar.update_absolute(1)

        full_log = "Processing Logs:\n" + "\n".join(self.log_messages) + f"\n\nStatus: {status}"
        return (tensor, urls, full_log)


def _pad_and_cat(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Pad a list of (1, H, W, C) tensors to the max HxW, then concat along batch dim."""
    if len(tensors) == 1:
        return tensors[0]
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    out = []
    for t in tensors:
        h, w = t.shape[1], t.shape[2]
        if h == max_h and w == max_w:
            out.append(t)
            continue
        pad_h, pad_w = max_h - h, max_w - w
        pl, pr = pad_w // 2, pad_w - pad_w // 2
        pt, pb = pad_h // 2, pad_h - pad_h // 2
        padded = torch.nn.functional.pad(
            t.permute(0, 3, 1, 2),
            (pl, pr, pt, pb),
            mode="constant", value=0,
        ).permute(0, 2, 3, 1)
        out.append(padded)
    return torch.cat(out, dim=0)
