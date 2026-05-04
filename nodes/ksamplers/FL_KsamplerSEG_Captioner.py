# FL_KsamplerSEG_Captioner: caption each region's padded crop via OpenRouter.
#
# Crops the source image at each region's padded bbox, JPEG/85 base64-encodes,
# and POSTs to OpenRouter chat-completions with vision content. Uses a
# Semaphore to bound parallel HTTP requests. Streams thumbnails+captions
# back to the on-node DOM widget as they arrive.

import os
import io
import base64
import asyncio
import logging
import json

import torch
import numpy as np
import aiohttp
from PIL import Image, ImageDraw, ImageFont

from server import PromptServer

from .FL_KsamplerSEG_common import unwrap_regions, attach_captions


CAPTIONER_MODELS = [
    "openai/gpt-4o-mini",
    "google/gemini-flash-1.5",
    "anthropic/claude-haiku-4.5",
]


class FL_KsamplerSEG_Captioner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "regions": ("SEG_REGIONS",),
                "source_image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (CAPTIONER_MODELS, {"default": "openai/gpt-4o-mini"}),
                "prompt_template": ("STRING", {
                    "default": "Describe the contents of this image region in 10-15 words. Focus on what's visible. Output only the description, nothing else.",
                    "multiline": True,
                }),
                "prefix": ("STRING", {"default": "", "multiline": False}),
                "suffix": ("STRING", {"default": "", "multiline": False}),
                "max_tokens": ("INT", {"default": 60, "min": 10, "max": 200, "step": 5}),
                "parallel_requests": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1}),
                "default_negative_prompt": ("STRING", {
                    "default": "blurry, low quality, artifacts",
                    "multiline": True,
                }),
                "caption_first_frame_only": ("BOOLEAN", {"default": True}),
                "show_preview": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("SEG_REGIONS", "IMAGE")
    RETURN_NAMES = ("regions", "preview_image")
    FUNCTION = "caption"
    CATEGORY = "🏵️Fill Nodes/Ksamplers"

    def caption(self, regions, source_image, api_key, model, prompt_template,
                prefix, suffix, max_tokens, parallel_requests,
                default_negative_prompt, caption_first_frame_only,
                show_preview, unique_id=None):
        regions = unwrap_regions(regions)

        key = (api_key or "").strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "FL_KsamplerSEG_Captioner: provide an OpenRouter API key in the "
                "widget or set OPENROUTER_API_KEY env var."
            )

        # Pull the captioning frame.
        if caption_first_frame_only or source_image.shape[0] == 1:
            frame_idx = 0
        else:
            logging.warning(
                "[FL_KsamplerSEG_Captioner] caption_first_frame_only=False on "
                "a multi-frame batch is expensive; consider toggling on."
            )
            frame_idx = 0  # Still use frame 0; per-frame mode is v1.1.

        frame = source_image[frame_idx]  # (H, W, C)
        frame_np = (frame.cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)

        # Validate sizing.
        H_img, W_img = frame_np.shape[0], frame_np.shape[1]
        H_reg, W_reg = regions["image_size"]
        if (H_img, W_img) != (H_reg, W_reg):
            logging.warning(
                f"[FL_KsamplerSEG_Captioner] source_image size {H_img}x{W_img} "
                f"!= regions image_size {H_reg}x{W_reg}; cropping by regions geometry."
            )

        N = regions["shape_masks"].shape[0]
        bboxes = regions["padded_bboxes"]

        # Pre-build per-region crops + base64.
        crops_b64 = []
        crops_pil = []
        for n in range(N):
            y0, x0, y1, x1 = bboxes[n]
            y0 = max(0, min(H_img, y0))
            x0 = max(0, min(W_img, x0))
            y1 = max(0, min(H_img, y1))
            x1 = max(0, min(W_img, x1))
            if y1 <= y0 or x1 <= x0:
                # Empty crop -> placeholder
                pil = Image.new("RGB", (16, 16), (32, 32, 32))
            else:
                pil = Image.fromarray(frame_np[y0:y1, x0:x1])
            crops_pil.append(pil)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            crops_b64.append(base64.b64encode(buf.getvalue()).decode())

        # Run captions concurrently.
        captions = self._run_captions(
            key, model, prompt_template, max_tokens, parallel_requests,
            crops_b64, crops_pil, unique_id, show_preview,
        )

        # Build the preview grid IMAGE.
        grid = self._build_grid(crops_pil, captions, max_cols=4)
        grid_np = np.array(grid).astype(np.float32) / 255.0
        grid_t = torch.from_numpy(grid_np).unsqueeze(0)

        out = attach_captions(
            regions, captions,
            prefix=prefix, suffix=suffix,
            default_negative=default_negative_prompt,
        )
        return (out, grid_t)

    # ------------------------------------------------------------------
    # Async caption runner
    # ------------------------------------------------------------------

    def _run_captions(self, api_key, model, template, max_tokens,
                      parallel, crops_b64, crops_pil, node_id, show_preview):
        """Synchronously run the async batch. Returns captions in region order."""
        async def runner():
            sem = asyncio.Semaphore(int(parallel))
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/filliptm/ComfyUI_Fill-Nodes",
                "X-Title": "FL_KsamplerSEG",
            }
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                async def one(idx, b64):
                    async with sem:
                        text = await self._caption_one(session, model, template,
                                                       max_tokens, b64)
                    if show_preview and node_id is not None:
                        try:
                            buf = io.BytesIO()
                            thumb = crops_pil[idx].copy()
                            thumb.thumbnail((192, 192), Image.Resampling.LANCZOS)
                            thumb.save(buf, format="JPEG", quality=80)
                            thumb_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
                            PromptServer.instance.send_sync(
                                "fl_seg_captioner_progress",
                                {
                                    "node": str(node_id),
                                    "region_index": idx,
                                    "caption": text,
                                    "thumbnail": thumb_uri,
                                    "total": len(crops_b64),
                                },
                            )
                        except Exception as e:
                            logging.warning(f"[FL_KsamplerSEG_Captioner] preview send failed: {e}")
                    return idx, text

                tasks = [one(i, b64) for i, b64 in enumerate(crops_b64)]
                results = await asyncio.gather(*tasks)
            results.sort(key=lambda r: r[0])
            return [text for _, text in results]

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(runner())
        finally:
            loop.close()

    async def _caption_one(self, session, model, template, max_tokens, b64):
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": model,
            "max_tokens": int(max_tokens),
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": template},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                    }},
                ]},
            ],
        }
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    return f"[caption failed: HTTP {resp.status} {body[:120]}]"
                data = await resp.json()
                choices = data.get("choices") or []
                if not choices:
                    return f"[caption failed: empty choices in {json.dumps(data)[:120]}]"
                msg = choices[0].get("message", {}).get("content")
                if not msg:
                    return "[caption failed: no message content]"
                return str(msg).strip()
        except asyncio.TimeoutError:
            return "[caption failed: timeout]"
        except Exception as e:
            return f"[caption failed: {type(e).__name__}: {e}]"

    # ------------------------------------------------------------------
    # Preview grid
    # ------------------------------------------------------------------

    @staticmethod
    def _build_grid(crops, captions, max_cols=4):
        """Compose a single IMAGE: thumbnails arranged in a grid, captions below."""
        n = len(crops)
        cols = min(max_cols, n)
        rows = (n + cols - 1) // cols
        thumb_size = 192
        text_height = 70
        cell_w = thumb_size + 16
        cell_h = thumb_size + text_height + 16

        out_w = cols * cell_w + 16
        out_h = rows * cell_h + 16

        canvas = Image.new("RGB", (out_w, out_h), (24, 24, 28))
        draw = ImageDraw.Draw(canvas)

        font = None
        for name in ("arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"):
            try:
                font = ImageFont.truetype(name, 11)
                break
            except IOError:
                continue
        if font is None:
            font = ImageFont.load_default()

        small_font = font

        for i, crop in enumerate(crops):
            r = i // cols
            c = i % cols
            cx = 16 + c * cell_w
            cy = 16 + r * cell_h
            thumb = crop.copy()
            thumb.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            tx = cx + (thumb_size - thumb.width) // 2
            ty = cy + (thumb_size - thumb.height) // 2
            canvas.paste(thumb, (tx, ty))

            # Region index in top-left of the cell.
            label = str(i)
            draw.rectangle([cx, cy, cx + 22, cy + 18], fill=(255, 240, 80))
            draw.text((cx + 5, cy + 2), label, fill=(0, 0, 0), font=font)

            # Caption text (wrap to fit).
            cap = captions[i] if i < len(captions) else ""
            cap_lines = FL_KsamplerSEG_Captioner._wrap_text(cap, small_font, draw, thumb_size)
            tx0 = cx
            ty0 = cy + thumb_size + 6
            for j, line in enumerate(cap_lines[:5]):
                draw.text((tx0, ty0 + j * 12), line,
                          fill=(220, 220, 220), font=small_font)

        return canvas

    @staticmethod
    def _wrap_text(text, font, draw, max_w):
        """Naive word-wrap to fit max_w pixels."""
        if not text:
            return [""]
        words = text.split()
        lines = []
        cur = ""
        for w in words:
            trial = (cur + " " + w).strip()
            try:
                bbox = draw.textbbox((0, 0), trial, font=font)
                width = bbox[2] - bbox[0]
            except AttributeError:
                width, _ = draw.textsize(trial, font=font)
            if width > max_w and cur:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines
