import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from server import PromptServer


class FL_RegionPNGOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "rotation_min": ("FLOAT", {"default": -5.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "rotation_max": ("FLOAT", {"default": 5.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "vignette_enabled": ("BOOLEAN", {"default": True}),
                "vignette_diameter": ("FLOAT", {"default": 0.85, "min": 0.05, "max": 3.0, "step": 0.01}),
                "vignette_skew": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 2.0, "step": 0.01}),
                "vignette_opacity": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vignette_offset_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "vignette_offset_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "vignette_rotation_offset": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "regions_json": ("STRING", {"default": "[]", "multiline": True}),
                "show_canvas": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "compose"
    CATEGORY = "🏵️Fill Nodes/Image"

    def compose(self, image, seed, rotation_min, rotation_max, vignette_enabled,
                vignette_diameter, vignette_skew, vignette_opacity,
                vignette_offset_x, vignette_offset_y, vignette_rotation_offset,
                regions_json, show_canvas=True, unique_id=None):
        regions = self._parse_regions(regions_json)
        vignette_settings = {
            "enabled": bool(vignette_enabled),
            "diameter": float(vignette_diameter),
            "skew": float(vignette_skew),
            "opacity": float(vignette_opacity),
            "offset_x": int(vignette_offset_x),
            "offset_y": int(vignette_offset_y),
            "rotation_offset": float(vignette_rotation_offset),
        }
        out = []

        for batch_index in range(int(image.shape[0])):
            base = self._tensor_to_pil(image[batch_index]).convert("RGBA")
            if show_canvas and batch_index == 0:
                self._send_canvas_preview(base, regions, unique_id)

            result = base.copy()
            for region_index, region in enumerate(regions):
                rng = np.random.default_rng(
                    (int(seed) + batch_index * 1000003 + region_index * 9176) & 0xFFFFFFFF
                )
                result = self._apply_region(
                    result,
                    region,
                    rng,
                    float(rotation_min),
                    float(rotation_max),
                    vignette_settings,
                )

            out.append(self._pil_to_tensor(result.convert("RGB")))

        return (torch.cat(out, dim=0),)

    def _apply_region(self, base, region, rng, rotation_min, rotation_max, vignette_settings):
        x, y, w, h = self._region_rect(region, base.size)
        if w < 2 or h < 2:
            return base

        directory = str(region.get("directory", "")).strip()
        if not directory:
            return base

        pngs = self._find_pngs(directory)
        if not pngs:
            print(f"[FL_RegionPNGOverlay] no PNG files found in region directory: {directory}")
            return base

        overlay_path = pngs[int(rng.integers(0, len(pngs)))]
        try:
            overlay = Image.open(overlay_path).convert("RGBA")
        except Exception as e:
            print(f"[FL_RegionPNGOverlay] failed to open {overlay_path}: {e}")
            return base

        angle = float(rng.uniform(min(rotation_min, rotation_max), max(rotation_min, rotation_max)))
        fitted = self._fit_rotate_overlay(overlay, w, h, angle)
        if fitted is None:
            return base

        avail_x = max(0, w - fitted.width)
        avail_y = max(0, h - fitted.height)
        paste_x = x + (int(rng.integers(0, avail_x + 1)) if avail_x > 0 else 0)
        paste_y = y + (int(rng.integers(0, avail_y + 1)) if avail_y > 0 else 0)

        if vignette_settings["enabled"] and vignette_settings["opacity"] > 0.0:
            center_x = (
                x + (w / 2.0)
                + float(region.get("vignette_offset_x", 0))
                + vignette_settings["offset_x"]
            )
            center_y = (
                y + (h / 2.0)
                + float(region.get("vignette_offset_y", 0))
                + vignette_settings["offset_y"]
            )
            diameter = max(1.0, min(w, h) * vignette_settings["diameter"])
            vignette = self._make_vignette(
                diameter=diameter,
                skew=vignette_settings["skew"],
                opacity=vignette_settings["opacity"],
                angle=angle + vignette_settings["rotation_offset"],
            )
            vx = int(round(center_x - vignette.width / 2.0))
            vy = int(round(center_y - vignette.height / 2.0))
            self._alpha_composite_clipped(base, vignette, vx, vy)

        base.alpha_composite(fitted, (paste_x, paste_y))
        return base

    @staticmethod
    def _make_vignette(diameter, skew, opacity, angle):
        ellipse_w = max(1.0, float(diameter))
        ellipse_h = max(1.0, float(diameter) * max(0.01, float(skew)))
        theta = math.radians(float(angle))
        bbox_w = int(math.ceil(abs(ellipse_w * math.cos(theta)) + abs(ellipse_h * math.sin(theta)))) + 4
        bbox_h = int(math.ceil(abs(ellipse_w * math.sin(theta)) + abs(ellipse_h * math.cos(theta)))) + 4
        bbox_w = max(2, bbox_w)
        bbox_h = max(2, bbox_h)

        yy, xx = np.mgrid[0:bbox_h, 0:bbox_w].astype(np.float32)
        cx = (bbox_w - 1) / 2.0
        cy = (bbox_h - 1) / 2.0
        dx = xx - cx
        dy = yy - cy

        cos_t = math.cos(-theta)
        sin_t = math.sin(-theta)
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t

        norm = np.sqrt((rx / (ellipse_w / 2.0)) ** 2 + (ry / (ellipse_h / 2.0)) ** 2)
        alpha = np.clip(1.0 - norm, 0.0, 1.0)
        alpha = (alpha ** 1.8) * np.clip(float(opacity), 0.0, 1.0) * 255.0

        arr = np.zeros((bbox_h, bbox_w, 4), dtype=np.uint8)
        arr[..., 3] = alpha.astype(np.uint8)
        return Image.fromarray(arr, mode="RGBA")

    @staticmethod
    def _alpha_composite_clipped(base, overlay, x, y):
        base_w, base_h = base.size
        src_x0 = max(0, -x)
        src_y0 = max(0, -y)
        dst_x = max(0, x)
        dst_y = max(0, y)
        width = min(overlay.width - src_x0, base_w - dst_x)
        height = min(overlay.height - src_y0, base_h - dst_y)
        if width <= 0 or height <= 0:
            return
        cropped = overlay.crop((src_x0, src_y0, src_x0 + width, src_y0 + height))
        base.alpha_composite(cropped, (dst_x, dst_y))

    @staticmethod
    def _fit_rotate_overlay(overlay, region_w, region_h, angle):
        ow, oh = overlay.size
        if ow < 1 or oh < 1:
            return None

        theta = math.radians(angle)
        rotated_w_factor = abs(ow * math.cos(theta)) + abs(oh * math.sin(theta))
        rotated_h_factor = abs(ow * math.sin(theta)) + abs(oh * math.cos(theta))
        if rotated_w_factor <= 0 or rotated_h_factor <= 0:
            return None

        scale = min(region_w / rotated_w_factor, region_h / rotated_h_factor)
        if scale <= 0:
            return None

        new_w = max(1, int(math.floor(ow * scale)))
        new_h = max(1, int(math.floor(oh * scale)))

        for _ in range(8):
            resized = overlay.resize((new_w, new_h), Image.Resampling.LANCZOS)
            rotated = resized.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
            if rotated.width <= region_w and rotated.height <= region_h:
                return rotated
            shrink = min(region_w / max(1, rotated.width), region_h / max(1, rotated.height)) * 0.98
            new_w = max(1, int(math.floor(new_w * shrink)))
            new_h = max(1, int(math.floor(new_h * shrink)))
            if new_w <= 1 or new_h <= 1:
                break

        return None

    @staticmethod
    def _parse_regions(regions_json):
        try:
            raw = json.loads(regions_json or "[]")
        except Exception as e:
            print(f"[FL_RegionPNGOverlay] invalid regions_json; returning input image unchanged: {e}")
            return []
        if not isinstance(raw, list):
            return []
        return [r for r in raw if isinstance(r, dict)]

    @staticmethod
    def _region_rect(region, image_size):
        img_w, img_h = image_size
        x = int(round(float(region.get("x", 0))))
        y = int(round(float(region.get("y", 0))))
        w = int(round(float(region.get("w", 0))))
        h = int(round(float(region.get("h", 0))))

        x = max(0, min(img_w, x))
        y = max(0, min(img_h, y))
        w = max(0, min(img_w - x, w))
        h = max(0, min(img_h - y, h))
        return x, y, w, h

    @staticmethod
    def _find_pngs(directory):
        try:
            root = Path(os.path.expandvars(os.path.expanduser(directory)))
            if not root.is_dir():
                return []
            return sorted(
                str(p) for p in root.iterdir()
                if p.is_file() and p.suffix.lower() == ".png"
            )
        except Exception as e:
            print(f"[FL_RegionPNGOverlay] failed to scan {directory}: {e}")
            return []

    @staticmethod
    def _tensor_to_pil(tensor):
        arr = tensor.detach().cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _pil_to_tensor(image):
        arr = np.asarray(image).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def _send_canvas_preview(self, image, regions, unique_id):
        try:
            preview = image.convert("RGB").copy()
            preview.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            import base64
            import io

            buf = io.BytesIO()
            preview.save(buf, format="PNG")
            PromptServer.instance.send_sync(
                "fl_region_png_overlay_canvas",
                {
                    "node": str(unique_id) if unique_id is not None else "",
                    "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
                    "source_size": [image.width, image.height],
                    "regions": regions,
                },
            )
        except Exception as e:
            print(f"[FL_RegionPNGOverlay] canvas preview send failed: {e}")
