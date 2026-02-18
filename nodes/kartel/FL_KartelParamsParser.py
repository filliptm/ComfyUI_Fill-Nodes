import io
import json
import re
import numpy as np
from PIL import Image
import requests
import torch


class FL_KartelParamsParser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "params_json": ("STRING", {"default": "{}", "multiline": True}),
            },
            "optional": {
                "string_key_1": ("STRING", {"default": ""}),
                "string_key_2": ("STRING", {"default": ""}),
                "string_key_3": ("STRING", {"default": ""}),
                "string_key_4": ("STRING", {"default": ""}),
                "int_key_1": ("STRING", {"default": ""}),
                "int_key_2": ("STRING", {"default": ""}),
                "float_key_1": ("STRING", {"default": ""}),
                "float_key_2": ("STRING", {"default": ""}),
                "bool_key_1": ("STRING", {"default": ""}),
                "image_url_key_1": ("STRING", {"default": ""}),
                "image_url_key_2": ("STRING", {"default": ""}),
                "image_url_key_3": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "INT", "FLOAT", "FLOAT", "BOOLEAN", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("string_1", "string_2", "string_3", "string_4", "int_1", "int_2", "float_1", "float_2", "bool_1", "image_1", "image_2", "image_3")
    FUNCTION = "parse"
    CATEGORY = "🏵️Fill Nodes/Kartel"

    def parse(self, params_json, string_key_1="", string_key_2="", string_key_3="",
              string_key_4="", int_key_1="", int_key_2="", float_key_1="",
              float_key_2="", bool_key_1="", image_url_key_1="", image_url_key_2="",
              image_url_key_3=""):
        print(f"[FL_KartelParamsParser] Raw params_json ({len(params_json)} chars): {params_json[:500]}")
        print(f"[FL_KartelParamsParser] image_url_key_1='{image_url_key_1}' image_url_key_2='{image_url_key_2}' image_url_key_3='{image_url_key_3}'")

        try:
            data = json.loads(params_json)
            print(f"[FL_KartelParamsParser] Parsed JSON OK, keys: {list(data.keys())}")
        except json.JSONDecodeError:
            # Strip trailing commas before } and ] then retry
            cleaned = re.sub(r',\s*([}\]])', r'\1', params_json)
            try:
                data = json.loads(cleaned)
                print(f"[FL_KartelParamsParser] Parsed JSON after stripping trailing commas, keys: {list(data.keys())}")
            except json.JSONDecodeError as e:
                print(f"[FL_KartelParamsParser] Invalid JSON: {e}")
                data = {}

        def get_str(key):
            if not key:
                return ""
            return str(data.get(key, ""))

        def get_int(key):
            if not key:
                return 0
            val = data.get(key, 0)
            try:
                return int(val)
            except (ValueError, TypeError):
                print(f"[FL_KartelParamsParser] Cannot convert '{key}' value '{val}' to INT, defaulting to 0")
                return 0

        def get_float(key):
            if not key:
                return 0.0
            val = data.get(key, 0.0)
            try:
                return float(val)
            except (ValueError, TypeError):
                print(f"[FL_KartelParamsParser] Cannot convert '{key}' value '{val}' to FLOAT, defaulting to 0.0")
                return 0.0

        def get_bool(key):
            if not key:
                return False
            val = data.get(key, False)
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ("true", "1", "yes")
            return bool(val)

        # 1x1 black pixel — valid IMAGE tensor placeholder when no image is available
        empty_image = torch.zeros(1, 1, 1, 3)

        def get_image(key):
            if not key:
                return empty_image
            url = data.get(key, "")
            if not url:
                return empty_image
            try:
                print(f"[FL_KartelParamsParser] Downloading image from '{key}': {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                pil_image = Image.open(io.BytesIO(response.content))
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                img_np = np.array(pil_image).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np).unsqueeze(0)
                print(f"[FL_KartelParamsParser] Downloaded image '{key}': {pil_image.width}x{pil_image.height}")
                return img_tensor
            except Exception as e:
                print(f"[FL_KartelParamsParser] Failed to download image for '{key}' ({url}): {e}")
                return empty_image

        return (
            get_str(string_key_1),
            get_str(string_key_2),
            get_str(string_key_3),
            get_str(string_key_4),
            get_int(int_key_1),
            get_int(int_key_2),
            get_float(float_key_1),
            get_float(float_key_2),
            get_bool(bool_key_1),
            get_image(image_url_key_1),
            get_image(image_url_key_2),
            get_image(image_url_key_3),
        )
