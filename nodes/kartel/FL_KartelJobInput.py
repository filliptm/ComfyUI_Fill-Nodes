import io
import json
import re
import numpy as np
from PIL import Image
import requests
import torch


# Standardized params_json schema:
# {
#     "string_1": "", "string_2": "", "string_3": "", "string_4": "",
#     "int_1": 0, "int_2": 0, "int_3": 0, "int_4": 0,
#     "float_1": 0.0, "float_2": 0.0, "float_3": 0.0, "float_4": 0.0,
#     "bool_1": false, "bool_2": false, "bool_3": false, "bool_4": false,
#     "image_url_1": "", "image_url_2": "", "image_url_3": "", "image_url_4": ""
# }


class FL_KartelJobInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_id": ("STRING", {"default": ""}),
                "user_id": ("STRING", {"default": ""}),
                "user_email": ("STRING", {"default": ""}),
                "app_name": ("STRING", {"default": ""}),
                "callback_url": ("STRING", {"default": ""}),
                "params_json": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    RETURN_TYPES = (
        "STRING", "STRING", "STRING", "STRING", "STRING",  # job metadata
        "STRING", "STRING", "STRING", "STRING",              # string_1-4
        "INT", "INT", "INT", "INT",                          # int_1-4
        "FLOAT", "FLOAT", "FLOAT", "FLOAT",                 # float_1-4
        "BOOLEAN", "BOOLEAN", "BOOLEAN", "BOOLEAN",          # bool_1-4
        "IMAGE", "IMAGE", "IMAGE", "IMAGE",                  # image_1-4
    )
    RETURN_NAMES = (
        "job_id", "user_id", "user_email", "app_name", "callback_url",
        "string_1", "string_2", "string_3", "string_4",
        "int_1", "int_2", "int_3", "int_4",
        "float_1", "float_2", "float_3", "float_4",
        "bool_1", "bool_2", "bool_3", "bool_4",
        "image_1", "image_2", "image_3", "image_4",
    )
    FUNCTION = "process"
    CATEGORY = "🏵️Fill Nodes/Kartel"

    def process(self, job_id, user_id, user_email, app_name, callback_url, params_json):
        print(f"[FL_KartelJobInput] === INCOMING DATA ===")
        print(f"[FL_KartelJobInput] job_id='{job_id}'")
        print(f"[FL_KartelJobInput] user_id='{user_id}'")
        print(f"[FL_KartelJobInput] user_email='{user_email}'")
        print(f"[FL_KartelJobInput] app_name='{app_name}'")
        print(f"[FL_KartelJobInput] callback_url='{callback_url}'")
        print(f"[FL_KartelJobInput] params_json='{params_json}'")
        print(f"[FL_KartelJobInput] === END INCOMING DATA ===")

        # Parse JSON with trailing comma tolerance
        try:
            data = json.loads(params_json)
            print(f"[FL_KartelJobInput] Parsed OK, keys: {list(data.keys())}")
            print(f"[FL_KartelJobInput] Full parsed data: {data}")
        except json.JSONDecodeError:
            cleaned = re.sub(r',\s*([}\]])', r'\1', params_json)
            try:
                data = json.loads(cleaned)
                print(f"[FL_KartelJobInput] Parsed after comma fix, keys: {list(data.keys())}")
                print(f"[FL_KartelJobInput] Full parsed data: {data}")
            except json.JSONDecodeError as e:
                print(f"[FL_KartelJobInput] Invalid JSON: {e}")
                data = {}

        # 1x1 black pixel placeholder for missing images
        empty_image = torch.zeros(1, 1, 1, 3)

        def get_str(key):
            val = data.get(key, "")
            return str(val) if val else ""

        def get_int(key):
            val = data.get(key, 0)
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0

        def get_float(key):
            val = data.get(key, 0.0)
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        def get_bool(key):
            val = data.get(key, False)
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ("true", "1", "yes")
            return bool(val)

        def get_image(key):
            url = data.get(key, "")
            if not url:
                return empty_image
            try:
                print(f"[FL_KartelJobInput] Downloading image '{key}': {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                pil_image = Image.open(io.BytesIO(response.content))
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                img_np = np.array(pil_image).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np).unsqueeze(0)
                print(f"[FL_KartelJobInput] Downloaded '{key}': {pil_image.width}x{pil_image.height}")
                return img_tensor
            except Exception as e:
                print(f"[FL_KartelJobInput] Failed to download '{key}' ({url}): {e}")
                return empty_image

        return (
            job_id, user_id, user_email, app_name, callback_url,
            get_str("string_1"), get_str("string_2"), get_str("string_3"), get_str("string_4"),
            get_int("int_1"), get_int("int_2"), get_int("int_3"), get_int("int_4"),
            get_float("float_1"), get_float("float_2"), get_float("float_3"), get_float("float_4"),
            get_bool("bool_1"), get_bool("bool_2"), get_bool("bool_3"), get_bool("bool_4"),
            get_image("image_url_1"), get_image("image_url_2"), get_image("image_url_3"), get_image("image_url_4"),
        )
