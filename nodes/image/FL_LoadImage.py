import os
import json
from PIL import Image
import torch
import numpy as np
from server import PromptServer
from aiohttp import web
import io


class FL_LoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root_directory": ("STRING", {"default": "./"}),
            },
            "optional": {
                "selected_file": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "browse_files"
    OUTPUT_NODE = True
    CATEGORY = "🏵️Fill Nodes/Image"

    def browse_files(self, root_directory, selected_file=""):
        if selected_file and os.path.isfile(selected_file):
            image = Image.open(selected_file)
            # Convert RGBA to RGB if needed (for WebP with transparency)
            if image.mode == 'RGBA':
                # Create white background and composite
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
            elif image.mode != 'RGB':
                # Convert other modes (P, L, etc.) to RGB
                image = image.convert('RGB')
            
            image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
            return (image_tensor, selected_file)
        else:
            return (torch.zeros(1, 1, 1, 3), "")

    @classmethod
    def IS_CHANGED(cls, root_directory, selected_file=""):
        return selected_file

    @classmethod
    def VALIDATE_INPUTS(cls, root_directory, selected_file=""):
        if not os.path.isdir(root_directory):
            return "Root directory does not exist"
        if selected_file and not os.path.isfile(selected_file):
            return "Selected file does not exist"
        return True


def get_directory_structure(path):
    structure = {"name": os.path.basename(path), "children": [], "path": path, "expanded": False}
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir():
                    structure["children"].append(get_directory_structure(entry.path))
    except PermissionError:
        pass
    return structure


def get_file_list(path):
    return [f for f in os.listdir(path) if
            os.path.isfile(os.path.join(path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))]


@PromptServer.instance.routes.post("/fl_file_browser/get_directory_structure")
async def api_get_directory_structure(request):
    data = await request.json()
    path = data.get("path", "./")

    if not os.path.isabs(path):
        path = os.path.abspath(path)

    if not os.path.exists(path):
        return web.json_response({"error": "Directory does not exist"}, status=400)

    structure = get_directory_structure(path)
    files = get_file_list(path)
    return web.json_response({"structure": structure, "files": files})


@PromptServer.instance.routes.post("/fl_file_browser/get_thumbnail")
async def api_get_thumbnail(request):
    data = await request.json()
    path = data.get("path", "./")
    file = data.get("file", "")

    if not os.path.isabs(path):
        path = os.path.abspath(path)

    full_path = os.path.join(path, file)
    if not os.path.exists(full_path):
        return web.json_response({"error": "File does not exist"}, status=400)

    try:
        with Image.open(full_path) as img:
            # Convert RGBA to RGB if needed (for WebP with transparency)
            if img.mode == 'RGBA':
                # Create white background and composite
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                img = background
            elif img.mode != 'RGB':
                # Convert other modes (P, L, etc.) to RGB
                img = img.convert('RGB')
            
            img.thumbnail((80, 80))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            return web.Response(body=buf.read(), content_type='image/png')
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)