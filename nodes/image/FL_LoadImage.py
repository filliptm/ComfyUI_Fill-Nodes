import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

import folder_paths
from comfy.utils import common_upscale


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_LOAD_SETTINGS = {
    "version": 1,
    "resize_mode": "original",
    "width": 0,
    "height": 0,
}
DEFAULT_SETTINGS_JSON = json.dumps(DEFAULT_LOAD_SETTINGS, separators=(",", ":"))


def available_image_files():
    input_dir = Path(folder_paths.get_input_directory()).resolve()
    os.makedirs(input_dir, exist_ok=True)
    files, _ = folder_paths.recursive_search(str(input_dir))
    result = []
    for filename in files:
        if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        path = (input_dir / filename).resolve()
        try:
            path.relative_to(input_dir)
        except ValueError:
            continue
        if path.is_file():
            result.append(Path(filename).as_posix())
    return sorted(result, key=str.casefold)


def _resolve_input_image(filename):
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Choose an image file.")

    filename = filename.strip()
    name, annotated_dir = folder_paths.annotated_filepath(filename)
    input_dir = Path(folder_paths.get_input_directory()).resolve()
    if annotated_dir is not None and Path(annotated_dir).resolve() != input_dir:
        raise ValueError("FL Load Image files must be inside the ComfyUI input directory.")

    try:
        path = Path(folder_paths.get_annotated_filepath(name, str(input_dir))).resolve()
    except ValueError as e:
        raise ValueError("FL Load Image files must be inside the ComfyUI input directory.") from e
    try:
        path.relative_to(input_dir)
    except ValueError as e:
        raise ValueError("FL Load Image files must be inside the ComfyUI input directory.") from e
    if not path.is_file():
        raise ValueError(f"Image file does not exist: {filename}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix or filename}")
    return path


def _resolve_legacy_image(root_directory, selected_file):
    if not isinstance(root_directory, str) or not root_directory.strip():
        raise ValueError("FL Load Image legacy root directory is empty.")
    if not isinstance(selected_file, str) or not selected_file.strip():
        raise ValueError("Choose an image file.")

    root = Path(root_directory).expanduser().resolve()
    path = Path(selected_file).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not root.is_dir():
        raise ValueError("FL Load Image legacy root directory does not exist.")
    try:
        path.relative_to(root)
    except ValueError as e:
        raise ValueError("FL Load Image legacy file must be inside its root directory.") from e
    if not path.is_file():
        raise ValueError(f"Image file does not exist: {selected_file}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix or selected_file}")
    return path


def resolve_image_path(root_directory, selected_file="", image=""):
    if image:
        return _resolve_input_image(image)
    return _resolve_legacy_image(root_directory, selected_file)


def _parse_settings(load_settings, width_override=None, height_override=None):
    try:
        configured = json.loads(load_settings)
    except (TypeError, json.JSONDecodeError) as e:
        raise ValueError("FL Load Image settings are not valid JSON.") from e
    if not isinstance(configured, dict):
        raise ValueError("FL Load Image settings must be a JSON object.")

    settings = DEFAULT_LOAD_SETTINGS.copy()
    settings.update(configured)
    version = settings["version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        raise ValueError(f"FL Load Image settings version {version} is unsupported.")

    for name in ("width", "height"):
        value = settings[name]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"FL Load Image {name} must be an integer.")
        if not 0 <= value <= 16384:
            raise ValueError("FL Load Image width and height must be between 0 and 16384.")

    resize_mode = settings["resize_mode"]
    if resize_mode not in ("original", "fit", "crop"):
        raise ValueError("FL Load Image resize_mode must be original, fit, or crop.")

    for name, value in (("width", width_override), ("height", height_override)):
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"FL Load Image {name} override must be an integer.")
        if not 0 <= value <= 16384:
            raise ValueError("FL Load Image width and height overrides must be between 0 and 16384.")
        settings[name] = value

    if resize_mode == "fit" and settings["width"] == 0 and settings["height"] == 0:
        raise ValueError("FL Load Image fit resize requires a width or height.")
    if resize_mode == "crop" and (settings["width"] == 0 or settings["height"] == 0):
        raise ValueError("FL Load Image crop resize requires both width and height.")
    return settings


def _target_dimensions(source_width, source_height, settings):
    if settings["resize_mode"] == "original":
        return source_width, source_height
    if settings["resize_mode"] == "crop":
        return settings["width"], settings["height"]

    width = settings["width"]
    height = settings["height"]
    if width == 0:
        scale = height / source_height
    elif height == 0:
        scale = width / source_width
    else:
        scale = min(width / source_width, height / source_height)
    return max(1, round(source_width * scale)), max(1, round(source_height * scale))


def _load_image(path):
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source)
        source_width, source_height = source.size
        has_alpha = "A" in source.getbands() or "transparency" in source.info
        if has_alpha:
            rgba = source.convert("RGBA")
            image = Image.new("RGB", rgba.size, (255, 255, 255))
            image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            image = source.convert("RGB")
        image_tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).unsqueeze(0)
    return image_tensor, source_width, source_height, has_alpha


def _resize_image(image, settings):
    source_height = int(image.shape[1])
    source_width = int(image.shape[2])
    width, height = _target_dimensions(source_width, source_height, settings)
    if (width, height) == (source_width, source_height):
        return image

    crop = "center" if settings["resize_mode"] == "crop" else "disabled"
    image = common_upscale(image.movedim(-1, 1), width, height, "lanczos", crop)
    return image.movedim(1, -1)


def _preview_reference(path):
    input_dir = Path(folder_paths.get_input_directory()).resolve()
    try:
        relative = path.relative_to(input_dir)
    except ValueError:
        return None
    return relative.name, relative.parent.as_posix() if relative.parent != Path(".") else ""


class FL_LoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        files = available_image_files()
        return {
            "required": {
                "root_directory": ("STRING", {"default": "./"}),
            },
            "optional": {
                "selected_file": ("STRING", {"default": ""}),
                "image": ([""] + files,),
                "load_settings": ("STRING", {"default": DEFAULT_SETTINGS_JSON, "multiline": False}),
                "width_override": ("INT", {"min": 0, "max": 16384, "forceInput": True}),
                "height_override": ("INT", {"min": 0, "max": 16384, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "file_path")
    FUNCTION = "browse_files"
    CATEGORY = "🏵️Fill Nodes/Image"
    DESCRIPTION = "Loads, previews, and resizes an image from ComfyUI input."

    def browse_files(
        self,
        root_directory,
        selected_file="",
        image="",
        load_settings=DEFAULT_SETTINGS_JSON,
        width_override=None,
        height_override=None,
    ):
        settings = _parse_settings(load_settings, width_override, height_override)
        path = resolve_image_path(root_directory, selected_file, image)
        loaded, source_width, source_height, has_alpha = _load_image(path)
        loaded = _resize_image(loaded, settings)

        preview = {
            "filename": path.name,
            "source_width": source_width,
            "source_height": source_height,
            "loaded_width": int(loaded.shape[2]),
            "loaded_height": int(loaded.shape[1]),
            "resize_mode": settings["resize_mode"],
            "requested_width": settings["width"],
            "requested_height": settings["height"],
            "source_has_alpha": has_alpha,
        }
        reference = _preview_reference(path)
        if reference is not None:
            preview.update({"filename": reference[0], "subfolder": reference[1], "type": "input"})
        else:
            preview["type"] = "legacy"

        return {
            "ui": {"fl_load_image": [preview]},
            "result": (loaded, str(path)),
        }

    @classmethod
    def IS_CHANGED(
        cls,
        root_directory,
        selected_file="",
        image="",
        load_settings=DEFAULT_SETTINGS_JSON,
        width_override=None,
        height_override=None,
    ):
        path = resolve_image_path(root_directory, selected_file, image)
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        root_directory,
        selected_file="",
        image="",
        load_settings=DEFAULT_SETTINGS_JSON,
        width_override=None,
        height_override=None,
    ):
        try:
            resolve_image_path(root_directory, selected_file, image)
            _parse_settings(load_settings, width_override, height_override)
        except ValueError as e:
            return str(e)
        return True
