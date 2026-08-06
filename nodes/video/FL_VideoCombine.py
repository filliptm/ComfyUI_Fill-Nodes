import json
import os
import secrets
import threading
from collections import OrderedDict
from fractions import Fraction

import torch
import torch.nn.functional as F

import folder_paths
from comfy.cli_args import args
from comfy_api.latest import InputImpl, Types


DEFAULT_RENDER_SETTINGS = {
    "version": 1,
    "filename_prefix": "FillVideo",
    "frame_rate": 24.0,
    "format": "mp4",
    "codec": "h264",
    "crf": 19,
    "bit_depth": 8,
    "include_audio": True,
    "trim_video_to_audio": False,
    "audio_gain_db": 0.0,
    "output_directory": "",
    "save_output": True,
    "save_metadata": True,
}
DEFAULT_SETTINGS_JSON = json.dumps(DEFAULT_RENDER_SETTINGS, separators=(",", ":"))

_MAX_PREVIEW_FILES = 64
_preview_files = OrderedDict()
_preview_files_lock = threading.Lock()


def register_preview_file(file_path):
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"FL Video Combine preview file does not exist: {file_path}")

    token = secrets.token_urlsafe(24)
    with _preview_files_lock:
        _preview_files[token] = file_path
        while len(_preview_files) > _MAX_PREVIEW_FILES:
            _preview_files.popitem(last=False)
    return token


def preview_file_for_token(token):
    if not isinstance(token, str) or not token:
        return None

    with _preview_files_lock:
        file_path = _preview_files.get(token)
        if file_path is None:
            return None
        if not os.path.isfile(file_path):
            del _preview_files[token]
            return None
        _preview_files.move_to_end(token)
        return file_path


def _parse_settings(render_settings):
    try:
        configured = json.loads(render_settings)
    except (TypeError, json.JSONDecodeError) as e:
        raise ValueError("FL Video Combine settings are not valid JSON.") from e
    if not isinstance(configured, dict):
        raise ValueError("FL Video Combine settings must be a JSON object.")

    settings = DEFAULT_RENDER_SETTINGS.copy()
    settings.update(configured)

    version = settings["version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        raise ValueError(f"FL Video Combine settings version {version} is unsupported.")

    filename_prefix = settings["filename_prefix"]
    if not isinstance(filename_prefix, str) or not filename_prefix.strip():
        raise ValueError("FL Video Combine filename_prefix must be a non-empty string.")

    frame_rate = settings["frame_rate"]
    if not isinstance(frame_rate, (int, float)) or isinstance(frame_rate, bool) or not 1 <= frame_rate <= 120:
        raise ValueError("FL Video Combine frame_rate must be between 1 and 120.")

    if settings["format"] != "mp4":
        raise ValueError("FL Video Combine currently supports only MP4 output.")
    if settings["codec"] != "h264":
        raise ValueError("FL Video Combine currently supports only H.264 video.")

    crf = settings["crf"]
    if not isinstance(crf, int) or isinstance(crf, bool) or not 0 <= crf <= 51:
        raise ValueError("FL Video Combine crf must be an integer between 0 and 51.")

    bit_depth = settings["bit_depth"]
    if not isinstance(bit_depth, int) or isinstance(bit_depth, bool) or bit_depth not in (8, 10):
        raise ValueError("FL Video Combine bit_depth must be 8 or 10.")

    audio_gain_db = settings["audio_gain_db"]
    if not isinstance(audio_gain_db, (int, float)) or isinstance(audio_gain_db, bool) or not -60 <= audio_gain_db <= 12:
        raise ValueError("FL Video Combine audio_gain_db must be between -60 and 12.")

    output_directory = settings["output_directory"]
    if not isinstance(output_directory, str):
        raise ValueError("FL Video Combine output_directory must be a string.")
    output_directory = os.path.expanduser(output_directory.strip())
    if output_directory:
        if "\0" in output_directory:
            raise ValueError("FL Video Combine output_directory contains an invalid null character.")
        if not os.path.isabs(output_directory):
            raise ValueError("FL Video Combine output_directory must be an absolute path.")
        output_directory = os.path.normpath(output_directory)

    for name in ("include_audio", "trim_video_to_audio", "save_output", "save_metadata"):
        if not isinstance(settings[name], bool):
            raise ValueError(f"FL Video Combine {name} must be true or false.")

    settings["frame_rate"] = float(frame_rate)
    settings["audio_gain_db"] = float(audio_gain_db)
    settings["output_directory"] = output_directory
    return settings


def _output_directory(settings):
    custom_directory = settings["output_directory"]
    if custom_directory:
        try:
            os.makedirs(custom_directory, exist_ok=True)
        except OSError as e:
            raise ValueError(f"FL Video Combine could not create output_directory: {e}") from e
        if not os.path.isdir(custom_directory):
            raise ValueError("FL Video Combine output_directory is not a directory.")
        return custom_directory, "custom"
    if settings["save_output"]:
        return folder_paths.get_output_directory(), "output"
    return folder_paths.get_temp_directory(), "temp"


def _prepare_images(images):
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] == 0:
        raise ValueError("FL Video Combine requires at least one image frame.")
    if images.shape[-1] not in (3, 4):
        raise ValueError("FL Video Combine supports RGB and RGBA image batches.")

    source_height = int(images.shape[1])
    source_width = int(images.shape[2])
    if images.shape[-1] == 4:
        images = images[..., :3]

    pad_width = source_width % 2
    pad_height = source_height % 2
    if pad_width or pad_height:
        images = images.movedim(-1, 1)
        images = F.pad(images, (0, pad_width, 0, pad_height), mode="replicate")
        images = images.movedim(1, -1)

    return images, source_width, source_height


def _prepare_audio(audio, include_audio, audio_gain_db):
    if audio is None or not include_audio:
        return None

    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError("FL Video Combine audio must have a [batch, channels, samples] waveform.")
    if waveform.shape[1] not in (1, 2, 6):
        raise ValueError("FL Video Combine supports mono, stereo, or 5.1 audio.")
    if audio_gain_db == 0:
        return audio

    gain = 10 ** (audio_gain_db / 20)
    return {
        "waveform": waveform * gain,
        "sample_rate": audio["sample_rate"],
    }


def _trim_images_to_audio(images, audio, frame_rate, enabled):
    if audio is None or not enabled:
        return images

    audio_frame_count = max(1, round(audio["waveform"].shape[-1] / audio["sample_rate"] * frame_rate))
    if audio_frame_count >= images.shape[0]:
        return images
    return images[:audio_frame_count]


def _build_metadata(prompt, extra_pnginfo, enabled):
    if not enabled or args.disable_metadata:
        return None

    metadata = {}
    if extra_pnginfo is not None:
        metadata.update(extra_pnginfo)
    if prompt is not None:
        metadata["prompt"] = prompt
    return metadata or None


class FL_VideoCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "render_settings": ("STRING", {"default": DEFAULT_SETTINGS_JSON, "multiline": False}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "combine_video"
    OUTPUT_NODE = True
    CATEGORY = "🏵️Fill Nodes/Video"
    DESCRIPTION = "Combines an image batch and optional audio into an MP4 video."

    def combine_video(self, images, render_settings=DEFAULT_SETTINGS_JSON, audio=None, prompt=None, extra_pnginfo=None):
        settings = _parse_settings(render_settings)
        images, source_width, source_height = _prepare_images(images)
        encoded_height = int(images.shape[1])
        encoded_width = int(images.shape[2])
        prepared_audio = _prepare_audio(audio, settings["include_audio"], settings["audio_gain_db"])
        images = _trim_images_to_audio(
            images,
            prepared_audio,
            settings["frame_rate"],
            settings["trim_video_to_audio"],
        )

        output_dir, output_type = _output_directory(settings)
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            settings["filename_prefix"],
            output_dir,
            encoded_width,
            encoded_height,
        )
        file = f"{filename}_{counter:05}_.mp4"
        output_path = os.path.join(full_output_folder, file)

        video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=images,
                audio=prepared_audio,
                frame_rate=Fraction(round(settings["frame_rate"] * 1000), 1000),
            ),
            bit_depth=settings["bit_depth"],
        )
        video.save_to(
            output_path,
            format=Types.VideoContainer.MP4,
            codec=Types.VideoCodec.H264,
            metadata=_build_metadata(prompt, extra_pnginfo, settings["save_metadata"]),
            crf=settings["crf"],
        )

        frame_count = int(images.shape[0])
        preview = {
            "filename": file,
            "subfolder": subfolder,
            "type": output_type,
            "frame_count": frame_count,
            "frame_rate": settings["frame_rate"],
            "duration": frame_count / settings["frame_rate"],
            "source_width": source_width,
            "source_height": source_height,
            "encoded_width": encoded_width,
            "encoded_height": encoded_height,
            "has_audio": prepared_audio is not None,
            "bit_depth": settings["bit_depth"],
        }
        if output_type == "custom":
            token = register_preview_file(output_path)
            preview["preview_url"] = f"/fl/video-combine/preview/{token}"
        return {
            "ui": {"fl_video_combine": [preview]},
            "result": (output_path,),
        }
