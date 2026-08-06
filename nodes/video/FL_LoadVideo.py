import json
import math
import os
from fractions import Fraction
from pathlib import Path

import av
import psutil
import torch

import folder_paths
from comfy.utils import common_upscale
from comfy_api.latest import InputImpl, Types


VIDEO_EXTENSIONS = {".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
DEFAULT_LOAD_SETTINGS = {
    "version": 1,
    "start_time": 0.0,
    "end_time": 0.0,
    "sample_mode": "source",
    "target_fps": 24.0,
    "select_every_nth": 1,
    "frame_load_cap": 0,
    "resize_mode": "original",
    "width": 0,
    "height": 0,
    "include_audio": True,
}
DEFAULT_SETTINGS_JSON = json.dumps(DEFAULT_LOAD_SETTINGS, separators=(",", ":"))


def available_video_files():
    input_dir = Path(folder_paths.get_input_directory()).resolve()
    os.makedirs(input_dir, exist_ok=True)
    files, _ = folder_paths.recursive_search(str(input_dir))
    result = []
    for filename in files:
        if Path(filename).suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        path = (input_dir / filename).resolve()
        try:
            path.relative_to(input_dir)
        except ValueError:
            continue
        if path.is_file():
            result.append(Path(filename).as_posix())
    return sorted(result, key=str.casefold)


def resolve_video_path(filename):
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Choose a video file.")

    filename = filename.strip()
    name, annotated_dir = folder_paths.annotated_filepath(filename)
    input_dir = Path(folder_paths.get_input_directory()).resolve()
    if annotated_dir is not None and Path(annotated_dir).resolve() != input_dir:
        raise ValueError("FL Load Video files must be inside the ComfyUI input directory.")

    try:
        path = Path(folder_paths.get_annotated_filepath(name, str(input_dir))).resolve()
    except ValueError as e:
        raise ValueError("FL Load Video files must be inside the ComfyUI input directory.") from e
    try:
        path.relative_to(input_dir)
    except ValueError as e:
        raise ValueError("FL Load Video files must be inside the ComfyUI input directory.") from e
    if not path.is_file():
        raise ValueError(f"Video file does not exist: {filename}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video format: {path.suffix or filename}")
    return path


def _parse_settings(load_settings):
    try:
        configured = json.loads(load_settings)
    except (TypeError, json.JSONDecodeError) as e:
        raise ValueError("FL Load Video settings are not valid JSON.") from e
    if not isinstance(configured, dict):
        raise ValueError("FL Load Video settings must be a JSON object.")

    settings = DEFAULT_LOAD_SETTINGS.copy()
    settings.update(configured)

    version = settings["version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        raise ValueError(f"FL Load Video settings version {version} is unsupported.")

    for name in ("start_time", "end_time", "target_fps"):
        value = settings[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"FL Load Video {name} must be a finite number.")

    start_time = float(settings["start_time"])
    end_time = float(settings["end_time"])
    target_fps = float(settings["target_fps"])
    if start_time < 0:
        raise ValueError("FL Load Video start_time cannot be negative.")
    if end_time < 0:
        raise ValueError("FL Load Video end_time cannot be negative.")
    if end_time and end_time <= start_time:
        raise ValueError("FL Load Video end_time must be greater than start_time.")
    if not 1 <= target_fps <= 120:
        raise ValueError("FL Load Video target_fps must be between 1 and 120.")

    sample_mode = settings["sample_mode"]
    if sample_mode not in ("source", "target_fps", "every_nth"):
        raise ValueError("FL Load Video sample_mode must be source, target_fps, or every_nth.")

    for name in ("select_every_nth", "frame_load_cap", "width", "height"):
        value = settings[name]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"FL Load Video {name} must be an integer.")

    if settings["select_every_nth"] < 1:
        raise ValueError("FL Load Video select_every_nth must be at least 1.")
    if settings["frame_load_cap"] < 0:
        raise ValueError("FL Load Video frame_load_cap cannot be negative.")
    if not 0 <= settings["width"] <= 16384 or not 0 <= settings["height"] <= 16384:
        raise ValueError("FL Load Video width and height must be between 0 and 16384.")

    resize_mode = settings["resize_mode"]
    if resize_mode not in ("original", "fit", "crop"):
        raise ValueError("FL Load Video resize_mode must be original, fit, or crop.")
    if resize_mode == "fit" and settings["width"] == 0 and settings["height"] == 0:
        raise ValueError("FL Load Video fit resize requires a width or height.")
    if resize_mode == "crop" and (settings["width"] == 0 or settings["height"] == 0):
        raise ValueError("FL Load Video crop resize requires both width and height.")
    if not isinstance(settings["include_audio"], bool):
        raise ValueError("FL Load Video include_audio must be true or false.")

    settings["start_time"] = start_time
    settings["end_time"] = end_time
    settings["target_fps"] = target_fps
    return settings


def probe_video(path):
    path = Path(path)
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in file: {path.name}")
        stream = container.streams.video[0]
        frame_rate = float(stream.average_rate) if stream.average_rate else 1.0
        if container.duration is not None:
            duration = float(container.duration / av.time_base)
        elif stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif stream.frames and frame_rate:
            duration = float(stream.frames / frame_rate)
        else:
            duration = 0.0

        frame_count_estimated = not bool(stream.frames)
        frame_count = int(stream.frames) if stream.frames else int(round(duration * frame_rate))
        codec = stream.codec_context.name if stream.codec_context is not None else ""
        container_format = container.format.name or ""
        width = int(stream.width)
        height = int(stream.height)
        has_audio = bool(container.streams.audio)

    bit_depth = int(InputImpl.VideoFromFile(str(path)).get_bit_depth())
    return {
        "width": width,
        "height": height,
        "duration": duration,
        "frame_rate": frame_rate,
        "frame_count": frame_count,
        "frame_count_estimated": frame_count_estimated,
        "bit_depth": bit_depth,
        "codec": codec,
        "container": container_format,
        "has_audio": has_audio,
        "size": path.stat().st_size,
    }


def _target_dimensions(source_width, source_height, settings):
    mode = settings["resize_mode"]
    if mode == "original":
        return source_width, source_height
    if mode == "crop":
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


def build_load_plan(probe, settings):
    source_duration = float(probe["duration"])
    start_time = settings["start_time"]
    if start_time >= source_duration:
        raise ValueError("FL Load Video start_time is beyond the end of the video.")
    end_time = min(settings["end_time"] or source_duration, source_duration)
    selected_duration = end_time - start_time
    source_fps = float(probe["frame_rate"])

    if settings["sample_mode"] == "target_fps":
        effective_fps = settings["target_fps"]
    elif settings["sample_mode"] == "every_nth":
        effective_fps = source_fps / settings["select_every_nth"]
    else:
        effective_fps = source_fps

    estimated_output_frames = max(1, math.ceil(selected_duration * effective_fps))
    frame_cap = settings["frame_load_cap"]
    if frame_cap:
        estimated_output_frames = min(estimated_output_frames, frame_cap)
        decode_duration = min(selected_duration, frame_cap / effective_fps)
    else:
        decode_duration = selected_duration

    output_width, output_height = _target_dimensions(probe["width"], probe["height"], settings)
    estimated_source_frames = max(1, math.ceil(decode_duration * source_fps))
    source_bytes = estimated_source_frames * probe["width"] * probe["height"] * 3 * 4
    output_bytes = estimated_output_frames * output_width * output_height * 3 * 4
    estimated_peak_bytes = source_bytes + output_bytes + min(source_bytes, output_bytes)
    return {
        "start_time": start_time,
        "end_time": end_time,
        "selected_duration": selected_duration,
        "decode_duration": decode_duration,
        "effective_fps": effective_fps,
        "estimated_source_frames": estimated_source_frames,
        "estimated_output_frames": estimated_output_frames,
        "output_width": output_width,
        "output_height": output_height,
        "estimated_peak_bytes": estimated_peak_bytes,
    }


def _check_memory(plan):
    available = psutil.virtual_memory().available
    headroom = min(1024 * 1024 * 1024, available // 4)
    if plan["estimated_peak_bytes"] > available - headroom:
        required_gb = plan["estimated_peak_bytes"] / (1024 ** 3)
        raise RuntimeError(
            f"FL Load Video requires approximately {required_gb:.1f} GB of RAM. "
            "Reduce the range, frame count, FPS, or resolution."
        )


def _sample_images(images, source_fps, settings):
    source_count = int(images.shape[0])
    if source_count == 0:
        raise RuntimeError("FL Load Video decoded no frames.")

    if settings["sample_mode"] == "target_fps":
        effective_fps = settings["target_fps"]
        output_count = max(1, math.ceil((source_count / source_fps) * effective_fps))
        indices = [
            min(source_count - 1, round(index * source_fps / effective_fps))
            for index in range(output_count)
        ]
    elif settings["sample_mode"] == "every_nth":
        nth = settings["select_every_nth"]
        effective_fps = source_fps / nth
        indices = list(range(0, source_count, nth))
    else:
        effective_fps = source_fps
        indices = list(range(source_count))

    frame_cap = settings["frame_load_cap"]
    if frame_cap:
        indices = indices[:frame_cap]
    if not indices:
        raise RuntimeError("FL Load Video settings selected no frames.")

    if len(indices) == source_count and all(index == value for index, value in enumerate(indices)):
        return images, effective_fps
    return images[indices], effective_fps


def _trim_audio(audio, duration, include_audio):
    if audio is None or not include_audio:
        return None
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    samples = min(int(waveform.shape[-1]), math.ceil(duration * sample_rate))
    if samples == waveform.shape[-1]:
        return audio
    return {
        "waveform": waveform[..., :samples].clone(),
        "sample_rate": sample_rate,
    }


def _resize_images(images, settings):
    source_height = int(images.shape[1])
    source_width = int(images.shape[2])
    width, height = _target_dimensions(source_width, source_height, settings)
    if (width, height) == (source_width, source_height):
        return images

    crop = "center" if settings["resize_mode"] == "crop" else "disabled"
    images = common_upscale(images.movedim(-1, 1), width, height, "lanczos", crop)
    return images.movedim(1, -1)


def _preview_reference(filename):
    name, _ = folder_paths.annotated_filepath(filename)
    normalized = Path(name)
    return normalized.name, normalized.parent.as_posix() if normalized.parent != Path(".") else ""


class FL_LoadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        files = available_video_files()
        return {
            "required": {
                "video": ([""] + files,),
                "load_settings": ("STRING", {"default": DEFAULT_SETTINGS_JSON, "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "VIDEO", "FLOAT", "INT")
    RETURN_NAMES = ("images", "audio", "video", "fps", "frame_count")
    FUNCTION = "load_video"
    CATEGORY = "🏵️Fill Nodes/Video"
    DESCRIPTION = "Loads, previews, trims, samples, and resizes a video from ComfyUI input."

    def load_video(self, video, load_settings=DEFAULT_SETTINGS_JSON):
        settings = _parse_settings(load_settings)
        path = resolve_video_path(video)
        probe = probe_video(path)
        plan = build_load_plan(probe, settings)
        _check_memory(plan)

        source = InputImpl.VideoFromFile(
            str(path),
            start_time=plan["start_time"],
            duration=plan["decode_duration"],
        )
        components = source.get_components()
        images, effective_fps = _sample_images(components.images, float(components.frame_rate), settings)
        images = _resize_images(images, settings)
        frame_count = int(images.shape[0])
        loaded_duration = frame_count / effective_fps
        audio = _trim_audio(components.audio, loaded_duration, settings["include_audio"])
        native_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=images,
                audio=audio,
                frame_rate=Fraction(round(effective_fps * 1000), 1000),
                metadata=components.metadata,
            ),
            bit_depth=probe["bit_depth"],
        )

        filename, subfolder = _preview_reference(video)
        preview = {
            "filename": filename,
            "subfolder": subfolder,
            "type": "input",
            "source_width": probe["width"],
            "source_height": probe["height"],
            "source_duration": probe["duration"],
            "source_fps": probe["frame_rate"],
            "source_frame_count": probe["frame_count"],
            "loaded_width": int(images.shape[2]),
            "loaded_height": int(images.shape[1]),
            "loaded_duration": loaded_duration,
            "loaded_fps": effective_fps,
            "loaded_frame_count": frame_count,
            "has_audio": audio is not None,
            "bit_depth": probe["bit_depth"],
        }
        return {
            "ui": {"fl_load_video": [preview]},
            "result": (images, audio, native_video, float(effective_fps), frame_count),
        }

    @classmethod
    def IS_CHANGED(cls, video, load_settings=DEFAULT_SETTINGS_JSON):
        path = resolve_video_path(video)
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    @classmethod
    def VALIDATE_INPUTS(cls, video, load_settings=DEFAULT_SETTINGS_JSON):
        try:
            resolve_video_path(video)
            _parse_settings(load_settings)
        except ValueError as e:
            return str(e)
        return True
