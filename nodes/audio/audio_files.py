import hashlib
import os
from pathlib import Path

import folder_paths
from comfy_extras.nodes_audio import load


def available_audio_files():
    input_dir = folder_paths.get_input_directory()
    os.makedirs(input_dir, exist_ok=True)
    return sorted(folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"]))


def resolve_audio_path(filename):
    if not filename:
        raise ValueError("Choose an audio file or connect beat_positions.")
    if not folder_paths.exists_annotated_filepath(filename):
        raise ValueError(f"Audio file does not exist: {filename}")

    path = Path(folder_paths.get_annotated_filepath(filename)).resolve()
    input_dir = Path(folder_paths.get_input_directory()).resolve()
    try:
        path.relative_to(input_dir)
    except ValueError as error:
        raise ValueError("Audio files must be inside the ComfyUI input directory.") from error
    if not path.is_file():
        raise ValueError(f"Audio file does not exist: {filename}")
    return path


def audio_file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as audio_file:
        for chunk in iter(lambda: audio_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_audio_file(filename):
    path = resolve_audio_path(filename)
    waveform, sample_rate = load(str(path))
    return path, {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
