import hashlib
import os
from pathlib import Path

import folder_paths
from comfy_extras.nodes_audio import load


def available_audio_files():
    input_dir = Path(folder_paths.get_input_directory()).resolve()
    os.makedirs(input_dir, exist_ok=True)
    files, _ = folder_paths.recursive_search(str(input_dir))
    media_files = folder_paths.filter_files_content_types(files, ["audio", "video"])
    result = []
    for filename in media_files:
        path = (input_dir / filename).resolve()
        try:
            path.relative_to(input_dir)
        except ValueError:
            continue
        if path.is_file():
            result.append(Path(filename).as_posix())
    return sorted(result, key=str.casefold)


def audio_library_entries():
    input_dir = Path(folder_paths.get_input_directory()).resolve()
    entries = []
    for filename in available_audio_files():
        path = (input_dir / filename).resolve()
        stat = path.stat()
        relative = Path(filename)
        entries.append({
            "path": relative.as_posix(),
            "folder": relative.parent.as_posix() if relative.parent != Path(".") else "",
            "size": stat.st_size,
            "modified": stat.st_mtime,
        })
    return entries


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
