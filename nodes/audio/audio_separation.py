import json
import math
from pathlib import Path

import soundfile
import torch
import torchaudio

import comfy.model_management
import folder_paths
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

from .audio_files import audio_file_hash, load_audio_file, resolve_audio_path


SEPARATION_MODEL = "hdemucs_high_musdb_plus"
STEM_NAMES = ("drums", "bass", "other", "vocals")


class SeparationCancelled(Exception):
    pass


def stem_cache_directory(filename):
    path = resolve_audio_path(filename)
    audio_hash = audio_file_hash(path)
    return (
        Path(folder_paths.get_user_directory())
        / "fl_audio_prompt_timeline"
        / "stems"
        / audio_hash
        / SEPARATION_MODEL
    )


def separation_manifest(filename):
    cache_dir = stem_cache_directory(filename)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("model") != SEPARATION_MODEL
        or manifest.get("stems") != list(STEM_NAMES)
        or any(not (cache_dir / f"{stem}.flac").is_file() for stem in STEM_NAMES)
    ):
        return None
    return manifest


def load_cached_stem(filename, stem):
    if stem not in STEM_NAMES:
        raise ValueError(f"Unknown analysis stem: {stem}")
    manifest = separation_manifest(filename)
    if manifest is None:
        raise ValueError(
            f"The {stem} stem has not been separated yet. Click Separate stems first."
        )
    path = stem_cache_directory(filename) / f"{stem}.flac"
    waveform, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    return {
        "waveform": torch.from_numpy(waveform.transpose()).unsqueeze(0),
        "sample_rate": sample_rate,
    }


def _ensure_stereo(waveform):
    if waveform.shape[0] == 2:
        return waveform
    if waveform.shape[0] == 1:
        return waveform.repeat(2, 1)
    return waveform[:2]


def _chunk_weights(length, overlap, first, last, dtype):
    weights = torch.ones(length, dtype=dtype)
    fade = min(overlap, length // 2)
    if fade and not first:
        weights[:fade] = torch.linspace(0.0, 1.0, fade, dtype=dtype)
    if fade and not last:
        weights[-fade:] = torch.linspace(1.0, 0.0, fade, dtype=dtype)
    return weights


def separate_audio_file(filename, progress=None, cancel_event=None, chunk_seconds=10.0, overlap_seconds=0.2):
    cached = separation_manifest(filename)
    if cached is not None:
        return cached

    path, audio = load_audio_file(filename)
    waveform = _ensure_stereo(audio["waveform"][0].float())
    input_sample_rate = int(audio["sample_rate"])
    sample_rate = HDEMUCS_HIGH_MUSDB_PLUS.sample_rate
    if input_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, input_sample_rate, sample_rate)

    reference = waveform.mean(0)
    reference_std = reference.std()
    if not torch.isfinite(reference_std) or reference_std <= 0:
        raise ValueError("Stem separation requires audio with a non-zero signal.")
    normalized = (waveform - reference.mean()) / reference_std
    length = normalized.shape[-1]
    chunk_length = max(1, round(chunk_seconds * sample_rate))
    overlap = max(0, round(overlap_seconds * sample_rate))
    step = max(1, chunk_length - overlap)
    chunk_count = max(1, math.ceil(max(0, length - overlap) / step))
    output = torch.zeros(len(STEM_NAMES), 2, length, dtype=normalized.dtype)
    weight_sum = torch.zeros(length, dtype=normalized.dtype)
    device = comfy.model_management.get_torch_device()
    model = None

    if progress:
        progress(0.02, "Loading Hybrid Demucs")
    try:
        model = HDEMUCS_HIGH_MUSDB_PLUS.get_model().eval().to(device)
        model_sources = list(model.sources)
        if set(model_sources) != set(STEM_NAMES):
            raise ValueError(f"Unexpected Hybrid Demucs stems: {model_sources}")
        source_indices = [model_sources.index(name) for name in STEM_NAMES]

        for index, start in enumerate(range(0, length, step)):
            if cancel_event is not None and cancel_event.is_set():
                raise SeparationCancelled("Stem separation cancelled.")
            end = min(length, start + chunk_length)
            chunk = normalized[:, start:end].unsqueeze(0).to(device)
            separated = model(chunk)[0, source_indices].detach().cpu()
            weights = _chunk_weights(
                end - start,
                overlap,
                first=start == 0,
                last=end == length,
                dtype=separated.dtype,
            )
            output[..., start:end].add_(separated * weights)
            weight_sum[start:end].add_(weights)
            if progress:
                progress(
                    0.05 + 0.85 * (index + 1) / chunk_count,
                    f"Separating chunk {index + 1}/{chunk_count}",
                )
            if end == length:
                break
    finally:
        if model is not None:
            model.to("cpu")
        comfy.model_management.soft_empty_cache()

    output.div_(weight_sum.clamp_min(1e-8))
    output.mul_(reference_std).add_(reference.mean())
    cache_dir = stem_cache_directory(filename)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = []
    try:
        for index, stem in enumerate(STEM_NAMES):
            if cancel_event is not None and cancel_event.is_set():
                raise SeparationCancelled("Stem separation cancelled.")
            temporary_path = cache_dir / f"{stem}.tmp.flac"
            soundfile.write(
                temporary_path,
                output[index].transpose(0, 1).numpy(),
                sample_rate,
                subtype="PCM_24",
            )
            temporary_paths.append(temporary_path)
        for stem, temporary_path in zip(STEM_NAMES, temporary_paths):
            temporary_path.replace(cache_dir / f"{stem}.flac")

        manifest = {
            "version": 1,
            "model": SEPARATION_MODEL,
            "audio_file": filename,
            "audio_sha256": audio_file_hash(path),
            "sample_rate": sample_rate,
            "duration": length / sample_rate,
            "stems": list(STEM_NAMES),
        }
        temporary_manifest = cache_dir / "manifest.tmp"
        temporary_manifest.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
        temporary_manifest.replace(cache_dir / "manifest.json")
        if progress:
            progress(1.0, "Stem separation complete")
        return manifest
    except Exception:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        raise
