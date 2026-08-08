import hashlib
import json
import math
from pathlib import Path

import librosa
import numpy as np
import torch

import folder_paths

from .audio_files import audio_file_hash, load_audio_file, resolve_audio_path
from .audio_separation import load_cached_stem
from .beat_this_detector import MODEL_FPS, MODEL_SHA256, analyze_beats as analyze_beat_this


ANALYSIS_VERSION = 7
DETECTOR_VERSION = f"beat-this-final0-{MODEL_SHA256[:12]}"
SOURCE_ANALYSIS_VERSION = 1
_WAVEFORM_BUCKETS_PER_SECOND = 60
_MAX_WAVEFORM_BUCKETS = 8192
_WAVEFORM_SCALE = 32767


def crop_audio(audio, fps, trim_start_frame=0, length_frames=0):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be greater than zero.")
    if trim_start_frame < 0 or trim_start_frame != round(trim_start_frame):
        raise ValueError("Trim start must be a whole frame count of zero or greater.")
    if length_frames < 0 or length_frames != round(length_frames):
        raise ValueError("Length must be a whole frame count of zero or greater.")

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError("Audio waveform must have shape [batch, channels, samples].")
    if waveform.shape[0] != 1:
        raise ValueError("Uploaded audio must contain one batch.")

    sample_count = waveform.shape[-1]
    source_duration = sample_count / sample_rate
    start_sample = round(trim_start_frame / fps * sample_rate)
    if start_sample >= sample_count:
        raise ValueError(
            f"Trim start frame {trim_start_frame} is outside the {source_duration:.3f}s audio file."
        )
    if length_frames:
        end_sample = start_sample + round(length_frames / fps * sample_rate)
        if end_sample > sample_count:
            available_frames = math.floor((sample_count - start_sample) / sample_rate * fps)
            raise ValueError(
                f"Length {length_frames} frames exceeds the {available_frames} frames available "
                "after the trim start."
            )
        resolved_frames = length_frames
    else:
        end_sample = sample_count
        resolved_frames = round((end_sample - start_sample) / sample_rate * fps)

    cropped = waveform[..., start_sample:end_sample].clone()
    duration = cropped.shape[-1] / sample_rate
    if duration <= 0:
        raise ValueError("The selected audio crop is empty.")
    return {
        "waveform": cropped,
        "sample_rate": sample_rate,
    }, {
        "source_duration": source_duration,
        "source_start": start_sample / sample_rate,
        "trim_start_frame": int(trim_start_frame),
        "length_frames": int(resolved_frames),
        "audio_duration": duration,
        "sample_rate": sample_rate,
    }


def mono_numpy(audio):
    waveform = audio["waveform"]
    waveform_np = waveform.detach().cpu().numpy() if isinstance(waveform, torch.Tensor) else np.asarray(waveform)
    if waveform_np.ndim == 3:
        if waveform_np.shape[0] != 1:
            raise ValueError("Audio analysis supports one batch.")
        waveform_np = waveform_np[0]
    if waveform_np.ndim == 2:
        waveform_np = waveform_np.mean(axis=0)
    if waveform_np.ndim != 1:
        raise ValueError("Audio analysis expects mono or channel-first audio.")
    return waveform_np.astype(np.float32, copy=False)


def waveform_preview(waveform, sample_rate):
    sample_count = len(waveform)
    duration = sample_count / sample_rate
    bucket_count = min(
        sample_count,
        _MAX_WAVEFORM_BUCKETS,
        max(1, math.ceil(duration * _WAVEFORM_BUCKETS_PER_SECOND)),
    )
    starts = np.linspace(0, sample_count, bucket_count + 1, dtype=np.int64)[:-1]
    minimums = np.minimum.reduceat(waveform, starts)
    maximums = np.maximum.reduceat(waveform, starts)
    peak = max(float(np.abs(minimums).max()), float(np.abs(maximums).max()))
    if peak > 0:
        minimums = minimums / peak
        maximums = maximums / peak

    peaks = np.empty(bucket_count * 2, dtype=np.int16)
    peaks[0::2] = np.rint(minimums * _WAVEFORM_SCALE).astype(np.int16)
    peaks[1::2] = np.rint(maximums * _WAVEFORM_SCALE).astype(np.int16)
    return {
        "version": 1,
        "duration": float(duration),
        "scale": _WAVEFORM_SCALE,
        "peaks": peaks.tolist(),
    }


def crop_waveform_preview(preview, start, duration):
    if not isinstance(preview, dict) or preview.get("version") != 1 or duration <= 0:
        return None
    peaks = preview.get("peaks")
    source_duration = preview.get("duration")
    if not isinstance(peaks, list) or len(peaks) < 2 or len(peaks) % 2 or not source_duration:
        return None
    bucket_count = len(peaks) // 2
    start_bucket = min(bucket_count - 1, max(0, math.floor(start / source_duration * bucket_count)))
    end_bucket = min(
        bucket_count,
        max(start_bucket + 1, math.ceil((start + duration) / source_duration * bucket_count)),
    )
    return {
        "version": 1,
        "duration": float(duration),
        "scale": preview["scale"],
        "peaks": peaks[start_bucket * 2:end_bucket * 2],
    }


def _detect_drums(waveform, sample_rate, onset_frames, onset_times):
    stft_mag = np.abs(librosa.stft(waveform))
    frequencies = librosa.fft_frequencies(sr=sample_rate)
    kick_low = (frequencies >= 30) & (frequencies <= 120)
    kick_mid = (frequencies >= 120) & (frequencies <= 300)
    snare_fundamental = (frequencies >= 150) & (frequencies <= 400)
    snare_noise = (frequencies >= 4000) & (frequencies <= 10000)
    hihat = frequencies >= 6000
    low_reject = frequencies <= 200
    kick_times = []
    snare_times = []
    hihat_times = []

    for onset_time, onset_frame in zip(onset_times, onset_frames):
        if onset_frame >= stft_mag.shape[1]:
            continue
        spectrum = stft_mag[:, onset_frame]
        total_energy = np.sum(spectrum) + 1e-10
        kick_low_energy = np.sum(spectrum[kick_low])
        kick_mid_energy = np.sum(spectrum[kick_mid])
        snare_fundamental_energy = np.sum(spectrum[snare_fundamental])
        snare_noise_energy = np.sum(spectrum[snare_noise])
        hihat_energy = np.sum(spectrum[hihat])
        low_energy = np.sum(spectrum[low_reject])
        kick_ratio = (kick_low_energy + kick_mid_energy) / total_energy
        snare_ratio = (snare_fundamental_energy + snare_noise_energy) / total_energy
        hihat_ratio = hihat_energy / total_energy
        low_ratio = low_energy / total_energy

        if kick_low_energy > 0 and kick_ratio > 0.15 and low_ratio > 0.25 and hihat_ratio < 0.4:
            kick_times.append(float(onset_time))
        elif (
            snare_fundamental_energy > 0
            and snare_noise_energy > 0
            and snare_ratio > 0.15
            and low_ratio < 0.6
        ):
            snare_times.append(float(onset_time))
        elif hihat_energy > 0 and hihat_ratio > 0.2 and low_ratio < 0.2:
            hihat_times.append(float(onset_time))

    return {
        "kick_times": kick_times,
        "snare_times": snare_times,
        "hihat_times": hihat_times,
        "sample_rate": int(sample_rate),
        "duration": float(len(waveform) / sample_rate),
        "total_kicks": len(kick_times),
        "total_snares": len(snare_times),
        "total_hihats": len(hihat_times),
    }


def apply_beat_offset(analysis, fps, beat_offset_ms=0, beat_grid_density="every_beat"):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be greater than zero.")
    if not math.isfinite(beat_offset_ms):
        raise ValueError("Beat offset must be a finite number.")
    if beat_grid_density not in {"every_2_beats", "every_beat", "half_beat"}:
        raise ValueError(f"Unknown beat grid density: {beat_grid_density}")

    duration = float(analysis["audio_duration"])
    offset = beat_offset_ms / 1000.0
    base_beat_times = list(
        analysis.get("base_beat_times", analysis.get("beat_times", []))
    )
    if not base_beat_times:
        raise ValueError("Beat analysis must contain at least one beat.")
    base_detected_beat_times = analysis.get(
        "base_detected_beat_times",
        analysis.get("detected_beat_times", []),
    )
    base_downbeat_times = list(
        analysis.get("base_downbeat_times", analysis.get("downbeat_times", []))
    )
    base_detected_downbeat_times = analysis.get(
        "base_detected_downbeat_times",
        analysis.get("detected_downbeat_times", base_downbeat_times),
    )
    downbeat_indices = set()
    for downbeat in base_downbeat_times:
        index = int(np.argmin(np.abs(np.asarray(base_beat_times) - downbeat)))
        if abs(base_beat_times[index] - downbeat) <= 1.0 / MODEL_FPS:
            downbeat_indices.add(index)

    base_grid_interval = float(analysis.get("base_grid_interval_seconds", 0.0))
    if base_grid_interval <= 0 and len(base_beat_times) > 1:
        base_grid_interval = float(np.median(np.diff(base_beat_times)))
    if base_grid_interval <= 0:
        bpm = float(analysis.get("bpm", 0.0))
        base_grid_interval = 60.0 / bpm if bpm > 0 else 0.0
    if not math.isfinite(base_grid_interval) or base_grid_interval <= 0:
        raise ValueError("Beat analysis must provide a valid beat interval or BPM.")

    if beat_grid_density == "every_2_beats":
        base_grid = [
            (base_beat_times[index], index in downbeat_indices)
            for index in range(0, len(base_beat_times), 2)
        ]
        grid_interval = base_grid_interval * 2.0
    elif beat_grid_density == "half_beat":
        base_grid = []
        for index, beat_time in enumerate(base_beat_times):
            base_grid.append((beat_time, index in downbeat_indices))
            if index + 1 < len(base_beat_times):
                base_grid.append(((beat_time + base_beat_times[index + 1]) / 2.0, False))
        grid_interval = base_grid_interval / 2.0
    else:
        base_grid = [
            (beat_time, index in downbeat_indices)
            for index, beat_time in enumerate(base_beat_times)
        ]
        grid_interval = base_grid_interval

    shifted_grid = [(beat_time + offset, is_downbeat) for beat_time, is_downbeat in base_grid]
    grid = [
        (beat_time, is_downbeat)
        for beat_time, is_downbeat in shifted_grid
        if 0.0 <= beat_time < duration
    ]
    if offset > 0:
        beat_time = shifted_grid[0][0] - grid_interval
        while beat_time >= 0:
            if beat_time < duration:
                grid.insert(0, (beat_time, False))
            beat_time -= grid_interval
    elif offset < 0:
        beat_time = shifted_grid[-1][0] + grid_interval
        while beat_time < duration:
            if beat_time >= 0:
                grid.append((beat_time, False))
            beat_time += grid_interval
    grid_beat_times = [beat_time for beat_time, _ in grid]
    grid_downbeat_times = [beat_time for beat_time, is_downbeat in grid if is_downbeat]

    result = dict(analysis)
    result["version"] = ANALYSIS_VERSION
    result["base_beat_times"] = base_beat_times
    result["base_detected_beat_times"] = list(base_detected_beat_times)
    result["base_downbeat_times"] = base_downbeat_times
    result["base_detected_downbeat_times"] = list(base_detected_downbeat_times)
    result["base_grid_interval_seconds"] = base_grid_interval
    result["beat_times"] = grid_beat_times
    result["downbeat_times"] = grid_downbeat_times
    result["detected_beat_times"] = list(base_detected_beat_times)
    result["detected_downbeat_times"] = list(base_detected_downbeat_times)
    result["beat_frames"] = [round(value * fps) for value in result["beat_times"]]
    result["downbeat_frames"] = [
        round(value * fps) for value in result["downbeat_times"]
    ]
    result["detected_beat_frames"] = [
        round(value * fps) for value in result["detected_beat_times"]
    ]
    result["num_beats"] = len(result["beat_times"])
    result["beat_offset_ms"] = int(round(beat_offset_ms))
    result["beat_grid_density"] = beat_grid_density
    result["grid_interval_seconds"] = grid_interval
    result["grid_bpm"] = 60.0 / grid_interval
    return result


def analyze_audio(audio, detect_beats=True, beat_audio=None):
    waveform = mono_numpy(audio)
    sample_rate = int(audio["sample_rate"])
    duration = len(waveform) / sample_rate

    onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        units="frames",
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
    drum_times = _detect_drums(waveform, sample_rate, onset_frames, onset_times)
    onsets = onset_times.tolist()

    analysis = {
        "version": ANALYSIS_VERSION,
        "detector_version": DETECTOR_VERSION,
        "onset_times": onsets,
        "sample_rate": sample_rate,
        "audio_duration": float(duration),
        "drum_times": drum_times,
        "waveform_preview": waveform_preview(waveform, sample_rate),
    }
    if not detect_beats:
        return analysis

    beat_waveform = mono_numpy(beat_audio) if beat_audio is not None else waveform
    beat_sample_rate = int(beat_audio["sample_rate"]) if beat_audio is not None else sample_rate
    detected = analyze_beat_this(beat_waveform, beat_sample_rate)
    beat_times = np.asarray(detected["beat_times"], dtype=np.float64)
    downbeat_times = np.asarray(detected["downbeat_times"], dtype=np.float64)
    beat_confidences = np.asarray(detected["beat_confidences"], dtype=np.float64)
    downbeat_confidences = np.asarray(
        detected["downbeat_confidences"], dtype=np.float64
    )
    if len(beat_times) < 2:
        raise ValueError("Beat This must detect at least two beats in the source audio.")

    interval = float(np.median(np.diff(beat_times)))
    analysis.update({
        "bpm": 60.0 / interval,
        "bpm_source": "beat_this_intervals_median",
        "base_grid_interval_seconds": interval,
        "beat_times": beat_times.tolist(),
        "downbeat_times": downbeat_times.tolist(),
        "detected_beat_times": beat_times.tolist(),
        "detected_downbeat_times": downbeat_times.tolist(),
        "base_detected_beat_confidences": beat_confidences.tolist(),
        "detected_beat_confidences": beat_confidences.tolist(),
        "base_detected_downbeat_confidences": downbeat_confidences.tolist(),
        "detected_downbeat_confidences": downbeat_confidences.tolist(),
        "num_beats": len(beat_times),
        "num_downbeats": len(downbeat_times),
        "detector": detected["detector"],
    })
    return analysis


def apply_half_time(analysis, half_time):
    result = dict(analysis)
    if not half_time or "beat_times" not in result:
        return result

    beat_times = list(result["beat_times"])
    beat_confidences = list(result.get("detected_beat_confidences", []))
    retained = list(range(0, len(beat_times), 2))
    if len(retained) < 2:
        raise ValueError("Beat This must detect at least two beats after half-time filtering.")
    retained_indices = set(retained)
    downbeat_times = list(result.get("downbeat_times", []))
    downbeat_confidences = list(result.get("detected_downbeat_confidences", []))
    retained_downbeats = []
    retained_downbeat_confidences = []
    for index, downbeat in enumerate(downbeat_times):
        nearest = min(range(len(beat_times)), key=lambda position: abs(beat_times[position] - downbeat))
        if nearest in retained_indices:
            retained_downbeats.append(downbeat)
            if index < len(downbeat_confidences):
                retained_downbeat_confidences.append(downbeat_confidences[index])

    retained_beats = [beat_times[index] for index in retained]
    retained_beat_confidences = [
        beat_confidences[index] for index in retained if index < len(beat_confidences)
    ]
    interval = float(np.median(np.diff(retained_beats)))
    result.update({
        "bpm": 60.0 / interval,
        "base_grid_interval_seconds": interval,
        "beat_times": retained_beats,
        "downbeat_times": retained_downbeats,
        "detected_beat_times": retained_beats,
        "detected_downbeat_times": retained_downbeats,
        "base_detected_beat_confidences": retained_beat_confidences,
        "detected_beat_confidences": retained_beat_confidences,
        "base_detected_downbeat_confidences": retained_downbeat_confidences,
        "detected_downbeat_confidences": retained_downbeat_confidences,
        "num_beats": len(retained_beats),
        "num_downbeats": len(retained_downbeats),
    })
    return result


def _crop_times(values, start, end):
    return [float(value - start) for value in values if start <= value < end]


def _crop_times_with_values(times, values, start, end):
    cropped_times = []
    cropped_values = []
    for index, value in enumerate(times):
        if start <= value < end:
            cropped_times.append(float(value - start))
            if index < len(values):
                cropped_values.append(values[index])
    return cropped_times, cropped_values


def project_analysis(analysis, crop, fps):
    start = float(crop["source_start"])
    duration = float(crop["audio_duration"])
    end = start + duration
    result = dict(analysis)

    for time_key, confidence_key in (
        ("base_detected_beat_times", "base_detected_beat_confidences"),
        ("detected_beat_times", "detected_beat_confidences"),
        ("base_detected_downbeat_times", "base_detected_downbeat_confidences"),
        ("detected_downbeat_times", "detected_downbeat_confidences"),
    ):
        times, confidences = _crop_times_with_values(
            analysis.get(time_key, []),
            analysis.get(confidence_key, []),
            start,
            end,
        )
        result[time_key] = times
        result[confidence_key] = confidences

    for key in (
        "base_beat_times",
        "beat_times",
        "base_downbeat_times",
        "downbeat_times",
        "onset_times",
    ):
        if key in analysis:
            result[key] = _crop_times(analysis[key], start, end)

    drums = dict(analysis.get("drum_times", {}))
    for key in ("kick_times", "snare_times", "hihat_times"):
        drums[key] = _crop_times(drums.get(key, []), start, end)
    drums.update({
        "duration": duration,
        "total_kicks": len(drums["kick_times"]),
        "total_snares": len(drums["snare_times"]),
        "total_hihats": len(drums["hihat_times"]),
    })

    result.update(crop)
    result.update({
        "beat_frames": [round(value * fps) for value in result.get("beat_times", [])],
        "downbeat_frames": [round(value * fps) for value in result.get("downbeat_times", [])],
        "detected_beat_frames": [
            round(value * fps) for value in result.get("detected_beat_times", [])
        ],
        "onset_frames": [round(value * fps) for value in result.get("onset_times", [])],
        "drum_times": drums,
        "num_beats": len(result.get("beat_times", [])),
        "num_downbeats": len(result.get("downbeat_times", [])),
        "waveform_preview": crop_waveform_preview(
            analysis.get("waveform_preview"),
            start,
            duration,
        ),
    })
    return result


def analysis_cache_key(
    path,
    analysis_source,
    detect_beats=True,
):
    values = {
        "analysis_version": ANALYSIS_VERSION,
        "audio_sha256": audio_file_hash(path),
        "analysis_source": analysis_source,
        "detect_beats": bool(detect_beats),
        "detector_version": DETECTOR_VERSION,
    }
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(cache_key):
    directory = Path(folder_paths.get_user_directory()) / "fl_audio_prompt_timeline" / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{cache_key}.json"


def cached_analysis_audio_file(cache_key):
    if (
        not isinstance(cache_key, str)
        or len(cache_key) != 64
        or any(character not in "0123456789abcdef" for character in cache_key.lower())
    ):
        return ""
    cache_path = _cache_path(cache_key)
    if not cache_path.is_file():
        return ""
    try:
        analysis = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    filename = analysis.get("audio_file") if isinstance(analysis, dict) else None
    if not isinstance(filename, str) or not filename:
        return ""
    try:
        resolve_audio_path(filename)
    except ValueError:
        return ""
    return filename


def analyze_audio_file(
    filename,
    fps,
    trim_start_frame=0,
    length_frames=0,
    half_time=False,
    beat_offset_ms=0,
    analysis_source="mix",
    beat_grid_density="every_beat",
    detect_beats=True,
):
    path = resolve_audio_path(filename)
    _, master_audio = load_audio_file(filename)
    cropped_audio, crop = crop_audio(master_audio, fps, trim_start_frame, length_frames)
    cache_key = analysis_cache_key(
        path,
        analysis_source,
        detect_beats,
    )
    cache_path = _cache_path(cache_key)
    cache_hit = cache_path.is_file()
    if cache_hit:
        source_analysis = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        analysis_audio = (
            master_audio
            if analysis_source == "mix"
            else load_cached_stem(filename, analysis_source)
        )
        source_analysis = analyze_audio(
            analysis_audio,
            detect_beats=detect_beats,
            beat_audio=master_audio if analysis_source != "mix" and detect_beats else None,
        )
        source_analysis.update({
            "audio_file": filename,
            "analysis_source": analysis_source,
            "beat_analysis_source": "mix" if detect_beats else None,
            "cache_key": cache_key,
            "audio_duration": crop["source_duration"],
            "source_duration": crop["source_duration"],
            "source_start": 0.0,
        })
        temporary_path = cache_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(source_analysis, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary_path.replace(cache_path)
    if detect_beats:
        analysis = apply_beat_offset(
            apply_half_time(source_analysis, half_time),
            fps,
            beat_offset_ms,
            beat_grid_density,
        )
    else:
        analysis = dict(source_analysis)
    analysis = project_analysis(analysis, crop, fps)
    analysis["source_analysis"] = {
        **source_analysis,
        "type": "fl_audio_source_analysis",
        "version": SOURCE_ANALYSIS_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "analysis_cache_hit": cache_hit,
    }
    analysis["analysis_cache_hit"] = cache_hit
    return analysis, cropped_audio
