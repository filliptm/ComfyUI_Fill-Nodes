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


ANALYSIS_VERSION = 3
DETECTOR_VERSION = "fl-audio-timeline-2"
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


def _tempo_value(tempo):
    if isinstance(tempo, np.ndarray):
        return float(tempo[0]) if tempo.size else 0.0
    return float(tempo)


def _regularize_beats(detected_beats, interval, duration):
    if len(detected_beats) < 2 or interval <= 0:
        return detected_beats

    beats = list(detected_beats)
    current = detected_beats[0] - interval
    while current > 0:
        beats.insert(0, current)
        current -= interval
    current = detected_beats[-1] + interval
    while current < duration:
        beats.append(current)
        current += interval

    ordered = np.sort(np.asarray(beats, dtype=np.float64))
    gaps = np.diff(ordered)
    if not np.any(gaps > interval * 1.5):
        return ordered

    filled = []
    for index, beat in enumerate(ordered[:-1]):
        filled.append(beat)
        gap = ordered[index + 1] - beat
        if gap > interval * 1.5:
            for step in range(1, int(gap / interval)):
                filled.append(beat + step * interval)
    filled.append(ordered[-1])
    return np.asarray(filled, dtype=np.float64)


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
    base_beat_times = analysis.get("base_beat_times", analysis.get("beat_times", []))
    base_detected_beat_times = analysis.get(
        "base_detected_beat_times",
        analysis.get("detected_beat_times", []),
    )

    if beat_grid_density == "every_2_beats":
        grid_beat_times = base_beat_times[::2]
        grid_bpm_scale = 0.5
    elif beat_grid_density == "half_beat":
        grid_beat_times = []
        for index, beat_time in enumerate(base_beat_times):
            grid_beat_times.append(beat_time)
            if index + 1 < len(base_beat_times):
                grid_beat_times.append((beat_time + base_beat_times[index + 1]) / 2.0)
        grid_bpm_scale = 2.0
    else:
        grid_beat_times = base_beat_times
        grid_bpm_scale = 1.0

    def shifted(values):
        if not values:
            return []
        times = np.asarray(values, dtype=np.float64)
        return np.unique(np.clip(times + offset, 0, duration)).tolist()

    result = dict(analysis)
    result["base_beat_times"] = list(base_beat_times)
    result["base_detected_beat_times"] = list(base_detected_beat_times)
    result["beat_times"] = shifted(grid_beat_times)
    result["detected_beat_times"] = shifted(base_detected_beat_times)
    result["beat_frames"] = [round(value * fps) for value in result["beat_times"]]
    result["detected_beat_frames"] = [
        round(value * fps) for value in result["detected_beat_times"]
    ]
    result["num_beats"] = len(result["beat_times"])
    result["beat_offset_ms"] = int(round(beat_offset_ms))
    result["beat_grid_density"] = beat_grid_density
    result["grid_bpm"] = float(result.get("bpm", 0.0)) * grid_bpm_scale
    return result


def analyze_audio(
    audio,
    fps,
    bpm_method="beat_intervals",
    half_time=False,
    beat_offset_ms=0,
    beat_grid_density="every_beat",
):
    if bpm_method not in {"beat_intervals", "onset_strength"}:
        raise ValueError(f"Unknown BPM method: {bpm_method}")
    waveform = mono_numpy(audio)
    sample_rate = int(audio["sample_rate"])
    duration = len(waveform) / sample_rate
    tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sample_rate, units="frames")
    detected_beats = librosa.frames_to_time(beat_frames, sr=sample_rate).astype(np.float64)
    onset_strength_bpm = _tempo_value(tempo)

    if len(detected_beats) > 1:
        interval = float(np.median(np.diff(detected_beats)))
        interval_bpm = 60.0 / interval
        if bpm_method == "beat_intervals":
            bpm = interval_bpm
            bpm_source = "beat_intervals_median"
        else:
            bpm = onset_strength_bpm
            bpm_source = "onset_strength"
    else:
        bpm = onset_strength_bpm
        bpm_source = "onset_strength"
        interval = 60.0 / bpm if bpm > 0 else 0.0

    if half_time:
        bpm /= 2.0
        interval *= 2.0
        detected_beats = detected_beats[::2]

    regularized_beats = _regularize_beats(detected_beats, interval, duration)
    detected_beats = np.unique(detected_beats)
    regularized_beats = np.unique(regularized_beats)
    if not len(regularized_beats):
        raise ValueError("No beats were detected in the selected audio range.")

    onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        units="frames",
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
    drum_times = _detect_drums(waveform, sample_rate, onset_frames, onset_times)
    beat_times = regularized_beats.tolist()
    detected_times = detected_beats.tolist()
    onsets = onset_times.tolist()

    analysis = {
        "version": ANALYSIS_VERSION,
        "detector_version": DETECTOR_VERSION,
        "bpm": float(bpm),
        "bpm_source": bpm_source,
        "beat_times": beat_times,
        "detected_beat_times": detected_times,
        "onset_times": onsets,
        "beat_frames": [round(value * fps) for value in beat_times],
        "detected_beat_frames": [round(value * fps) for value in detected_times],
        "onset_frames": [round(value * fps) for value in onsets],
        "num_beats": len(beat_times),
        "sample_rate": sample_rate,
        "audio_duration": float(duration),
        "drum_times": drum_times,
        "waveform_preview": waveform_preview(waveform, sample_rate),
    }
    return apply_beat_offset(analysis, fps, beat_offset_ms, beat_grid_density)


def analysis_cache_key(path, fps, trim_start_frame, length_frames, bpm_method, half_time, analysis_source):
    values = {
        "audio_sha256": audio_file_hash(path),
        "fps": float(fps),
        "trim_start_frame": int(trim_start_frame),
        "length_frames": int(length_frames),
        "bpm_method": bpm_method,
        "half_time": bool(half_time),
        "analysis_source": analysis_source,
        "detector_version": DETECTOR_VERSION,
    }
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(cache_key):
    directory = Path(folder_paths.get_user_directory()) / "fl_audio_prompt_timeline" / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{cache_key}.json"


def analyze_audio_file(
    filename,
    fps,
    trim_start_frame=0,
    length_frames=0,
    bpm_method="beat_intervals",
    half_time=False,
    beat_offset_ms=0,
    analysis_source="mix",
    beat_grid_density="every_beat",
):
    path = resolve_audio_path(filename)
    _, master_audio = load_audio_file(filename)
    cropped_audio, crop = crop_audio(master_audio, fps, trim_start_frame, length_frames)
    if analysis_source == "mix":
        analysis_audio = cropped_audio
    else:
        stem_audio = load_cached_stem(filename, analysis_source)
        analysis_audio, _ = crop_audio(stem_audio, fps, trim_start_frame, length_frames)
    cache_key = analysis_cache_key(
        path,
        fps,
        trim_start_frame,
        length_frames,
        bpm_method,
        half_time,
        analysis_source,
    )
    cache_path = _cache_path(cache_key)
    if cache_path.is_file():
        analysis = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        analysis = analyze_audio(
            analysis_audio,
            fps,
            bpm_method,
            half_time,
        )
        analysis.update(crop)
        analysis.update({
            "audio_file": filename,
            "analysis_source": analysis_source,
            "cache_key": cache_key,
        })
        temporary_path = cache_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(analysis, separators=(",", ":")), encoding="utf-8")
        temporary_path.replace(cache_path)
    return apply_beat_offset(
        analysis,
        fps,
        beat_offset_ms,
        beat_grid_density,
    ), cropped_audio
