import hashlib
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import requests
import torch

import comfy.model_management
import folder_paths


try:
    from beat_this.inference import Audio2Frames
    from beat_this.model.postprocessor import Postprocessor
except ImportError as error:
    Audio2Frames = None
    Postprocessor = None
    _IMPORT_ERROR = str(error)
else:
    _IMPORT_ERROR = ""


MODEL_NAME = "Beat This final0"
MODEL_PACKAGE = "beat-this"
MODEL_PACKAGE_VERSION = "1.1.0"
MODEL_FILENAME = "beat_this-final0.ckpt"
MODEL_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
MODEL_SIZE = 81058141
MODEL_SHA256 = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
MODEL_FPS = 50
MODEL_MEMORY_REQUIRED = 512 * 1024 * 1024


class BeatThisError(RuntimeError):
    pass


_download_lock = threading.Lock()
_inference_lock = threading.Lock()
_status_lock = threading.Lock()
_status = {
    "state": "missing",
    "progress": 0.0,
    "downloaded_bytes": 0,
    "total_bytes": MODEL_SIZE,
    "device": None,
    "message": "Beat This downloads on first analysis",
}


def checkpoint_path():
    return Path(folder_paths.models_dir) / "beat_this" / MODEL_FILENAME


def _package_version():
    try:
        return version(MODEL_PACKAGE)
    except PackageNotFoundError:
        return None


def _set_status(state, message, **values):
    with _status_lock:
        _status.update({
            "state": state,
            "message": message,
            **values,
        })


def model_status():
    with _status_lock:
        status = dict(_status)
    installed_version = _package_version()
    path = checkpoint_path()
    if _IMPORT_ERROR:
        status.update({
            "state": "unavailable",
            "message": f"Install {MODEL_PACKAGE}=={MODEL_PACKAGE_VERSION}: {_IMPORT_ERROR}",
        })
    elif status["state"] == "missing" and path.is_file() and path.stat().st_size == MODEL_SIZE:
        status.update({
            "state": "ready",
            "progress": 1.0,
            "downloaded_bytes": MODEL_SIZE,
            "message": "Beat This checkpoint ready",
        })
    return {
        "version": 1,
        "model": MODEL_NAME,
        "package": MODEL_PACKAGE,
        "package_version": installed_version,
        "expected_package_version": MODEL_PACKAGE_VERSION,
        "checkpoint": MODEL_FILENAME,
        "checkpoint_ready": path.is_file() and path.stat().st_size == MODEL_SIZE,
        **status,
    }


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_checkpoint(path):
    return (
        path.is_file()
        and path.stat().st_size == MODEL_SIZE
        and _file_sha256(path) == MODEL_SHA256
    )


def _download_checkpoint(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".part")
    temporary_path.unlink(missing_ok=True)
    downloaded = 0
    digest = hashlib.sha256()
    _set_status(
        "downloading",
        f"Downloading {MODEL_NAME}",
        progress=0.0,
        downloaded_bytes=0,
        total_bytes=MODEL_SIZE,
        device=None,
    )
    try:
        with requests.get(MODEL_URL, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    file.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    _set_status(
                        "downloading",
                        f"Downloading {MODEL_NAME}",
                        progress=min(1.0, downloaded / MODEL_SIZE),
                        downloaded_bytes=downloaded,
                    )
        _set_status(
            "verifying",
            "Verifying Beat This checkpoint",
            progress=1.0,
            downloaded_bytes=downloaded,
        )
        if downloaded != MODEL_SIZE:
            raise BeatThisError(
                f"Beat This checkpoint has {downloaded} bytes; expected {MODEL_SIZE}."
            )
        checksum = digest.hexdigest()
        if checksum != MODEL_SHA256:
            raise BeatThisError(
                f"Beat This checkpoint checksum mismatch: {checksum}."
            )
        temporary_path.replace(path)
    except (OSError, requests.RequestException, BeatThisError):
        temporary_path.unlink(missing_ok=True)
        raise


def ensure_checkpoint():
    if _IMPORT_ERROR:
        raise BeatThisError(
            f"Install {MODEL_PACKAGE}=={MODEL_PACKAGE_VERSION} to use Beat This: {_IMPORT_ERROR}"
        )
    installed_version = _package_version()
    if installed_version != MODEL_PACKAGE_VERSION:
        raise BeatThisError(
            f"Beat This requires {MODEL_PACKAGE}=={MODEL_PACKAGE_VERSION}; found "
            f"{installed_version or 'no installed package'}."
        )

    path = checkpoint_path()
    with _download_lock:
        if path.is_file():
            _set_status("verifying", "Verifying Beat This checkpoint", device=None)
            if _verified_checkpoint(path):
                _set_status(
                    "ready",
                    "Beat This checkpoint ready",
                    progress=1.0,
                    downloaded_bytes=MODEL_SIZE,
                    device=None,
                )
                return path
            path.unlink()
        try:
            _download_checkpoint(path)
        except (OSError, requests.RequestException, BeatThisError) as error:
            _set_status("error", str(error), device=None)
            raise BeatThisError(str(error)) from error
        _set_status(
            "ready",
            "Beat This checkpoint ready",
            progress=1.0,
            downloaded_bytes=MODEL_SIZE,
            device=None,
        )
        return path


def _event_confidences(logits, times):
    probabilities = logits.float().sigmoid().cpu().numpy()
    indices = np.clip(
        np.rint(np.asarray(times, dtype=np.float64) * MODEL_FPS).astype(int),
        0,
        len(probabilities) - 1,
    )
    return probabilities[indices].astype(np.float64).tolist()


def analyze_beats(waveform, sample_rate):
    path = ensure_checkpoint()
    if not _inference_lock.acquire(blocking=False):
        _set_status("waiting", "Waiting for another Beat This analysis")
        _inference_lock.acquire()

    tracker = None
    spectrogram = None
    beat_logits = None
    downbeat_logits = None
    started = time.perf_counter()
    device = None
    device_name = "unknown"
    try:
        device = comfy.model_management.get_torch_device()
        device_name = str(device)
        if getattr(device, "type", None) == "cuda":
            device_name = torch.cuda.get_device_name(device)
        _set_status("loading", f"Loading {MODEL_NAME}", device=device_name)
        comfy.model_management.free_memory(MODEL_MEMORY_REQUIRED, device)
        tracker = Audio2Frames(checkpoint_path=str(path), device=device, float16=False)
        _set_status("analyzing", "Analyzing beats and downbeats", device=device_name)
        spectrogram = tracker.signal2spect(waveform, sample_rate)
        beat_logits, downbeat_logits = tracker.spect2frames(spectrogram)
        beats, downbeats = Postprocessor(type="minimal")(beat_logits, downbeat_logits)
        beats = np.asarray(beats, dtype=np.float64)
        downbeats = np.asarray(downbeats, dtype=np.float64)
        if not len(beats):
            raise BeatThisError("Beat This did not detect any beats in the selected audio range.")
        result = {
            "beat_times": beats.tolist(),
            "downbeat_times": downbeats.tolist(),
            "beat_confidences": _event_confidences(beat_logits, beats),
            "downbeat_confidences": _event_confidences(downbeat_logits, downbeats),
            "detector": {
                "name": "beat_this",
                "model": MODEL_NAME,
                "checkpoint": "final0",
                "package_version": MODEL_PACKAGE_VERSION,
                "checkpoint_sha256": MODEL_SHA256,
                "postprocessor": "minimal",
                "frame_rate": MODEL_FPS,
                "device": device_name,
                "inference_seconds": time.perf_counter() - started,
            },
        }
    except BeatThisError as error:
        _set_status("error", str(error), device=device_name)
        raise
    except Exception as error:
        message = f"Beat This analysis failed: {error}"
        _set_status("error", message, device=device_name)
        raise BeatThisError(message) from error
    finally:
        try:
            if tracker is not None:
                tracker.model.to("cpu")
        finally:
            del tracker, spectrogram, beat_logits, downbeat_logits
            try:
                comfy.model_management.soft_empty_cache()
            finally:
                _inference_lock.release()

    _set_status(
        "ready",
        "Beat This checkpoint ready",
        progress=1.0,
        downloaded_bytes=MODEL_SIZE,
        device=device_name,
        last_inference_seconds=result["detector"]["inference_seconds"],
    )
    return result
