import asyncio
import threading
import uuid

from aiohttp import web
from server import PromptServer

from ..nodes.audio.audio_separation import (
    SeparationCancelled,
    separate_audio_file,
    separation_manifest,
)
from ..nodes.audio.audio_files import audio_library_entries
from ..nodes.audio.audio_timeline import analyze_audio_file
from ..nodes.audio.beat_this_detector import BeatThisError, model_status


_separation_jobs = {}
_separation_lock = threading.Lock()
_active_separation_job = None


def _public_job(job):
    return {
        key: value
        for key, value in job.items()
        if key not in {"cancel_event", "task"}
    }


def _update_separation_job(job_id, progress, message):
    with _separation_lock:
        job = _separation_jobs.get(job_id)
        if job is None:
            return
        job["progress"] = float(progress)
        job["message"] = message


async def _run_separation_job(job_id, filename):
    global _active_separation_job
    job = _separation_jobs[job_id]
    with _separation_lock:
        job["status"] = "running"
        job["message"] = "Loading stem model"
    try:
        manifest = await asyncio.to_thread(
            separate_audio_file,
            filename,
            lambda progress, message: _update_separation_job(job_id, progress, message),
            job["cancel_event"],
        )
        with _separation_lock:
            job["status"] = "completed"
            job["progress"] = 1.0
            job["message"] = "Stem separation complete"
            job["manifest"] = manifest
    except SeparationCancelled as error:
        with _separation_lock:
            job["status"] = "cancelled"
            job["message"] = str(error)
    except Exception as error:
        with _separation_lock:
            job["status"] = "error"
            job["message"] = str(error)
    finally:
        with _separation_lock:
            if _active_separation_job == job_id:
                _active_separation_job = None


@PromptServer.instance.routes.post("/fl/audio-prompt-timeline/analyze")
async def analyze_audio_timeline(request):
    try:
        values = await request.json()
        analysis, _ = await asyncio.to_thread(
            analyze_audio_file,
            values.get("audio_file"),
            float(values.get("fps", 24.0)),
            int(values.get("trim_start_frame", 0)),
            int(values.get("length_frames", 0)),
            bool(values.get("half_time", False)),
            int(values.get("beat_offset_ms", 0)),
            values.get("analysis_source", "mix"),
        )
        return web.json_response(analysis)
    except (BeatThisError, TypeError, ValueError) as error:
        return web.json_response({"error": str(error)}, status=400)


@PromptServer.instance.routes.get("/fl/audio-prompt-timeline/beat-model/status")
async def beat_model_status(_request):
    return web.json_response(model_status())


@PromptServer.instance.routes.get("/fl/audio-prompt-timeline/files")
async def audio_timeline_files(_request):
    files = await asyncio.to_thread(audio_library_entries)
    return web.json_response({"files": files})


@PromptServer.instance.routes.post("/fl/audio-prompt-timeline/separate")
async def start_audio_separation(request):
    global _active_separation_job
    try:
        values = await request.json()
        filename = values.get("audio_file")
        cached = await asyncio.to_thread(separation_manifest, filename)
        if cached is not None:
            return web.json_response({
                "status": "completed",
                "progress": 1.0,
                "message": "Using cached stems",
                "manifest": cached,
            })

        with _separation_lock:
            if _active_separation_job is not None:
                active = _separation_jobs[_active_separation_job]
                return web.json_response(
                    {
                        "error": "Another stem separation is already running.",
                        "job": _public_job(active),
                    },
                    status=409,
                )
            completed = [
                job_id
                for job_id, job in _separation_jobs.items()
                if job["status"] in {"completed", "cancelled", "error"}
            ]
            for old_job_id in completed[:-20]:
                del _separation_jobs[old_job_id]
            job_id = str(uuid.uuid4())
            job = {
                "job_id": job_id,
                "audio_file": filename,
                "status": "queued",
                "progress": 0.0,
                "message": "Stem separation queued",
                "cancel_event": threading.Event(),
            }
            _separation_jobs[job_id] = job
            _active_separation_job = job_id
        job["task"] = asyncio.create_task(_run_separation_job(job_id, filename))
        return web.json_response(_public_job(job), status=202)
    except (TypeError, ValueError) as error:
        return web.json_response({"error": str(error)}, status=400)


@PromptServer.instance.routes.get("/fl/audio-prompt-timeline/separate/{job_id}")
async def get_audio_separation(request):
    job_id = request.match_info["job_id"]
    with _separation_lock:
        job = _separation_jobs.get(job_id)
        if job is None:
            return web.json_response({"error": "Stem separation job was not found."}, status=404)
        return web.json_response(_public_job(job))


@PromptServer.instance.routes.post("/fl/audio-prompt-timeline/separate/{job_id}/cancel")
async def cancel_audio_separation(request):
    job_id = request.match_info["job_id"]
    with _separation_lock:
        job = _separation_jobs.get(job_id)
        if job is None:
            return web.json_response({"error": "Stem separation job was not found."}, status=404)
        if job["status"] in {"completed", "cancelled", "error"}:
            return web.json_response(_public_job(job))
        job["cancel_event"].set()
        job["message"] = "Cancelling after the current chunk"
        return web.json_response(_public_job(job))
