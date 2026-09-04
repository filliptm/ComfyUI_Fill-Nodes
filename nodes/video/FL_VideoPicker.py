import math
import os
import threading
import time
import uuid
from fractions import Fraction

import torch
from aiohttp import web

import comfy.model_management
import comfy.utils
import folder_paths
import nodes
from comfy_api.latest import InputImpl, Types
from server import PromptServer


_PREVIEW_MAX_SIDE = 480
_PREVIEW_MAX_FPS = 12.0
_pending_video_selections = {}
_pending_video_selections_lock = threading.Lock()


def split_video_grid(images, x_slices, y_slices):
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] == 0:
        raise ValueError("FL Video Picker requires at least one IMAGE frame.")
    if images.shape[-1] not in (3, 4):
        raise ValueError("FL Video Picker supports RGB and RGBA IMAGE batches.")

    height = int(images.shape[1])
    width = int(images.shape[2])
    if x_slices > width or y_slices > height:
        raise ValueError("FL Video Picker cannot create slices smaller than one pixel.")

    x_edges = [round(index * width / x_slices) for index in range(x_slices + 1)]
    y_edges = [round(index * height / y_slices) for index in range(y_slices + 1)]
    return [
        images[:, y_edges[row]:y_edges[row + 1], x_edges[column]:x_edges[column + 1]]
        for row in range(y_slices)
        for column in range(x_slices)
    ]


def _preview_images(images, frame_rate):
    stride = max(1, math.ceil(frame_rate / _PREVIEW_MAX_FPS))
    preview = images[::stride]
    height = int(preview.shape[1])
    width = int(preview.shape[2])
    if max(width, height) > _PREVIEW_MAX_SIDE:
        scale = _PREVIEW_MAX_SIDE / max(width, height)
        target_width = max(2, round(width * scale))
        target_height = max(2, round(height * scale))
        preview = comfy.utils.common_upscale(
            preview.movedim(-1, 1),
            target_width,
            target_height,
            "bilinear",
            "disabled",
        ).movedim(1, -1)
    return preview.cpu(), frame_rate / stride


def _write_preview(images, frame_rate, path):
    preview, preview_rate = _preview_images(images, frame_rate)
    video = InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=preview[..., :3],
            audio=None,
            frame_rate=Fraction(round(preview_rate * 1000), 1000),
        ),
        bit_depth=8,
    )
    video.save_to(
        path,
        format=Types.VideoContainer.MP4,
        codec=Types.VideoCodec.H264,
        metadata=None,
        crf=28,
    )


def _remove_previews(paths, directory):
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError:
            pass
    try:
        os.rmdir(directory)
    except OSError:
        pass


def _interrupt(message):
    nodes.interrupt_processing()
    comfy.model_management.throw_exception_if_processing_interrupted()
    raise RuntimeError(message)


class FL_VideoPicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "x_slices": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "y_slices": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "timeout_seconds": ("INT", {"default": 300, "min": 30, "max": 3600, "step": 10}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("selected_videos", "selection_info", "selected_indexes")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "select_videos"
    CATEGORY = "🏵️Fill Nodes/Video"
    DESCRIPTION = (
        "Splits a decoded video mosaic into a spatial grid and pauses execution for interactive "
        "multi-selection. Designed for zero-overlap FL KSampler Plus grids; route audio around "
        "this node so ComfyUI can pair the shared audio with every selected clip."
    )

    def select_videos(self, images, x_slices, y_slices, frame_rate, timeout_seconds):
        videos = split_video_grid(images, x_slices, y_slices)
        session_id = uuid.uuid4().hex
        preview_directory = os.path.join(folder_paths.get_temp_directory(), "fl_video_picker", session_id)
        os.makedirs(preview_directory, exist_ok=False)

        preview_paths = []
        candidates = []
        try:
            for index, video in enumerate(videos):
                path = os.path.join(preview_directory, f"slice_{index}.mp4")
                preview_paths.append(path)
                _write_preview(video, frame_rate, path)
                candidates.append({
                    "index": index,
                    "row": index // x_slices,
                    "column": index % x_slices,
                    "width": int(video.shape[2]),
                    "height": int(video.shape[1]),
                    "frame_count": int(video.shape[0]),
                    "duration": float(video.shape[0] / frame_rate),
                    "preview_url": f"/fl_video_picker/preview/{session_id}/{index}",
                })

            event = threading.Event()
            session = {
                "event": event,
                "selection": None,
                "cancelled": False,
                "preview_paths": preview_paths,
                "count": len(videos),
            }
            with _pending_video_selections_lock:
                _pending_video_selections[session_id] = session

            PromptServer.instance.send_sync("fl_video_picker_show", {
                "session_id": session_id,
                "candidates": candidates,
                "x_slices": x_slices,
                "y_slices": y_slices,
                "frame_rate": frame_rate,
                "timeout_seconds": timeout_seconds,
            })

            deadline = time.monotonic() + timeout_seconds
            while not event.wait(0.25):
                comfy.model_management.throw_exception_if_processing_interrupted()
                if time.monotonic() >= deadline:
                    _interrupt("Video selection timed out.")

            if session["cancelled"]:
                _interrupt("Video selection was cancelled.")

            selected = session["selection"]
            if not selected:
                selected = list(range(len(videos)))
                message = f"No selection: returned all {len(videos)} videos"
            else:
                message = f"Selected {len(selected)} of {len(videos)} videos: indices {selected}"

            selected_videos = [videos[index].clone() for index in selected]
            return (selected_videos, message, ",".join(str(index) for index in selected))
        finally:
            with _pending_video_selections_lock:
                _pending_video_selections.pop(session_id, None)
            _remove_previews(preview_paths, preview_directory)


@PromptServer.instance.routes.post("/fl_video_picker/select")
async def receive_video_selection(request):
    data = await request.json()
    session_id = data.get("session_id")
    selection = data.get("selection", [])
    cancelled = data.get("cancelled", False)

    if not isinstance(session_id, str) or not isinstance(selection, list) or not isinstance(cancelled, bool):
        return web.json_response({"status": "error", "message": "Invalid selection request."}, status=400)

    with _pending_video_selections_lock:
        session = _pending_video_selections.get(session_id)
        if session is None:
            return web.json_response({"status": "error", "message": "Session not found."}, status=404)
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= session["count"] for index in selection):
            return web.json_response({"status": "error", "message": "Selection contains an invalid index."}, status=400)
        session["selection"] = sorted(set(selection))
        session["cancelled"] = cancelled
        session["event"].set()

    return web.json_response({"status": "ok", "received": len(session["selection"])})


@PromptServer.instance.routes.get("/fl_video_picker/status/{session_id}")
async def video_selection_status(request):
    session_id = request.match_info["session_id"]
    with _pending_video_selections_lock:
        active = session_id in _pending_video_selections
    return web.json_response({"status": "active" if active else "inactive"})


@PromptServer.instance.routes.get("/fl_video_picker/preview/{session_id}/{index}")
async def video_selection_preview(request):
    session_id = request.match_info["session_id"]
    try:
        index = int(request.match_info["index"])
    except ValueError:
        return web.json_response({"status": "error", "message": "Invalid preview index."}, status=400)

    with _pending_video_selections_lock:
        session = _pending_video_selections.get(session_id)
        if session is None:
            return web.json_response({"status": "error", "message": "Session not found."}, status=404)
        if index < 0 or index >= session["count"]:
            return web.json_response({"status": "error", "message": "Invalid preview index."}, status=400)
        path = session["preview_paths"][index]

    return web.FileResponse(path, headers={"Cache-Control": "no-store"})
