import asyncio

import av
from aiohttp import web
from server import PromptServer

from ..nodes.video.FL_LoadVideo import probe_video, resolve_video_path


@PromptServer.instance.routes.get("/fl/load-video/info")
async def load_video_info(request):
    try:
        path = resolve_video_path(request.query.get("filename", ""))
        info = await asyncio.to_thread(probe_video, path)
    except (OSError, ValueError, av.error.FFmpegError) as e:
        return web.json_response({"error": str(e)}, status=400)
    return web.json_response(info)
