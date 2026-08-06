from aiohttp import web
from server import PromptServer

from ..nodes.video.FL_VideoCombine import preview_file_for_token


@PromptServer.instance.routes.get("/fl/video-combine/preview/{token}")
async def video_combine_preview(request):
    file_path = preview_file_for_token(request.match_info["token"])
    if file_path is None:
        raise web.HTTPNotFound()
    return web.FileResponse(file_path)
