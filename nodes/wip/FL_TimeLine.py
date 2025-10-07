import json
import torch
from PIL import Image
import numpy as np
from server import PromptServer
from aiohttp import web

class FL_TimeLine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "timeline_data": ("STRING", {"multiline": True}),
            },
            "optional": {
                "ipadapter_preset": (["LIGHT - SD1.5 only (low strength)", "STANDARD (medium strength)", "VIT-G (medium strength)", "PLUS (high strength)", "PLUS FACE (portraits)", "FULL FACE - SD1.5 only (portraits stronger)"], {"default": "LIGHT - SD1.5 only (low strength)"}),
                "video_width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "video_height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "interpolation_mode": (["Linear", "Ease_in", "Ease_out", "Ease_in_out"], {"default": "Linear"}),
                "number_animation_frames": ("INT", {"default": 96, "min": 1, "max": 1000, "step": 1}),
                "frames_per_second": ("INT", {"default": 12, "min": 1, "max": 60, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "process_timeline"
    CATEGORY = "🏵️Fill Nodes/WIP"

    def process_timeline(self, model, timeline_data, ipadapter_preset, video_width, video_height, interpolation_mode, number_animation_frames, frames_per_second):
        # Parse the timeline data
        timeline = json.loads(timeline_data)

        # Process timeline data here
        # For now, we'll just return the model as-is
        return (model,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

# API route for handling timeline data
@PromptServer.instance.routes.post("/fl_timeline/data")
async def handle_timeline_data(request):
    data = await request.json()
    print("Received timeline data:", data)
    return web.json_response({"status": "success"})