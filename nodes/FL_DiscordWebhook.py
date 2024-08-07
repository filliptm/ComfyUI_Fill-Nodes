import os
import numpy as np
import requests
from PIL import Image
from moviepy.editor import ImageSequenceClip
import torch
import tempfile
import json


class FL_SendToDiscordWebhook:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "webhook_url": ("STRING", {"default": "https://discord.com/api/webhooks/YOUR_WEBHOOK_HASH"}),
                "frame_rate": ("INT", {"default": 12, "min": 1, "max": 60, "step": 1}),
                "save_locally": ("BOOLEAN", {"default": True}),
                "bot_username": ("STRING", {"default": "ComfyUI Bot"}),
                "message": ("STRING", {"default": "Here's your image/video:", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_and_upload"
    CATEGORY = "🏵️Fill Nodes/Discord"
    OUTPUT_NODE = True

    def generate_and_upload(self, images, webhook_url: str, frame_rate: int, save_locally: bool, bot_username: str,
                            message: str):
        if save_locally:
            output_dir = os.path.join(os.path.dirname(__file__), "outputs")
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = tempfile.gettempdir()

        filename = f"discord_upload_{int(torch.rand(1).item() * 10000)}"

        # Prepare the webhook data
        webhook_data = {
            "username": bot_username,
            "content": message,
        }

        if len(images) == 1:
            file_path = os.path.join(output_dir, f"{filename}.png")
            single_image = 255.0 * images[0].cpu().numpy()
            single_image_pil = Image.fromarray(single_image.astype(np.uint8))
            single_image_pil.save(file_path)

            with open(file_path, "rb") as file_data:
                files = {
                    "payload_json": (None, json.dumps(webhook_data)),
                    "file": (f"{filename}.png", file_data)
                }
                response = requests.post(webhook_url, files=files)
        else:
            frames = [255.0 * image.cpu().numpy() for image in images]
            file_path = os.path.join(output_dir, f"{filename}.mp4")

            clip = ImageSequenceClip(frames, fps=frame_rate)
            clip.write_videofile(file_path, codec="libx264", fps=frame_rate)

            with open(file_path, 'rb') as file_data:
                files = {
                    "payload_json": (None, json.dumps(webhook_data)),
                    "file": (f"{filename}.mp4", file_data)
                }
                response = requests.post(webhook_url, files=files)

        if response.status_code == 204:
            message = "Successfully uploaded to Discord."
        else:
            message = f"Failed to upload. Status code: {response.status_code} - {response.text}"

        if not save_locally:
            os.remove(file_path)

        return (message,)