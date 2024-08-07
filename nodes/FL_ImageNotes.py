import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from comfy.utils import ProgressBar

class FL_ImageNotes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"default": "Text Here", "multiline": False}),
                "bar_height": ("INT", {"default": 50, "min": 10, "max": 200, "step": 2}),
                "text_size": ("INT", {"default": 24, "min": 10, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_notes"
    CATEGORY = "🏵️Fill Nodes/utility"

    def add_notes(self, images, text, bar_height, text_size):
        result = []
        total_images = len(images)
        pbar = ProgressBar(total_images)
        for i, image in enumerate(images, start=1):
            img = self.t2p(image)
            result_img = self.add_text_bar(img, text, bar_height, text_size)
            result_img = self.p2t(result_img)
            result.append(result_img)

            pbar.update_absolute(i)

        return (torch.cat(result, dim=0),)

    def add_text_bar(self, image, text, bar_height, text_size):
        width, height = image.size
        new_height = height + bar_height
        new_image = Image.new("RGB", (width, new_height), color="black")
        new_image.paste(image, (0, bar_height))

        draw = ImageDraw.Draw(new_image)
        font = ImageFont.truetype("arial.ttf", text_size)
        text_width, text_height = self.get_text_size(text, font)
        x = (width - text_width) // 2
        y = (bar_height - text_height) // 2
        draw.text((x, y), text, font=font, fill="white")

        return new_image

    def get_text_size(self, text, font):
        ascent, descent = font.getmetrics()
        text_width = font.getmask(text).getbbox()[2]
        text_height = font.getmask(text).getbbox()[3] + descent
        return text_width, text_height

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def p2t(self, p):
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            t = torch.from_numpy(i).unsqueeze(0)
        return t