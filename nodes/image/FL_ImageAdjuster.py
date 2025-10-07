import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io
from server import PromptServer
from ..utils import tensor_to_pil, pil_to_tensor


class FL_ImageAdjuster:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "hue": ("FLOAT", {"default": 0, "min": -180, "max": 180, "step": 1}),
                "saturation": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "brightness": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "contrast": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "sharpness": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_image"
    CATEGORY = "🏵️Fill Nodes/Image"
    OUTPUT_NODE = True

    def adjust_image(self, image, hue, saturation, brightness, contrast, sharpness):
        # Convert tensor to PIL Image
        pil_image = tensor_to_pil(image)

        # Apply adjustments
        adjusted_image = self.apply_adjustments(pil_image, hue, saturation, brightness, contrast, sharpness)

        # Convert back to tensor
        tensor_image = pil_to_tensor(adjusted_image)

        # Prepare image for display
        display_image = self.prepare_image_for_display(adjusted_image)

        # Send the image to the frontend
        PromptServer.instance.send_sync("fl_image_adjuster", {"image": display_image})

        return (tensor_image,)

    def apply_adjustments(self, image, hue, saturation, brightness, contrast, sharpness):
        # Convert to HSV for hue and saturation adjustments
        hsv_image = image.convert('HSV')
        h, s, v = hsv_image.split()

        # Hue adjustment
        h = h.point(lambda x: (x + hue) % 256)

        # Saturation adjustment
        s = s.point(lambda x: max(0, min(255, x + saturation * 255 / 100)))

        # Merge channels
        hsv_image = Image.merge('HSV', (h, s, v))
        rgb_image = hsv_image.convert('RGB')

        # Brightness adjustment
        enhancer = ImageEnhance.Brightness(rgb_image)
        rgb_image = enhancer.enhance(1 + brightness / 100)

        # Contrast adjustment
        enhancer = ImageEnhance.Contrast(rgb_image)
        rgb_image = enhancer.enhance(1 + contrast / 100)

        # Sharpness adjustment
        if sharpness > 0:
            # Convert sharpness to an integer percentage between 100 and 200
            sharpness_percent = int(100 + sharpness)
            rgb_image = rgb_image.filter(ImageFilter.UnsharpMask(radius=2, percent=sharpness_percent, threshold=3))

        return rgb_image

    def prepare_image_for_display(self, pil_image):
        # Convert PIL Image to base64 string
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"