import json
import numpy as np
import torch
from PIL import Image


class FL_GradGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "color_mode": (["RGB", "HSV"],),
                "interpolation": (["Linear", "Ease In", "Ease Out", "Ease In-Out"],),
                "gradient_colors": ("STRING", {"default": "[]"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_gradient"
    CATEGORY = "🏵️Fill Nodes/experiments"

    def generate_gradient(self, width, height, color_mode, interpolation, gradient_colors):
        # Parse gradient colors
        gradient_colors = json.loads(gradient_colors)
        colors = []
        positions = []
        for stop in gradient_colors:
            pos = stop['pos']
            color = stop['color']
            positions.append(pos)
            colors.append(color)

        # Create the gradient image
        image = np.zeros((height, width, 3), dtype=np.float32)

        # Apply interpolation to positions if needed
        if interpolation != "Linear":
            x = np.linspace(0, 1, width)
            if interpolation == "Ease In":
                x = x ** 2
            elif interpolation == "Ease Out":
                x = 1 - (1 - x) ** 2
            elif interpolation == "Ease In-Out":
                x = np.where(x < 0.5, 2 * x ** 2, 1 - (-2 * x + 2) ** 2 / 2)
            positions = np.interp(x, [0, 1], [0, 1])

        # Generate gradient
        for i in range(width):
            pos = i / (width - 1)
            if pos <= positions[0]:
                color = colors[0]
            elif pos >= positions[-1]:
                color = colors[-1]
            else:
                for j in range(len(positions) - 1):
                    if positions[j] <= pos < positions[j + 1]:
                        t = (pos - positions[j]) / (positions[j + 1] - positions[j])
                        color = [
                            (1 - t) * colors[j][k] + t * colors[j + 1][k]
                            for k in range(3)
                        ]
                        break
            image[:, i] = [c / 255.0 for c in color]  # Normalize color values to [0, 1]

        # Convert to HSV if needed
        if color_mode == "HSV":
            image_rgb = (image * 255).astype(np.uint8)
            image_hsv = Image.fromarray(image_rgb, mode="RGB").convert("HSV")
            image = np.array(image_hsv).astype(np.float32) / 255.0

        # Convert to PyTorch tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # Add batch dimension
        return (image_tensor,)


NODE_CLASS_MAPPINGS = {
    "GradientImageGenerator": FL_GradGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GradientImageGenerator": "Gradient Image Generator"
}