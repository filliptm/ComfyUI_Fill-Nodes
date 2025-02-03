import torch
import numpy as np
from PIL import Image
import cv2
import comfy.model_management


class FL_AnimeLineExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "line_threshold": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "line_width": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1
                }),
                "detail_level": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "noise_reduction": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "invert_output": ("BOOLEAN", {
                    "default": False,
                    "label": "White Background"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extract_lines"
    CATEGORY = "🏵️Fill Nodes/Image"

    def extract_lines(self, image, line_threshold, line_width, detail_level, noise_reduction, invert_output):
        if isinstance(image, torch.Tensor):
            image = (image.cpu().numpy() * 255).astype(np.uint8)

        if len(image.shape) == 4:
            processed_images = []
            for img in image:
                processed = self._process_single_image(img, line_threshold, line_width, detail_level, noise_reduction,
                                                       invert_output)
                processed_images.append(processed)
            result = np.stack(processed_images)
        else:
            result = self._process_single_image(image, line_threshold, line_width, detail_level, noise_reduction,
                                                invert_output)

        return (torch.from_numpy(result.astype(np.float32) / 255.0),)

    def _process_single_image(self, img, line_threshold, line_width, detail_level, noise_reduction, invert_output):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Apply noise reduction (Gaussian blur)
        if noise_reduction > 0:
            blur_radius = int(noise_reduction * 10)
            if blur_radius % 2 == 0:
                blur_radius += 1
            gray = cv2.GaussianBlur(gray, (blur_radius, blur_radius), 0)

        # Calculate adaptive threshold parameters
        block_size = int((1 - detail_level) * 50) + 11
        if block_size % 2 == 0:
            block_size += 1

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            int(line_threshold * 20)
        )

        # Clean up small noise
        kernel_size = max(1, int((1 - detail_level) * 3))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Adjust line width
        if line_width > 1:
            kernel = np.ones((line_width, line_width), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)

        # Invert if requested
        if invert_output:
            binary = cv2.bitwise_not(binary)

        # Convert back to RGB
        lines_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        return lines_rgb