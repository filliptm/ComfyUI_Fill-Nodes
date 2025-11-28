import torch
import numpy as np
from PIL import Image, ImageDraw
import random
import math


class FL_RandomShapeGenerator:
    """
    Generate images with random shapes, colors, and patterns on a white background.
    Outputs both the image tensor and a latent representation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "num_shapes": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "min_shape_size": ("INT", {"default": 20, "min": 5, "max": 500, "step": 5}),
                "max_shape_size": ("INT", {"default": 150, "min": 10, "max": 1000, "step": 5}),
                "shape_types": ([
                    "all",
                    "circles",
                    "rectangles",
                    "triangles",
                    "ellipses",
                    "lines",
                    "polygons",
                    "circles_rectangles",
                    "geometric_only",
                ], {"default": "all"}),
                "color_mode": ([
                    "random",
                    "vibrant",
                    "pastel",
                    "monochrome",
                    "warm",
                    "cool",
                    "grayscale",
                ], {"default": "random"}),
                "opacity_min": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "opacity_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "background_color": ([
                    "white",
                    "black",
                    "gray",
                    "random",
                ], {"default": "white"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
            "optional": {
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "generate_shapes"
    CATEGORY = "🏵️Fill Nodes/utility"

    def get_background_color(self, mode):
        """Get background color based on mode"""
        if mode == "white":
            return (255, 255, 255)
        elif mode == "black":
            return (0, 0, 0)
        elif mode == "gray":
            return (128, 128, 128)
        else:  # random
            return (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))

    def get_random_color(self, mode):
        """Generate a random color based on the color mode"""
        if mode == "random":
            return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        elif mode == "vibrant":
            # High saturation, random hue
            h = random.random()
            s = random.uniform(0.8, 1.0)
            v = random.uniform(0.8, 1.0)
            return self._hsv_to_rgb(h, s, v)
        elif mode == "pastel":
            # Low saturation, high value
            h = random.random()
            s = random.uniform(0.2, 0.4)
            v = random.uniform(0.85, 1.0)
            return self._hsv_to_rgb(h, s, v)
        elif mode == "monochrome":
            # Random hue, varying saturation and value
            h = random.uniform(0, 0.1)  # Reddish tones
            s = random.uniform(0.5, 1.0)
            v = random.uniform(0.3, 1.0)
            return self._hsv_to_rgb(h, s, v)
        elif mode == "warm":
            # Red, orange, yellow range
            h = random.uniform(0, 0.12)
            s = random.uniform(0.6, 1.0)
            v = random.uniform(0.6, 1.0)
            return self._hsv_to_rgb(h, s, v)
        elif mode == "cool":
            # Blue, cyan, purple range
            h = random.uniform(0.5, 0.75)
            s = random.uniform(0.6, 1.0)
            v = random.uniform(0.6, 1.0)
            return self._hsv_to_rgb(h, s, v)
        elif mode == "grayscale":
            gray = random.randint(0, 255)
            return (gray, gray, gray)
        else:
            return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB"""
        if s == 0.0:
            return (int(v * 255), int(v * 255), int(v * 255))

        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6

        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q

        return (int(r * 255), int(g * 255), int(b * 255))

    def get_shape_types(self, mode):
        """Get list of shape types based on mode"""
        all_shapes = ["circle", "rectangle", "triangle", "ellipse", "line", "polygon"]

        if mode == "all":
            return all_shapes
        elif mode == "circles":
            return ["circle"]
        elif mode == "rectangles":
            return ["rectangle"]
        elif mode == "triangles":
            return ["triangle"]
        elif mode == "ellipses":
            return ["ellipse"]
        elif mode == "lines":
            return ["line"]
        elif mode == "polygons":
            return ["polygon"]
        elif mode == "circles_rectangles":
            return ["circle", "rectangle"]
        elif mode == "geometric_only":
            return ["circle", "rectangle", "triangle"]
        else:
            return all_shapes

    def draw_shape(self, draw, shape_type, x, y, size, color, opacity):
        """Draw a shape on the image"""
        # Create RGBA color with opacity
        rgba_color = color + (int(opacity * 255),)

        if shape_type == "circle":
            bbox = [x - size // 2, y - size // 2, x + size // 2, y + size // 2]
            draw.ellipse(bbox, fill=rgba_color)

        elif shape_type == "rectangle":
            w = random.randint(size // 2, size)
            h = random.randint(size // 2, size)
            bbox = [x - w // 2, y - h // 2, x + w // 2, y + h // 2]
            draw.rectangle(bbox, fill=rgba_color)

        elif shape_type == "triangle":
            # Random triangle
            points = [
                (x, y - size // 2),
                (x - size // 2, y + size // 2),
                (x + size // 2, y + size // 2)
            ]
            # Rotate randomly
            angle = random.uniform(0, 2 * math.pi)
            center_x, center_y = x, y
            rotated_points = []
            for px, py in points:
                dx, dy = px - center_x, py - center_y
                new_x = center_x + dx * math.cos(angle) - dy * math.sin(angle)
                new_y = center_y + dx * math.sin(angle) + dy * math.cos(angle)
                rotated_points.append((new_x, new_y))
            draw.polygon(rotated_points, fill=rgba_color)

        elif shape_type == "ellipse":
            w = random.randint(size // 3, size)
            h = random.randint(size // 3, size)
            bbox = [x - w // 2, y - h // 2, x + w // 2, y + h // 2]
            draw.ellipse(bbox, fill=rgba_color)

        elif shape_type == "line":
            # Random line with thickness
            angle = random.uniform(0, 2 * math.pi)
            end_x = x + int(size * math.cos(angle))
            end_y = y + int(size * math.sin(angle))
            thickness = random.randint(2, max(3, size // 10))
            draw.line([(x, y), (end_x, end_y)], fill=rgba_color, width=thickness)

        elif shape_type == "polygon":
            # Random polygon with 5-8 sides
            num_sides = random.randint(5, 8)
            points = []
            for i in range(num_sides):
                angle = (2 * math.pi * i / num_sides) + random.uniform(-0.3, 0.3)
                radius = size // 2 * random.uniform(0.7, 1.0)
                px = x + int(radius * math.cos(angle))
                py = y + int(radius * math.sin(angle))
                points.append((px, py))
            draw.polygon(points, fill=rgba_color)

    def generate_single_image(self, width, height, num_shapes, min_size, max_size,
                              shape_types, color_mode, opacity_min, opacity_max, bg_color):
        """Generate a single image with random shapes"""
        # Create RGBA image for proper alpha blending
        image = Image.new("RGBA", (width, height), bg_color + (255,))

        # Create a separate layer for shapes with transparency
        shape_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shape_layer)

        available_shapes = self.get_shape_types(shape_types)

        for _ in range(num_shapes):
            # Random position
            x = random.randint(0, width)
            y = random.randint(0, height)

            # Random size
            size = random.randint(min_size, max_size)

            # Random shape
            shape_type = random.choice(available_shapes)

            # Random color
            color = self.get_random_color(color_mode)

            # Random opacity
            opacity = random.uniform(opacity_min, opacity_max)

            # Draw the shape
            self.draw_shape(draw, shape_type, x, y, size, color, opacity)

        # Composite shape layer onto background
        image = Image.alpha_composite(image, shape_layer)

        # Convert to RGB
        image = image.convert("RGB")

        return image

    def generate_shapes(self, width, height, batch_size, num_shapes, min_shape_size,
                        max_shape_size, shape_types, color_mode, opacity_min, opacity_max,
                        background_color, seed, vae=None):
        """Generate batch of images with random shapes"""

        # Set seed for reproducibility
        if seed != 0:
            random.seed(seed)
            np.random.seed(seed)

        images = []

        for i in range(batch_size):
            # Use different seed for each image in batch if seed is set
            if seed != 0:
                random.seed(seed + i)
                np.random.seed(seed + i)

            bg_color = self.get_background_color(background_color)

            pil_image = self.generate_single_image(
                width, height, num_shapes, min_shape_size, max_shape_size,
                shape_types, color_mode, opacity_min, opacity_max, bg_color
            )

            # Convert PIL image to tensor
            np_image = np.array(pil_image).astype(np.float32) / 255.0
            images.append(np_image)

        # Stack images into batch tensor [B, H, W, C]
        image_tensor = torch.from_numpy(np.stack(images, axis=0))

        # Generate latent if VAE is provided
        latent = {"samples": torch.zeros(batch_size, 4, height // 8, width // 8)}

        if vae is not None:
            try:
                # Encode image to latent space
                # VAE expects [B, C, H, W] format
                image_for_vae = image_tensor.permute(0, 3, 1, 2)
                latent_samples = vae.encode(image_for_vae[:, :3, :, :])
                latent = {"samples": latent_samples}
                print(f"[FL_RandomShapeGenerator] Encoded {batch_size} images to latent space")
            except Exception as e:
                print(f"[FL_RandomShapeGenerator] Warning: Could not encode to latent: {e}")
                # Return zero latent as fallback
                latent = {"samples": torch.zeros(batch_size, 4, height // 8, width // 8)}

        print(f"[FL_RandomShapeGenerator] Generated {batch_size} images at {width}x{height} with {num_shapes} shapes each")

        return (image_tensor, latent)


NODE_CLASS_MAPPINGS = {
    "FL_RandomShapeGenerator": FL_RandomShapeGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_RandomShapeGenerator": "FL Random Shape Generator"
}
