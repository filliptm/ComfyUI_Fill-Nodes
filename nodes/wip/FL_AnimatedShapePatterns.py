import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math

def pil2tensor(image):
    """Convert PIL Image to tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def parse_color(color):
    """Parse color string to RGB tuple"""
    if isinstance(color, str):
        if ',' in color:
            return tuple(int(c.strip()) for c in color.split(','))
        else:
            # Handle named colors
            from PIL import ImageColor
            try:
                return ImageColor.getrgb(color)
            except:
                return (255, 255, 255)
    return color

class FL_AnimatedShapePatterns:

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "generate_patterns"
    CATEGORY = "🏵️Fill Nodes/WIP"
    DESCRIPTION = """
Creates animated geometric shape patterns with various algorithms.
Generates a batch of images with shapes arranged in interesting patterns.
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pattern_type": ([
                    'grid_wave',
                    'grid_ripple',
                    'circle_array',
                    'orbital',
                    'spiral',
                    'lissajous',
                ], {"default": 'grid_wave'}),
                "frame_count": ("INT", {"default": 30, "min": 1, "max": 500, "step": 1}),
                "frame_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "shape": ([
                    'circle',
                    'square',
                    'triangle',
                    'hexagon',
                    'star',
                ], {"default": 'circle'}),
                "shape_size": ("INT", {"default": 20, "min": 2, "max": 500, "step": 1}),
                "shape_color": ("STRING", {"default": 'white'}),
                "bg_color": ("STRING", {"default": 'black'}),
                "count": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "amplitude": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 500.0, "step": 1.0}),
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "trail_length": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "size_variation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "phase_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 6.28, "step": 0.1}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "border_width": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
                "border_color": ("STRING", {"default": 'white'}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    def draw_shape(self, draw, shape, center_x, center_y, size, rotation, fill_color, border_width=0, border_color='white'):
        """Draw a shape at the specified location"""
        half_size = size / 2

        if shape == 'circle':
            bbox = [center_x - half_size, center_y - half_size,
                   center_x + half_size, center_y + half_size]
            if border_width > 0:
                draw.ellipse(bbox, fill=fill_color, outline=border_color, width=border_width)
            else:
                draw.ellipse(bbox, fill=fill_color)

        elif shape == 'square':
            bbox = [center_x - half_size, center_y - half_size,
                   center_x + half_size, center_y + half_size]
            if border_width > 0:
                draw.rectangle(bbox, fill=fill_color, outline=border_color, width=border_width)
            else:
                draw.rectangle(bbox, fill=fill_color)

        elif shape == 'triangle':
            points = [
                (center_x, center_y - half_size),  # top
                (center_x - half_size, center_y + half_size),  # bottom left
                (center_x + half_size, center_y + half_size),  # bottom right
            ]
            if rotation != 0:
                points = self.rotate_points(points, center_x, center_y, rotation)
            if border_width > 0:
                draw.polygon(points, fill=fill_color, outline=border_color, width=border_width)
            else:
                draw.polygon(points, fill=fill_color)

        elif shape == 'hexagon':
            points = []
            for i in range(6):
                angle = math.radians(60 * i + rotation)
                x = center_x + half_size * math.cos(angle)
                y = center_y + half_size * math.sin(angle)
                points.append((x, y))
            if border_width > 0:
                draw.polygon(points, fill=fill_color, outline=border_color, width=border_width)
            else:
                draw.polygon(points, fill=fill_color)

        elif shape == 'star':
            points = []
            for i in range(10):
                angle = math.radians(36 * i + rotation)
                r = half_size if i % 2 == 0 else half_size * 0.4
                x = center_x + r * math.cos(angle - math.pi / 2)
                y = center_y + r * math.sin(angle - math.pi / 2)
                points.append((x, y))
            if border_width > 0:
                draw.polygon(points, fill=fill_color, outline=border_color, width=border_width)
            else:
                draw.polygon(points, fill=fill_color)

    def rotate_points(self, points, cx, cy, angle):
        """Rotate points around a center"""
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        rotated = []
        for x, y in points:
            x -= cx
            y -= cy
            new_x = x * cos_a - y * sin_a + cx
            new_y = x * sin_a + y * cos_a + cy
            rotated.append((new_x, new_y))
        return rotated

    def generate_grid_wave(self, frame, frame_count, width, height, count, speed, amplitude, frequency, phase_offset):
        """Generate positions for grid wave pattern"""
        positions = []
        grid_size = int(math.sqrt(count))
        spacing_x = width / (grid_size + 1)
        spacing_y = height / (grid_size + 1)

        t = (frame / frame_count) * speed * 2 * math.pi

        for i in range(grid_size):
            for j in range(grid_size):
                base_x = (i + 1) * spacing_x
                base_y = (j + 1) * spacing_y

                # Wave offset
                offset_x = amplitude * math.sin(t + j * phase_offset)
                offset_y = amplitude * math.cos(t + i * phase_offset)

                x = base_x + offset_x
                y = base_y + offset_y
                positions.append((x, y, i * grid_size + j))

        return positions

    def generate_grid_ripple(self, frame, frame_count, width, height, count, speed, amplitude, frequency, phase_offset):
        """Generate positions for ripple pattern emanating from center"""
        positions = []
        grid_size = int(math.sqrt(count))
        spacing_x = width / (grid_size + 1)
        spacing_y = height / (grid_size + 1)

        center_x = width / 2
        center_y = height / 2
        t = (frame / frame_count) * speed * 2 * math.pi

        for i in range(grid_size):
            for j in range(grid_size):
                base_x = (i + 1) * spacing_x
                base_y = (j + 1) * spacing_y

                # Calculate distance from center
                dx = base_x - center_x
                dy = base_y - center_y
                distance = math.sqrt(dx * dx + dy * dy)

                # Ripple effect
                ripple = amplitude * math.sin(t - distance * frequency * 0.01)

                # Apply ripple in radial direction
                if distance > 0:
                    x = base_x + (dx / distance) * ripple
                    y = base_y + (dy / distance) * ripple
                else:
                    x, y = base_x, base_y

                positions.append((x, y, i * grid_size + j))

        return positions

    def generate_circle_array(self, frame, frame_count, width, height, count, speed, amplitude, frequency, phase_offset):
        """Generate positions for shapes arranged in a circle"""
        positions = []
        center_x = width / 2
        center_y = height / 2
        radius = min(width, height) * 0.35

        t = (frame / frame_count) * speed * 2 * math.pi

        for i in range(count):
            angle = (i / count) * 2 * math.pi + t

            # Add amplitude variation
            r = radius + amplitude * math.sin(t * frequency + i * phase_offset)

            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            positions.append((x, y, i))

        return positions

    def generate_orbital(self, frame, frame_count, width, height, count, speed, amplitude, frequency, phase_offset):
        """Generate positions for orbital pattern with multiple rings"""
        positions = []
        center_x = width / 2
        center_y = height / 2
        max_radius = min(width, height) * 0.4

        t = (frame / frame_count) * speed * 2 * math.pi

        # Determine number of rings
        num_rings = max(2, int(math.sqrt(count)))
        shapes_per_ring = count // num_rings

        for ring in range(num_rings):
            radius = max_radius * ((ring + 1) / num_rings)
            ring_speed = 1.0 + ring * 0.3  # Outer rings move faster

            for i in range(shapes_per_ring):
                angle = (i / shapes_per_ring) * 2 * math.pi + t * ring_speed + ring * phase_offset

                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                positions.append((x, y, ring * shapes_per_ring + i))

        return positions

    def generate_spiral(self, frame, frame_count, width, height, count, speed, amplitude, frequency, phase_offset):
        """Generate positions for spiral pattern"""
        positions = []
        center_x = width / 2
        center_y = height / 2
        max_radius = min(width, height) * 0.4

        t = (frame / frame_count) * speed * 2 * math.pi

        for i in range(count):
            progress = i / count
            angle = progress * 4 * math.pi + t  # 2 full rotations
            radius = max_radius * progress

            # Add wave variation
            radius += amplitude * 0.1 * math.sin(t * frequency + i * phase_offset)

            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((x, y, i))

        return positions

    def generate_lissajous(self, frame, frame_count, width, height, count, speed, amplitude, frequency, phase_offset):
        """Generate positions for Lissajous curve pattern"""
        positions = []
        center_x = width / 2
        center_y = height / 2

        t = (frame / frame_count) * speed * 2 * math.pi
        scale_x = width * 0.4
        scale_y = height * 0.4

        for i in range(count):
            progress = (i / count) * 2 * math.pi

            # Lissajous curve with adjustable frequency
            x = center_x + scale_x * math.sin(frequency * progress + t)
            y = center_y + scale_y * math.sin((frequency + 1) * progress + t + phase_offset)

            positions.append((x, y, i))

        return positions

    def generate_patterns(self, pattern_type, frame_count, frame_width, frame_height,
                         shape, shape_size, shape_color, bg_color, count, speed,
                         amplitude, frequency, blur_radius=0.0, trail_length=0.0,
                         size_variation=0.0, rotation=0.0, phase_offset=0.0,
                         intensity=1.0, border_width=0, border_color='white', seed=0):

        # Parse colors
        shape_color = parse_color(shape_color)
        bg_color = parse_color(bg_color)
        border_color = parse_color(border_color)

        # Set random seed
        if seed > 0:
            np.random.seed(seed)

        images_list = []
        masks_list = []
        previous_output = None

        for frame in range(frame_count):
            # Create blank image
            image = Image.new("RGB", (frame_width, frame_height), bg_color)
            draw = ImageDraw.Draw(image)

            # Generate positions based on pattern type
            if pattern_type == 'grid_wave':
                positions = self.generate_grid_wave(frame, frame_count, frame_width, frame_height,
                                                    count, speed, amplitude, frequency, phase_offset)
            elif pattern_type == 'grid_ripple':
                positions = self.generate_grid_ripple(frame, frame_count, frame_width, frame_height,
                                                      count, speed, amplitude, frequency, phase_offset)
            elif pattern_type == 'circle_array':
                positions = self.generate_circle_array(frame, frame_count, frame_width, frame_height,
                                                       count, speed, amplitude, frequency, phase_offset)
            elif pattern_type == 'orbital':
                positions = self.generate_orbital(frame, frame_count, frame_width, frame_height,
                                                  count, speed, amplitude, frequency, phase_offset)
            elif pattern_type == 'spiral':
                positions = self.generate_spiral(frame, frame_count, frame_width, frame_height,
                                                count, speed, amplitude, frequency, phase_offset)
            elif pattern_type == 'lissajous':
                positions = self.generate_lissajous(frame, frame_count, frame_width, frame_height,
                                                   count, speed, amplitude, frequency, phase_offset)
            else:
                positions = []

            # Draw shapes at calculated positions
            for x, y, index in positions:
                # Calculate size with variation
                current_size = shape_size
                if size_variation > 0:
                    t = (frame / frame_count) * 2 * math.pi
                    size_mod = 1.0 + size_variation * math.sin(t + index * 0.5)
                    current_size = int(shape_size * size_mod)

                # Calculate rotation
                current_rotation = rotation + (frame / frame_count) * rotation

                # Draw the shape
                self.draw_shape(draw, shape, x, y, current_size, current_rotation,
                              shape_color, border_width, border_color)

            # Apply blur
            if blur_radius > 0:
                image = image.filter(ImageFilter.GaussianBlur(blur_radius))

            # Convert to tensor
            image_tensor = pil2tensor(image)

            # Apply trailing effect
            if trail_length > 0 and previous_output is not None:
                image_tensor = image_tensor + trail_length * previous_output
                image_tensor = image_tensor / image_tensor.max()

            previous_output = image_tensor.clone()

            # Apply intensity
            image_tensor = image_tensor * intensity
            image_tensor = torch.clamp(image_tensor, 0.0, 1.0)

            # Extract mask from red channel
            mask = image_tensor[:, :, :, 0]

            images_list.append(image_tensor)
            masks_list.append(mask)

        # Concatenate all frames
        out_images = torch.cat(images_list, dim=0)
        out_masks = torch.cat(masks_list, dim=0)

        return (out_images, out_masks)
