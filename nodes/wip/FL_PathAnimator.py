import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math
import json

def pil2tensor(image):
    """Convert PIL Image to tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(tensor):
    """Convert tensor to PIL Image"""
    return Image.fromarray(np.clip(255. * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def parse_color(color):
    """Parse color string to RGB tuple"""
    if isinstance(color, str):
        if ',' in color:
            return tuple(int(c.strip()) for c in color.split(','))
        else:
            from PIL import ImageColor
            try:
                return ImageColor.getrgb(color)
            except:
                return (255, 255, 255)
    return color

class FL_PathAnimator:

    RETURN_TYPES = ("IMAGE", "MASK", "STRING",)
    RETURN_NAMES = ("image", "mask", "coordinates",)
    FUNCTION = "animate_paths"
    CATEGORY = "🏵️Fill Nodes/WIP"
    DESCRIPTION = """
Creates animated shapes that follow user-drawn paths.
Open the path editor to draw trajectories on a reference image, then shapes will follow these paths over time.
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "frame_count": ("INT", {"default": 30, "min": 1, "max": 500, "step": 1}),
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
            },
            "optional": {
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "trail_length": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rotation_speed": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "border_width": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
                "border_color": ("STRING", {"default": 'white'}),
                "paths_data": ("STRING", {"default": '{"paths": [], "image_size": {"width": 512, "height": 512}}', "multiline": True}),
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
                (center_x, center_y - half_size),
                (center_x - half_size, center_y + half_size),
                (center_x + half_size, center_y + half_size),
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

    def interpolate_path(self, points, t):
        """
        Interpolate position along a path at time t (0.0 to 1.0)
        Returns (x, y) coordinates
        """
        if len(points) == 0:
            return (0, 0)
        if len(points) == 1:
            return (points[0]['x'], points[0]['y'])

        # Calculate total path length
        total_length = 0
        segment_lengths = []
        for i in range(len(points) - 1):
            dx = points[i + 1]['x'] - points[i]['x']
            dy = points[i + 1]['y'] - points[i]['y']
            length = math.sqrt(dx * dx + dy * dy)
            segment_lengths.append(length)
            total_length += length

        if total_length == 0:
            return (points[0]['x'], points[0]['y'])

        # Find target distance along path
        target_distance = t * total_length

        # Find which segment contains target distance
        current_distance = 0
        for i, seg_length in enumerate(segment_lengths):
            if current_distance + seg_length >= target_distance:
                # Interpolate within this segment
                segment_t = (target_distance - current_distance) / seg_length if seg_length > 0 else 0
                x = points[i]['x'] + (points[i + 1]['x'] - points[i]['x']) * segment_t
                y = points[i]['y'] + (points[i + 1]['y'] - points[i]['y']) * segment_t
                return (x, y)
            current_distance += seg_length

        # Return last point if we've gone past the end
        return (points[-1]['x'], points[-1]['y'])

    def animate_paths(self, frame_width, frame_height, frame_count, shape, shape_size,
                     shape_color, bg_color, blur_radius=0.0, trail_length=0.0,
                     rotation_speed=0.0, border_width=0, border_color='white',
                     paths_data='{"paths": [], "image_size": {"width": 512, "height": 512}}'):

        # Parse colors
        shape_color = parse_color(shape_color)
        bg_color = parse_color(bg_color)
        border_color = parse_color(border_color)

        # Parse paths data
        try:
            paths_obj = json.loads(paths_data)
            paths = paths_obj.get('paths', [])
            canvas_size = paths_obj.get('canvas_size', {'width': frame_width, 'height': frame_height})
        except json.JSONDecodeError:
            print("FL_PathAnimator: Invalid JSON in paths_data, using empty paths")
            paths = []
            canvas_size = {'width': frame_width, 'height': frame_height}

        # Calculate scaling factors to transform from canvas coordinates to frame coordinates
        canvas_width = canvas_size.get('width', frame_width)
        canvas_height = canvas_size.get('height', frame_height)
        scale_x = frame_width / canvas_width if canvas_width > 0 else 1.0
        scale_y = frame_height / canvas_height if canvas_height > 0 else 1.0

        # Scale all path coordinates
        scaled_paths = []
        for path in paths:
            scaled_path = path.copy()
            scaled_points = []
            for point in path.get('points', []):
                scaled_points.append({
                    'x': point['x'] * scale_x,
                    'y': point['y'] * scale_y
                })
            scaled_path['points'] = scaled_points
            scaled_paths.append(scaled_path)

        images_list = []
        masks_list = []
        previous_output = None

        for frame in range(frame_count):
            # Create blank image with bg_color
            image = Image.new("RGB", (frame_width, frame_height), bg_color)
            draw = ImageDraw.Draw(image)

            # Calculate time along path (0.0 to 1.0)
            t = frame / max(frame_count - 1, 1)

            # Draw each path's shape
            for path_idx, path in enumerate(scaled_paths):
                points = path.get('points', [])
                if len(points) == 0:
                    continue

                # Get position along this path
                x, y = self.interpolate_path(points, t)

                # Calculate rotation
                current_rotation = rotation_speed * t * 360.0

                # Draw the shape
                self.draw_shape(draw, shape, x, y, shape_size, current_rotation,
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

            # Clamp values
            image_tensor = torch.clamp(image_tensor, 0.0, 1.0)

            # Extract mask from red channel
            mask = image_tensor[:, :, :, 0]

            images_list.append(image_tensor)
            masks_list.append(mask)

        # Concatenate all frames
        out_images = torch.cat(images_list, dim=0)
        out_masks = torch.cat(masks_list, dim=0)

        # Format coordinate string output
        # Convert scaled paths to coordinate string format (list of tracks)
        # Each path becomes a separate track so they all animate simultaneously
        coord_string = json.dumps([
            [{"x": int(point["x"]), "y": int(point["y"])} for point in path.get("points", [])]
            for path in scaled_paths
        ])

        return (out_images, out_masks, coord_string)
