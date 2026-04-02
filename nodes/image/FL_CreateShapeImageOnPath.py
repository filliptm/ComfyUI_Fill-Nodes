import torch
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def parse_color(color):
    if isinstance(color, str) and ',' in color:
        return tuple(int(c.strip()) for c in color.split(','))
    return color


def parse_json_tracks(tracks):
    tracks_data = []
    try:
        if isinstance(tracks, str):
            parsed = json.loads(tracks.replace("'", '"'))
            tracks_data.extend(parsed)
        else:
            for track_str in tracks:
                parsed = json.loads(track_str.replace("'", '"'))
                tracks_data.append(parsed)

        if tracks_data and isinstance(tracks_data[0], dict) and 'x' in tracks_data[0]:
            tracks_data = [tracks_data]
        elif tracks_data and isinstance(tracks_data[0], list) and tracks_data[0] and isinstance(tracks_data[0][0], dict) and 'x' in tracks_data[0][0]:
            pass
        else:
            print(f"Warning: Unexpected track format: {type(tracks_data[0])}")

    except json.JSONDecodeError as e:
        print(f"Error parsing tracks JSON: {e}")
        tracks_data = []

    return tracks_data


class FL_CreateShapeImageOnPath:

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "createshapemask"
    CATEGORY = "🏵️Fill Nodes/Image"
    DESCRIPTION = """
Creates an image or batch of images with the specified shape.
Locations are center locations.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shape": (
                    ['circle', 'square', 'triangle'],
                    {"default": 'circle'}
                ),
                "coordinates": ("STRING", {"forceInput": True}),
                "frame_width": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
                "shape_width": ("INT", {"default": 128, "min": 2, "max": 4096, "step": 1}),
                "shape_height": ("INT", {"default": 128, "min": 2, "max": 4096, "step": 1}),
                "shape_color": ("STRING", {"default": 'white'}),
                "bg_color": ("STRING", {"default": 'black'}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100, "step": 0.1}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "size_multiplier": ("FLOAT", {"default": [1.0], "forceInput": True}),
                "trailing": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "border_width": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "border_color": ("STRING", {"default": 'black'}),
            }
        }

    def createshapemask(self, coordinates, frame_width, frame_height, shape_width, shape_height, shape_color,
                        bg_color, blur_radius, shape, intensity, size_multiplier=[1.0], trailing=1.0, border_width=0, border_color='black'):

        shape_color = parse_color(shape_color)
        border_color = parse_color(border_color)
        bg_color = parse_color(bg_color)
        coords_list = parse_json_tracks(coordinates)

        batch_size = len(coords_list[0])
        images_list = []
        masks_list = []

        if not size_multiplier or len(size_multiplier) != batch_size:
            size_multiplier = [1] * batch_size
        else:
            size_multiplier = size_multiplier * (batch_size // len(size_multiplier)) + size_multiplier[:batch_size % len(size_multiplier)]

        previous_output = None

        for i in range(batch_size):
            image = Image.new("RGB", (frame_width, frame_height), bg_color)
            draw = ImageDraw.Draw(image)

            current_width = shape_width * size_multiplier[i]
            current_height = shape_height * size_multiplier[i]

            for coords in coords_list:
                location_x = coords[i]['x']
                location_y = coords[i]['y']

                if shape == 'circle' or shape == 'square':
                    left_up_point = (location_x - current_width // 2, location_y - current_height // 2)
                    right_down_point = (location_x + current_width // 2, location_y + current_height // 2)
                    two_points = [left_up_point, right_down_point]

                    if shape == 'circle':
                        if border_width > 0:
                            draw.ellipse(two_points, fill=shape_color, outline=border_color, width=border_width)
                        else:
                            draw.ellipse(two_points, fill=shape_color)
                    elif shape == 'square':
                        if border_width > 0:
                            draw.rectangle(two_points, fill=shape_color, outline=border_color, width=border_width)
                        else:
                            draw.rectangle(two_points, fill=shape_color)

                elif shape == 'triangle':
                    left_up_point = (location_x - current_width // 2, location_y + current_height // 2)
                    right_down_point = (location_x + current_width // 2, location_y + current_height // 2)
                    top_point = (location_x, location_y - current_height // 2)

                    if border_width > 0:
                        draw.polygon([top_point, left_up_point, right_down_point], fill=shape_color, outline=border_color, width=border_width)
                    else:
                        draw.polygon([top_point, left_up_point, right_down_point], fill=shape_color)

            if blur_radius != 0:
                image = image.filter(ImageFilter.GaussianBlur(blur_radius))

            image = pil2tensor(image)
            if trailing != 1.0 and previous_output is not None:
                image += trailing * previous_output
                image = image / image.max()
            previous_output = image
            image = image * intensity
            mask = image[:, :, :, 0]
            masks_list.append(mask)
            images_list.append(image)

        out_images = torch.cat(images_list, dim=0).cpu().float()
        out_masks = torch.cat(masks_list, dim=0)
        return (out_images, out_masks)
