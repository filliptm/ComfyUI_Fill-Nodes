import torch
from nodes import common_ksampler, VAEDecode, VAEEncode
import comfy.samplers
import comfy.utils
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..utils import tensor_to_pil, pil_to_tensor

class FL_KSamplerXYZPlot:
    # Positioning and style parameters
    CELL_MARGIN = 70
    AXIS_LABEL_MARGIN = 150
    FONT_SIZE = 40
    AXIS_LABEL_OFFSET_X = 70
    AXIS_LABEL_OFFSET_Y = 70

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "input_type": (["latent", "image"],),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "x_axis": (["steps", "cfg", "denoise", "sampler_name", "scheduler"],),
                "x_values": ("STRING", {"default": "20,30,40"}),
                "y_axis": (["steps", "cfg", "denoise", "sampler_name", "scheduler"],),
                "y_values": ("STRING", {"default": "7,8,9"}),
                "z_axis": (["none", "steps", "cfg", "denoise", "sampler_name", "scheduler"],),
                "z_values": ("STRING", {"default": ""}),
                "z_stack_mode": (["vertical", "horizontal"],),
            },
            "optional": {
                "latent_image": ("LATENT",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample_xyz_plot"
    CATEGORY = "🏵️Fill Nodes/Ksamplers"

    def parse_range(self, value_string, param_type):
        if param_type in ["sampler_name", "scheduler"]:
            return [x.strip() for x in value_string.split(',')]
        return [float(x.strip()) for x in value_string.split(',')]

    def generate_grid(self, x_values, y_values, z_values):
        if z_values:
            return [(x, y, z) for z in z_values for y in y_values for x in x_values]
        return [(x, y, None) for y in y_values for x in x_values]

    def create_image_grid(self, images, rows, cols, x_values, y_values, x_axis, y_axis):
        if not images:
            raise ValueError("No images provided to create grid")

        cell_width, cell_height = images[0].size
        grid_w = cols * (cell_width + self.CELL_MARGIN) + self.CELL_MARGIN + self.AXIS_LABEL_MARGIN
        grid_h = rows * (cell_height + self.CELL_MARGIN) + self.CELL_MARGIN + self.AXIS_LABEL_MARGIN

        grid = Image.new('RGB', size=(grid_w, grid_h), color='white')
        draw = ImageDraw.Draw(grid)

        try:
            font = ImageFont.truetype("arial.ttf", self.FONT_SIZE)
        except IOError:
            font = ImageFont.load_default()

        # Draw red axes
        draw.line([(self.AXIS_LABEL_MARGIN, grid_h - self.AXIS_LABEL_MARGIN), (grid_w, grid_h - self.AXIS_LABEL_MARGIN)], fill='red', width=2)  # X-axis
        draw.line([(self.AXIS_LABEL_MARGIN, 0), (self.AXIS_LABEL_MARGIN, grid_h - self.AXIS_LABEL_MARGIN)], fill='red', width=2)  # Y-axis

        for i, img in enumerate(images):
            col = i % cols
            row = rows - 1 - (i // cols)  # Flip the row order
            x = col * (cell_width + self.CELL_MARGIN) + self.CELL_MARGIN + self.AXIS_LABEL_MARGIN
            y = row * (cell_height + self.CELL_MARGIN) + self.CELL_MARGIN
            grid.paste(img, (x, y))

        # Add x-axis labels
        for i, label in enumerate(x_values):
            x = i * (cell_width + self.CELL_MARGIN) + self.CELL_MARGIN + self.AXIS_LABEL_MARGIN + cell_width // 2
            y = grid_h - self.AXIS_LABEL_MARGIN + 5
            draw.text((x, y), str(label), fill='black', font=font, anchor='mt')

        # Add y-axis labels
        for i, label in enumerate(y_values):
            x = self.AXIS_LABEL_MARGIN - 5
            y = (rows - 1 - i) * (cell_height + self.CELL_MARGIN) + self.CELL_MARGIN + cell_height // 2
            draw.text((x, y), str(label), fill='black', font=font, anchor='rm')

        # Add axis titles with adjustable positioning
        draw.text((grid_w // 2, grid_h - self.AXIS_LABEL_OFFSET_Y), x_axis, fill='black', font=font, anchor='ms')
        draw.text((self.AXIS_LABEL_OFFSET_X, grid_h // 2), y_axis, fill='black', font=font, anchor='ms', rotation=90)

        return grid

    def sample_xyz_plot(self, model, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise,
                        x_axis, x_values, y_axis, y_values, z_axis, z_values, input_type, z_stack_mode,
                        latent_image=None, image=None, vae=None):
        try:
            device = comfy.model_management.get_torch_device()

            x_values = self.parse_range(x_values, x_axis)
            y_values = self.parse_range(y_values, y_axis)
            z_values = self.parse_range(z_values, z_axis) if z_axis != "none" else [None]

            param_grid = self.generate_grid(x_values, y_values, z_values)

            # Input selection and error handling
            if input_type == "latent":
                if latent_image is None:
                    raise ValueError("Latent input is selected, but no latent image is provided.")
                input_list = [{"samples": latent_image["samples"][i:i+1]} for i in range(latent_image["samples"].shape[0])]
            elif input_type == "image":
                if image is None:
                    raise ValueError("Image input is selected, but no image is provided.")
                if vae is None:
                    raise ValueError("Image input is selected, but no VAE is provided for encoding.")
                input_list = [image[i:i+1] for i in range(image.shape[0])]
            else:
                raise ValueError(f"Invalid input type: {input_type}")

            final_grids = []

            for idx, item in enumerate(input_list):
                results = []
                for x_val, y_val, z_val in param_grid:
                    current_params = {
                        "steps": steps,
                        "cfg": cfg,
                        "denoise": denoise,
                        "sampler_name": sampler_name,
                        "scheduler": scheduler
                    }
                    current_params[x_axis] = x_val
                    current_params[y_axis] = y_val
                    if z_axis != "none":
                        current_params[z_axis] = z_val

                    if input_type == "image":
                        latent = VAEEncode().encode(vae, item)[0]
                    else:
                        latent = item

                    samples = common_ksampler(model, seed + idx, int(current_params["steps"]), current_params["cfg"],
                                              current_params["sampler_name"], current_params["scheduler"],
                                              positive, negative, latent,
                                              denoise=current_params["denoise"])[0]

                    if vae is not None:
                        vae_decoder = VAEDecode()
                        output_image = vae_decoder.decode(vae, samples)[0]
                        results.append(tensor_to_pil(output_image))

                if z_axis != "none":
                    grids = []
                    for i, z_val in enumerate(z_values):
                        start = i * len(x_values) * len(y_values)
                        end = start + len(x_values) * len(y_values)
                        grid = self.create_image_grid(
                            results[start:end], len(y_values), len(x_values),
                            x_values, y_values, x_axis, y_axis
                        )
                        draw = ImageDraw.Draw(grid)
                        font = ImageFont.truetype("arial.ttf", self.FONT_SIZE) if self.FONT_SIZE else ImageFont.load_default()
                        draw.text((200, 10), f"{z_axis}: {z_val}", fill='black', font=font)
                        grids.append(grid)

                    if z_stack_mode == "vertical":
                        total_width = max(grid.width for grid in grids)
                        total_height = sum(grid.height for grid in grids)
                        final_grid = Image.new('RGB', (total_width, total_height), color='white')
                        y_offset = 0
                        for grid in grids:
                            final_grid.paste(grid, (0, y_offset))
                            y_offset += grid.height
                    else:  # horizontal
                        total_width = sum(grid.width for grid in grids)
                        total_height = max(grid.height for grid in grids)
                        final_grid = Image.new('RGB', (total_width, total_height), color='white')
                        x_offset = 0
                        for grid in grids:
                            final_grid.paste(grid, (x_offset, 0))
                            x_offset += grid.width
                else:
                    final_grid = self.create_image_grid(
                        results, len(y_values), len(x_values),
                        x_values, y_values, x_axis, y_axis
                    )

                final_grids.append(pil_to_tensor(final_grid))

            return (torch.cat(final_grids, dim=0),)

        except Exception as e:
            logging.error(f"Error in FL_KsamplerXYZPlot: {str(e)}")
            raise