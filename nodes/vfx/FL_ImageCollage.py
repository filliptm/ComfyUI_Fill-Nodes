import torch
import numpy as np
from PIL import Image
import sys
from comfy.utils import ProgressBar

class FL_ImageCollage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "tile_image": ("IMAGE",),
                "tile_size": ("INT", {"default": 32, "min": 8, "max": 256, "step": 8}),
                "spacing": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_collage"
    CATEGORY = "🏵️Fill Nodes/VFX"

    def create_collage(self, base_image, tile_image, tile_size, spacing):
        base_batch_size = len(base_image)
        tile_batch_size = len(tile_image)

        if tile_batch_size == 1:
            # Duplicate the single tile image to match the base image batch size
            tile_image = tile_image.repeat(base_batch_size, 1, 1, 1)
        elif tile_batch_size != base_batch_size:
            raise ValueError(f"The number of tile images ({tile_batch_size}) does not match the number of base images ({base_batch_size}).")

        result = []
        pbar = ProgressBar(base_batch_size)

        for i, (base_img, tile_img) in enumerate(zip(base_image, tile_image), start=1):
            base_img = self.t2p(base_img)
            tile_img = self.t2p(tile_img)
            result_img = self.create_collage_image(base_img, tile_img, tile_size, spacing)
            result_img = self.p2t(result_img)
            result.append(result_img)

            # Update the print log
            progress = i / base_batch_size * 100
            sys.stdout.write(f"\rProcessing images: {progress:.2f}%")
            sys.stdout.flush()

        # Print a new line after the progress log
        print()

        return (torch.cat(result, dim=0),)

    def create_collage_image(self, base_image, tile_image, tile_size, spacing):
        base_width, base_height = base_image.size
        tile_width, tile_height = tile_image.size

        # Calculate the aspect ratio of the tile image
        aspect_ratio = tile_width / tile_height

        # Calculate the new dimensions of the tile image while maintaining the aspect ratio
        if tile_width > tile_height:
            new_tile_width = tile_size
            new_tile_height = int(tile_size / aspect_ratio)
        else:
            new_tile_width = int(tile_size * aspect_ratio)
            new_tile_height = tile_size

        # Resize the tile image to the new dimensions
        tile_image = tile_image.resize((new_tile_width, new_tile_height), Image.Resampling.LANCZOS)

        # Create a new blank image for the collage
        collage_image = Image.new("RGB", base_image.size)

        for y in range(0, base_height, new_tile_height + spacing):
            for x in range(0, base_width, new_tile_width + spacing):
                # Get the average color of the corresponding region in the base image
                region = base_image.crop((x, y, x + new_tile_width, y + new_tile_height))
                avg_color = tuple(np.array(region).mean(axis=(0, 1)).astype(int))

                # Create a mask based on the brightness of the tile image
                tile_mask = Image.new("L", (new_tile_width, new_tile_height), 0)
                tile_mask_data = np.array(tile_image.convert("L"))
                tile_mask_data = (tile_mask_data / 255.0) ** 2  # Adjust the brightness sensitivity
                tile_mask.putdata(np.uint8(tile_mask_data.flatten() * 255))

                # Colorize the tile image based on the average color of the base image region
                colorized_tile = Image.new("RGB", (new_tile_width, new_tile_height), avg_color)
                colorized_tile.putalpha(tile_mask)

                # Paste the colorized tile onto the collage image
                collage_image.paste(colorized_tile, (x, y), mask=tile_mask)

        return collage_image

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