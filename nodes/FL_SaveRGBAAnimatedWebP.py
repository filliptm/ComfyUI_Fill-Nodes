import folder_paths
from comfy.cli_args import args

from PIL import Image, ImageDraw
import numpy as np
import os
import zipfile
import json
import cv2

class FL_SaveRGBAAnimatedWebP:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    methods = {"default": 4, "fastest": 0, "slowest": 6}
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images_rgb": ("IMAGE", ),
                     "images_alpha": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                     "fps": ("FLOAT", {"default": 16.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                     "lossless": ("BOOLEAN", {"default": True}),
                     "quality": ("INT", {"default": 80, "min": 0, "max": 100}),
                     "method": (list(s.methods.keys()),),
                     # "num_frames": ("INT", {"default": 0, "min": 0, "max": 8192}),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "🏵️Fill Nodes/Image"

    def save_images(self, images_rgb, images_alpha, fps, filename_prefix, lossless, quality, method, num_frames=0, prompt=None, extra_pnginfo=None):
        method = self.methods.get(method)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images_rgb[0].shape[1], images_rgb[0].shape[0])
        results: list[FileLocator] = []

        def create_checkerboard(size=30, pattern_size=(830, 480), color1=(140, 140, 140), color2=(113, 113, 113)):
            img = Image.new('RGB', (pattern_size[0], pattern_size[1]), color1)
            draw = ImageDraw.Draw(img)
            for i in range(0, pattern_size[0], size):
                for j in range(0, pattern_size[1], size):
                    if (i + j) // size % 2 == 0:
                        draw.rectangle([i, j, i+size, j+size], fill=color2)
            return img

        pil_images = []
        np_images = []
        for image_rgb, image_alpha in zip(images_rgb, images_alpha):

            def image_to_np(image):
                i = 255. * image.cpu().numpy()
                return np.clip(i, 0, 255)
            
            def image_to_single_color_np(image):
                i = 1. * image.cpu().numpy()
                i = (i[:,:,0:1] + i[:,:,1:2] + i[:,:,2:3]) / 3.
                return np.clip(i, 0, 1)
            
            image_rgb = image_to_np(image_rgb)
            alpha_channel = image_to_single_color_np(image_alpha)

            np_images.append(np.concatenate([image_rgb[:,:,::-1], alpha_channel*255], axis=2).astype(np.uint8))
            checkerboard = create_checkerboard(pattern_size=(image_rgb.shape[1], image_rgb.shape[0]))

            images = image_rgb * alpha_channel + checkerboard * (1-alpha_channel)
            img = Image.fromarray(images.astype(np.uint8))
            pil_images.append(img)

        metadata = pil_images[0].getexif()
        if not args.disable_metadata:
            if prompt is not None:
                metadata[0x0110] = "prompt:{}".format(json.dumps(prompt))
            if extra_pnginfo is not None:
                inital_exif = 0x010f
                for x in extra_pnginfo:
                    metadata[inital_exif] = "{}:{}".format(x, json.dumps(extra_pnginfo[x]))
                    inital_exif -= 1

        if num_frames == 0:
            num_frames = len(pil_images)

        c = len(pil_images)
        for i in range(0, c, num_frames):
            file = f"{filename}_{counter:05}_.webp"
            pil_images[i].save(os.path.join(full_output_folder, file), save_all=True, duration=int(1000.0/fps), append_images=pil_images[i + 1:i + num_frames], exif=metadata, lossless=lossless, quality=quality, method=method)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
        
        zip_path = os.path.join(full_output_folder, f"{filename}_{counter-1:05}.zip") 
        print("Saving zip tp", zip_path)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for idx, img in enumerate(np_images):
                success, buffer = cv2.imencode(".png", img)
                if not success:
                    print(f"Failed to encode image {idx}, skipping...")
                    continue
                
                filename = f"img_{idx:03d}.png"
                zipf.writestr(filename, buffer.tobytes())

        animated = num_frames != 1
        return { "ui": { "images": results, "animated": (animated,) }}


