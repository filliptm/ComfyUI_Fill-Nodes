import os
from PIL import Image, ImageOps


class FL_MirrorAndAppendCaptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_directory": ("STRING", {"default": "X://path/to/images"}),
                "caption_extension": ([".caption", ".txt"], {"default": ".txt"}),
                "additional_text": ("STRING", {"default": "Frame"}),
                "text_position": (["append", "prepend"], {"default": "append"}),
                "create_mirrors": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("directory",)
    OUTPUT_NODE = True
    FUNCTION = "start"

    CATEGORY = "🏵️Fill Nodes/Captioning"

    def start(self, image_directory, caption_extension, additional_text, text_position, create_mirrors):
        if not os.path.exists(image_directory):
            raise Exception(f"Directory {image_directory} does not exist")

        image_files = [f for f in os.listdir(image_directory) if f.lower().endswith((".png", ".jpg", ".webp", ".jpeg"))]
        image_files.sort()  # Ensure consistent order

        new_images = []
        new_captions = []
        frame_count = 0

        for index, image_file in enumerate(image_files):
            image_path = os.path.join(image_directory, image_file)
            caption_path = os.path.splitext(image_path)[0] + caption_extension

            # Process original image
            pil_image = Image.open(image_path)
            new_images.append((pil_image, image_file))

            # Process original caption
            if os.path.exists(caption_path):
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()

                frame_text = f"{additional_text}_{frame_count}"
                
                if text_position == "append":
                    new_caption = f"{caption}, {frame_text}"
                else:  # prepend
                    new_caption = f"{frame_text}, {caption}"

                new_captions.append((new_caption, image_file))
                frame_count += 1

            # Process mirrored image and caption if enabled
            if create_mirrors:
                mirrored_image = ImageOps.mirror(pil_image)
                mirrored_image_file = os.path.splitext(image_file)[0] + "_Mirror" + os.path.splitext(image_file)[1]
                new_images.append((mirrored_image, mirrored_image_file))

                if os.path.exists(caption_path):
                    mirrored_frame_text = f"{additional_text}_{frame_count}"
                    
                    if text_position == "append":
                        new_mirrored_caption = f"{caption}, {mirrored_frame_text}"
                    else:  # prepend
                        new_mirrored_caption = f"{mirrored_frame_text}, {caption}"

                    new_captions.append((new_mirrored_caption, mirrored_image_file))
                    frame_count += 1

        # Save new images and captions
        for img, filename in new_images:
            img.save(os.path.join(image_directory, filename))

        for caption, filename in new_captions:
            caption_filename = os.path.splitext(filename)[0] + caption_extension
            with open(os.path.join(image_directory, caption_filename), "w", encoding="utf-8") as f:
                f.write(caption)

        return (image_directory,)


# Register the node in the ComfyUI system
def register_node():
    return FL_MirrorAndAppendCaptions