from .fl_image_randomizer import FL_ImageRandomizer
from .fl_image_caption_saver import FL_ImageCaptionSaver

NODE_CLASS_MAPPINGS = {
    "FL_ImageRandomizer": FL_ImageRandomizer,
    "FL_ImageCaptionSaver": FL_ImageCaptionSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ImageRandomizer": "FL Image Randomizer",
    "FL_ImageCaptionSaver": "FL Image Caption Saver"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
