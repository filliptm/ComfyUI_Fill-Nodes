from .fl_image_randomizer import FL_ImageRandomizer
from .fl_image_caption_saver import FL_ImageCaptionSaver
from .fl_image_dimension_display import FL_ImageDimensionDisplay
from .fl_audio_frame_calculator import FL_AudioFrameCalculator
from .fl_audio_preview import FL_AudioPreview
from .fl_image_duration_sync import FL_ImageDurationSync
from .fl_vhs_audio import FL_AudioConverter
from .fl_code_node import FL_CodeNode
from .fl_image_pixelator import FL_ImagePixelator
from .fl_directorycrawl import FL_DirectoryCrawl

NODE_CLASS_MAPPINGS = {
    "FL_ImageRandomizer": FL_ImageRandomizer,
    "FL_ImageCaptionSaver": FL_ImageCaptionSaver,
    "FL_ImageDimensionDisplay": FL_ImageDimensionDisplay,
    "FL_AudioPreview": FL_AudioPreview,
    "FL_ImageDurationSync": FL_ImageDurationSync,
    "FL_AudioConverter": FL_AudioConverter,
    "FL_AudioFrameCalculator": FL_AudioFrameCalculator,
    "FL_CodeNode": FL_CodeNode,
    "FL_ImagePixelator": FL_ImagePixelator,
    "FL_DirectoryCrawl": FL_DirectoryCrawl
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ImageRandomizer": "FL Image Randomizer",
    "FL_ImageCaptionSaver": "FL Image Caption Saver",
    "FL_ImageDimensionDisplay": "FL Image Size",
    "FL_AudioPreview": "FL Audio Preview",
    "FL_ImageDurationSync": "FL Image Duration Sync",
    "FL_AudioConverter": "FL VHS Audio Converter",
    "FL_AudioFrameCalculator": "FL Audio Scanner",
    "FL_CodeNode": "FL Code Node",
    "FL_ImagePixelator": "FL Image Pixelator",
    "FL_DirectoryCrawl": "FL DirectoryCrawl"
}



WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']



ascii_art = """

███╗   ███╗ █████╗  ██████╗██╗  ██╗██╗███╗   ██╗███████╗    ██████╗ ███████╗██╗     ██╗   ██╗███████╗██╗ ██████╗ ███╗   ██╗███████╗
████╗ ████║██╔══██╗██╔════╝██║  ██║██║████╗  ██║██╔════╝    ██╔══██╗██╔════╝██║     ██║   ██║██╔════╝██║██╔═══██╗████╗  ██║██╔════╝
██╔████╔██║███████║██║     ███████║██║██╔██╗ ██║█████╗      ██║  ██║█████╗  ██║     ██║   ██║███████╗██║██║   ██║██╔██╗ ██║███████╗
██║╚██╔╝██║██╔══██║██║     ██╔══██║██║██║╚██╗██║██╔══╝      ██║  ██║██╔══╝  ██║     ██║   ██║╚════██║██║██║   ██║██║╚██╗██║╚════██║
██║ ╚═╝ ██║██║  ██║╚██████╗██║  ██║██║██║ ╚████║███████╗    ██████╔╝███████╗███████╗╚██████╔╝███████║██║╚██████╔╝██║ ╚████║███████║
╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝    ╚═════╝ ╚══════╝╚══════╝ ╚═════╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝

"""
print(ascii_art)

WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']