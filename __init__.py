from .nodes.fl_image_randomizer import FL_ImageRandomizer
from .nodes.fl_image_caption_saver import FL_ImageCaptionSaver
from .nodes.fl_image_dimension_display import FL_ImageDimensionDisplay
#from .nodes.fl_audio_frame_calculator import FL_AudioFrameCalculator
#from .nodes.fl_audio_preview import FL_AudioPreview
#from .nodes.fl_image_duration_sync import FL_ImageDurationSync
#from .nodes.fl_vhs_audio import FL_AudioConverter
from .nodes.fl_code_node import FL_CodeNode
from .nodes.fl_image_pixelator import FL_ImagePixelator
from .nodes.fl_directorycrawl import FL_DirectoryCrawl
from .nodes.fl_ascii import FL_Ascii
from .nodes.fl_glitch import FL_Glitch
from .nodes.fl_ripple import FL_Ripple
from .nodes.fl_pixelsort import FL_PixelSort
from .nodes.fl_hexagonalpattern import FL_HexagonalPattern
from .nodes.fl_nftgenerator import FL_NFTGenerator
from .nodes.fl_halftone import FL_HalftonePattern
from .nodes.fl_randomrange import FL_RandomNumber
from .nodes.fl_promptselector import FL_PromptSelector
from .nodes.fl_shader import FL_Shadertoy
from .nodes.fl_pixelshader import FL_PixelArtShader
from .nodes.fl_infinitezoom import FL_InfiniteZoom
from .nodes.fl_paperdrawn import FL_PaperDrawn
from .nodes.fl_imagenotes import FL_ImageNotes
from .nodes.fl_imagecollage import FL_ImageCollage
from .nodes.fl_ksamplersettings import FL_KsamplerSettings
from .nodes.fl_retroeffect import FL_RetroEffect
from .nodes.fl_td_ksampler import FL_TD_KSampler
from .nodes.fl_inpaintcrop import FL_InpaintCrop
from .nodes.fl_inpaintcrop import FL_Inpaint_Stitch

NODE_CLASS_MAPPINGS = {
    "FL_ImageRandomizer": FL_ImageRandomizer,
    "FL_ImageCaptionSaver": FL_ImageCaptionSaver,
    "FL_ImageDimensionDisplay": FL_ImageDimensionDisplay,
    #"FL_AudioPreview": FL_AudioPreview,
    #"FL_ImageDurationSync": FL_ImageDurationSync,
    #"FL_AudioConverter": FL_AudioConverter,
    #"FL_AudioFrameCalculator": FL_AudioFrameCalculator,
    "FL_CodeNode": FL_CodeNode,
    "FL_ImagePixelator": FL_ImagePixelator,
    "FL_DirectoryCrawl": FL_DirectoryCrawl,
    "FL_Ascii": FL_Ascii,
    "FL_Glitch": FL_Glitch,
    "FL_Ripple": FL_Ripple,
    "FL_PixelSort": FL_PixelSort,
    "FL_HexagonalPattern": FL_HexagonalPattern,
    "FL_NFTGenerator": FL_NFTGenerator,
    "FL_HalftonePattern": FL_HalftonePattern,
    "FL_RandomNumber": FL_RandomNumber,
    "FL_PromptSelector": FL_PromptSelector,
    "FL_Shadertoy": FL_Shadertoy,
    "FL_PixelArtShader": FL_PixelArtShader,
    "FL_InfiniteZoom": FL_InfiniteZoom,
    "FL_PaperDrawn": FL_PaperDrawn,
    "FL_ImageNotes": FL_ImageNotes,
    "FL_ImageCollage": FL_ImageCollage,
    "FL_KsamplerSettings": FL_KsamplerSettings,
    "FL_RetroEffect": FL_RetroEffect,
    "FL_TD_Sampler": FL_TD_KSampler,
    "FL_InpaintCrop": FL_InpaintCrop,
    "FL_Inpaint_Stitch": FL_Inpaint_Stitch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ImageRandomizer": "FL Image Randomizer",
    "FL_ImageCaptionSaver": "FL Image Caption Saver",
    "FL_ImageDimensionDisplay": "FL Image Size",
    #"FL_AudioPreview": "FL Audio Preview",
    #"FL_ImageDurationSync": "FL Image Duration Sync",
    #"FL_AudioConverter": "FL VHS Audio Converter",
    #"FL_AudioFrameCalculator": "FL Audio Scanner",
    "FL_CodeNode": "FL Code Node",
    "FL_ImagePixelator": "FL Image Pixelator",
    "FL_DirectoryCrawl": "FL Directory Crawl",
    "FL_Ascii": "FL Ascii",
    "FL_Glitch": "FL Glitch",
    "FL_Ripple": "FL Ripple",
    "FL_PixelSort": "FL PixelSort",
    "FL_HexagonalPattern": "FL Hexagonal Pattern",
    "FL_NFTGenerator": "FL NFT Generator",
    "FL_HalftonePattern": "FL Halftone",
    "FL_RandomNumber": "FL Random Number",
    "FL_PromptSelector": "FL Prompt Selector",
    "FL_Shadertoy": "FL Shadertoy",
    "FL_PixelArtShader": "FL Pixel Art",
    "FL_InfiniteZoom": "FL Infinite Zoom",
    "FL_PaperDrawn": "FL Paper Drawn",
    "FL_ImageNotes": "FL Image Notes",
    "FL_ImageCollage": "FL Image Collage",
    "FL_KsamplerSettings": "FL Ksampler Settings",
    "FL_RetroEffect": "FL Retro Effect",
    "FL_TD_Ksampler": "FL TD Ksampler",
    "FL_InpaintCrop": "FL Inpaint Crop",
    "FL_Inpaint_Stitch": "FL Inpaint Stitch"
}


ascii_art = """

███╗   ███╗ █████╗  ██████╗██╗  ██╗██╗███╗   ██╗███████╗               
████╗ ████║██╔══██╗██╔════╝██║  ██║██║████╗  ██║██╔════╝               
██╔████╔██║███████║██║     ███████║██║██╔██╗ ██║█████╗                 
██║╚██╔╝██║██╔══██║██║     ██╔══██║██║██║╚██╗██║██╔══╝                 
██║ ╚═╝ ██║██║  ██║╚██████╗██║  ██║██║██║ ╚████║███████╗               
╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝               
                                                                       
██████╗ ███████╗██╗     ██╗   ██╗███████╗██╗ ██████╗ ███╗   ██╗███████╗
██╔══██╗██╔════╝██║     ██║   ██║██╔════╝██║██╔═══██╗████╗  ██║██╔════╝
██║  ██║█████╗  ██║     ██║   ██║███████╗██║██║   ██║██╔██╗ ██║███████╗
██║  ██║██╔══╝  ██║     ██║   ██║╚════██║██║██║   ██║██║╚██╗██║╚════██║
██████╔╝███████╗███████╗╚██████╔╝███████║██║╚██████╔╝██║ ╚████║███████║
╚═════╝ ╚══════╝╚══════╝ ╚═════╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
                                                                       

"""
print(ascii_art)

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
