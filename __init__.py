from .nodes.fl_image_randomizer import FL_ImageRandomizer
from .nodes.FL_Image_Caption_Saver import FL_ImageCaptionSaver
from .nodes.fl_image_dimension_display import FL_ImageDimensionDisplay
from .nodes.FL_Code_Node import FL_CodeNode
from .nodes.fl_image_pixelator import FL_ImagePixelator
from .nodes.FL_DirectoryCrawl import FL_DirectoryCrawl
from .nodes.FL_Ascii import FL_Ascii
from .nodes.FL_Glitch import FL_Glitch
from .nodes.fl_ripple import FL_Ripple
from .nodes.fl_pixelsort import FL_PixelSort
from .nodes.FL_HexagonalPattern import FL_HexagonalPattern
from .nodes.fl_nftgenerator import FL_NFTGenerator
from .nodes.FL_HalfTone import FL_HalftonePattern
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
from .nodes.fl_sdultimate_slices import FL_SDUltimate_Slices
from .nodes.FL_BatchAligned import FL_BatchAlign
from .nodes.fl_videocropnstitch import FL_VideoCropMask
from .nodes.fl_videocropnstitch import FL_VideoRecompose
from .nodes.fl_separatemasks import FL_SeparateMaskComponents
from .nodes.fl_pasteoncanvas import FL_PasteOnCanvas
from .nodes.FL_BulletHellGame import FL_BulletHellGame
from .nodes.fl_tetrisgame import FL_TetrisGame
from .nodes.FL_Dither import FL_Dither
from .nodes.FL_SystemCheck import FL_SystemCheck
from .nodes.FL_ColorPicker import FL_ColorPicker
from .nodes.GradientImageGenerator import GradientImageGenerator
from .nodes.FL_MirrorAndAppendCaptions import FL_MirrorAndAppendCaptions
from .nodes.FL_ImageCaptionLayout import FL_ImageCaptionLayout
from .nodes.FL_HFHubModelUploader import FL_HFHubModelUploader
from .nodes.FL_ZipDirectory import FL_ZipDirectory
from .nodes.FL_ZipSave import FL_ZipSave
from .nodes.FL_GPT_Vision import FL_GPT_Vision
from .nodes.FL_TimeLine import FL_TimeLine
from .nodes.FL_SimpleGPTVision import FL_SimpleGPTVision
from .nodes.FL_DiscordWebhook import FL_SendToDiscordWebhook
from .nodes.FL_HF_Character import FL_HF_Character
from .nodes.FL_CaptionToCSV import FL_CaptionToCSV
from .nodes.FL_KsamplerPlus import FL_KsamplerPlus




NODE_CLASS_MAPPINGS = {
    "FL_ImageRandomizer": FL_ImageRandomizer,
    "FL_ImageCaptionSaver": FL_ImageCaptionSaver,
    "FL_ImageDimensionDisplay": FL_ImageDimensionDisplay,
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
    "FL_Inpaint_Stitch": FL_Inpaint_Stitch,
    "FL_SDUltimate_Slices": FL_SDUltimate_Slices,
    "FL_BatchAlign": FL_BatchAlign,
    "FL_VideoRecompose": FL_VideoRecompose,
    "FL_VideoCropMask": FL_VideoCropMask,
    "FL_SeparateMaskComponents": FL_SeparateMaskComponents,
    "FL_PasteOnCanvas": FL_PasteOnCanvas,
    "FL_BulletHellGame": FL_BulletHellGame,
    "FL_TetrisGame": FL_TetrisGame,
    "FL_Dither": FL_Dither,
    "FL_SystemCheck": FL_SystemCheck,
    "FL_ColorPicker": FL_ColorPicker,
    "GradientImageGenerator": GradientImageGenerator,
    "FL_MirrorAndAppendCaptions": FL_MirrorAndAppendCaptions,
    "FL_ImageCaptionLayout": FL_ImageCaptionLayout,
    "FL_HFHubModelUploader": FL_HFHubModelUploader,
    "FL_ZipDirectory": FL_ZipDirectory,
    "FL_ZipSave": FL_ZipSave,
    "FL_GPT_Vision": FL_GPT_Vision,
    "FL_TimeLine": FL_TimeLine,
    "FL_SimpleGPTVision": FL_SimpleGPTVision,
    "FL_SendToDiscordWebhook": FL_SendToDiscordWebhook,
    "FL_HF_Character": FL_HF_Character,
    "FL_CaptionToCSV": FL_CaptionToCSV,
    "FL_KsamplerPlus": FL_KsamplerPlus,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ImageRandomizer": "FL Image Randomizer",
    "FL_ImageCaptionSaver": "FL Image Caption Saver",
    "FL_ImageDimensionDisplay": "FL Image Size",
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
    "FL_Inpaint_Stitch": "FL Inpaint Stitch",
    "FL_SDUltimate_Slices": "FL SDUltimate Slices",
    "FL_BatchAlign": "FL Batch Align",
    "FL_VideoCropMask": "FL Video CropMask",
    "FL_VideoRecompose": "FL Video Recompose",
    "FL_SeparateMaskComponents": "FL Separate Mask Components",
    "FL_PasteOnCanvas": "FL Paste On Canvas",
    "FL_BulletHellGame": "FL BulletHell Game",
    "FL_TetrisGame": "FL Tetris Game",
    "FL_Dither": "FL Dither",
    "FL_SystemCheck": "FL System Check",
    "FL_ColorPicker": "FL Color Picker",
    "GradientImageGenerator": "GradientImageGenerator",
    "FL_MirrorAndAppendCaptions": "FL Mirror And Append Captions",
    "FL_ImageCaptionLayout": "FL Image Caption Layout",
    "FL_HFHubModelUploader": "FL HFHub Model Uploader",
    "FL_ZipDirectory": "FL Zip Directory",
    "FL_ZipSave": "FL_ZipSave",
    "FL_GPT_Vision": "FL GPT Captions",
    "FL_TimeLine": "FL Time Line",
    "FL_SimpleGPTVision": "FL Simple GPT Vision",
    "FL_SendToDiscordWebhook": "FL Kytra Discord Webhook",
    "FL_HF_Character": "FL HF Character",
    "FL_CaptionToCSV": "FL Caption To CSV",
    "FL_KsamplerPlus": "FL Ksampler Plus",

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
