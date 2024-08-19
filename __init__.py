from .nodes.FL_Image_Randomizer import FL_ImageRandomizer
from .nodes.FL_Image_Caption_Saver import FL_ImageCaptionSaver
from .nodes.FL_Image_Dimension_Display import FL_ImageDimensionDisplay
from .nodes.FL_Code_Node import FL_CodeNode
from .nodes.FL_Image_Pixelator import FL_ImagePixelator
from .nodes.FL_DirectoryCrawl import FL_DirectoryCrawl
from .nodes.FL_Ascii import FL_Ascii
from .nodes.FL_Glitch import FL_Glitch
from .nodes.FL_Ripple import FL_Ripple
from .nodes.FL_PixelSort import FL_PixelSort
from .nodes.FL_HexagonalPattern import FL_HexagonalPattern
from .nodes.FL_NFTGenerator import FL_NFTGenerator
from .nodes.FL_HalfTone import FL_HalftonePattern
from .nodes.FL_RandomRange import FL_RandomNumber
from .nodes.FL_PromptSelector import FL_PromptSelector
from .nodes.FL_Shader import FL_Shadertoy
from .nodes.FL_PixelArt import FL_PixelArtShader
from .nodes.FL_InfiniteZoom import FL_InfiniteZoom
from .nodes.FL_PaperDrawn import FL_PaperDrawn
from .nodes.FL_ImageNotes import FL_ImageNotes
from .nodes.FL_ImageCollage import FL_ImageCollage
from .nodes.FL_KsamplerSettings import FL_KsamplerSettings
from .nodes.FL_RetroEffect import FL_RetroEffect
from .nodes.FL_InpaintCrop import FL_InpaintCrop
from .nodes.FL_InpaintCrop import FL_Inpaint_Stitch
from .nodes.FL_SD_Slices import FL_SDUltimate_Slices
from .nodes.FL_BatchAligned import FL_BatchAlign
from .nodes.FL_VideoCropNStitch import FL_VideoCropMask
from .nodes.FL_VideoCropNStitch import FL_VideoRecompose
from .nodes.FL_SeparateMasks import FL_SeparateMaskComponents
from .nodes.FL_PasteOnCanvas import FL_PasteOnCanvas
from .nodes.FL_BulletHellGame import FL_BulletHellGame
from .nodes.FL_TetrisGame import FL_TetrisGame
from .nodes.FL_Dither import FL_Dither
from .nodes.FL_SystemCheck import FL_SystemCheck
from .nodes.FL_ColorPicker import FL_ColorPicker
from .nodes.FL_GradGen import GradientImageGenerator
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
from .nodes.FL_KsamplerBasic import FL_KsamplerBasic
from .nodes.FL_KsamplerFractals import FL_FractalKSampler
from .nodes.FL_UpscaleModel import FL_UpscaleModel
from .nodes.FL_SaveCSV import FL_SaveCSV




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
    "FL_KsamplerBasic": FL_KsamplerBasic,
    "FL_FractalKSampler": FL_FractalKSampler,
    "FL_UpscaleModel": FL_UpscaleModel,
    "FL_SaveCSV": FL_SaveCSV,

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
    "FL_KsamplerSettings": "FL KSampler Settings",
    "FL_RetroEffect": "FL Retro Effect",
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
    "FL_KsamplerPlus": "FL KSampler Plus",
    "FL_KsamplerBasic": "FL KSampler Basic",
    "FL_FractalKSampler": "FL Fractal KSampler",
    "FL_UpscaleModel": "FL Upscale Model",
    "FL_SaveCSV": "FL Save CSV",

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
