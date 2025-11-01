# AI NODES
from .nodes.ai.FL_Fal_Gemini_ImageEdit import FL_Fal_Gemini_ImageEdit
from .nodes.ai.FL_Fal_Kling_AIAvatar import FL_Fal_Kling_AIAvatar
from .nodes.ai.FL_Fal_Kontext import FL_Fal_Kontext
from .nodes.ai.FL_Fal_Pixverse import FL_Fal_Pixverse
from .nodes.ai.FL_Fal_Pixverse_LipSync import FL_Fal_Pixverse_LipSync
from .nodes.ai.FL_Fal_Pixverse_Transition import FL_Fal_Pixverse_Transition
from .nodes.ai.FL_Fal_Seedance_i2v import FL_Fal_Seedance_i2v
from .nodes.ai.FL_Fal_Seedream_Edit import FL_Fal_Seedream_Edit
from .nodes.ai.FL_Fal_SeedVR_Upscale import FL_Fal_SeedVR_Upscale
from .nodes.ai.FL_Fal_Sora import FL_Fal_Sora
from .nodes.ai.FL_GeminiImageEditor import FL_GeminiImageEditor
from .nodes.ai.FL_GeminiImageGenADV import FL_GeminiImageGenADV
from .nodes.ai.FL_GeminiTextAPI import FL_GeminiTextAPI
from .nodes.ai.FL_GeminiVideoCaptioner import FL_GeminiVideoCaptioner
from .nodes.ai.FL_Hedra_API import FL_Hedra_API
from .nodes.ai.FL_HunyuanDelight import FL_HunyuanDelight
from .nodes.ai.FL_PixVerseAPI import FL_PixVerseAPI
from .nodes.ai.FL_RunwayAct2 import FL_RunwayAct2
from .nodes.ai.FL_RunwayImageAPI import FL_RunwayImageAPI
from .nodes.ai.FL_VertexGemini25FlashImage import FL_VertexGemini25FlashImage
from .nodes.ai.FL_VertexVeo3 import FL_Veo3VideoGen

# API_TOOLS NODES
from .nodes.api_tools.FL_API_Base64_ImageLoader import FL_API_Base64_ImageLoader
from .nodes.api_tools.FL_API_ImageSaver import FL_API_ImageSaver

# AUDIO NODES
from .nodes.audio.FL_Audio_BPM_Analyzer import FL_Audio_BPM_Analyzer
from .nodes.audio.FL_Audio_Beat_Visualizer import FL_Audio_Beat_Visualizer
from .nodes.audio.FL_Audio_Crop import FL_Audio_Crop
from .nodes.audio.FL_Audio_Drum_Detector import FL_Audio_Drum_Detector
from .nodes.audio.FL_Audio_Envelope_Visualizer import FL_Audio_Envelope_Visualizer
from .nodes.audio.FL_Audio_Music_Video_Sequencer import FL_Audio_Music_Video_Sequencer
from .nodes.audio.FL_Audio_Reactive_Brightness import FL_Audio_Reactive_Brightness
from .nodes.audio.FL_Audio_Reactive_Edge_Glow import FL_Audio_Reactive_Edge_Glow
from .nodes.audio.FL_Audio_Reactive_Envelope import FL_Audio_Reactive_Envelope
from .nodes.audio.FL_Audio_Reactive_Saturation import FL_Audio_Reactive_Saturation
from .nodes.audio.FL_Audio_Reactive_Scale import FL_Audio_Reactive_Scale
from .nodes.audio.FL_Audio_Reactive_Speed import FL_Audio_Reactive_Speed
from .nodes.audio.FL_Audio_Segment_Extractor import FL_Audio_Segment_Extractor
from .nodes.audio.FL_Audio_Separation import FL_Audio_Separation
from .nodes.audio.FL_Audio_Shot_Iterator import FL_Audio_Shot_Iterator

# CAPTIONING NODES
from .nodes.captioning.FL_CSVExtractor import FL_CSVExtractor
from .nodes.captioning.FL_CaptionToCSV import FL_CaptionToCSV
from .nodes.captioning.FL_Caption_Saver_V2 import FL_CaptionSaver_V2
from .nodes.captioning.FL_ImageCaptionLayout import FL_ImageCaptionLayout
from .nodes.captioning.FL_ImageCaptionLayoutPDF import FL_ImageCaptionLayoutPDF
from .nodes.captioning.FL_Image_Caption_Saver import FL_ImageCaptionSaver
from .nodes.captioning.FL_LoadCSV import FL_LoadCSV
from .nodes.captioning.FL_MirrorAndAppendCaptions import FL_MirrorAndAppendCaptions
from .nodes.captioning.FL_OllamaCaptioner import FL_OllamaCaptioner
from .nodes.captioning.FL_SaveCSV import FL_SaveCSV
from .nodes.captioning.FL_Video_Caption_Saver import FL_VideoCaptionSaver
from .nodes.captioning.FL_WordFrequencyGraph import FL_WordFrequencyGraph

# DISCORD NODES
from .nodes.discord.FL_DiscordWebhook import FL_SendToDiscordWebhook

# EXPERIMENTS NODES
from .nodes.experiments.FL_BatchAligned import FL_BatchAlign
from .nodes.experiments.FL_ColorPicker import FL_ColorPicker
from .nodes.experiments.FL_GradGen import FL_GradGenerator
from .nodes.experiments.FL_NFTGenerator import FL_NFTGenerator

# FILE_OPERATIONS NODES
from .nodes.file_operations.FL_ZipDirectory import FL_ZipDirectory
from .nodes.file_operations.FL_ZipSave import FL_ZipSave

# GAMES NODES
from .nodes.games.FL_BulletHellGame import FL_BulletHellGame
from .nodes.games.FL_TetrisGame import FL_TetrisGame

# GOOGLE_DRIVE NODES
from .nodes.google_drive.FL_GoogleCloudStorage import FL_GoogleCloudStorage
from .nodes.google_drive.FL_GoogleDriveDownloader import FL_GoogleDriveDownloader
from .nodes.google_drive.FL_GoogleDriveImageDownloader import FL_GoogleDriveImageDownloader

# GPT NODES
from .nodes.gpt.FL_Dalle3 import FL_Dalle3
from .nodes.gpt.FL_GPT_Image1 import FL_GPT_Image1
from .nodes.gpt.FL_GPT_Image1_ADV import FL_GPT_Image1_ADV
from .nodes.gpt.FL_GPT_Text import FL_GPT_Text
from .nodes.gpt.FL_GPT_Vision import FL_GPT_Vision
from .nodes.gpt.FL_SimpleGPTVision import FL_SimpleGPTVision

# HUGGING_FACE NODES
from .nodes.hugging_face.FL_HFDatasetDownloader import FL_HFDatasetDownloader
from .nodes.hugging_face.FL_HFHubModelUploader import FL_HFHubModelUploader
from .nodes.hugging_face.FL_HF_Character import FL_HF_Character
from .nodes.hugging_face.FL_HF_UploaderAbsolute import FL_HF_UploaderAbsolute

# IMAGE NODES
from .nodes.image.FL_AnimeLineExtractor import FL_AnimeLineExtractor
from .nodes.image.FL_ApplyMask import FL_ApplyMask
from .nodes.image.FL_BlackFrameReject import FL_BlackFrameReject
from .nodes.image.FL_ImageAddNoise import FL_ImageAddNoise
from .nodes.image.FL_ImageAdjuster import FL_ImageAdjuster
from .nodes.image.FL_ImageAspectCropper import FL_ImageAspectCropper
from .nodes.image.FL_ImageBatch import FL_ImageBatch
from .nodes.image.FL_ImageBatchListConverter import FL_ImageListToImageBatch, FL_ImageBatchToImageList
from .nodes.image.FL_ImageBatchToGrid import FL_ImageBatchToGrid
from .nodes.image.FL_ImageNotes import FL_ImageNotes
from .nodes.image.FL_ImageOverlay import FL_ImageOverlay
from .nodes.image.FL_ImageSelector import FL_ImageSelector
from .nodes.image.FL_ImageSlicer import FL_ImageSlicer
from .nodes.image.FL_Image_AddToBatch import FL_ImageAddToBatch
from .nodes.image.FL_Image_Blank import FL_ImageBlank
from .nodes.image.FL_Image_Crop import FL_ImageCrop
from .nodes.image.FL_Image_Dimension_Display import FL_ImageDimensionDisplay
from .nodes.image.FL_Image_Randomizer import FL_ImageRandomizer
from .nodes.image.FL_LoadImage import FL_LoadImage
from .nodes.image.FL_PaddingRemover import FL_PaddingRemover
from .nodes.image.FL_ReplaceColor import FL_ReplaceColor
from .nodes.image.FL_SaveAndDisplayImage import FL_SaveAndDisplayImage
from .nodes.image.FL_SaveImages import FL_SaveImages
from .nodes.image.FL_SaveRGBAAnimatedWebP import FL_SaveRGBAAnimatedWebP
from .nodes.image.FL_SaveWebM import FL_SaveWebM
from .nodes.image.FL_SaveWebpImages import FL_SaveWebPImage

# KSAMPLERS NODES
from .nodes.ksamplers.FL_KsamplerBasic import FL_KsamplerBasic
from .nodes.ksamplers.FL_KsamplerPlus import FL_KsamplerPlus
from .nodes.ksamplers.FL_KsamplerPlusV2 import FL_KsamplerPlusV2
from .nodes.ksamplers.FL_KsamplerSettings import FL_KsamplerSettings
from .nodes.ksamplers.FL_SamplerStrings import FL_SamplerStrings
from .nodes.ksamplers.FL_SchedulerStrings import FL_SchedulerStrings
from .nodes.ksamplers.FL_KSamplerXYZPlot import FL_KSamplerXYZPlot

# LOADERS NODES
from .nodes.loaders.FL_NodeLoader import FL_NodeLoader
from .nodes.loaders.FL_NodePackLoader import FL_NodePackLoader
from .nodes.loaders.FL_UpscaleModel import FL_UpscaleModel

# PDF NODES
from .nodes.pdf.FL_BulkPDFLoader import FL_BulkPDFLoader
from .nodes.pdf.FL_ImagesToPDF import FL_ImagesToPDF
from .nodes.pdf.FL_PDFEncryptor import FL_PDFEncryptor
from .nodes.pdf.FL_PDFImageExtractor import FL_PDFImageExtractor
from .nodes.pdf.FL_PDFLoader import FL_PDFLoader
from .nodes.pdf.FL_PDFMerger import FL_PDFMerger
from .nodes.pdf.FL_PDFSaver import FL_PDFSaver
from .nodes.pdf.FL_PDFTextExtractor import FL_PDFTextExtractor
from .nodes.pdf.FL_PDFToImage import FL_PDFToImages
from .nodes.pdf.FL_TextToPDF import FL_TextToPDF

# PROMPTING NODES
from .nodes.prompting.FL_MadLibGenerator import FL_MadLibGenerator
from .nodes.prompting.FL_Prompt import FL_PromptBasic
from .nodes.prompting.FL_PromptMulti import FL_PromptMulti
from .nodes.prompting.FL_PromptSelector import FL_PromptSelector
from .nodes.prompting.FL_PromptSelectorBasic import FL_PromptSelectorBasic

# UTILITY NODES
from .nodes.utility.FL_ClipScanner import FL_ClipScanner
from .nodes.utility.FL_Code_Node import FL_CodeNode
from .nodes.utility.FL_DirectoryCrawl import FL_DirectoryCrawl
from .nodes.utility.FL_Float import FL_Float
from .nodes.utility.FL_InpaintCrop import FL_InpaintCrop
from .nodes.utility.FL_InpaintCrop import FL_Inpaint_Stitch
from .nodes.utility.FL_JS import FL_JS
from .nodes.utility.FL_Math import FL_Math
from .nodes.utility.FL_ModelInspector import FL_ModelInspector
from .nodes.utility.FL_NumberConverter import FL_IntToFloat, FL_FloatToInt
from .nodes.utility.FL_Padding import FL_Padding
from .nodes.utility.FL_PasteByMask import FL_PasteByMask
from .nodes.utility.FL_PasteOnCanvas import FL_PasteOnCanvas
from .nodes.utility.FL_PathTypeChecker import FL_PathTypeChecker
from .nodes.utility.FL_RandomRange import FL_RandomNumber
from .nodes.utility.FL_SD_Slices import FL_SDUltimate_Slices
from .nodes.utility.FL_SeparateMasks import FL_SeparateMaskComponents
from .nodes.utility.FL_Switch import FL_Switch
from .nodes.utility.FL_Switch_Big import FL_Switch_Big
from .nodes.utility.FL_SystemCheck import FL_SystemCheck
from .nodes.utility.FL_UnloadModel import FL_UnloadModel, FL_UnloadAllModels
from .nodes.utility.FL_VideoCropNStitch import FL_VideoCropMask
from .nodes.utility.FL_VideoCropNStitch import FL_VideoRecompose

# VFX NODES
from .nodes.vfx.FL_Ascii import FL_Ascii
from .nodes.vfx.FL_Dither import FL_Dither
from .nodes.vfx.FL_Glitch import FL_Glitch
from .nodes.vfx.FL_HalfTone import FL_HalftonePattern
from .nodes.vfx.FL_HexagonalPattern import FL_HexagonalPattern
from .nodes.vfx.FL_ImageCollage import FL_ImageCollage
from .nodes.vfx.FL_Image_Pixelator import FL_ImagePixelator
from .nodes.vfx.FL_InfiniteZoom import FL_InfiniteZoom
from .nodes.vfx.FL_PaperDrawn import FL_PaperDrawn
from .nodes.vfx.FL_PixelArt import FL_PixelArtShader
from .nodes.vfx.FL_PixelSort import FL_PixelSort
from .nodes.vfx.FL_RetroEffect import FL_RetroEffect
from .nodes.vfx.FL_Ripple import FL_Ripple
from .nodes.vfx.FL_Shader import FL_Shadertoy
from .nodes.vfx.FL_TextOverlay import FL_TextOverlayNode

# VIDEO NODES
from .nodes.video.FL_FILM import FL_FILM
from .nodes.video.FL_ProResVideo import FL_ProResVideo
from .nodes.video.FL_RIFE import FL_RIFE
from .nodes.video.FL_VideoBatchSplitter import FL_VideoBatchSplitter
from .nodes.video.FL_VideoCadence import FL_VideoCadence
from .nodes.video.FL_VideoCadenceCompile import FL_VideoCadenceCompile
from .nodes.video.FL_VideoCrossfade import FL_VideoCrossfade
from .nodes.video.FL_VideoCut import FL_VideoCut
from .nodes.video.FL_VideoTrim import FL_VideoTrim

# WIP NODES
from .nodes.wip.FL_KsamplerFractals import FL_FractalKSampler
from .nodes.wip.FL_TimeLine import FL_TimeLine
from .nodes.wip.FL_WF_Agent import FL_WF_Agent
from .nodes.wip.FL_WanFirstLastFrameToVideo import FL_WanFirstLastFrameToVideo
from .nodes.wip.FL_QwenImageEditStrength import FL_QwenImageEditStrength
from .nodes.wip.FL_WanVaceToVideoMultiRef import FL_WanVaceToVideoMultiRef
from .nodes.wip.FL_AnimatedShapePatterns import FL_AnimatedShapePatterns
from .nodes.wip.FL_PathAnimator import FL_PathAnimator

NODE_CLASS_MAPPINGS = {
    "FL_SaveWebM": FL_SaveWebM,
    "FL_TextOverlayNode": FL_TextOverlayNode,
    "FL_ImageBlank": FL_ImageBlank,
    "FL_ImageRandomizer": FL_ImageRandomizer,
    "FL_ImageCaptionSaver": FL_ImageCaptionSaver,
    "FL_VideoCaptionSaver": FL_VideoCaptionSaver,
    "FL_ImageDimensionDisplay": FL_ImageDimensionDisplay,
    "FL_GeminiVideoCaptioner": FL_GeminiVideoCaptioner,
    "FL_GeminiImageEditor": FL_GeminiImageEditor,
    "FL_GPT_Image1": FL_GPT_Image1,
    "FL_CodeNode": FL_CodeNode,
    "FL_ImagePixelator": FL_ImagePixelator,
    "FL_ImageAddToBatch": FL_ImageAddToBatch,
    "FL_DirectoryCrawl": FL_DirectoryCrawl,
    "FL_Ascii": FL_Ascii,
    "FL_ReplaceColor": FL_ReplaceColor,
    "FL_ImageAddNoise": FL_ImageAddNoise,
    "FL_WordFrequencyGraph": FL_WordFrequencyGraph,
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
    "FL_GradGenerator": FL_GradGenerator,
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
    "FL_KsamplerPlusV2": FL_KsamplerPlusV2,
    "FL_KsamplerBasic": FL_KsamplerBasic,
    "FL_FractalKSampler": FL_FractalKSampler,
    "FL_UpscaleModel": FL_UpscaleModel,
    "FL_SaveCSV": FL_SaveCSV,
    "FL_LoadCSV": FL_LoadCSV,
    "FL_CSVExtractor": FL_CSVExtractor,
    "FL_KSamplerXYZPlot": FL_KSamplerXYZPlot,
    "FL_SamplerStrings": FL_SamplerStrings,
    "FL_SchedulerStrings": FL_SchedulerStrings,
    "FL_ImageCaptionLayoutPDF": FL_ImageCaptionLayoutPDF,
    "FL_Dalle3": FL_Dalle3,
    "FL_SaveImages": FL_SaveImages,
    "FL_LoadImage": FL_LoadImage,
    "FL_PDFLoader": FL_PDFLoader,
    "FL_PDFToImages": FL_PDFToImages,
    "FL_PDFSaver": FL_PDFSaver,
    "FL_ImagesToPDF": FL_ImagesToPDF,
    "FL_PDFMerger": FL_PDFMerger,
    "FL_PDFTextExtractor": FL_PDFTextExtractor,
    "FL_PDFImageExtractor": FL_PDFImageExtractor,
    "FL_BulkPDFLoader": FL_BulkPDFLoader,
    "FL_TextToPDF": FL_TextToPDF,
    "FL_PDFEncryptor": FL_PDFEncryptor,
    "FL_SaveAndDisplayImage": FL_SaveAndDisplayImage,
    "FL_OllamaCaptioner": FL_OllamaCaptioner,
    "FL_ImageAdjuster": FL_ImageAdjuster,
    "FL_CaptionSaver_V2": FL_CaptionSaver_V2,
    "FL_PathTypeChecker": FL_PathTypeChecker,
    "FL_SaveWebPImage(SaveImage)": FL_SaveWebPImage,
    "FL_MadLibGenerator": FL_MadLibGenerator,
    "FL_Math": FL_Math,
    "FL_ImageSlicer": FL_ImageSlicer,
    "FL_ImageSelector": FL_ImageSelector,
    "FL_ImageOverlay": FL_ImageOverlay,
    "FL_ImageAspectCropper": FL_ImageAspectCropper,
    "FL_HF_UploaderAbsolute": FL_HF_UploaderAbsolute,
    "FL_ImageListToImageBatch": FL_ImageListToImageBatch,
    "FL_ImageBatchToImageList": FL_ImageBatchToImageList,
    "FL_ImageBatchToGrid": FL_ImageBatchToGrid,
    "FL_ApplyMask": FL_ApplyMask,
    "FL_ProResVideo": FL_ProResVideo,
    "FL_Padding": FL_Padding,
    "FL_GoogleDriveDownloader": FL_GoogleDriveDownloader,
    "FL_NodeLoader": FL_NodeLoader,
    "FL_NodePackLoader": FL_NodePackLoader,
    "FL_API_Base64_ImageLoader": FL_API_Base64_ImageLoader,
    "FL_API_ImageSaver": FL_API_ImageSaver,
    "FL_GoogleDriveImageDownloader": FL_GoogleDriveImageDownloader,
    "FL_AnimeLineExtractor": FL_AnimeLineExtractor,
    "FL_HunyuanDelight": FL_HunyuanDelight,
    "FL_ClipScanner": FL_ClipScanner,
    "FL_VideoCut": FL_VideoCut,
    "FL_GeminiTextAPI": FL_GeminiTextAPI,
    "FL_JS": FL_JS,
    "FL_HFDatasetDownloader": FL_HFDatasetDownloader,
    "FL_WF_Agent": FL_WF_Agent,
    "FL_BlackFrameReject": FL_BlackFrameReject,
    "FL_PixVerseAPI": FL_PixVerseAPI,
    "FL_Fal_Pixverse": FL_Fal_Pixverse,
    "FL_Fal_Kontext": FL_Fal_Kontext,
    "FL_Fal_Gemini_ImageEdit": FL_Fal_Gemini_ImageEdit,
    "FL_Fal_Seedance_i2v": FL_Fal_Seedance_i2v,
    "FL_Fal_Seedream_Edit": FL_Fal_Seedream_Edit,
    "FL_Fal_SeedVR_Upscale": FL_Fal_SeedVR_Upscale,
    "FL_Fal_Pixverse_Transition": FL_Fal_Pixverse_Transition,
    "FL_Fal_Pixverse_LipSync": FL_Fal_Pixverse_LipSync,
    "FL_Fal_Kling_AIAvatar": FL_Fal_Kling_AIAvatar,
    "FL_Fal_Sora": FL_Fal_Sora,
    "FL_PromptBasic": FL_PromptBasic,
    "FL_PromptMulti": FL_PromptMulti,
    "FL_PromptSelectorBasic": FL_PromptSelectorBasic,
    "FL_PaddingRemover": FL_PaddingRemover,
    "FL_GPT_Text": FL_GPT_Text,
    "FL_GoogleCloudStorage": FL_GoogleCloudStorage,
    "FL_Switch": FL_Switch,
    "FL_Switch_Big": FL_Switch_Big,
    "FL_PasteByMask": FL_PasteByMask,
    "FL_ModelInspector": FL_ModelInspector,
    "FL_Float": FL_Float,
    "FL_UnloadModel": FL_UnloadModel,
    "FL_UnloadAllModels": FL_UnloadAllModels,
    "FL_VideoTrim": FL_VideoTrim,
    "FL_VideoBatchSplitter": FL_VideoBatchSplitter,
    "FL_VideoCrossfade": FL_VideoCrossfade,
    "FL_VideoCadence": FL_VideoCadence,
    "FL_VideoCadenceCompile": FL_VideoCadenceCompile,
    "FL_WanVaceToVideoMultiRef": FL_WanVaceToVideoMultiRef,
    "FL_RIFE": FL_RIFE,
    "FL_FILM": FL_FILM,
    "FL_GeminiImageGenADV": FL_GeminiImageGenADV,
    "FL_GPT_Image1_ADV": FL_GPT_Image1_ADV,
    "FL_ImageBatch": FL_ImageBatch,
    "FL_Hedra_API": FL_Hedra_API,
    "FL_RunwayImageAPI": FL_RunwayImageAPI,
    "FL_RunwayAct2": FL_RunwayAct2,
    "FL_ImageCrop": FL_ImageCrop,
    "FL_WanFirstLastFrameToVideo": FL_WanFirstLastFrameToVideo,
    "FL_Veo3VideoGen": FL_Veo3VideoGen,
    "FL_VertexGemini25FlashImage": FL_VertexGemini25FlashImage,
    "FL_SaveRGBAAnimatedWebP": FL_SaveRGBAAnimatedWebP,
    "FL_Audio_BPM_Analyzer": FL_Audio_BPM_Analyzer,
    "FL_Audio_Beat_Visualizer": FL_Audio_Beat_Visualizer,
    "FL_Audio_Crop": FL_Audio_Crop,
    "FL_Audio_Drum_Detector": FL_Audio_Drum_Detector,
    "FL_Audio_Envelope_Visualizer": FL_Audio_Envelope_Visualizer,
    "FL_Audio_Music_Video_Sequencer": FL_Audio_Music_Video_Sequencer,
    "FL_Audio_Reactive_Brightness": FL_Audio_Reactive_Brightness,
    "FL_Audio_Reactive_Edge_Glow": FL_Audio_Reactive_Edge_Glow,
    "FL_Audio_Reactive_Envelope": FL_Audio_Reactive_Envelope,
    "FL_Audio_Reactive_Saturation": FL_Audio_Reactive_Saturation,
    "FL_Audio_Reactive_Scale": FL_Audio_Reactive_Scale,
    "FL_Audio_Reactive_Speed": FL_Audio_Reactive_Speed,
    "FL_Audio_Segment_Extractor": FL_Audio_Segment_Extractor,
    "FL_Audio_Separation": FL_Audio_Separation,
    "FL_Audio_Shot_Iterator": FL_Audio_Shot_Iterator,
    "FL_QwenImageEditStrength": FL_QwenImageEditStrength,
    "FL_IntToFloat": FL_IntToFloat,
    "FL_FloatToInt": FL_FloatToInt,
    "FL_AnimatedShapePatterns": FL_AnimatedShapePatterns,
    "FL_PathAnimator": FL_PathAnimator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ImageBlank": "FL Image Blank",
    "FL_ImageRandomizer": "FL Image Randomizer",
    "FL_ImageCaptionSaver": "FL Image Caption Saver",
    "FL_VideoCaptionSaver": "FL Video Caption Saver",
    "FL_ImageDimensionDisplay": "FL Image Size",
    "FL_GeminiVideoCaptioner": "FL Gemini Video Captioner",
    "FL_GeminiImageEditor": "FL Gemini Image Editor",
    "FL_GPT_Image1": "FL GPT Image-1",
    "FL_CodeNode": "FL Code Node",
    "FL_ImagePixelator": "FL Image Pixelator",
    "FL_ImageAddToBatch": "FL Image Add To Batch",
    "FL_DirectoryCrawl": "FL Directory Crawl",
    "FL_Ascii": "FL Ascii",
    "FL_ReplaceColor": "FL Replace Color",
    "FL_ImageAddNoise": "FL Image Add Noise",
    "FL_WordFrequencyGraph": "FL Word Frequency Graph",
    "FL_Glitch": "FL Glitch",
    "FL_Ripple": "FL Ripple",
    "FL_PixelSort": "FL PixelSort",
    "FL_HexagonalPattern": "FL Hexagonal Pattern",
    "FL_NFTGenerator": "FL NFT Generator",
    "FL_HalftonePattern": "FL Halftone",
    "FL_RandomNumber": "FL Random Number",
    "FL_PromptSelector": "FL Prompt Selector",
    "FL_PromptSelectorBasic": "FL Prompt Selector Basic",
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
    "FL_GradGenerator": "FL Grad Generator",
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
    "FL_KsamplerPlusV2": "FL KSampler Plus V2",
    "FL_KsamplerBasic": "FL KSampler Basic",
    "FL_FractalKSampler": "FL Fractal KSampler",
    "FL_UpscaleModel": "FL Upscale Model",
    "FL_SaveCSV": "FL Save CSV",
    "FL_LoadCSV": "FL Load CSV",
    "FL_CSVExtractor": "FL CSV Extractor",
    "FL_KSamplerXYZPlot": "FL KSampler XYZ Plot",
    "FL_SamplerStrings": "FL Sampler String XYZ",
    "FL_SchedulerStrings": "FL Scheduler String XYZ",
    "FL_ImageCaptionLayoutPDF": "FL Image Caption Layout PDF",
    "FL_Dalle3": "FL Dalle 3",
    "FL_SaveImages": "FL Save Images",
    "FL_LoadImage": "FL Load Image",
    "FL_PDFLoader": "FL PDF Loader",
    "FL_PDFToImages": "FL PDF To Images",
    "FL_PDFSaver": "FL PDF Saver",
    "FL_ImagesToPDF": "FL Images To PDF",
    "FL_PDFMerger": "FL PDF Merger",
    "FL_PDFTextExtractor": "FL PDF Text Extractor",
    "FL_PDFImageExtractor": "FL PDF Image Extractor",
    "FL_BulkPDFLoader": "FL Bulk PDF Loader",
    "FL_TextToPDF": "FL Text To PDF",
    "FL_PDFEncryptor": "FL PDF Encryptor",
    "FL_SaveAndDisplayImage": "FL Save And Display Image",
    "FL_OllamaCaptioner": "FL Ollama Captioner by Cosmic",
    "FL_ImageAdjuster": "FL Image Adjuster",
    "FL_CaptionSaver_V2": "FL Caption Saver V2",
    "FL_PathTypeChecker": "FL Path Type Checker",
    "FL_SaveWebPImage(SaveImage)": "FL Save WebP Image",
    "FL_MadLibGenerator": "FL MadLib Generator",
    "FL_Math": "FL Math",
    "FL_ImageSlicer": "FL Image Slicer",
    "FL_ImageSelector": "FL Image Selector",
    "FL_ImageOverlay": "FL Image Overlay",
    "FL_ImageAspectCropper": "FL Image Aspect Cropper",
    "FL_HF_UploaderAbsolute": "FL HF Uploader Absolute",
    "FL_ImageListToImageBatch": "FL Image List To Image Batch",
    "FL_ImageBatchToImageList": "FL Image Batch To Image List",
    "FL_ImageBatchToGrid": "FL Image Batch To Grid",
    "FL_ApplyMask": "FL Apply Mask",
    "FL_ProResVideo": "FL ProRes Video",
    "FL_Padding": "FL Padding",
    "FL_GoogleDriveDownloader": "FL Google Drive Downloader",
    "FL_NodeLoader": "FL Node Loader",
    "FL_NodePackLoader": "FL Node Pack Loader",
    "FL_API_Base64_ImageLoader": "FL API Base64 Image Loader",
    "FL_API_ImageSaver": "FL API Image Saver",
    "FL_GoogleDriveImageDownloader": "FL Google Drive Image Downloader",
    "FL_AnimeLineExtractor": "FL Anime Line Extractor",
    "FL_HunyuanDelight": "FL Hunyuan Delight",
    "FL_ClipScanner": "FL Clip Scanner (Kytra)",
    "FL_VideoCut": "FL Video Cut Detector",
    "FL_GeminiTextAPI": "FL Gemini Text API",
    "FL_JS": "FL JavaScript",
    "FL_HFDatasetDownloader": "FL HF Dataset Downloader",
    "FL_WF_Agent": "FL Workflow Agent",
    "FL_BlackFrameReject": "FL Black Frame Reject",
    "FL_PixVerseAPI": "FL PixVerse API",
    "FL_Fal_Pixverse": "FL Fal Pixverse API",
    "FL_Fal_Kontext": "FL Fal Kontext API",
    "FL_Fal_Gemini_ImageEdit": "FL Fal Gemini Image Edit",
    "FL_Fal_Seedance_i2v": "FL Fal Seedance i2v",
    "FL_Fal_Seedream_Edit": "FL Fal Seedream Edit",
    "FL_Fal_SeedVR_Upscale": "FL Fal SeedVR Upscale",
    "FL_Fal_Pixverse_Transition": "FL Fal Pixverse Transition",
    "FL_Fal_Pixverse_LipSync": "FL Fal Pixverse LipSync",
    "FL_Fal_Kling_AIAvatar": "FL Fal Kling AI Avatar",
    "FL_Fal_Sora": "FL Fal Sora 2",
    "FL_PromptBasic": "FL Prompt Basic",
    "FL_PromptMulti": "FL Prompt Multi",
    "FL_PromptSelectorBasic": "FL Prompt Selector Basic",
    "FL_PaddingRemover": "FL Padding Remover",
    "FL_GPT_Text": "FL GPT Text",
    "FL_GoogleCloudStorage": "FL Google Cloud Storage Uploader",
    "FL_Switch": "FL Switch",
    "FL_Switch_Big": "FL Switch Big",
    "FL_PasteByMask": "FL Paste By Mask",
    "FL_Float": "FL Float",
    "FL_UnloadModel": "FL Unload Model",
    "FL_UnloadAllModels": "FL Unload All Models",
    "FL_ModelInspector": "FL Model Inspector",
    "FL_VideoTrim": "FL Video Trim",
    "FL_VideoBatchSplitter": "FL Video Batch Splitter",
    "FL_VideoCrossfade": "FL Video Crossfade",
    "FL_VideoCadence": "FL Video Cadence",
    "FL_WanVaceToVideoMultiRef": "FL Wan Vace To Video Multi Reference",
    "FL_VideoCadenceCompile": "FL Video Cadence Compile",
    "FL_FILM": "FL FILM Frame Interpolation",
    "FL_RIFE": "FL RIFE Frame Interpolation",
    "FL_GeminiImageGenADV": "FL Gemini Image Gen ADV",
    "FL_GPT_Image1_ADV": "FL GPT Image1 ADV",
    "FL_ImageBatch": "FL Image Batch",
    "FL_Hedra_API": "FL Hedra API",
    "FL_RunwayImageAPI": "FL Runway Image API",
    "FL_RunwayAct2": "FL Runway Act2",
    "FL_TextOverlayNode": "FL Text Overlay",
    "FL_SaveWebM": "FL Save WebM",
    "FL_ImageCrop": "FL Image Crop",
    "FL_WanFirstLastFrameToVideo": "FL Wan First Frame Last Frame",
    "FL_Veo3VideoGen": "FL Vertex Veo3",
    "FL_VertexGemini25FlashImage": "FL Vertex Gemini 2.5 Flash Image",
    "FL_SaveRGBAAnimatedWebP": "FL Save RGBA Animated WebP",
    "FL_Audio_BPM_Analyzer": "FL Audio BPM Analyzer",
    "FL_Audio_Beat_Visualizer": "FL Audio Beat Visualizer",
    "FL_Audio_Crop": "FL Audio Crop",
    "FL_Audio_Drum_Detector": "FL Audio Drum Detector",
    "FL_Audio_Envelope_Visualizer": "FL Audio Envelope Visualizer",
    "FL_Audio_Music_Video_Sequencer": "FL Audio Music Video Sequencer",
    "FL_Audio_Reactive_Brightness": "FL Audio Reactive Brightness",
    "FL_Audio_Reactive_Edge_Glow": "FL Audio Reactive Edge Glow",
    "FL_Audio_Reactive_Envelope": "FL Audio Reactive Envelope",
    "FL_Audio_Reactive_Saturation": "FL Audio Reactive Saturation",
    "FL_Audio_Reactive_Scale": "FL Audio Reactive Scale",
    "FL_Audio_Reactive_Speed": "FL Audio Reactive Speed",
    "FL_Audio_Segment_Extractor": "FL Audio Segment Extractor",
    "FL_Audio_Separation": "FL Audio Separation",
    "FL_Audio_Shot_Iterator": "FL Audio Shot Iterator",
    "FL_QwenImageEditStrength": "FL Qwen Image Edit with Strength",
    "FL_IntToFloat": "FL Int to Float",
    "FL_FloatToInt": "FL Float to Int",
    "FL_AnimatedShapePatterns": "FL Animated Shape Patterns",
    "FL_PathAnimator": "FL Path Animator",
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

# Using OpenCV for video processing

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]