from .nodes.FL_Image_Randomizer import FL_ImageRandomizer
from .nodes.FL_Image_Caption_Saver import FL_ImageCaptionSaver
from .nodes.FL_Image_Dimension_Display import FL_ImageDimensionDisplay
from .nodes.FL_GeminiVideoCaptioner import FL_GeminiVideoCaptioner
from .nodes.FL_GeminiImageEditor import FL_GeminiImageEditor
from .nodes.FL_GPT_Image1 import FL_GPT_Image1
from .nodes.FL_Code_Node import FL_CodeNode
from .nodes.FL_Video_Caption_Saver import FL_VideoCaptionSaver
from .nodes.FL_Image_Pixelator import FL_ImagePixelator
from .nodes.FL_Image_AddToBatch import FL_ImageAddToBatch
from .nodes.FL_DirectoryCrawl import FL_DirectoryCrawl
from .nodes.FL_Ascii import FL_Ascii
from .nodes.FL_ReplaceColor import FL_ReplaceColor
from .nodes.FL_ImageAddNoise import FL_ImageAddNoise
from .nodes.FL_WordFrequencyGraph import FL_WordFrequencyGraph
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
from .nodes.FL_GradGen import FL_GradGenerator
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
from .nodes.FL_KsamplerPlusV2 import FL_KsamplerPlusV2
from .nodes.FL_KsamplerBasic import FL_KsamplerBasic
from .nodes.FL_KsamplerFractals import FL_FractalKSampler
from .nodes.FL_UpscaleModel import FL_UpscaleModel
from .nodes.FL_SaveCSV import FL_SaveCSV
from. nodes.FL_KSamplerXYZPlot import FL_KSamplerXYZPlot
from .nodes.FL_SamplerStrings import FL_SamplerStrings
from .nodes.FL_SchedulerStrings import FL_SchedulerStrings
from .nodes.FL_ImageCaptionLayoutPDF import FL_ImageCaptionLayoutPDF
from .nodes.FL_Dalle3 import FL_Dalle3
from .nodes.FL_SaveImages import FL_SaveImages
from .nodes.FL_LoadImage import FL_LoadImage
from .nodes.FL_PDFLoader import FL_PDFLoader
from .nodes.FL_PDFToImage import FL_PDFToImages
from .nodes.FL_PDFSaver import FL_PDFSaver
from .nodes.FL_ImagesToPDF import FL_ImagesToPDF
from .nodes.FL_PDFMerger import FL_PDFMerger
from .nodes.FL_PDFTextExtractor import FL_PDFTextExtractor
from .nodes.FL_PDFImageExtractor import FL_PDFImageExtractor
from .nodes.FL_BulkPDFLoader import FL_BulkPDFLoader
from .nodes.FL_TextToPDF import FL_TextToPDF
from .nodes.FL_PDFEncryptor import FL_PDFEncryptor
from .nodes.FL_SaveAndDisplayImage import FL_SaveAndDisplayImage
from .nodes.FL_OllamaCaptioner import FL_OllamaCaptioner
from .nodes.FL_ImageAdjuster import FL_ImageAdjuster
from .nodes.FL_Caption_Saver_V2 import FL_CaptionSaver_V2
from .nodes.FL_PathTypeChecker import FL_PathTypeChecker
from .nodes.FL_SaveWebpImages import FL_SaveWebPImage
from .nodes.FL_MadLibGenerator import FL_MadLibGenerator
from .nodes.FL_Math import FL_Math
from .nodes.FL_ImageSlicer import FL_ImageSlicer
from .nodes.FL_ImageSelector import FL_ImageSelector
from .nodes.FL_ImageAspectCropper import FL_ImageAspectCropper
from .nodes.FL_HF_UploaderAbsolute import FL_HF_UploaderAbsolute
from .nodes.FL_ImageBatchListConverter import FL_ImageListToImageBatch, FL_ImageBatchToImageList
from .nodes.FL_ImageBatchToGrid import FL_ImageBatchToGrid
from .nodes.FL_ApplyMask import FL_ApplyMask
from .nodes.FL_ProResVideo import FL_ProResVideo
from .nodes.FL_Padding import FL_Padding
from .nodes.FL_GoogleDriveDownloader import FL_GoogleDriveDownloader
from .nodes.FL_NodeLoader import FL_NodeLoader
from .nodes.FL_NodePackLoader import FL_NodePackLoader
from .nodes.FL_API_Base64_ImageLoader import FL_API_Base64_ImageLoader
from .nodes.FL_API_ImageSaver import FL_API_ImageSaver
from .nodes.FL_GoogleDriveImageDownloader import FL_GoogleDriveImageDownloader
from .nodes.FL_AnimeLineExtractor import FL_AnimeLineExtractor
from .nodes.FL_HunyuanDelight import FL_HunyuanDelight
from .nodes.FL_ClipScanner import FL_ClipScanner
from .nodes.FL_VideoCut import FL_VideoCut
from .nodes.FL_GeminiTextAPI import FL_GeminiTextAPI
from .nodes.FL_JS import FL_JS
from .nodes.FL_HFDatasetDownloader import FL_HFDatasetDownloader
from .nodes.FL_WF_Agent import FL_WF_Agent
from .nodes.FL_BlackFrameReject import FL_BlackFrameReject
from .nodes.FL_PixVerseAPI import FL_PixVerseAPI
from .nodes.FL_Fal_Pixverse import FL_Fal_Pixverse
from .nodes.FL_Fal_Kontext import FL_Fal_Kontext
from .nodes.FL_Fal_Gemini_ImageEdit import FL_Fal_Gemini_ImageEdit
from .nodes.FL_Fal_Seedance_i2v import FL_Fal_Seedance_i2v
from .nodes.FL_Fal_Seedream_Edit import FL_Fal_Seedream_Edit
from .nodes.FL_Fal_Pixverse_Transition import FL_Fal_Pixverse_Transition
from .nodes.FL_Fal_Pixverse_LipSync import FL_Fal_Pixverse_LipSync
from .nodes.FL_Fal_Kling_AIAvatar import FL_Fal_Kling_AIAvatar
from .nodes.FL_Prompt import FL_PromptBasic
from .nodes.FL_PromptMulti import FL_PromptMulti
from .nodes.FL_PaddingRemover import FL_PaddingRemover
from .nodes.FL_GPT_Text import FL_GPT_Text
from .nodes.FL_GoogleCloudStorage import FL_GoogleCloudStorage
from .nodes.FL_Switch import FL_Switch
from .nodes.FL_Switch_Big import FL_Switch_Big
from .nodes.FL_PasteByMask import FL_PasteByMask
from .nodes.FL_ModelInspector import FL_ModelInspector
from .nodes.FL_Float import FL_Float
from .nodes.FL_UnloadModel import FL_UnloadModel, FL_UnloadAllModels
from .nodes.FL_VideoTrim import FL_VideoTrim
from .nodes.FL_VideoBatchSplitter import FL_VideoBatchSplitter
from .nodes.FL_VideoCrossfade import FL_VideoCrossfade
from .nodes.FL_VideoCadence import FL_VideoCadence
from .nodes.FL_VideoCadenceCompile import FL_VideoCadenceCompile
from .nodes.FL_GeminiImageGenADV import FL_GeminiImageGenADV
from .nodes.FL_GPT_Image1_ADV import FL_GPT_Image1_ADV
from .nodes.FL_ImageBatch import FL_ImageBatch
from .nodes.FL_Hedra_API import FL_Hedra_API
from .nodes.FL_RunwayImageAPI import FL_RunwayImageAPI
from .nodes.FL_RunwayAct2 import FL_RunwayAct2
from .nodes.FL_Image_Blank import FL_ImageBlank
from .nodes.FL_TextOverlay import FL_TextOverlayNode
from .nodes.FL_SaveWebM import FL_SaveWebM
from .nodes.FL_Image_Crop import FL_ImageCrop
from .nodes.FL_WanFirstLastFrameToVideo import FL_WanFirstLastFrameToVideo


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
    "FL_Fal_Pixverse_Transition": FL_Fal_Pixverse_Transition,
    "FL_Fal_Pixverse_LipSync": FL_Fal_Pixverse_LipSync,
    "FL_Fal_Kling_AIAvatar": FL_Fal_Kling_AIAvatar,
    "FL_PromptBasic": FL_PromptBasic,
    "FL_PromptMulti": FL_PromptMulti,
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
    "FL_GeminiImageGenADV": FL_GeminiImageGenADV,
    "FL_GPT_Image1_ADV": FL_GPT_Image1_ADV,
    "FL_ImageBatch": FL_ImageBatch,
    "FL_Hedra_API": FL_Hedra_API,
    "FL_RunwayImageAPI": FL_RunwayImageAPI,
    "FL_RunwayAct2": FL_RunwayAct2,
    "FL_ImageCrop": FL_ImageCrop,
    "FL_WanFirstLastFrameToVideo": FL_WanFirstLastFrameToVideo,
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
    "FL_Fal_Pixverse_Transition": "FL Fal Pixverse Transition",
    "FL_Fal_Pixverse_LipSync": "FL Fal Pixverse LipSync",
    "FL_Fal_Kling_AIAvatar": "FL Fal Kling AI Avatar",
    "FL_PromptBasic": "FL Prompt Basic",
    "FL_PromptMulti": "FL Prompt Multi",
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
    "FL_VideoCadenceCompile": "FL Video Cadence Compile",
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