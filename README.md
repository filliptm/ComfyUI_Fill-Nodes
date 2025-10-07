# 🏵️ Fill-Nodes

If you enjoy this project, consider supporting me on Patreon!
<p align="left">
  <a href="https://www.patreon.com/c/Machinedelusions">
    <img src="images\\nodes\\Patreon.png" width="150px" alt="Patreon">
  </a>
</p>

Fill-Nodes is a versatile collection of custom nodes for ComfyUI that extends functionality across multiple domains. Features include advanced image processing, visual effects generation, comprehensive file handling (PDF creation/extraction, Google Drive integration), AI model interfaces (GPT, DALL-E, Hugging Face, Runway, Gemini, and more), utility nodes for workflow enhancement, audio-reactive visual effects, and specialized tools for video processing, captioning, and batch operations. The pack provides both practical workflow solutions and creative tools within a unified node collection.

## 🔍 Table of Contents

<table>
  <tr>
    <td valign="top">
      <ul>
        <li><a href="#image-nodes">🖼️ Image Nodes</a></li>
        <li><a href="#captioning-nodes">📝 Captioning Nodes</a></li>
        <li><a href="#vfx-nodes">✨ VFX Nodes</a></li>
        <li><a href="#utility-nodes">🛠️ Utility Nodes</a></li>
        <li><a href="#ksamplers-nodes">🎲 KSamplers Nodes</a></li>
        <li><a href="#pdf-nodes">📄 PDF Nodes</a></li>
        <li><a href="#gpt-nodes">🤖 GPT Nodes</a></li>
      </ul>
    </td>
    <td valign="top">
      <ul>
        <li><a href="#ai-nodes">🧠 AI Nodes</a></li>
        <li><a href="#audio-nodes">🔊 Audio Nodes</a></li>
        <li><a href="#experimental-nodes">🧪 Experimental Nodes</a></li>
        <li><a href="#prompting-nodes">💬 Prompting Nodes</a></li>
        <li><a href="#file-operation-nodes">📂 File Operation Nodes</a></li>
        <li><a href="#google-drive-nodes">☁️ Google Drive Nodes</a></li>
        <li><a href="#api-tool-nodes">🔌 API Tool Nodes</a></li>
      </ul>
    </td>
    <td valign="top">
      <ul>
        <li><a href="#hugging-face-nodes">🤗 Hugging Face Nodes</a></li>
        <li><a href="#loader-nodes">⏬ Loader Nodes</a></li>
        <li><a href="#discord-nodes">💬 Discord Nodes</a></li>
        <li><a href="#work-in-progress-nodes">🚧 Work-in-Progress Nodes</a></li>
        <li><a href="#game-nodes">🎮 Game Nodes</a></li>
        <li><a href="#video-nodes">🎬 Video Nodes</a></li>
      </ul>
    </td>
  </tr>
</table>

---

<details>
<summary><h2 id="-image-nodes">🖼️ Image Nodes</h2></summary>

> *Nodes for manipulating, analyzing, and working with images.*

| Node | Description |
|------|-------------|
| `FL_AnimeLineExtractor` | Extracts line art from anime-style images using adaptive thresholding and morphological operations. Allows fine control over line thickness, detail level, and noise reduction with options to invert output for white backgrounds. |
| `FL_ApplyMask` | Applies a mask to an image by setting it as the alpha channel. Automatically handles different batch sizes and spatial dimensions by interpolating the mask if needed. |
| `FL_BlackFrameReject` | A node that filters out images from a batch based on the percentage of black pixels. Images with a black pixel percentage above the threshold are rejected. |
| `FL_ImageAddNoise` | Node for imageaddnoise functionality. |
| `FL_ImageAddToBatch` | Node for imageaddtobatch functionality. |
| `FL_ImageAdjuster` | Provides comprehensive image adjustment controls for hue, saturation, brightness, contrast, and sharpness with real-time preview capability. Converts images between tensor and PIL formats to apply precise adjustments while maintaining proper color spaces. |
| `FL_ImageAspectCropper` | Node for imageaspectcropper functionality. |
| `FL_ImageBatch` | Combines multiple images into a batched tensor. |
| `FL_ImageBatchToGrid` | Arranges batched image tensors into a single grid layout with configurable number of images per row. Creates a visually organized grid by calculating rows and columns automatically based on batch size, placing images in row-major order. |
| `FL_ImageBlank` | Creates blank/solid color images with configurable dimensions. |
| `FL_ImageCrop` | Crops images to specified dimensions with offset positions. |
| `FL_ImageDimensionDisplay` | Node for imagedimensiondisplay functionality. |
| `FL_ImageListToImageBatch` | Node for imagelisttoimagebatch functionality. |
| `FL_ImageNotes` | Adds a text bar with custom notes to images, supporting batch processing. Allows configuration of bar height and text size with automatic text centering on the bar, maintaining the original image proportions below the added text. |
| `FL_ImageRandomizer` | Node for imagerandomizer functionality. |
| `FL_ImageSelector` | Node for imageselector functionality. |
| `FL_ImageSlicer` | Slices input images into a configurable grid of smaller image pieces with equal dimensions. Divides each input image based on specified X and Y subdivision counts, returning all slices as a batched tensor for further processing. |
| `FL_LoadImage` | Interactive node with a file browser interface for selecting and loading images directly within ComfyUI. Includes REST API endpoints for directory navigation, file listing, and thumbnail generation with proper file validation and error handling. |
| `FL_PaddingRemover` | Automatically detects and removes uniform padding from images. |
| `FL_ReplaceColor` | Node for replacecolor functionality. |
| `FL_SaveImages` | Saves batch-processed images to disk with support for complex folder structures defined through JSON configuration. Features sequential file naming with configurable templates, optional metadata storage, and collision avoidance through automatic index incrementation. |
| `FL_SaveRGBAAnimatedWebP` | Saves animated WebP images with RGBA support. |
| `FL_SaveWebM` | Saves image sequences as WebM videos with alpha support. |

### 📷 Screenshots & Examples

![🖼️ Image Nodes Example](images/nodes/image_nodes.png)

</details>

---

<details>
<summary><h2 id="-captioning-nodes">📝 Captioning Nodes</h2></summary>

> *Nodes for generating, saving, and manipulating image and video captions.*

| Node | Description |
|------|-------------|
| `FL_CSVExtractor` | Node for csvextractor functionality. |
| `FL_CaptionSaver_V2` | Node for captionsaver v2 functionality. |
| `FL_CaptionToCSV` | Scans a directory for image files and their corresponding text captions, then compiles them into a CSV file with image filename and caption columns. Automatically handles missing caption files and sorts entries for consistency. |
| `FL_ImageCaptionLayout` | Creates visual layouts combining images with their corresponding text captions from a directory. Supports customizable grid arrangements with configurable image size, caption height, font size, and padding, automatically wrapping text to fit within the allocated caption area. |
| `FL_ImageCaptionLayoutPDF` | Creates PDF documents displaying a grid of images with their corresponding captions, with options for horizontal or vertical orientation. Provides both the PDF output and a preview image tensor with configurable display size, caption height, font size, and padding. |
| `FL_ImageCaptionSaver` | Node for imagecaptionsaver functionality. |
| `FL_LoadCSV` | Loads CSV files with captions and metadata. |
| `FL_MirrorAndAppendCaptions` | Processes a directory of images to create horizontally mirrored copies and updates caption files with additional text. Supports both appending and prepending frame identifiers to captions with proper file extension handling and consistent frame numbering across the collection. |
| `FL_OllamaCaptioner` | Integrates with Ollama local LLM API to generate descriptive captions for images. Encodes images as base64 for API submission, saves both images and generated captions to disk with configurable overwrite protection and customizable model selection. |
| `FL_SaveCSV` | Saves caption data to CSV format. |
| `FL_VideoCaptionSaver` | Node for videocaptionsaver functionality. |
| `FL_WordFrequencyGraph` | Generates word frequency visualizations from text. |

### 📷 Screenshots & Examples

![📝 Captioning Nodes Example](images/nodes/captioning_nodes.png)

</details>

---

<details>
<summary><h2 id="-vfx-nodes">✨ VFX Nodes</h2></summary>

> *Nodes for applying visual effects to images.*

| Node | Description |
|------|-------------|
| `FL_Ascii` | Converts images to ASCII art using customizable characters, fonts and spacing. Supports using characters in sequence or mapping them by pixel intensity, with options to use system or local fonts. |
| `FL_Dither` | Applies various dithering algorithms (Floyd-Steinberg, Random, Ordered, Bayer) to images with adjustable color quantization. Supports batch processing and preserves original image dimensions while reducing the color palette to create retro-style visual effects. |
| `FL_Glitch` | Creates digital glitch effects on images using the glitch_this library with adjustable intensity and optional color offset. Implements a double-pass glitch effect with rotation between passes to create more complex distortions, and supports deterministic output through seed control. |
| `FL_HalftonePattern` | Node for halftonepattern functionality. |
| `FL_HexagonalPattern` | Creates a mosaic of hexagon-shaped image segments with customizable sizing, spacing, rotation, and shadow effects. Transforms input images into an artistic hexagonal grid pattern with adjustable parameters for visual styling. |
| `FL_ImageCollage` | Creates collages by tiling a smaller image across a base image, with the tiles colored based on the average color of the corresponding region. Supports customizable tile size and spacing with automatic handling of mismatched batch sizes and aspect ratio preservation. |
| `FL_ImagePixelator` | Node for imagepixelator functionality. |
| `FL_InfiniteZoom` | Creates mesmerizing infinite zoom effects using OpenGL shaders with customizable scale, mirror effects, and animation speed. Processes both individual images and batches with progressive time shifts to create seamless zoom animations when combined into video. |
| `FL_PaperDrawn` | Applies a realistic hand-drawn paper effect using OpenGL shaders with adjustable parameters for line quality, sampling, and vignetting. Processes images through a custom gradient-based algorithm that simulates pen strokes with configurable density and temporal modulation for animations. |
| `FL_PixelArtShader` | Node for pixelartshader functionality. |
| `FL_PixelSort` | Applies pixel sorting effects based on saturation values with adjustable threshold, smoothing, and rotation parameters. Creates glitch art aesthetics by identifying intervals in the image and sorting pixels within those intervals according to their color properties. |
| `FL_RetroEffect` | Applies retro visual effects to images including color channel offset, scanlines, vignetting, and noise with adjustable strength parameters. Creates nostalgic aesthetics reminiscent of old CRT displays and vintage photography through multiple image processing techniques. |
| `FL_Ripple` | Creates dynamic ripple and wave effects emanating from a configurable center point with adjustable amplitude, frequency, and phase settings. Supports temporal modulation for animated sequences and provides precise control over the distortion pattern and intensity. |
| `FL_Shadertoy` | Node for shadertoy functionality. |
| `FL_TextOverlayNode` | Node for textoverlaynode functionality. |

### 📷 Screenshots & Examples

![✨ VFX Nodes Example](images/nodes/vfx_nodes.png)

</details>

---

<details>
<summary><h2 id="-utility-nodes">🛠️ Utility Nodes</h2></summary>

> *General utility nodes for various tasks.*

| Node | Description |
|------|-------------|
| `FL_ClipScanner` | Analyzes text prompts using different CLIP models (supporting SDXL, SD 1.5, and FLUX configurations) to show how they are tokenized. Provides detailed output showing the exact tokens processed, their count, and the specific model configuration used. |
| `FL_CodeNode` | Node for codenode functionality. |
| `FL_DirectoryCrawl` | Recursively searches directories for image or text files and loads them as a batch. Supports configurable file type filtering, automatic image resizing to match dimensions within batches, and limits to prevent overloading memory with large directories. |
| `FL_Float` | Simple float value input node. |
| `FL_InpaintCrop` | Crops images for inpainting based on mask regions with advanced context expansion options and automatic padding controls. Provides both free-size and forced-size modes with optional rescaling, hole filling, and mask inversion for precise control over the inpainting area. |
| `FL_JS` | Executes custom JavaScript code. |
| `FL_Math` | Evaluates mathematical expressions using three input variables (A, B, C) with support for basic operations, powers, and negative values. Provides both integer and floating-point outputs with proper error handling for invalid expressions. |
| `FL_ModelInspector` | Inspects and displays model information. |
| `FL_Padding` | Adds customizable padding to images with independently configurable top, bottom, left, and right border widths. Supports RGB color selection for padding areas and optional alpha channel preservation with proper tensor handling. |
| `FL_PasteByMask` | Pastes `image_to_paste` onto `image_base` using `mask` to determine the location. The `resize_behavior` parameter determines how the image to paste is resized to fit the mask. If `mask_mapping_optional` obtained from a 'Separate Mask Components' node is used, it will control which image gets pasted onto which base image. |
| `FL_PasteOnCanvas` | Pastes images onto a canvas with alpha masking, supporting custom canvas dimensions, background color, padding, and multiple resize algorithms. Automatically handles batch processing with proper alpha blending and optional use of background images. |
| `FL_PathTypeChecker` | Analyzes input paths and identifies their type (absolute, relative, drive-specific, UNC, or URL-like). Provides path validation and classification to help identify potential issues with file system references. |
| `FL_RandomNumber` | Node for randomnumber functionality. |
| `FL_SDUltimate_Slices` | Node for sdultimate slices functionality. |
| `FL_SeparateMaskComponents` | Analyzes masks and separates them into individual component masks based on connected components analysis. Outputs both the separated mask tensors and mappings to their original batch indices, enabling advanced mask manipulation for selective processing. |
| `FL_Switch` | Routes input based on boolean condition. |
| `FL_Switch_Big` | Extended switch with multiple routing options. |
| `FL_SystemCheck` | Performs comprehensive system diagnostics gathering detailed information about Python version, operating system, hardware specifications, and installed libraries. Exposes a web API endpoint for retrieving system information in JSON format to aid with troubleshooting and compatibility verification. |
| `FL_UnloadModel` | Unloads models from memory to free VRAM. |
| `FL_VideoCropMask` | Node for videocropmask functionality. |

### 📷 Screenshots & Examples

![🛠️ Utility Nodes Example](images/nodes/utility_nodes.png)

</details>

---

<details>
<summary><h2 id="-ksamplers-nodes">🎲 KSamplers Nodes</h2></summary>

> *Nodes for sampling in the generation process.*

| Node | Description |
|------|-------------|
| `FL_KSamplerXYZPlot` | Creates comprehensive grid comparisons by varying up to three KSampler parameters simultaneously. Generates labeled visual matrices showing the impact of changing steps, CFG, denoise values, samplers or schedulers, with professional-looking axes, labels and optional Z-dimension stacking in vertical or horizontal arrangements. |
| `FL_KsamplerBasic` | Provides a streamlined implementation of ComfyUI's KSampler with support for both latent and direct image inputs. Handles all essential sampling parameters with proper error handling and integration with VAE encoding/decoding functions. |
| `FL_KsamplerPlus` | Advanced KSampler that processes images by dividing them into overlapping slices and seamlessly blending the results. Supports configurable grid dimensions, overlap percentages, optional sliced conditioning for each region, and batch processing with progressive memory management. |
| `FL_KsamplerPlusV2` | Enhanced KSamplerPlus with improved algorithms. |
| `FL_KsamplerSettings` | Provides a comprehensive preset system for aspect ratios and dimensions optimized for different model types (SD, SV3D, SD2, XL). Outputs coordinated settings for two-pass workflows with separate steps, CFG values, and denoise settings along with properly configured sampler and scheduler selections. |
| `FL_SamplerStrings` | Generates comma-separated strings of sampler names based on boolean selections, providing a streamlined interface for sampler combination and configuration. Creates standardized sampler strings compatible with ComfyUI's KSampler nodes for consistent workflow integration. |
| `FL_SchedulerStrings` | Generates comma-separated strings of scheduler names based on boolean selections, providing a streamlined interface for scheduler configuration. Creates standardized scheduler strings compatible with ComfyUI's KSampler nodes for consistent workflow integration. |

### 📷 Screenshots & Examples

![🎲 KSamplers Nodes Example](images/nodes/ksamplers_nodes.png)

</details>

---

<details>
<summary><h2 id="-pdf-nodes">📄 PDF Nodes</h2></summary>

> *Nodes for working with PDF files.*

| Node | Description |
|------|-------------|
| `FL_BulkPDFLoader` | Loads all PDF files from a specified directory with error handling and reporting. Returns metadata including file path, filename, page count, and binary content for each successfully loaded PDF. |
| `FL_ImagesToPDF` | Converts batched image tensors into a PDF document with one image per page. Supports configurable DPI and standard page sizes (A4, Letter, Legal) with automatic image resizing and centering to fit the selected page format. |
| `FL_PDFEncryptor` | Encrypts PDFs with password protection. |
| `FL_PDFImageExtractor` | Extracts embedded images from PDF files with support for various image encoding formats including JPEG, PNG, and other compression methods. Provides filtering by minimum width/height dimensions and returns extracted images as tensors ready for processing. |
| `FL_PDFLoader` | Loads PDF files from disk and returns structured data including the file path, page count, and binary content. Performs format validation to ensure the input file is a valid PDF before loading. |
| `FL_PDFMerger` | Combines two PDF files into a single document by appending the pages of the second PDF after the first. Returns the merged PDF content in memory with updated page count information for further processing or saving. |
| `FL_PDFSaver` | Saves PDF data to disk with configurable output directory and filename options. Ensures proper file extension handling and directory creation if needed, returning the full path to the saved PDF file. |
| `FL_PDFTextExtractor` | Extracts all text content from PDF documents, combining text from multiple pages with proper spacing. Returns the complete extracted text as a single string with page separators for structured extraction. |
| `FL_PDFToImages` | Node for pdftoimages functionality. |
| `FL_TextToPDF` | Converts text to formatted PDF documents. |

### 📷 Screenshots & Examples

![📄 PDF Nodes Example](images/nodes/pdf_nodes.png)

</details>

---

<details>
<summary><h2 id="-gpt-nodes">🤖 GPT Nodes</h2></summary>

> *Nodes for integrating with GPT and OpenAI models.*

| Node | Description |
|------|-------------|
| `FL_Dalle3` | Integrates OpenAI's DALL-E 3 image generation API into ComfyUI with support for different resolutions, quality settings, and style options. Includes batch processing, automatic image saving, retry functionality, and returns both generated images and the revised prompts used by the DALL-E 3 model. |
| `FL_GPT_Image1` | Generates images using OpenAI GPT. |
| `FL_GPT_Image1_ADV` | Advanced GPT image generation with extended controls. |
| `FL_GPT_Text` | Generates text using OpenAI GPT models. |
| `FL_GPT_Vision` | Integrates with OpenAI's GPT Vision models (GPT-4o, GPT-4o-mini, GPT-4-vision-preview) to analyze and caption images. Features batch processing with configurable system prompts, detail levels, and temperature settings, while saving both images and associated captions to disk with optional overwrite protection. |
| `FL_SimpleGPTVision` | Integrates OpenAI's GPT Vision models (GPT-4o-mini, GPT-4o, GPT-4-vision-preview) to analyze images and generate text descriptions. Features customizable system prompts, user requests, and generation parameters while using environment variables for secure API key handling. |

### 📷 Screenshots & Examples

![🤖 GPT Nodes Example](images/nodes/gpt_nodes.png)

</details>

---

<details>
<summary><h2 id="-ai-nodes">🧠 AI Nodes</h2></summary>

> *Nodes that integrate with various AI models and services.*

| Node | Description |
|------|-------------|
| `FL_Fal_Gemini_ImageEdit` | A ComfyUI node for the Fal AI Gemini 2.5 Flash Image Edit API. Takes multiple images and a prompt to edit them using Gemini's multimodal capabilities. |
| `FL_Fal_Kling_AIAvatar` | A ComfyUI node for the Fal AI Kling Video AI Avatar API. Takes an image and audio to generate realistic avatar videos. |
| `FL_Fal_Kontext` | A ComfyUI node for the Fal AI Flux Pro Kontext API. Takes multiple image/prompt pairs and generates new images using Fal AI's flux-pro/kontext endpoint. Supports async processing for multiple inputs. |
| `FL_Fal_Pixverse` | A ComfyUI node for the Fal AI Image-to-Video API. Takes an image and converts it to a video using Fal AI's pixverse/v4/image-to-video endpoint. Downloads the video, extracts frames, and returns them as image tensors. |
| `FL_Fal_Pixverse_LipSync` | A ComfyUI node for the Fal AI Pixverse LipSync API. Takes a video and audio/text and generates realistic lipsync animations. |
| `FL_Fal_Pixverse_Transition` | A ComfyUI node for the Fal AI Pixverse v5 Transition API. Takes two images and creates a transition video between them using Fal AI's transition endpoint. Downloads the video, extracts frames, and returns them as image tensors. |
| `FL_Fal_Seedance_i2v` | Converts images to videos using Fal AI Seedance image-to-video API. |
| `FL_Fal_Seedream_Edit` | A ComfyUI node for the Fal AI ByteDance Seedream v4 Edit API. Takes multiple images and a prompt to edit them using Seedream's capabilities. |
| `FL_Fal_Sora` | Generates videos from prompts using OpenAI Sora via Fal.ai API. |
| `FL_GeminiImageEditor` | Edits images using Google Gemini AI with natural language instructions. |
| `FL_GeminiImageGenADV` | Advanced image generation using Google Gemini AI. |
| `FL_GeminiTextAPI` | Generates text using Google Gemini AI models. |
| `FL_GeminiVideoCaptioner` | Node for captioning videos using Google's Gemini API. Note: All videos (from file or image batch) are converted to WebM format with a size limit of just under 30MB to ensure compatibility with the Gemini API payload limitations. Video quality will be adjusted automatically to meet the size requirement. |
| `FL_Hedra_API` | Integrates with Hedra API for AI-powered audio/video generation. |
| `FL_HunyuanDelight` | Integrates Tencent's Hunyuan3D-2 model for image-to-image transformations with automatic model downloading and loading. Supports batch processing with adjustable parameters for image guidance, steps, and multiple processing loops for iterative refinement. |
| `FL_PixVerseAPI` | A ComfyUI node for the PixVerse Image-to-Video API. Takes an image and converts it to a video using PixVerse's API. Downloads the video, extracts frames, and returns them as image tensors. |
| `FL_RunwayAct2` | Generates video using Runway Act-One API. |
| `FL_RunwayImageAPI` | Creates images using Runway ML Gen-2/Gen-3 API. |
| `FL_Veo3VideoGen` | Generates videos using Google Veo 3 via Vertex AI. |
| `FL_VertexGemini25FlashImage` | Fast image generation using Google Vertex AI Gemini 2.5 Flash. |

### 📷 Screenshots & Examples

![🧠 AI Nodes Example](images/nodes/ai_nodes.png)

</details>

---

<details>
<summary><h2 id="-audio-nodes">🔊 Audio Nodes</h2></summary>

> *Nodes for audio analysis, processing, and audio-reactive visual effects.*

| Node | Description |
|------|-------------|
| `FL_Audio_BPM_Analyzer` | A ComfyUI node for BPM and beat detection using Librosa. Analyzes the entire audio once and outputs beat positions for use in segmentation. |
| `FL_Audio_Beat_Visualizer` | A ComfyUI node for generating beat visualization frames. Creates frames that alternate between black and white on beat switches. |
| `FL_Audio_Crop` | A ComfyUI node for cropping (trimming) audio to a specific start and end time. |
| `FL_Audio_Drum_Detector` | A ComfyUI node for detecting drum elements (kicks, snares, hi-hats) from audio. Uses onset detection with frequency band analysis to classify drum types. |
| `FL_Audio_Envelope_Visualizer` | A ComfyUI node for visualizing audio envelopes as frames. Creates white frames that fade to black based on envelope values. |
| `FL_Audio_Music_Video_Sequencer` | A ComfyUI node for generating complete music video shot sequences. Takes beat positions and a pattern, outputs a full edit list for the entire song. |
| `FL_Audio_Reactive_Brightness` | A ComfyUI node for applying audio-reactive brightness/luminance changes to frames. Adjusts brightness based on envelope values from drum detection. |
| `FL_Audio_Reactive_Edge_Glow` | A ComfyUI node for applying audio-reactive edge detection and glow effect. Detects edges and adds glowing outline that pulses with the audio. |
| `FL_Audio_Reactive_Envelope` | A ComfyUI node for generating per-frame control envelopes from drum detections. Creates ADSR envelopes for kicks, snares, and hi-hats across the entire song. |
| `FL_Audio_Reactive_Saturation` | A ComfyUI node for applying audio-reactive saturation changes to frames. Adjusts color saturation based on envelope values from drum detection. |
| `FL_Audio_Reactive_Scale` | A ComfyUI node for applying audio-reactive scale/zoom effect to frames. Scales frames based on envelope values from drum detection. |
| `FL_Audio_Reactive_Speed` | A ComfyUI node for applying audio-reactive speed/time remapping to frames. Speeds up or slows down playback based on envelope values from drum detection. |
| `FL_Audio_Segment_Extractor` | A ComfyUI node for extracting audio segments based on pre-analyzed beat positions. Takes beat positions from FL_Audio_BPM_Analyzer and extracts specific beat ranges. |
| `FL_Audio_Separation` | A ComfyUI node for separating audio into four sources: bass, drums, other, and vocals. Uses the Hybrid Demucs model from torchaudio. |
| `FL_Audio_Shot_Iterator` | A ComfyUI node for extracting individual shot data from a music video sequence. Takes sequence JSON and shot index, outputs frame count and shot details. |

### 📷 Screenshots & Examples

![🔊 Audio Nodes Example](images/nodes/audio_nodes.png)

</details>

---

<details>
<summary><h2 id="-experimental-nodes">🧪 Experimental Nodes</h2></summary>

> *Experimental nodes with various functionalities.*

| Node | Description |
|------|-------------|
| `FL_BatchAlign` | Node for batchalign functionality. |
| `FL_ColorPicker` | Simple interface for selecting and outputting color values in hexadecimal format. Takes a hex color string input (e.g., #FF0000) and passes it through to other nodes, enabling color selection within workflows. |
| `FL_GradGenerator` | Node for gradgenerator functionality. |
| `FL_NFTGenerator` | Selects images from a directory based on rarity percentages encoded in filenames following a specific format. Uses seed-based randomization to determine selection probability, automatically finds corresponding mask files, and outputs both the selected image and its mask as tensors. |

### 📷 Screenshots & Examples

![🧪 Experimental Nodes Example](images/nodes/experiments_nodes.png)

</details>

---

<details>
<summary><h2 id="-prompting-nodes">💬 Prompting Nodes</h2></summary>

> *Nodes for generating and manipulating prompts.*

| Node | Description |
|------|-------------|
| `FL_MadLibGenerator` | Creates randomized text by replacing delimiters in a template with words from five configurable word lists. Uses consistent seeding per list to ensure reproducible results, with support for custom delimiters and separate RNG states for each word list. |
| `FL_PromptBasic` | Basic text prompt input node. |
| `FL_PromptMulti` | Multi-line prompt editor with weighting support. |
| `FL_PromptSelector` | Selects a single prompt from a multi-line text input based on a specified index, with options to prepend and append additional text. Provides error handling for index out-of-range conditions and automatic text formatting with proper spacing. |

### 📷 Screenshots & Examples

![💬 Prompting Nodes Example](images/nodes/prompting_nodes.png)

</details>

---

<details>
<summary><h2 id="-file-operation-nodes">📂 File Operation Nodes</h2></summary>

> *Nodes for file operations.*

| Node | Description |
|------|-------------|
| `FL_ZipDirectory` | Compresses entire directory structures into zip archives with relative path preservation and efficient compression settings. Processes files using a temporary storage area and returns the resulting zip data for further handling in workflows. |
| `FL_ZipSave` | Compresses files from a specified input directory into a zip archive and saves it to a target location with customizable filename. Creates the output directory if it doesn't exist and ensures proper ZIP extension, returning the full path to the created archive. |

### 📷 Screenshots & Examples

![📂 File Operation Nodes Example](images/nodes/fileoperations_nodes.png)

</details>

---

<details>
<summary><h2 id="-google-drive-nodes">☁️ Google Drive Nodes</h2></summary>

> *Nodes for Google Cloud services integration.*

| Node | Description |
|------|-------------|
| `FL_GoogleCloudStorage` | A ComfyUI node for uploading images and videos to Google Cloud Storage. Can handle single images, batches of images, and optionally compile batches into videos. |
| `FL_GoogleDriveDownloader` | Downloads files from Google Drive using share links with automatic file ID extraction. Handles ZIP files by automatically extracting them to the specified output directory, with error handling and reporting for each step of the download process. |
| `FL_GoogleDriveImageDownloader` | Downloads image files specifically from Google Drive and automatically converts them to ComfyUI-compatible tensor format. Supports various image formats with automatic RGB conversion and proper tensor dimensioning, returning ready-to-use images for immediate integration into workflows. |

### 📷 Screenshots & Examples

![☁️ Google Drive Nodes Example](images/nodes/googledrive_nodes.png)

</details>

---

<details>
<summary><h2 id="-api-tool-nodes">🔌 API Tool Nodes</h2></summary>

> *Nodes for API interactions.*

| Node | Description |
|------|-------------|
| `FL_API_Base64_ImageLoader` | Loads Base64-encoded images with support for automatic data URL prefix removal and image resizing. Preserves metadata like job_id, user_id, and category for API integration workflows. |
| `FL_API_ImageSaver` | Saves images to a categorized directory structure based on user_id and category parameters. Supports different image formats (PNG, JPEG, WebP) with configurable quality settings for web API integration. |

### 📷 Screenshots & Examples

![🔌 API Tool Nodes Example](images/nodes/apitools_nodes.png)

</details>

---

<details>
<summary><h2 id="-hugging-face-nodes">🤗 Hugging Face Nodes</h2></summary>

> *Nodes for integrating with Hugging Face.*

| Node | Description |
|------|-------------|
| `FL_HFDatasetDownloader` | Downloads datasets from Hugging Face Hub. |
| `FL_HFHubModelUploader` | Uploads models and associated files to Hugging Face Hub with support for creating new repositories or using existing ones. Features comprehensive upload capabilities for model files, images, ZIP archives, and README documentation with progress tracking and retry mechanisms. |
| `FL_HF_Character` | Uploads character-related content to Hugging Face repositories with structured organization by studio, project, and character name. Supports multiple file types including LoRA models, datasets, image layouts, PDFs, and CSV files with comprehensive progress tracking. |
| `FL_HF_UploaderAbsolute` | Uploads files to Hugging Face repositories using absolute paths and reads API keys from environment variables. Supports various content types including LoRA models, datasets, images, PDFs, and CSV files with a simplified directory structure. |

### 📷 Screenshots & Examples

![🤗 Hugging Face Nodes Example](images/nodes/huggingface_nodes.png)

</details>

---

<details>
<summary><h2 id="-loader-nodes">⏬ Loader Nodes</h2></summary>

> *Nodes for loading various resources.*

| Node | Description |
|------|-------------|
| `FL_NodeLoader` | Simple pass-through node that accepts and returns a TRIGGER input, designed to ensure custom nodes are loaded when a workflow is executed. Acts as a lightweight utility for controlling workflow execution order. |
| `FL_NodePackLoader` | Enhanced trigger node that forces processing on every execution regardless of input changes. Uses NaN for change detection to ensure the node always executes when triggered, providing a reliable mechanism for loading node packs. |
| `FL_UpscaleModel` | Processes images through upscaling models with support for batch processing, precision control, and optional downscaling for fine-tuned results. Features progress tracking for large batches and automatic handling of device-specific optimizations for both CPU and GPU processing. |

### 📷 Screenshots & Examples

![⏬ Loader Nodes Example](images/nodes/loaders_nodes.png)

</details>

---

<details>
<summary><h2 id="-discord-nodes">💬 Discord Nodes</h2></summary>

> *Nodes for Discord integration.*

| Node | Description |
|------|-------------|
| `FL_SendToDiscordWebhook` | Node for sendtodiscordwebhook functionality. |

### 📷 Screenshots & Examples

![💬 Discord Nodes Example](images/nodes/discord_nodes.png)

</details>

---

<details>
<summary><h2 id="-work-in-progress-nodes">🚧 Work-in-Progress Nodes</h2></summary>

> *Nodes that are still in development.*

| Node | Description |
|------|-------------|
| `FL_FractalKSampler` | Recursive fractal sampling with progressive upscaling. |
| `FL_TimeLine` | Processes timeline data for creating animated sequences with support for different interpolation modes, resolution settings, and frame rate controls. Includes an API endpoint for handling timeline data within the ComfyUI server architecture, enabling advanced animation workflows. |
| `FL_WF_Agent` | A node that uses Gemini AI to generate and execute JavaScript code for workflow manipulation |
| `FL_WanFirstLastFrameToVideo` | Interpolates video between first and last frames. |

### 📷 Screenshots & Examples

![🚧 Work-in-Progress Nodes Example](images/nodes/wip_nodes.png)

</details>

---

<details>
<summary><h2 id="-game-nodes">🎮 Game Nodes</h2></summary>

> *Nodes implementing games.*

| Node | Description |
|------|-------------|
| `FL_BulletHellGame` | Implements a playable bullet hell-style shooter game within the ComfyUI interface where players control a ship with mouse movements and combat enemy ships that fire various bullet patterns. Features include multiple enemy ships with different attack patterns, player-guided bullets that track enemies, level progression, and score tracking. |
| `FL_TetrisGame` | Implements a fully playable Tetris game within ComfyUI's interface using standard keyboard controls (arrow keys) for movement, rotation, and acceleration. Features include complete tetromino collision detection, line clearing mechanics, game over detection, and a responsive canvas that adjusts to the node's dimensions. |

### 📷 Screenshots & Examples

![🎮 Game Nodes Example](images/nodes/games_nodes.png)

</details>

---

<details>
<summary><h2 id="-video-nodes">🎬 Video Nodes</h2></summary>

> *Nodes for video processing and frame interpolation.*

| Node | Description |
|------|-------------|
| `FL_FILM` | FILM (Frame Interpolation for Large Motion) frame interpolation node. Generates intermediate frames between input frames, especially good for large motion. Downloads model to cache folder on first use. |
| `FL_ProResVideo` | Creates professional-quality ProRes videos from image sequences with configurable FPS and output settings. Uses a two-step process with temporary MP4 creation followed by FFmpeg conversion to ProRes 4444 format with high-quality settings optimized for post-production workflows. |
| `FL_RIFE` | RIFE (Real-Time Intermediate Flow Estimation) frame interpolation node. Generates intermediate frames between input frames for smooth slow-motion effects. Downloads models to cache folder on first use. |
| `FL_VideoBatchSplitter` | Splits video batches into smaller segments. |
| `FL_VideoCadence` | Adjusts video playback cadence by frame patterns. |
| `FL_VideoCadenceCompile` | Compiles frames with custom cadence patterns. |
| `FL_VideoCrossfade` | Creates crossfade transitions between clips. |
| `FL_VideoCut` | A node that detects scene cuts in a batch of images (video frames) and outputs the segmented clips as MP4 files to a specified folder. |
| `FL_VideoTrim` | Trims video sequences to specified frame ranges. |

### 📷 Screenshots & Examples

![🎬 Video Nodes Example](images/nodes/video_nodes.png)

</details>

---
