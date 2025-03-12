# 🏵️ Fill-Nodes Documentation

![Fill-Nodes Banner](../images/banner.png)

> *A comprehensive collection of custom nodes for ComfyUI providing a wide range of functionality from image manipulation to PDF handling, AI integration, and more.*

## 🔍 Table of Contents

<div style="columns: 3; column-gap: 20px;">

- [🖼️ Image Nodes](#image-nodes)
- [📝 Captioning Nodes](#captioning-nodes)
- [✨ VFX Nodes](#vfx-nodes)
- [🛠️ Utility Nodes](#utility-nodes)
- [🎲 KSamplers Nodes](#ksamplers-nodes)
- [📄 PDF Nodes](#pdf-nodes)
- [🤖 GPT Nodes](#gpt-nodes)
- [🧪 Experimental Nodes](#experimental-nodes)
- [💬 Prompting Nodes](#prompting-nodes)
- [📂 File Operation Nodes](#file-operation-nodes)
- [☁️ Google Drive Nodes](#google-drive-nodes)
- [🔌 API Tool Nodes](#api-tool-nodes)
- [🤗 Hugging Face Nodes](#hugging-face-nodes)
- [📊 Analysis Nodes](#analysis-nodes)
- [⏬ Loader Nodes](#loader-nodes)
- [💬 Discord Nodes](#discord-nodes)
- [🚧 Work-in-Progress Nodes](#work-in-progress-nodes)
- [🎮 Game Nodes](#game-nodes)
- [🧮 Math Nodes](#math-nodes)
- [🎬 Video Nodes](#video-nodes)
- [🧠 AI Nodes](#ai-nodes)
- [🔧 Utils](#utils)

</div>

---

<details>
<summary><h2 id="image-nodes">🖼️ Image Nodes</h2></summary>

> *Nodes for manipulating, analyzing, and working with images.*

| Node | Description |
|------|-------------|
| `FL_AnimeLineExtractor` | Extracts line art from anime-style images |
| `FL_ApplyMask` | Applies a mask to an image |
| `FL_Image_Dimension_Display` | Displays image dimensions |
| `FL_Image_Pixelator` | Creates pixelated versions of images |
| `FL_Image_Randomizer` | Randomizes aspects of images |
| `FL_ImageAdjuster` | Adjusts image properties like brightness and contrast |
| `FL_ImageBatchListConverter` | Converts between image batches and lists |
| `FL_ImageBatchToGrid` | Arranges image batches into a grid layout |
| `FL_ImageNotes` | Adds notes to images |
| `FL_ImageSlicer` | Slices images into smaller pieces |
| `FL_LoadImage` | Loads images from disk |
| `FL_SaveAndDisplayImage` | Saves and displays images |

### 📷 Screenshots & Examples

![Image Nodes Example](../images/image_nodes_example.png)

</details>

---

<details>
<summary><h2 id="captioning-nodes">📝 Captioning Nodes</h2></summary>

> *Nodes for generating, saving, and manipulating image and video captions.*

| Node | Description |
|------|-------------|
| `FL_Caption_Saver_V2` | Enhanced version of caption saver |
| `FL_CaptionToCSV` | Converts captions to CSV format |
| `FL_Image_Caption_Saver` | Saves captions for images |
| `FL_ImageCaptionLayout` | Creates layouts combining images and captions |
| `FL_ImageCaptionLayoutPDF` | Creates PDF layouts with images and captions |
| `FL_MirrorAndAppendCaptions` | Mirrors and appends captions to images |
| `FL_OllamaCaptioner` | Generates captions using Ollama models |
| `FL_Video_Caption_Saver` | Saves captions for videos |

### 📷 Screenshots & Examples

![Captioning Nodes Example](../images/captioning_nodes_example.png)

</details>

---

<details>
<summary><h2 id="vfx-nodes">✨ VFX Nodes</h2></summary>

> *Nodes for applying visual effects to images.*

| Node | Description |
|------|-------------|
| `FL_Ascii` | Converts images to ASCII art |
| `FL_Dither` | Applies dithering effects to images |
| `FL_Glitch` | Creates glitch effects |
| `FL_HalfTone` | Applies halftone patterns to images |
| `FL_HexagonalPattern` | Creates hexagonal patterns from images |
| `FL_InfiniteZoom` | Creates infinite zoom effects |
| `FL_PaperDrawn` | Applies paper drawing effects |
| `FL_PixelArt` | Transforms images into pixel art style |
| `FL_PixelSort` | Applies pixel sorting algorithms |
| `FL_RetroEffect` | Applies retro-style effects |
| `FL_Ripple` | Creates ripple/wave effects |
| `FL_Shader` | Applies custom shaders to images |

### 📷 Screenshots & Examples

![VFX Nodes Example](../images/vfx_nodes_example.png)

</details>

---

<details>
<summary><h2 id="utility-nodes">🛠️ Utility Nodes</h2></summary>

> *General utility nodes for various tasks.*

| Node | Description |
|------|-------------|
| `FL_Code_Node` | Runs custom code |
| `FL_DirectoryCrawl` | Crawls directories for files |
| `FL_ImageCollage` | Creates collages from multiple images |
| `FL_InpaintCrop` | Crops images for inpainting |
| `FL_Padding` | Adds padding to images |
| `FL_PasteOnCanvas` | Pastes images onto a canvas |
| `FL_PathTypeChecker` | Checks path types |
| `FL_RandomRange` | Generates random numbers in a range |
| `FL_SaveImages` | Saves images to disk |
| `FL_SaveWebPImage` | Saves images in WebP format |
| `FL_SD_Slices` | Creates slices for Stable Diffusion processing |
| `FL_SeparateMaskComponents` | Separates mask components |
| `FL_SystemCheck` | Performs system checks |

### 📷 Screenshots & Examples

![Utility Nodes Example](../images/utility_nodes_example.png)

</details>

---

<details>
<summary><h2 id="ksamplers-nodes">🎲 KSamplers Nodes</h2></summary>

> *Nodes for sampling in the generation process.*

| Node | Description |
|------|-------------|
| `FL_KsamplerBasic` | Basic KSampler implementation |
| `FL_KsamplerFractals` | KSampler for fractals |
| `FL_KsamplerPlus` | Enhanced KSampler with additional features |
| `FL_KsamplerSettings` | Manages KSampler settings |
| `FL_KSamplerXYZPlot` | Creates XYZ plots with KSampler |
| `FL_SamplerStrings` | Manages sampler string options |
| `FL_SchedulerStrings` | Manages scheduler string options |

### 📷 Screenshots & Examples

![KSamplers Nodes Example](../images/ksamplers_nodes_example.png)

</details>

---

<details>
<summary><h2 id="pdf-nodes">📄 PDF Nodes</h2></summary>

> *Nodes for working with PDF files.*

| Node | Description |
|------|-------------|
| `FL_BulkPDFLoader` | Loads multiple PDF files |
| `FL_ImagesToPDF` | Converts images to PDF |
| `FL_PDFImageExtractor` | Extracts images from PDF files |
| `FL_PDFLoader` | Loads PDF files |
| `FL_PDFMerger` | Merges multiple PDF files |
| `FL_PDFSaver` | Saves PDF files |
| `FL_PDFTextExtractor` | Extracts text from PDF files |
| `FL_PDFToImage` | Converts PDF to images |

### 📷 Screenshots & Examples

![PDF Nodes Example](../images/pdf_nodes_example.png)

</details>

---

<details>
<summary><h2 id="gpt-nodes">🤖 GPT Nodes</h2></summary>

> *Nodes for integrating with GPT and other AI models.*

| Node | Description |
|------|-------------|
| `FL_Dalle3` | Integrates with DALL-E 3 |
| `FL_GPT_Vision` | Integrates with GPT Vision models |
| `FL_SimpleGPTVision` | Simplified GPT Vision integration |
| `FL_GeminiVideoCaptioner` | Captions videos using Gemini models |

### 📷 Screenshots & Examples

![GPT Nodes Example](../images/gpt_nodes_example.png)

</details>

---

<details>
<summary><h2 id="experimental-nodes">🧪 Experimental Nodes</h2></summary>

> *Experimental nodes with various functionalities.*

| Node | Description |
|------|-------------|
| `FL_BatchAligned` | Aligns batches of images |
| `FL_ColorPicker` | Picks colors from images |
| `FL_GradGen` | Generates color gradients |
| `FL_NFTGenerator` | Generates NFT-style images |

### 📷 Screenshots & Examples

![Experimental Nodes Example](../images/experimental_nodes_example.png)

</details>

---

<details>
<summary><h2 id="prompting-nodes">💬 Prompting Nodes</h2></summary>

> *Nodes for generating and manipulating prompts.*

| Node | Description |
|------|-------------|
| `FL_MadLibGenerator` | Generates prompts using MadLib style |
| `FL_PromptSelector` | Selects prompts from a collection |

### 📷 Screenshots & Examples

![Prompting Nodes Example](../images/prompting_nodes_example.png)

</details>

---

<details>
<summary><h2 id="file-operation-nodes">📂 File Operation Nodes</h2></summary>

> *Nodes for file operations.*

| Node | Description |
|------|-------------|
| `FL_ZipDirectory` | Zips directories |
| `FL_ZipSave` | Saves files in zip format |

### 📷 Screenshots & Examples

![File Operation Nodes Example](../images/file_operation_nodes_example.png)

</details>

---

<details>
<summary><h2 id="google-drive-nodes">☁️ Google Drive Nodes</h2></summary>

> *Nodes for integrating with Google Drive.*

| Node | Description |
|------|-------------|
| `FL_GoogleDriveDownloader` | Downloads files from Google Drive |
| `FL_GoogleDriveImageDownloader` | Downloads images from Google Drive |

### 📷 Screenshots & Examples

![Google Drive Nodes Example](../images/google_drive_nodes_example.png)

</details>

---

<details>
<summary><h2 id="api-tool-nodes">🔌 API Tool Nodes</h2></summary>

> *Nodes for API interactions.*

| Node | Description |
|------|-------------|
| `FL_API_Base64_ImageLoader` | Loads Base64-encoded images via API |
| `FL_API_ImageSaver` | Saves images via API |

### 📷 Screenshots & Examples

![API Tool Nodes Example](../images/api_tool_nodes_example.png)

</details>

---

<details>
<summary><h2 id="hugging-face-nodes">🤗 Hugging Face Nodes</h2></summary>

> *Nodes for integrating with Hugging Face.*

| Node | Description |
|------|-------------|
| `FL_HF_Character` | Creates characters using Hugging Face models |
| `FL_HF_UploaderAbsolute` | Uploads to Hugging Face with absolute paths |
| `FL_HFHubModelUploader` | Uploads models to Hugging Face Hub |

### 📷 Screenshots & Examples

![Hugging Face Nodes Example](../images/hugging_face_nodes_example.png)

</details>

---

<details>
<summary><h2 id="analysis-nodes">📊 Analysis Nodes</h2></summary>

> *Nodes for analyzing images and other data.*

| Node | Description |
|------|-------------|
| `FL_ClipScanner` | Scans images using CLIP models |

### 📷 Screenshots & Examples

![Analysis Nodes Example](../images/analysis_nodes_example.png)

</details>

---

<details>
<summary><h2 id="loader-nodes">⏬ Loader Nodes</h2></summary>

> *Nodes for loading various resources.*

| Node | Description |
|------|-------------|
| `FL_NodeLoader` | Loads custom nodes |
| `FL_UpscaleModel` | Loads upscale models |

### 📷 Screenshots & Examples

![Loader Nodes Example](../images/loader_nodes_example.png)

</details>

---

<details>
<summary><h2 id="discord-nodes">💬 Discord Nodes</h2></summary>

> *Nodes for Discord integration.*

| Node | Description |
|------|-------------|
| `FL_DiscordWebhook` | Sends content to Discord via webhooks |

### 📷 Screenshots & Examples

![Discord Nodes Example](../images/discord_nodes_example.png)

</details>

---

<details>
<summary><h2 id="work-in-progress-nodes">🚧 Work-in-Progress Nodes</h2></summary>

> *Nodes that are still in development.*

| Node | Description |
|------|-------------|
| `FL_HunyuanDelight` | Integration with Hunyuan AI models |
| `FL_TimeLine` | Timeline management for workflows |

### 📷 Screenshots & Examples

![WIP Nodes Example](../images/wip_nodes_example.png)

</details>

---

<details>
<summary><h2 id="game-nodes">🎮 Game Nodes</h2></summary>

> *Nodes implementing games.*

| Node | Description |
|------|-------------|
| `FL_BulletHellGame` | Implements a bullet hell game |
| `FL_TetrisGame` | Implements a Tetris game |

### 📷 Screenshots & Examples

![Game Nodes Example](../images/game_nodes_example.png)

</details>

---

<details>
<summary><h2 id="math-nodes">🧮 Math Nodes</h2></summary>

> *Nodes for mathematical operations.*

| Node | Description |
|------|-------------|
| `FL_Math` | Performs various mathematical operations |

### 📷 Screenshots & Examples

![Math Nodes Example](../images/math_nodes_example.png)

</details>

---

<details>
<summary><h2 id="video-nodes">🎬 Video Nodes</h2></summary>

> *Nodes for video processing.*

| Node | Description |
|------|-------------|
| `FL_ProResVideo` | Works with ProRes video format |
| `FL_SceneCut` | Detects scene cuts in videos |
| `FL_VideoCropNStitch` | Crops and stitches video frames |

### 📷 Screenshots & Examples

![Video Nodes Example](../images/video_nodes_example.png)

</details>

---

<details>
<summary><h2 id="ai-nodes">🧠 AI Nodes</h2></summary>

> *Nodes that integrate with various AI models.*

| Node | Description |
|------|-------------|
| `FL_HunyuanDelight` | Integration with Hunyuan AI models |

### 📷 Screenshots & Examples

![AI Nodes Example](../images/ai_nodes_example.png)

</details>

---

<details>
<summary><h2 id="utils">🔧 Utils</h2></summary>

> *Utility nodes for the system.*

| Node | Description |
|------|-------------|
| `FL_NodePackLoader` | Loads packs of nodes |

### 📷 Screenshots & Examples

![Utils Example](../images/utils_example.png)

</details>

---

## 🤝 Contributing

If you'd like to contribute to Fill-Nodes, please see our [contribution guidelines](../CONTRIBUTING.md).

## 📜 License

This project is licensed under the [MIT License](../LICENSE).

## 🙏 Acknowledgements

Thanks to all contributors and the ComfyUI community.

<div align="center">
</div>
