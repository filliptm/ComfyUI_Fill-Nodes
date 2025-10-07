import os
import torch
import numpy as np
from PIL import Image
import folder_paths
import tempfile
import importlib
import requests
from tqdm import tqdm
import cv2
import torchaudio
import subprocess

class FL_Fal_Sora:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fal_api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "A dramatic scene..."}),
                "use_pro": ("BOOLEAN", {"default": False}),
                "resolution": (["auto", "720p", "1080p"], {"default": "auto"}),
                "aspect_ratio": (["auto", "16:9", "9:16"], {"default": "16:9"}),
                "duration": ([4, 8, 12], {"default": 4}),
                "openai_api_key": ("STRING", {"default": ""}),
                "nth_frame": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("frames", "audio", "video_url", "status_msg")
    FUNCTION = "generate_video"
    CATEGORY = "🏵️Fill Nodes/AI"

    def generate_video(self, fal_api_key, prompt, use_pro, resolution, aspect_ratio, duration, openai_api_key, nth_frame, image=None):
        # Helper function for empty audio
        def empty_audio():
            return {"waveform": torch.zeros((1, 1, 0)), "sample_rate": 44100}

        try:
            # Validate API key
            if not fal_api_key or fal_api_key.strip() == "":
                return (torch.zeros(1, 64, 64, 3), empty_audio(), "", "❌ Error: Fal API key is required")

            # Validate prompt
            if not prompt or prompt.strip() == "":
                return (torch.zeros(1, 64, 64, 3), empty_audio(), "", "❌ Error: Prompt is required")

            # Clear and set FAL_KEY environment variable
            if 'FAL_KEY' in os.environ:
                del os.environ['FAL_KEY']
            os.environ['FAL_KEY'] = fal_api_key

            # Reload fal_client to ensure it picks up the new API key
            import sys
            if 'fal_client' in sys.modules:
                del sys.modules['fal_client']

            import fal_client

            # Determine which endpoint to use
            is_image_to_video = image is not None
            original_resolution = resolution
            original_aspect_ratio = aspect_ratio

            if is_image_to_video:
                if use_pro:
                    endpoint = "fal-ai/sora-2/image-to-video/pro"
                    valid_resolutions = ["auto", "720p", "1080p"]
                else:
                    endpoint = "fal-ai/sora-2/image-to-video"
                    valid_resolutions = ["auto", "720p"]
            else:
                if use_pro:
                    endpoint = "fal-ai/sora-2/text-to-video/pro"
                    valid_resolutions = ["720p", "1080p"]
                    # For text-to-video, default to 720p if "auto" is selected
                    if resolution == "auto":
                        resolution = "720p"
                        print("⚠️  'auto' resolution not supported for text-to-video, using '720p'")
                else:
                    endpoint = "fal-ai/sora-2/text-to-video"
                    valid_resolutions = ["720p"]
                    if resolution != "720p":
                        print(f"⚠️  Resolution '{resolution}' not available for non-PRO text-to-video, using '720p'")
                        resolution = "720p"

            # Validate resolution
            if resolution not in valid_resolutions:
                old_res = resolution
                resolution = valid_resolutions[0]
                print(f"⚠️  Resolution '{old_res}' not valid for {endpoint}, using '{resolution}'")
                print(f"   Valid resolutions: {', '.join(valid_resolutions)}")

            # Validate aspect_ratio for text-to-video (no auto option)
            if not is_image_to_video and aspect_ratio == "auto":
                aspect_ratio = "16:9"
                print("⚠️  'auto' aspect ratio not supported for text-to-video, using '16:9'")

            print(f"\n{'='*60}")
            print(f"📹 Using endpoint: {endpoint}")
            print(f"📝 Prompt: {prompt[:100]}...")
            print(f"🎬 Settings: {resolution}, {aspect_ratio}, {duration}s")
            if original_resolution != resolution or original_aspect_ratio != aspect_ratio:
                print(f"ℹ️  Note: Some parameters were auto-corrected for this endpoint")
            print(f"{'='*60}\n")

            # Build arguments
            arguments = {
                "prompt": prompt,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
            }

            # Add OpenAI API key if provided (to avoid billing from Fal)
            if openai_api_key and openai_api_key.strip():
                arguments["api_key"] = openai_api_key
                print("🔑 Using OpenAI API key (billed to your OpenAI account)")

            # Handle image input for image-to-video
            image_url = None
            temp_image_path = None
            if is_image_to_video:
                print("🖼️  Image provided - using image-to-video mode")

                # Convert tensor to PIL Image
                i = 255. * image[0].cpu().numpy()
                img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                # Save to temporary file
                temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
                img_pil.save(temp_image_path)

                # Upload image to Fal
                print("📤 Uploading image to Fal...")
                image_url = fal_client.upload_file(temp_image_path)
                arguments["image_url"] = image_url
                print(f"✅ Image uploaded: {image_url}")

            # Call the API
            print(f"🚀 Calling Fal API: {endpoint}")

            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        print(f"📝 {log['message']}")

            result = fal_client.subscribe(
                endpoint,
                arguments=arguments,
                with_logs=True,
                on_queue_update=on_queue_update,
            )

            # Clean up temp image file
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)

            # Get video URL from result
            if not result:
                error_msg = "❌ API returned empty result"
                print(error_msg)
                return (torch.zeros(1, 64, 64, 3), empty_audio(), "", error_msg)

            if 'video' not in result:
                error_msg = f"❌ No 'video' field in API response. Response keys: {list(result.keys())}"
                print(error_msg)
                return (torch.zeros(1, 64, 64, 3), empty_audio(), "", error_msg)

            if 'url' not in result['video']:
                error_msg = f"❌ No 'url' in video response. Video keys: {list(result['video'].keys())}"
                print(error_msg)
                return (torch.zeros(1, 64, 64, 3), empty_audio(), "", error_msg)

            video_url = result['video']['url']
            print(f"✅ Video generated: {video_url}")

            # Download the video
            print("⬇️  Downloading video...")
            response = requests.get(video_url, stream=True)

            if response.status_code != 200:
                error_msg = f"❌ Failed to download video. HTTP Status: {response.status_code}"
                print(error_msg)
                return (torch.zeros(1, 64, 64, 3), empty_audio(), video_url, error_msg)

            total_size = int(response.headers.get('content-length', 0))

            temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

            with open(temp_video_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)

            print(f"✅ Video downloaded to: {temp_video_path}")

            # Extract audio from video using ffmpeg
            print("🎵 Extracting audio from video...")
            temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name

            try:
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-i', temp_video_path,
                    '-vn',  # No video
                    '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
                    '-ar', '44100',  # 44.1kHz sample rate
                    '-ac', '2',  # Stereo
                    '-y',  # Overwrite output file
                    temp_audio_path
                ]

                result_ffmpeg = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

                if result_ffmpeg.returncode != 0:
                    print(f"⚠️  FFmpeg warning: {result_ffmpeg.stderr}")
                    print("⚠️  Video may not contain audio, using empty audio")
                    audio_output = empty_audio()
                else:
                    # Load audio using torchaudio
                    waveform, sample_rate = torchaudio.load(temp_audio_path)
                    audio_output = {
                        "waveform": waveform.unsqueeze(0),  # Add batch dimension
                        "sample_rate": sample_rate
                    }
                    print(f"✅ Audio extracted: {waveform.shape[1]} samples at {sample_rate}Hz")

                # Clean up temp audio file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

            except FileNotFoundError:
                print("⚠️  FFmpeg not found. Install ffmpeg to extract audio. Using empty audio.")
                audio_output = empty_audio()
            except Exception as e:
                print(f"⚠️  Error extracting audio: {str(e)}. Using empty audio.")
                audio_output = empty_audio()

            # Extract frames from video
            print(f"🎞️  Extracting frames (every {nth_frame} frame(s))...")
            cap = cv2.VideoCapture(temp_video_path)

            if not cap.isOpened():
                error_msg = f"❌ Failed to open video file: {temp_video_path}"
                print(error_msg)
                return (torch.zeros(1, 64, 64, 3), audio_output, video_url, error_msg)

            frames = []
            frame_count = 0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                cap.release()
                error_msg = "❌ Video file contains 0 frames"
                print(error_msg)
                return (torch.zeros(1, 64, 64, 3), audio_output, video_url, error_msg)

            with tqdm(total=total_frames, desc="Extracting frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % nth_frame == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)

                    frame_count += 1
                    pbar.update(1)

            cap.release()

            # Clean up temp video file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

            if not frames:
                error_msg = "❌ No frames extracted from video"
                print(error_msg)
                return (torch.zeros(1, 64, 64, 3), audio_output, video_url, error_msg)

            print(f"✅ Extracted {len(frames)} frames")

            # Convert frames to tensor
            frames_np = np.array(frames).astype(np.float32) / 255.0
            frames_tensor = torch.from_numpy(frames_np)

            success_msg = f"✅ Successfully generated video with {len(frames)} frames using {endpoint}"
            print(success_msg)

            return (frames_tensor, audio_output, video_url, success_msg)

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            # Remove large base64 data URIs from error message for readability
            if "data:image" in error_msg or "data:audio" in error_msg:
                error_msg = error_msg[:500] + "... (truncated)"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return (torch.zeros(1, 64, 64, 3), empty_audio(), "", error_msg)

NODE_CLASS_MAPPINGS = {
    "FL_Fal_Sora": FL_Fal_Sora
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_Fal_Sora": "FL Fal Sora 2"
}
