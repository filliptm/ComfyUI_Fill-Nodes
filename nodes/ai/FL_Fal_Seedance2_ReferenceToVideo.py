import io
import logging
import os
import re
import time
import wave

import fal_client
import requests
import torch
from PIL import Image

import comfy.model_management
from comfy_api.latest import InputImpl, Types, io as comfy_io
from server import PromptServer


logger = logging.getLogger("fl_fill_nodes")

ENDPOINT = "bytedance/seedance-2.0/reference-to-video"
EVENT_NAME = "fl_fal_seedance2_progress"

MAX_IMAGE_BYTES = 30_000_000
MAX_AUDIO_BYTES = 15_000_000
MAX_VIDEO_BYTES = 50_000_000
MIN_VIDEO_PIXELS = 640 * 640
MAX_VIDEO_PIXELS = 834 * 1112

REFERENCE_PATTERN = re.compile(r"@(Image|Video|Audio)(\d+)", re.IGNORECASE)


def _ordered_references(references, prefix):
    if not references:
        return []

    indexed = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for name, value in references.items():
        match = pattern.match(name)
        if match and value is not None:
            indexed.append((int(match.group(1)), value))
    return [value for _, value in sorted(indexed)]


def _validate_reference_tags(prompt, image_count, video_count, audio_count):
    counts = {
        "image": image_count,
        "video": video_count,
        "audio": audio_count,
    }
    for kind, raw_index in REFERENCE_PATTERN.findall(prompt):
        index = int(raw_index)
        available = counts[kind.lower()]
        if index < 1 or index > available:
            raise ValueError(
                f"Prompt references @{kind.title()}{index}, but only {available} "
                f"{kind.lower()} reference(s) are connected."
            )


def _validate_reference_counts(images, videos, audios):
    if len(images) > 9:
        raise ValueError("Seedance 2.0 accepts at most 9 reference images.")
    if len(videos) > 3:
        raise ValueError("Seedance 2.0 accepts at most 3 reference videos.")
    if len(audios) > 3:
        raise ValueError("Seedance 2.0 accepts at most 3 reference audio files.")
    if len(images) + len(videos) + len(audios) > 12:
        raise ValueError("Seedance 2.0 accepts at most 12 reference files in total.")
    if audios and not images and not videos:
        raise ValueError("Reference audio requires at least one reference image or video.")


def _image_to_png(image, index):
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"@Image{index} must contain exactly one image.")

    array = image[0].detach().cpu().clamp(0, 1).mul(255).byte().numpy()
    if array.shape[-1] == 1:
        array = array[..., 0]
    elif array.shape[-1] not in (3, 4):
        raise ValueError(f"@Image{index} has an unsupported channel count: {array.shape[-1]}.")

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    data = buffer.getvalue()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"@Image{index} is larger than 30 MB after PNG encoding.")
    return data


def _video_to_mp4(video, index):
    duration = float(video.get_duration())
    if duration <= 0:
        raise ValueError(f"@Video{index} has no playable duration.")

    width, height = video.get_dimensions()
    pixels = int(width) * int(height)
    if pixels < MIN_VIDEO_PIXELS or pixels > MAX_VIDEO_PIXELS:
        raise ValueError(
            f"@Video{index} is {width}x{height}. Fal requires reference videos between "
            f"approximately 640x640 and 834x1112 in pixel area."
        )

    buffer = io.BytesIO()
    video.save_to(
        buffer,
        format=Types.VideoContainer.MP4,
        codec=Types.VideoCodec.H264,
    )
    data = buffer.getvalue()
    return data, duration


def _audio_to_wav(audio, index):
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])

    if waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError(f"@Audio{index} must contain exactly one audio clip.")
    if sample_rate <= 0:
        raise ValueError(f"@Audio{index} has an invalid sample rate.")

    samples = waveform[0].detach().cpu().float().clamp(-1, 1)
    if samples.shape[0] < 1:
        raise ValueError(f"@Audio{index} has no channels.")

    duration = samples.shape[-1] / sample_rate
    if duration <= 0:
        raise ValueError(f"@Audio{index} has no playable duration.")

    pcm = (
        samples.transpose(0, 1)
        .contiguous()
        .mul(32767)
        .round()
        .to(dtype=torch.int16)
        .numpy()
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(samples.shape[0])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

    data = buffer.getvalue()
    if len(data) > MAX_AUDIO_BYTES:
        raise ValueError(f"@Audio{index} is larger than 15 MB after WAV encoding.")
    return data, duration


def _prepare_references(images, videos, audios, node_id):
    prepared_images = []
    prepared_videos = []
    prepared_audios = []
    total_items = len(images) + len(videos) + len(audios)
    completed = 0

    for index, image in enumerate(images, start=1):
        _check_interrupted(None, node_id)
        prepared_images.append(_image_to_png(image, index))
        completed += 1
        _emit(node_id, "preparing", value=completed, max=total_items)

    video_duration = 0.0
    video_bytes = 0
    for index, video in enumerate(videos, start=1):
        _check_interrupted(None, node_id)
        data, duration = _video_to_mp4(video, index)
        prepared_videos.append(data)
        video_duration += duration
        video_bytes += len(data)
        completed += 1
        _emit(node_id, "preparing", value=completed, max=total_items)

    if videos and not 2 <= video_duration <= 15:
        raise ValueError(
            f"Combined reference video duration is {video_duration:.2f}s; Fal requires 2 to 15 seconds."
        )
    if video_bytes >= MAX_VIDEO_BYTES:
        raise ValueError("Combined reference videos must be smaller than 50 MB after MP4 encoding.")

    audio_duration = 0.0
    for index, audio in enumerate(audios, start=1):
        _check_interrupted(None, node_id)
        data, duration = _audio_to_wav(audio, index)
        prepared_audios.append(data)
        audio_duration += duration
        completed += 1
        _emit(node_id, "preparing", value=completed, max=total_items)

    if audio_duration > 15:
        raise ValueError(
            f"Combined reference audio duration is {audio_duration:.2f}s; Fal allows at most 15 seconds."
        )

    return prepared_images, prepared_videos, prepared_audios


def _build_arguments(
    prompt,
    resolution,
    duration,
    aspect_ratio,
    generate_audio,
    bitrate_mode,
    end_user_id,
    image_urls,
    video_urls,
    audio_urls,
):
    arguments = {
        "prompt": prompt,
        "resolution": resolution,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "generate_audio": bool(generate_audio),
        "bitrate_mode": bitrate_mode,
    }
    if image_urls:
        arguments["image_urls"] = image_urls
    if video_urls:
        arguments["video_urls"] = video_urls
    if audio_urls:
        arguments["audio_urls"] = audio_urls
    if end_user_id.strip():
        arguments["end_user_id"] = end_user_id.strip()
    return arguments


def _latest_log(status):
    logs = status.logs or []
    for entry in reversed(logs):
        if isinstance(entry, dict):
            message = entry.get("message")
            if message:
                return str(message)[:500]
    return ""


def _emit(node_id, phase, **details):
    if PromptServer.instance is None:
        return
    payload = {"node": node_id, "phase": phase, **details}
    try:
        PromptServer.instance.send_sync(EVENT_NAME, payload)
    except Exception:
        logger.debug("Unable to send Seedance 2.0 UI progress event.", exc_info=True)


def _cancel_request(handle):
    try:
        handle.cancel()
    except Exception:
        logger.debug("Unable to cancel Fal request.", exc_info=True)


def _check_interrupted(handle, node_id):
    if not comfy.model_management.processing_interrupted():
        return
    if handle is not None:
        _cancel_request(handle)
    _emit(node_id, "cancelled")
    comfy.model_management.throw_exception_if_processing_interrupted()


def _upload_references(client, images, videos, audios, node_id):
    urls = {"images": [], "videos": [], "audios": []}
    total = len(images) + len(videos) + len(audios)
    completed = 0

    for index, data in enumerate(images, start=1):
        _check_interrupted(None, node_id)
        urls["images"].append(
            client.upload(data, content_type="image/png", file_name=f"image_{index}.png")
        )
        completed += 1
        _emit(node_id, "uploading", value=completed, max=total)

    for index, data in enumerate(videos, start=1):
        _check_interrupted(None, node_id)
        urls["videos"].append(
            client.upload(data, content_type="video/mp4", file_name=f"video_{index}.mp4")
        )
        completed += 1
        _emit(node_id, "uploading", value=completed, max=total)

    for index, data in enumerate(audios, start=1):
        _check_interrupted(None, node_id)
        urls["audios"].append(
            client.upload(data, content_type="audio/wav", file_name=f"audio_{index}.wav")
        )
        completed += 1
        _emit(node_id, "uploading", value=completed, max=total)

    return urls


def _poll_request(handle, node_id):
    last_log = ""
    in_progress_emitted = False
    status_failures = 0
    while True:
        _check_interrupted(handle, node_id)
        try:
            status = handle.status(with_logs=True)
            status_failures = 0
        except Exception:
            status_failures += 1
            if status_failures >= 3:
                _cancel_request(handle)
                raise RuntimeError(
                    f"Unable to retrieve Fal status after 3 attempts for request {handle.request_id}."
                )
            _emit(
                node_id,
                "generating",
                log=f"Status check failed; retrying ({status_failures}/3)…",
                request_id=handle.request_id,
            )
            time.sleep(2**status_failures)
            continue
        if isinstance(status, fal_client.Queued):
            _emit(
                node_id,
                "queued",
                queue_position=status.position,
                request_id=handle.request_id,
            )
        elif isinstance(status, fal_client.InProgress):
            latest_log = _latest_log(status)
            if not in_progress_emitted or latest_log != last_log:
                in_progress_emitted = True
                last_log = latest_log
                _emit(
                    node_id,
                    "generating",
                    log=latest_log,
                    request_id=handle.request_id,
                )
        elif isinstance(status, fal_client.Completed):
            return
        time.sleep(1)


def _download_video(url, node_id, handle):
    buffer = io.BytesIO()
    with requests.get(url, stream=True, timeout=(10, 300)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        downloaded = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            _check_interrupted(handle, node_id)
            if not chunk:
                continue
            buffer.write(chunk)
            downloaded += len(chunk)
            _emit(node_id, "downloading", value=downloaded, max=total)
    buffer.seek(0)
    return InputImpl.VideoFromFile(buffer)


def _safe_error(error, api_key):
    message = str(error)
    if api_key:
        message = message.replace(api_key, "***")
    return message


class FL_Fal_Seedance2_ReferenceToVideo(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="FL_Fal_Seedance2_ReferenceToVideo",
            display_name="FL Fal Seedance 2.0 Reference to Video",
            category="🏵️Fill Nodes/AI",
            description=(
                "Generate a Seedance 2.0 video through Fal using prompt, image, video, "
                "and audio references. Refer to connected assets as @Image1, @Video1, and @Audio1."
            ),
            inputs=[
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip=(
                        "Describe the video. Refer to connected assets in order as "
                        "@Image1, @Video1, @Audio1, and so on."
                    ),
                ),
                comfy_io.Autogrow.Input(
                    "reference_images",
                    template=comfy_io.Autogrow.TemplateNames(
                        comfy_io.Image.Input("reference_image"),
                        names=[f"image_{index}" for index in range(1, 10)],
                        min=0,
                    ),
                    tooltip="Up to 9 reference images. JPEG, PNG, and WebP are supported by Fal.",
                ),
                comfy_io.Autogrow.Input(
                    "reference_videos",
                    template=comfy_io.Autogrow.TemplateNames(
                        comfy_io.Video.Input("reference_video"),
                        names=[f"video_{index}" for index in range(1, 4)],
                        min=0,
                    ),
                    tooltip="Up to 3 native VIDEO references with 2 to 15 seconds combined duration.",
                ),
                comfy_io.Autogrow.Input(
                    "reference_audios",
                    template=comfy_io.Autogrow.TemplateNames(
                        comfy_io.Audio.Input("reference_audio"),
                        names=[f"audio_{index}" for index in range(1, 4)],
                        min=0,
                    ),
                    tooltip=(
                        "Up to 3 audio references with at most 15 seconds combined duration. "
                        "Audio requires at least one image or video reference."
                    ),
                ),
                comfy_io.Combo.Input(
                    "resolution",
                    options=["480p", "720p", "1080p", "4k"],
                    default="720p",
                    tooltip=(
                        "Output resolution. 4k is present in Fal's current machine-readable schema."
                    ),
                ),
                comfy_io.Combo.Input(
                    "duration",
                    options=["auto", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"],
                    default="auto",
                    tooltip="Output duration in seconds, or auto to let Seedance decide.",
                ),
                comfy_io.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
                    default="auto",
                ),
                comfy_io.Boolean.Input(
                    "generate_audio",
                    default=True,
                    tooltip="Generate synchronized sound effects, ambience, and speech.",
                ),
                comfy_io.Combo.Input(
                    "bitrate_mode",
                    options=["standard", "high"],
                    default="standard",
                    advanced=True,
                    tooltip="High requests a larger, higher-bitrate output file.",
                ),
                comfy_io.String.Input(
                    "end_user_id",
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip="Optional end-user identifier sent to Fal.",
                ),
                comfy_io.String.Input(
                    "fal_api_key",
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip="Optional Fal key override. Leave blank to use the FAL_KEY environment variable.",
                ),
            ],
            outputs=[
                comfy_io.Video.Output("video"),
                comfy_io.String.Output("video_url"),
                comfy_io.Int.Output("seed"),
                comfy_io.String.Output("request_id"),
            ],
            hidden=[comfy_io.Hidden.unique_id],
            is_api_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(
        cls,
        prompt,
        reference_images,
        reference_videos,
        reference_audios,
        resolution,
        duration,
        aspect_ratio,
        generate_audio,
        bitrate_mode,
        end_user_id="",
        fal_api_key="",
    ):
        node_id = cls.hidden.unique_id if cls.hidden is not None else None
        api_key = fal_api_key.strip() or os.environ.get("FAL_KEY", "").strip()
        handle = None

        try:
            if not prompt.strip():
                raise ValueError("Prompt cannot be empty.")
            if not api_key:
                raise ValueError(
                    "A Fal API key is required. Set FAL_KEY or enter fal_api_key in the node."
                )

            images = _ordered_references(reference_images, "image")
            videos = _ordered_references(reference_videos, "video")
            audios = _ordered_references(reference_audios, "audio")
            _validate_reference_counts(images, videos, audios)
            _validate_reference_tags(prompt, len(images), len(videos), len(audios))

            counts = {
                "images": len(images),
                "videos": len(videos),
                "audios": len(audios),
            }
            _emit(node_id, "preparing", value=0, max=sum(counts.values()), counts=counts)
            prepared = _prepare_references(images, videos, audios, node_id)

            client = fal_client.SyncClient(key=api_key)
            urls = _upload_references(
                client,
                prepared[0],
                prepared[1],
                prepared[2],
                node_id,
            )
            arguments = _build_arguments(
                prompt.strip(),
                resolution,
                duration,
                aspect_ratio,
                generate_audio,
                bitrate_mode,
                end_user_id,
                urls["images"],
                urls["videos"],
                urls["audios"],
            )

            _check_interrupted(None, node_id)
            handle = client.submit(ENDPOINT, arguments)
            _emit(node_id, "queued", request_id=handle.request_id)
            _poll_request(handle, node_id)

            result = handle.get()
            video_data = result.get("video") if isinstance(result, dict) else None
            video_url = video_data.get("url") if isinstance(video_data, dict) else None
            seed = result.get("seed") if isinstance(result, dict) else None
            if not video_url:
                raise RuntimeError("Fal completed the request without returning a video URL.")
            if not isinstance(seed, int):
                raise RuntimeError("Fal completed the request without returning a seed.")

            video = _download_video(video_url, node_id, handle)
            _emit(
                node_id,
                "complete",
                video_url=video_url,
                seed=seed,
                request_id=handle.request_id,
                counts=counts,
            )
            return comfy_io.NodeOutput(video, video_url, seed, handle.request_id)
        except Exception as error:
            message = _safe_error(error, api_key)
            _emit(
                node_id,
                "error",
                message=message,
                request_id=handle.request_id if handle is not None else "",
            )
            raise RuntimeError(f"Fal Seedance 2.0 failed: {message}") from None
