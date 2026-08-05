import logging
import math
import re

import torch

import comfy.nested_tensor
import node_helpers
import nodes
from comfy_api.latest import io
from comfy_extras import nodes_minimax_h3 as minimax_h3


H3Timeline = io.Custom("FL_H3_TIMELINE")
FLPromptSchedule = io.Custom("FL_PROMPT_SCHEDULE")
FLPromptEnvelope = io.Custom("FL_PROMPT_ENVELOPE")
_HEADER = re.compile(r"^\s*\[\s*([0-9:.]+)\s*-\s*([0-9:.]+)\s*\]\s*$")
_VIDEO_FRAMES_PER_TOKEN = (1, 4, 4, 4, 4)
_EPS = 1e-6


def _parse_time(value, time_unit, bpm, line):
    try:
        if ":" in value:
            if time_unit != "seconds":
                raise ValueError("timecodes are only valid in seconds mode")
            parts = value.split(":")
            if len(parts) == 2:
                minutes, seconds = parts
                return float(minutes) * 60.0 + float(seconds)
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)
            raise ValueError("timecodes must be MM:SS or HH:MM:SS")

        number = float(value)
        if time_unit == "seconds":
            return number
        if time_unit == "frames":
            if not number.is_integer():
                raise ValueError("frame positions must be whole numbers")
            return number / minimax_h3.FPS
        if bpm <= 0:
            raise ValueError("BPM must be greater than zero")
        return number * 60.0 / bpm
    except ValueError as error:
        raise ValueError(f"MiniMax H3 timeline line {line}: invalid time '{value}' ({error}).") from error


def _parse_timeline(text, time_unit, bpm):
    sections = []
    current = None
    body = []

    def finish():
        if current is None:
            return
        prompt = "\n".join(body).strip()
        if not prompt:
            raise ValueError(f"MiniMax H3 timeline line {current['line']}: section prompt is empty.")
        sections.append({
            "line": current["line"],
            "start": current["start"],
            "end": current["end"],
            "prompt": prompt,
        })

    for line_number, line in enumerate(text.splitlines(), 1):
        match = _HEADER.match(line)
        if match:
            finish()
            body = []
            start = _parse_time(match.group(1), time_unit, bpm, line_number)
            end = _parse_time(match.group(2), time_unit, bpm, line_number)
            if start < 0:
                raise ValueError(f"MiniMax H3 timeline line {line_number}: start time cannot be negative.")
            if end <= start:
                raise ValueError(f"MiniMax H3 timeline line {line_number}: end time must be after start time.")
            current = {"line": line_number, "start": start, "end": end}
            continue

        if line.lstrip().startswith("["):
            raise ValueError(
                f"MiniMax H3 timeline line {line_number}: expected a header like [00:00.000 - 00:02.000]."
            )
        if current is None:
            if line.strip():
                raise ValueError(
                    f"MiniMax H3 timeline line {line_number}: prompt text must follow a time header."
                )
            continue
        body.append(line)

    finish()
    return sections


def _resolve_sections(sections, duration, duration_policy, transition_mode, transition_frames):
    if not sections:
        return []

    previous = None
    for section in sections:
        if previous is not None:
            if section["start"] < previous["start"]:
                raise ValueError(
                    f"MiniMax H3 timeline line {section['line']}: sections must be ordered by start time."
                )
            if section["start"] < previous["end"] - _EPS:
                raise ValueError(
                    f"MiniMax H3 timeline line {section['line']}: section overlaps the previous section."
                )
        previous = section

    last_end = sections[-1]["end"]
    if duration_policy == "strict" and last_end > duration + _EPS:
        raise ValueError(
            f"MiniMax H3 timeline ends at {last_end:.3f}s, but the aligned H3 latent is {duration:.3f}s. "
            "Shorten the schedule or use clamp/fit."
        )

    scale = duration / last_end if duration_policy == "fit" else 1.0
    resolved = []
    for section in sections:
        start = section["start"] * scale
        end = section["end"] * scale
        fade_in_end = section.get("fade_in_end")
        fade_out_start = section.get("fade_out_start")
        crossfade_start = section.get("crossfade_start")
        crossfade_end = section.get("crossfade_end")
        if fade_in_end is not None:
            fade_in_end *= scale
        if fade_out_start is not None:
            fade_out_start *= scale
        if crossfade_start is not None:
            crossfade_start *= scale
            crossfade_end *= scale
        if duration_policy == "clamp":
            start = min(start, duration)
            end = min(end, duration)
            if fade_in_end is not None:
                fade_in_end = min(fade_in_end, duration)
            if fade_out_start is not None:
                fade_out_start = min(fade_out_start, duration)
            if crossfade_start is not None:
                crossfade_start = min(crossfade_start, duration)
                crossfade_end = min(crossfade_end, duration)
        if end <= start + _EPS:
            continue
        resolved_section = {**section, "start": start, "end": end}
        if fade_in_end is not None:
            resolved_section["fade_in_end"] = min(max(fade_in_end, start), end)
            resolved_section["fade_out_start"] = min(max(fade_out_start, start), end)
        if crossfade_start is not None:
            resolved_section["crossfade_start"] = max(0.0, crossfade_start)
            resolved_section["crossfade_end"] = min(end, max(crossfade_start, crossfade_end))
        resolved.append(resolved_section)

    if transition_mode != "hard" and transition_frames > 0:
        transition_seconds = transition_frames / minimax_h3.FPS
        for first, second in zip(resolved, resolved[1:]):
            if abs(first["end"] - second["start"]) <= _EPS:
                shortest = min(first["end"] - first["start"], second["end"] - second["start"])
                if transition_seconds > shortest + _EPS:
                    raise ValueError(
                        f"MiniMax H3 timeline line {second['line']}: transition is longer than an adjacent section."
                    )
    return resolved


def _schedule_sections(schedule):
    if not isinstance(schedule, dict) or schedule.get("type") != "fl_prompt_schedule":
        raise TypeError("FL MiniMax H3 Prompt Timeline received an invalid prompt schedule.")
    version = schedule.get("version")
    if version not in {1, 2}:
        raise ValueError("FL MiniMax H3 Prompt Timeline supports FL prompt schedule versions 1 and 2.")

    values = schedule.get("sections")
    if not isinstance(values, list):
        raise ValueError("FL prompt schedule sections must be a list.")

    sections = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"FL prompt schedule section {index + 1} must be an object.")
        try:
            start = float(value["start"])
            end = float(value["end"])
            fade_in_end = float(value["fade_in_end"])
            fade_out_start = float(value["fade_out_start"])
            crossfade_start = float(value.get("crossfade_start", start))
            crossfade_end = float(value.get("crossfade_end", start))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"FL prompt schedule section {index + 1} has invalid timing values."
            ) from error
        prompt = value.get("prompt")
        curve = value.get("curve")
        if not all(math.isfinite(number) for number in (
            start,
            end,
            fade_in_end,
            fade_out_start,
            crossfade_start,
            crossfade_end,
        )):
            raise ValueError(f"FL prompt schedule section {index + 1} timing must be finite.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"FL prompt schedule section {index + 1} prompt is empty.")
        if curve not in {"linear", "cosine"}:
            raise ValueError(
                f"FL prompt schedule section {index + 1} curve must be linear or cosine."
            )
        if start < 0 or end <= start:
            raise ValueError(f"FL prompt schedule section {index + 1} has an invalid range.")
        if not start - _EPS <= fade_in_end <= fade_out_start + _EPS <= end + _EPS:
            raise ValueError(
                f"FL prompt schedule section {index + 1} has invalid fade boundaries."
            )
        if not 0 <= crossfade_start <= start + _EPS <= crossfade_end + _EPS <= end + _EPS:
            raise ValueError(
                f"FL prompt schedule section {index + 1} has invalid crossfade boundaries."
            )
        sections.append({
            "line": value.get("line", index + 1),
            "start": start,
            "end": end,
            "prompt": prompt.strip(),
            "fade_in_end": fade_in_end,
            "fade_out_start": fade_out_start,
            "crossfade_start": crossfade_start,
            "crossfade_end": crossfade_end,
            "curve": curve,
        })
    for index, section in enumerate(sections):
        if section["crossfade_end"] <= section["crossfade_start"] + _EPS:
            continue
        if index == 0:
            raise ValueError("FL prompt schedule first section cannot crossfade.")
        previous = sections[index - 1]
        if abs(previous["end"] - section["start"]) > _EPS:
            raise ValueError(
                f"FL prompt schedule section {index + 1} crossfade requires a touching previous section."
            )
        if section["crossfade_start"] < previous["start"] - _EPS:
            raise ValueError(
                f"FL prompt schedule section {index + 1} crossfade exceeds the previous section."
            )
    return sections


def _prompt_envelopes(prompt_envelopes):
    envelopes = []
    for position, value in enumerate((prompt_envelopes or {}).values(), 1):
        if not isinstance(value, dict) or value.get("type") != "fl_prompt_envelope":
            raise TypeError(
                f"FL MiniMax H3 Prompt Timeline received an invalid prompt envelope at input {position}."
            )
        if value.get("version") != 1:
            raise ValueError("FL MiniMax H3 Prompt Timeline supports FL prompt envelope version 1.")

        prompt = value.get("prompt")
        weights = value.get("weights")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"FL prompt envelope {position} prompt is empty.")
        if not isinstance(weights, list) or not weights:
            raise ValueError(f"FL prompt envelope {position} weights must be a non-empty list.")

        resolved_weights = []
        for index, weight in enumerate(weights):
            try:
                number = float(weight)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"FL prompt envelope {position} weight {index} must be a number."
                ) from error
            if not math.isfinite(number) or number < 0:
                raise ValueError(
                    f"FL prompt envelope {position} weight {index} must be finite and non-negative."
                )
            resolved_weights.append(number)

        try:
            fps = float(value["fps"])
            duration = float(value["duration"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"FL prompt envelope {position} has invalid fps or duration."
            ) from error
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"FL prompt envelope {position} fps must be greater than zero.")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"FL prompt envelope {position} duration must be greater than zero.")

        envelopes.append({
            "prompt": prompt.strip(),
            "weights": resolved_weights,
            "fps": fps,
            "duration": duration,
        })
    return envelopes


def _format_time(seconds):
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:06.3f}"


def _combine_prompt(global_prompt, prompt):
    return "\n\n".join(part for part in (global_prompt.strip(), prompt.strip()) if part)


def _semantic_prompt(global_prompt, sections, prompt_envelopes=None):
    if not sections and not prompt_envelopes:
        return global_prompt.strip()
    entries = []
    if sections:
        entries.append("Timeline:")
        for section in sections:
            entries.append(
                f"[{_format_time(section['start'])} - {_format_time(section['end'])}]\n{section['prompt']}"
            )
    if prompt_envelopes:
        entries.append("Audio-reactive accents:")
        entries.extend(dict.fromkeys(envelope["prompt"] for envelope in prompt_envelopes))
    return _combine_prompt(global_prompt, "\n\n".join(entries))


def _prepare_references(vae, audio_vae, width, height, frame_count, ref_image_size,
                        ref_images, ref_videos, ref_video_audios, ref_audios):
    ref_items = []
    ref_blocks = []

    for image in (ref_images or {}).values():
        if image is None:
            continue
        image_height, image_width = image.shape[1], image.shape[2]
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((width * height) / (image_width * image_height)))
        else:
            scale = min(1.0, minimax_h3.REF_IMAGE_SHORT_EDGE / min(image_width, image_height))
        target_width = max(
            minimax_h3.CANVAS_MULTIPLE,
            round(image_width * scale / minimax_h3.CANVAS_MULTIPLE) * minimax_h3.CANVAS_MULTIPLE,
        )
        target_height = max(
            minimax_h3.CANVAS_MULTIPLE,
            round(image_height * scale / minimax_h3.CANVAS_MULTIPLE) * minimax_h3.CANVAS_MULTIPLE,
        )
        resized = minimax_h3._resize(image[:1], target_width, target_height, "disabled")
        latent = vae.encode(resized)
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({
            "kind": "image",
            "latent_h": target_height // 16,
            "latent_w": target_width // 16,
            "latent": latent,
        })

    ref_video_audios = ref_video_audios or {}
    for name, video_frames in (ref_videos or {}).items():
        if video_frames is None:
            continue
        soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        video_height, video_width = video_frames.shape[1], video_frames.shape[2]
        canvas_width, canvas_height = minimax_h3.adapt_canvas(video_width, video_height)
        if video_width * video_height < canvas_width * canvas_height:
            canvas_width = max(
                minimax_h3.CANVAS_MULTIPLE,
                round(video_width / minimax_h3.CANVAS_MULTIPLE) * minimax_h3.CANVAS_MULTIPLE,
            )
            canvas_height = max(
                minimax_h3.CANVAS_MULTIPLE,
                round(video_height / minimax_h3.CANVAS_MULTIPLE) * minimax_h3.CANVAS_MULTIPLE,
            )
        frames = minimax_h3._resize(video_frames, canvas_width, canvas_height, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        count = frames.shape[0]
        if count < 5:
            raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps).")
        while count % 17 != 5:
            count -= 1
        frames = frames[:count]
        latent = vae.encode(frames)

        audio_latent, audio_length = None, 0
        if soundtrack is not None:
            audio_latent, audio_length = minimax_h3.MiniMaxH3ReferenceToVideo._encode_ref_audio(
                audio_vae, soundtrack
            )
            ref_items.append({"type": "audio"})

        sample_indices = list(range(0, frames.shape[0], minimax_h3.FPS // 2))
        ref_items.append({
            "type": "video",
            "data": frames[sample_indices],
            "timestamps": [index / 2.0 for index in range(len(sample_indices))],
        })
        ref_blocks.append({
            "kind": "video_audio" if audio_length else "video",
            "latent_t": latent.shape[2],
            "latent_h": canvas_height // 16,
            "latent_w": canvas_width // 16,
            "ref_audio_t": audio_length,
            "latent": latent,
            "audio_latent": audio_latent,
        })

    for audio in (ref_audios or {}).values():
        if audio is None:
            continue
        audio_latent, audio_length = minimax_h3.MiniMaxH3ReferenceToVideo._encode_ref_audio(
            audio_vae, audio
        )
        ref_items.append({"type": "audio"})
        ref_blocks.append({
            "kind": "audio",
            "ref_audio_t": audio_length,
            "audio_latent": audio_latent,
        })

    return ref_items, ref_blocks


def _encode_prompt(clip, prompt, ref_items, ref_blocks):
    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if ref_blocks:
        conditioning = node_helpers.conditioning_set_values(
            conditioning, {"minimax_refs": ref_blocks}
        )
    return conditioning


def _curve(value, transition_mode):
    if transition_mode == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * value)
    return value


def _weights_at_time(sections, seconds, transition_mode, transition_seconds):
    weights = [0.0] * len(sections)
    if sections and "fade_in_end" in sections[0]:
        for index, (first, second) in enumerate(zip(sections, sections[1:])):
            crossfade_start = second.get("crossfade_start", second["start"])
            crossfade_end = second.get("crossfade_end", second["start"])
            if crossfade_start <= seconds < crossfade_end:
                amount = _curve(
                    (seconds - crossfade_start) / (crossfade_end - crossfade_start),
                    second["curve"],
                )
                weights[index] = 1.0 - amount
                weights[index + 1] = amount
                return weights
        for index, section in enumerate(sections):
            if not section["start"] <= seconds < section["end"]:
                continue
            weight = 1.0
            if section["fade_in_end"] > section["start"] + _EPS and seconds < section["fade_in_end"]:
                amount = (seconds - section["start"]) / (
                    section["fade_in_end"] - section["start"]
                )
                weight = _curve(amount, section["curve"])
            if section["fade_out_start"] < section["end"] - _EPS and seconds >= section["fade_out_start"]:
                amount = (seconds - section["fade_out_start"]) / (
                    section["end"] - section["fade_out_start"]
                )
                weight = min(weight, 1.0 - _curve(amount, section["curve"]))
            weights[index] = weight
        return weights

    if transition_mode != "hard" and transition_seconds > 0:
        half = transition_seconds * 0.5
        for index, (first, second) in enumerate(zip(sections, sections[1:])):
            if abs(first["end"] - second["start"]) > _EPS:
                continue
            boundary = first["end"]
            if boundary - half <= seconds < boundary + half:
                amount = _curve((seconds - boundary + half) / transition_seconds, transition_mode)
                weights[index] = 1.0 - amount
                weights[index + 1] = amount
                return weights

    for index, section in enumerate(sections):
        if section["start"] <= seconds < section["end"]:
            weights[index] = 1.0
            break
    return weights


def _temporal_weights(sections, video_t, audio_t, transition_mode, transition_frames, affect_audio):
    transition_seconds = (
        transition_frames / minimax_h3.FPS if transition_mode != "hard" else 0.0
    )
    video_weights = [[0.0] * video_t for _ in sections]
    audio_weights = [[0.0] * audio_t for _ in sections]

    frame = 0
    for token in range(video_t):
        token_frames = _VIDEO_FRAMES_PER_TOKEN[token % len(_VIDEO_FRAMES_PER_TOKEN)]
        totals = [0.0] * len(sections)
        for pixel_frame in range(frame, frame + token_frames):
            weights = _weights_at_time(
                sections,
                (pixel_frame + 0.5) / minimax_h3.FPS,
                transition_mode,
                transition_seconds,
            )
            for index, weight in enumerate(weights):
                totals[index] += weight
        for index, total in enumerate(totals):
            video_weights[index][token] = total / token_frames
        frame += token_frames

    if affect_audio == "video and audio":
        for token in range(audio_t):
            weights = _weights_at_time(
                sections,
                (token + 0.5) / minimax_h3.AUDIO_LATENT_FPS,
                transition_mode,
                transition_seconds,
            )
            for index, weight in enumerate(weights):
                audio_weights[index][token] = weight

    return video_weights, audio_weights


def _envelope_at_time(envelope, seconds):
    if seconds < 0 or seconds >= envelope["duration"]:
        return 0.0
    values = envelope["weights"]
    position = seconds * envelope["fps"] - 0.5
    if position <= 0:
        return values[0]
    if position >= len(values) - 1:
        return values[-1]
    first = int(math.floor(position))
    amount = position - first
    return values[first] * (1.0 - amount) + values[first + 1] * amount


def _envelope_temporal_weights(envelope, video_t, audio_t, affect_audio):
    video_weights = [0.0] * video_t
    audio_weights = [0.0] * audio_t

    frame = 0
    for token in range(video_t):
        token_frames = _VIDEO_FRAMES_PER_TOKEN[token % len(_VIDEO_FRAMES_PER_TOKEN)]
        total = 0.0
        for pixel_frame in range(frame, frame + token_frames):
            total += _envelope_at_time(
                envelope,
                (pixel_frame + 0.5) / minimax_h3.FPS,
            )
        video_weights[token] = total / token_frames
        frame += token_frames

    if affect_audio == "video and audio":
        for token in range(audio_t):
            audio_weights[token] = _envelope_at_time(
                envelope,
                (token + 0.5) / minimax_h3.AUDIO_LATENT_FPS,
            )
    return video_weights, audio_weights


def _conditioning_groups(clip, global_prompt, sections, ref_items, ref_blocks):
    groups = {}
    for index, section in enumerate(sections):
        prompt = _combine_prompt(global_prompt, section["prompt"])
        group = groups.get(prompt)
        if group is None:
            group = {
                "prompt": prompt,
                "section_indices": [],
                "conditioning": _encode_prompt(clip, prompt, ref_items, ref_blocks),
            }
            groups[prompt] = group
        group["section_indices"].append(index)
    return list(groups.values())


def _prompt_envelope_groups(clip, global_prompt, prompt_envelopes, ref_items, ref_blocks):
    groups = {}
    for index, envelope in enumerate(prompt_envelopes):
        prompt = _combine_prompt(global_prompt, envelope["prompt"])
        group = groups.get(prompt)
        if group is None:
            group = {
                "prompt": prompt,
                "envelope_indices": [],
                "conditioning": _encode_prompt(clip, prompt, ref_items, ref_blocks),
            }
            groups[prompt] = group
        group["envelope_indices"].append(index)
    return list(groups.values())


def _merge_weights(weights, indices):
    if not indices:
        return []
    return [
        max(weights[index][position] for index in indices)
        for position in range(len(weights[0]))
    ]


def _merge_section_weights(weights, indices):
    if not indices:
        return []
    return [
        min(1.0, sum(weights[index][position] for index in indices))
        for position in range(len(weights[0]))
    ]


def _h3_tensors(latent):
    samples = latent.get("samples")
    if not isinstance(samples, comfy.nested_tensor.NestedTensor):
        raise TypeError("FL MiniMax H3 timeline expects a nested H3 video/audio latent.")
    tensors = samples.unbind()
    if len(tensors) != 2:
        raise ValueError("FL MiniMax H3 timeline expects exactly one video and one audio latent.")
    video, audio = tensors
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
        raise ValueError("FL MiniMax H3 timeline expects video latent shape [1, 24, T, H, W].")
    if audio.ndim != 4 or audio.shape[0] != 1 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError("FL MiniMax H3 timeline expects audio latent shape [1, 32, 2, T40].")
    return video, audio


def _flatten_mask(video_shape, audio_shape, video_weights, audio_weights):
    _, video_channels, video_t, video_height, video_width = video_shape
    _, audio_channels, audio_tracks, audio_t = audio_shape
    video = torch.tensor(video_weights, dtype=torch.float16).view(1, video_t, 1, 1)
    video = video.expand(video_channels, video_t, video_height, video_width).reshape(-1)
    audio = torch.tensor(audio_weights, dtype=torch.float16).view(1, 1, audio_t)
    audio = audio.expand(audio_channels, audio_tracks, audio_t).reshape(-1)
    return torch.cat((video, audio)).unsqueeze(0)


def _apply_timeline(timeline, latent):
    if not isinstance(timeline, dict) or timeline.get("type") != "minimax_h3_prompt_timeline":
        raise TypeError("FL MiniMax H3 Apply Timeline received an invalid timeline object.")

    video, audio = _h3_tensors(latent)
    if video.shape[2] != timeline["video_t"] or audio.shape[-1] != timeline["audio_t"]:
        raise ValueError(
            "FL MiniMax H3 Apply Timeline can change spatial resolution, but the target latent must have "
            "the same video and audio duration as the source timeline."
        )

    sections = timeline["sections"]
    prompt_envelopes = timeline.get("prompt_envelopes", [])
    prompt_envelope_groups = timeline.get("prompt_envelope_groups", [])
    if not sections and not prompt_envelopes:
        return timeline["global_conditioning"]

    conditioning = []
    if sections:
        video_weights, audio_weights = _temporal_weights(
            sections,
            video.shape[2],
            audio.shape[-1],
            timeline["transition_mode"],
            timeline["transition_frames"],
            timeline["affect_audio"],
        )
        for group in timeline["conditioning_groups"]:
            group_video = _merge_section_weights(video_weights, group["section_indices"])
            group_audio = _merge_section_weights(audio_weights, group["section_indices"])
            mask = _flatten_mask(video.shape, audio.shape, group_video, group_audio)
            if torch.count_nonzero(mask):
                conditioning.extend(
                    node_helpers.conditioning_set_values(
                        group["conditioning"], {"mask": mask}
                    )
                )

    if prompt_envelopes:
        envelope_video_weights = []
        envelope_audio_weights = []
        for envelope in prompt_envelopes:
            envelope_video, envelope_audio = _envelope_temporal_weights(
                envelope,
                video.shape[2],
                audio.shape[-1],
                timeline["affect_audio"],
            )
            envelope_video_weights.append(envelope_video)
            envelope_audio_weights.append(envelope_audio)
        for group in prompt_envelope_groups:
            group_video = _merge_weights(
                envelope_video_weights,
                group["envelope_indices"],
            )
            group_audio = _merge_weights(
                envelope_audio_weights,
                group["envelope_indices"],
            )
            mask = _flatten_mask(video.shape, audio.shape, group_video, group_audio)
            if torch.count_nonzero(mask):
                conditioning.extend(
                    node_helpers.conditioning_set_values(
                        group["conditioning"], {"mask": mask}
                    )
                )

    if not conditioning:
        return timeline["global_conditioning"]
    conditioning.extend(
        node_helpers.conditioning_set_values(
            timeline["global_conditioning"], {"default": True}
        )
    )
    return conditioning


class FL_MiniMaxH3PromptTimeline(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3PromptTimeline",
            display_name="FL MiniMax H3 Prompt Timeline",
            category="🏵️Fill Nodes/Prompting",
            description=(
                "Creates MiniMax H3 video/audio latents with both strict latent-time prompt scheduling "
                "and a fast semantic timeline conditioning."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip="MiniMax H3 text encoder used for the global and scheduled prompts.",
                ),
                io.Vae.Input(
                    "vae",
                    tooltip="MiniMax H3 video VAE used for reference images and reference videos.",
                ),
                io.Vae.Input(
                    "audio_vae",
                    tooltip="MiniMax H3 audio VAE used for audio references and video soundtracks.",
                ),
                io.String.Input(
                    "global_prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip=(
                        "Persistent subject, style, and scene context. This is the default conditioning "
                        "under scheduled masks and is prepended to every scheduled prompt."
                    ),
                ),
                io.String.Input(
                    "timeline",
                    optional=True,
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    placeholder="[00:00.000 - 00:02.000]\nDescribe the first section.",
                    tooltip=(
                        "Legacy manual ordered time ranges followed by prompt text. Used only when "
                        "prompt_schedule is not connected."
                    ),
                ),
                FLPromptSchedule.Input(
                    "prompt_schedule",
                    optional=True,
                    tooltip=(
                        "Optional exact-time schedule, such as the output of FL Audio Beat Prompt Schedule. "
                        "When connected, it overrides the manual timeline."
                    ),
                ),
                io.Autogrow.Input(
                    "prompt_envelopes",
                    optional=True,
                    tooltip=(
                        "Optional audio-reactive prompt envelopes. Each unique prompt adds one conditioning "
                        "evaluation per sampling step."
                    ),
                    template=io.Autogrow.TemplatePrefix(
                        input=FLPromptEnvelope.Input(
                            "prompt_envelope",
                            tooltip="Reactive prompt envelope from an FL audio prompt-envelope node.",
                        ),
                        prefix="prompt_envelope_",
                        min=0,
                        max=10,
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=1344,
                    min=32,
                    max=nodes.MAX_RESOLUTION,
                    step=32,
                    tooltip="Width of the native H3 video latent in output pixels.",
                ),
                io.Int.Input(
                    "height",
                    default=768,
                    min=32,
                    max=nodes.MAX_RESOLUTION,
                    step=32,
                    tooltip="Height of the native H3 video latent in output pixels.",
                ),
                io.Int.Input(
                    "length",
                    default=124,
                    min=5,
                    max=3600,
                    step=17,
                    tooltip="Requested frame count at H3's fixed 24 fps. H3 snaps it to the 17k+5 frame grid.",
                ),
                io.Combo.Input(
                    "time_unit",
                    options=["seconds", "frames", "beats"],
                    default="seconds",
                    tooltip=(
                        "Legacy unit used only by manual timeline headers. Connected prompt schedules "
                        "own their time unit; H3 video always runs at 24 fps."
                    ),
                ),
                io.Float.Input(
                    "bpm",
                    default=120.0,
                    min=1.0,
                    max=999.0,
                    step=0.1,
                    tooltip=(
                        "Legacy constant BPM used only for manual timelines in beats mode. Connected "
                        "prompt schedules carry exact detected timing."
                    ),
                ),
                io.Combo.Input(
                    "transition_mode",
                    options=["hard", "linear", "cosine"],
                    default="cosine",
                    tooltip="Transition curve between contiguous sections in the manual timeline.",
                ),
                io.Int.Input(
                    "transition_frames",
                    default=4,
                    min=0,
                    max=96,
                    step=1,
                    tooltip="Manual timeline transition width in 24 fps video frames.",
                ),
                io.Combo.Input(
                    "affect_audio",
                    options=["video only", "video and audio"],
                    default="video only",
                    tooltip=(
                        "Choose whether scheduled text masks affect only H3 video tokens or both generated "
                        "video and audio tokens."
                    ),
                ),
                io.Combo.Input(
                    "duration_policy",
                    options=["strict", "clamp", "fit"],
                    default="strict",
                    tooltip=(
                        "strict rejects ranges past the H3 duration; clamp trims them; fit scales the full "
                        "schedule to the aligned H3 duration."
                    ),
                ),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                    tooltip=(
                        "match limits reference image area to the output canvas; max uses H3's maximum "
                        "reference short edge."
                    ),
                ),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    tooltip="Optional H3 reference images. Add sockets as needed.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_image",
                            tooltip="Reference image used by MiniMax H3.",
                        ),
                        prefix="ref_image_",
                        min=0,
                        max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos",
                    optional=True,
                    tooltip="Optional H3 reference videos. Add sockets as needed.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_video",
                            tooltip="Reference video frames used by MiniMax H3.",
                        ),
                        prefix="ref_video_",
                        min=0,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios",
                    optional=True,
                    tooltip="Optional soundtracks paired by index with reference videos.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input(
                            "ref_video_audio",
                            tooltip="Audio paired with the reference video of the same index.",
                        ),
                        prefix="ref_video_audio_",
                        min=0,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios",
                    optional=True,
                    tooltip="Optional standalone H3 audio references. Add sockets as needed.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input(
                            "ref_audio",
                            tooltip="Standalone audio reference used by MiniMax H3.",
                        ),
                        prefix="ref_audio_",
                        min=0,
                        max=3,
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="scheduled",
                    tooltip="Strict latent-time conditioning for the first H3 sampling pass.",
                ),
                io.Latent.Output(
                    display_name="latent",
                    tooltip="Native nested MiniMax H3 video/audio latent.",
                ),
                io.Conditioning.Output(
                    display_name="semantic",
                    tooltip=(
                        "Single semantic timeline conditioning for faster low-denoise refinement passes."
                    ),
                ),
                H3Timeline.Output(
                    display_name="timeline",
                    tooltip="Reusable encoded H3 timeline for applying the same schedule after spatial resizing.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        vae,
        audio_vae,
        global_prompt,
        width,
        height,
        length,
        time_unit,
        bpm,
        transition_mode,
        transition_frames,
        affect_audio,
        duration_policy,
        ref_image_size,
        timeline=None,
        prompt_schedule=None,
        prompt_envelopes=None,
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    ):
        latent, frame_count = minimax_h3._empty_av_latent(width, height, length)
        video, audio = _h3_tensors(latent)
        duration = frame_count / minimax_h3.FPS
        if prompt_schedule is not None:
            parsed_sections = _schedule_sections(prompt_schedule)
            resolved_transition_mode = "hard"
            resolved_transition_frames = 0
        else:
            parsed_sections = _parse_timeline(timeline or "", time_unit, bpm)
            resolved_transition_mode = transition_mode
            resolved_transition_frames = transition_frames
        sections = _resolve_sections(
            parsed_sections,
            duration,
            duration_policy,
            resolved_transition_mode,
            resolved_transition_frames,
        )
        resolved_prompt_envelopes = _prompt_envelopes(prompt_envelopes)

        ref_items, ref_blocks = _prepare_references(
            vae,
            audio_vae,
            width,
            height,
            frame_count,
            ref_image_size,
            ref_images,
            ref_videos,
            ref_video_audios,
            ref_audios,
        )
        global_conditioning = _encode_prompt(
            clip, global_prompt.strip(), ref_items, ref_blocks
        )
        semantic_conditioning = _encode_prompt(
            clip,
            _semantic_prompt(global_prompt, sections, resolved_prompt_envelopes),
            ref_items,
            ref_blocks,
        )

        conditioning_groups = _conditioning_groups(
            clip,
            global_prompt,
            sections,
            ref_items,
            ref_blocks,
        )
        prompt_envelope_groups = _prompt_envelope_groups(
            clip,
            global_prompt,
            resolved_prompt_envelopes,
            ref_items,
            ref_blocks,
        )

        timeline_object = {
            "type": "minimax_h3_prompt_timeline",
            "frame_count": frame_count,
            "video_t": video.shape[2],
            "audio_t": audio.shape[-1],
            "duration": duration,
            "sections": sections,
            "conditioning_groups": conditioning_groups,
            "prompt_envelopes": resolved_prompt_envelopes,
            "prompt_envelope_groups": prompt_envelope_groups,
            "global_conditioning": global_conditioning,
            "semantic_conditioning": semantic_conditioning,
            "transition_mode": resolved_transition_mode,
            "transition_frames": resolved_transition_frames,
            "affect_audio": affect_audio,
        }
        scheduled_conditioning = _apply_timeline(timeline_object, latent)
        logging.info(
            "FL MiniMax H3 timeline: %d sections, %d unique timeline prompts, "
            "%d reactive envelopes, %d unique reactive prompts, %d frames, %.3fs.",
            len(sections),
            len(conditioning_groups),
            len(resolved_prompt_envelopes),
            len(prompt_envelope_groups),
            frame_count,
            duration,
        )
        return io.NodeOutput(
            scheduled_conditioning,
            latent,
            semantic_conditioning,
            timeline_object,
        )


class FL_MiniMaxH3ApplyTimeline(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3ApplyTimeline",
            display_name="FL MiniMax H3 Apply Timeline",
            category="🏵️Fill Nodes/Prompting",
            description=(
                "Rebuilds a MiniMax H3 prompt timeline for a spatially resized H3 video/audio latent."
            ),
            inputs=[
                H3Timeline.Input(
                    "timeline",
                    tooltip="Encoded timeline from FL MiniMax H3 Prompt Timeline.",
                ),
                io.Latent.Input(
                    "latent",
                    tooltip=(
                        "Spatially resized native H3 video/audio latent. Its video and audio duration "
                        "must match the source timeline."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="scheduled",
                    tooltip="Strict prompt schedule rebuilt for the target latent's spatial dimensions.",
                )
            ],
        )

    @classmethod
    def execute(cls, timeline, latent):
        return io.NodeOutput(_apply_timeline(timeline, latent))
