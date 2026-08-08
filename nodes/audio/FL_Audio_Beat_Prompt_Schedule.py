import json
import math
import re

from comfy_api.latest import io

from .audio_files import (
    audio_file_hash,
    available_audio_files,
    resolve_audio_path,
)
from .audio_timeline import (
    ANALYSIS_VERSION,
    DETECTOR_VERSION,
    analyze_audio_file,
    apply_beat_offset,
    cached_analysis_audio_file,
)


FLPromptSchedule = io.Custom("FL_PROMPT_SCHEDULE")
_HEADER = re.compile(
    r"^\s*\[\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)"
    r"(?:\s*\|\s*(.*?))?\s*\]\s*$"
)
_HEADER_START = re.compile(r"^\s*\[\s*[0-9]+(?:\.[0-9]+)?\s*-")
_EPS = 1e-6
_DEFAULT_TIMELINE = (
    "[0 - 48 | fade_in=6 | fade_out=6]\n"
    "The subject slowly turns toward camera.\n\n"
    "[48 - 96 | fade_in=6 | fade_out=6]\n"
    "The camera pushes forward on the beat."
)


def _number(value, name, line):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Beat prompt schedule line {line}: {name} must be a number.") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Beat prompt schedule line {line}: {name} must be zero or greater.")
    return number


def _load_beats(beat_positions):
    data = _load_beat_data(beat_positions)
    return data["beat_times"], data["audio_duration"]


def _parse_beat_payload(beat_positions):
    if isinstance(beat_positions, dict):
        return beat_positions
    try:
        return json.loads(beat_positions)
    except json.JSONDecodeError as error:
        raise ValueError(f"Beat positions is not valid JSON: {error.msg}.") from error


def _load_beat_data(beat_positions):
    data = _parse_beat_payload(beat_positions)
    if not isinstance(data, dict):
        raise ValueError("Beat positions must be the JSON object from FL Audio BPM Analyzer.")

    values = data.get("beat_times")
    if not isinstance(values, list) or not values:
        raise ValueError("Beat positions must contain a non-empty beat_times list.")

    beat_times = []
    for index, value in enumerate(values):
        beat_time = _number(value, f"beat_times[{index}]", 0)
        if beat_times and beat_time < beat_times[-1] - _EPS:
            raise ValueError("Beat positions beat_times must be ordered from earliest to latest.")
        beat_times.append(beat_time)

    duration = _number(data.get("audio_duration"), "audio_duration", 0)
    if duration <= 0:
        raise ValueError("Beat positions audio_duration must be greater than zero.")
    if beat_times[-1] > duration + _EPS:
        raise ValueError("Beat positions contains a beat after audio_duration.")

    bpm = _number(data.get("bpm", 0.0), "bpm", 0)
    base_beat_times = data.get("base_beat_times", beat_times)
    if not isinstance(base_beat_times, list):
        base_beat_times = beat_times
    base_detected_beat_times = data.get(
        "base_detected_beat_times",
        data.get("detected_beat_times", []),
    )
    if not isinstance(base_detected_beat_times, list):
        base_detected_beat_times = []
    downbeat_times = data.get("downbeat_times", [])
    if not isinstance(downbeat_times, list):
        downbeat_times = []
    base_downbeat_times = data.get("base_downbeat_times", downbeat_times)
    if not isinstance(base_downbeat_times, list):
        base_downbeat_times = []
    detected_downbeat_times = data.get("detected_downbeat_times", downbeat_times)
    if not isinstance(detected_downbeat_times, list):
        detected_downbeat_times = []
    base_detected_downbeat_times = data.get(
        "base_detected_downbeat_times",
        detected_downbeat_times,
    )
    if not isinstance(base_detected_downbeat_times, list):
        base_detected_downbeat_times = []
    detected_beat_confidences = data.get("detected_beat_confidences", [])
    if not isinstance(detected_beat_confidences, list):
        detected_beat_confidences = []
    base_detected_beat_confidences = data.get(
        "base_detected_beat_confidences",
        detected_beat_confidences,
    )
    if not isinstance(base_detected_beat_confidences, list):
        base_detected_beat_confidences = []
    detected_downbeat_confidences = data.get("detected_downbeat_confidences", [])
    if not isinstance(detected_downbeat_confidences, list):
        detected_downbeat_confidences = []
    base_detected_downbeat_confidences = data.get(
        "base_detected_downbeat_confidences",
        detected_downbeat_confidences,
    )
    if not isinstance(base_detected_downbeat_confidences, list):
        base_detected_downbeat_confidences = []
    return {
        "bpm": bpm,
        "grid_bpm": data.get("grid_bpm", bpm),
        "base_grid_interval_seconds": data.get("base_grid_interval_seconds", 0.0),
        "grid_interval_seconds": data.get("grid_interval_seconds", 0.0),
        "beat_grid_density": data.get("beat_grid_density", "every_beat"),
        "beat_times": beat_times,
        "base_beat_times": base_beat_times,
        "downbeat_times": downbeat_times,
        "base_downbeat_times": base_downbeat_times,
        "audio_duration": duration,
        "detected_beat_times": data.get("detected_beat_times", []),
        "base_detected_beat_times": base_detected_beat_times,
        "detected_downbeat_times": detected_downbeat_times,
        "base_detected_downbeat_times": base_detected_downbeat_times,
        "detected_beat_confidences": detected_beat_confidences,
        "base_detected_beat_confidences": base_detected_beat_confidences,
        "detected_downbeat_confidences": detected_downbeat_confidences,
        "base_detected_downbeat_confidences": base_detected_downbeat_confidences,
        "onset_times": data.get("onset_times", []),
        "drum_times": data.get("drum_times", {}),
        "bpm_source": data.get("bpm_source", ""),
        "analysis_source": data.get("analysis_source", ""),
        "beat_analysis_source": data.get("beat_analysis_source", ""),
        "detector_version": data.get("detector_version", ""),
        "detector": data.get("detector") if isinstance(data.get("detector"), dict) else None,
        "analysis_cache_hit": bool(data.get("analysis_cache_hit", False)),
        "waveform_preview": (
            data.get("waveform_preview")
            if isinstance(data.get("waveform_preview"), dict)
            else None
        ),
    }


def _parse_options(options, line, default_fade_in, default_fade_out):
    values = {
        "fade_in": default_fade_in,
        "fade_out": default_fade_out,
        "crossfade": 0.0,
    }
    if not options:
        return values

    for option in options.split("|"):
        if "=" not in option:
            raise ValueError(
                f"Beat prompt schedule line {line}: options must use "
                "fade_in=value, fade_out=value, or crossfade=value."
            )
        name, value = (part.strip() for part in option.split("=", 1))
        if name not in values:
            raise ValueError(
                f"Beat prompt schedule line {line}: unknown option '{name}'. "
                "Use fade_in, fade_out, or crossfade."
            )
        values[name] = _number(value, name, line)
    return values


def _parse_schedule(text, default_fade_in, default_fade_out, time_unit="beats"):
    if time_unit not in {"beats", "seconds", "frames"}:
        raise ValueError(f"Beat prompt schedule has an invalid time unit '{time_unit}'.")

    sections = []
    current = None
    body = []
    unit_name = time_unit[:-1] if time_unit != "seconds" else "second"

    def finish():
        if current is None:
            return
        prompt = "\n".join(body).strip()
        if not prompt:
            raise ValueError(f"Beat prompt schedule line {current['line']}: section prompt is empty.")
        sections.append({**current, "prompt": prompt})

    for line_number, line in enumerate(text.splitlines(), 1):
        match = _HEADER.match(line)
        if match:
            finish()
            body = []
            start = _number(match.group(1), f"start {unit_name}", line_number)
            end = _number(match.group(2), f"end {unit_name}", line_number)
            if end <= start:
                raise ValueError(
                    f"Beat prompt schedule line {line_number}: end {unit_name} must be after "
                    f"start {unit_name}."
                )
            options = _parse_options(
                match.group(3), line_number, default_fade_in, default_fade_out
            )
            if time_unit == "frames":
                frame_values = {
                    "start": start,
                    "end": end,
                    "fade_in": options["fade_in"],
                    "fade_out": options["fade_out"],
                    "crossfade": options["crossfade"],
                }
                for name, value in frame_values.items():
                    if abs(value - round(value)) > _EPS:
                        raise ValueError(
                            f"Beat prompt schedule line {line_number}: {name} must be a whole "
                            "frame in frames mode."
                        )
            if options["fade_in"] + options["fade_out"] > end - start + _EPS:
                raise ValueError(
                    f"Beat prompt schedule line {line_number}: fade_in and fade_out exceed the section."
                )
            current = {
                "line": line_number,
                "start_position": start,
                "end_position": end,
                "start_beat": start,
                "end_beat": end,
                **options,
            }
            continue

        if _HEADER_START.match(line):
            raise ValueError(
                f"Beat prompt schedule line {line_number}: expected a header like "
                "[0 - 8 | fade_in=0.5 | fade_out=0.5]."
            )
        if current is None:
            if line.strip():
                raise ValueError(
                    f"Beat prompt schedule line {line_number}: prompt text must follow a beat header."
                )
            continue
        body.append(line)

    finish()
    if not sections:
        raise ValueError("Beat prompt schedule has no sections.")

    previous = None
    for section in sections:
        if previous is not None:
            if section["start_position"] < previous["start_position"]:
                raise ValueError(
                    f"Beat prompt schedule line {section['line']}: sections must be ordered."
                )
            if section["start_position"] < previous["end_position"] - _EPS:
                raise ValueError(
                    f"Beat prompt schedule line {section['line']}: section overlaps the previous section."
                )
            if section["crossfade"] > _EPS:
                if abs(section["start_position"] - previous["end_position"]) > _EPS:
                    raise ValueError(
                        f"Beat prompt schedule line {section['line']}: crossfade requires "
                        "a touching previous section."
                    )
                shortest = min(
                    previous["end_position"] - previous["start_position"],
                    section["end_position"] - section["start_position"],
                )
                if section["crossfade"] > shortest + _EPS:
                    raise ValueError(
                        f"Beat prompt schedule line {section['line']}: crossfade exceeds "
                        "the shorter adjacent section."
                    )
                previous["fade_out"] = 0.0
                section["fade_in"] = 0.0
        elif section["crossfade"] > _EPS:
            raise ValueError(
                f"Beat prompt schedule line {section['line']}: the first section cannot crossfade."
            )
        previous = section
    return sections


def _apply_render_groups(sections, value):
    if value is None or value == "":
        return sections
    try:
        payload = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as error:
        raise ValueError(f"Beat prompt render groups is not valid JSON: {error.msg}.") from error
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Beat prompt render groups must be a version 1 object.")
    groups = payload.get("section_groups")
    if not isinstance(groups, list) or len(groups) != len(sections):
        raise ValueError(
            "Beat prompt render groups must contain one section_groups entry per prompt section."
        )

    positions = {}
    for index, group in enumerate(groups):
        if group is None:
            continue
        if isinstance(group, bool) or not isinstance(group, int) or group < 1:
            raise ValueError(
                f"Beat prompt render group for section {index + 1} must be a positive integer or null."
            )
        positions.setdefault(group, []).append(index)

    for group, indices in positions.items():
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"Beat prompt render group {group} must contain consecutive sections.")
        for previous, current in zip(indices, indices[1:]):
            if abs(sections[previous]["end_position"] - sections[current]["start_position"]) > _EPS:
                raise ValueError(f"Beat prompt render group {group} requires touching prompt sections.")
        for index in indices:
            sections[index]["render_group"] = group
    return sections


def _beat_to_seconds(position, beat_times, duration, line):
    maximum = len(beat_times)
    if position > maximum + _EPS:
        raise ValueError(
            f"Beat prompt schedule line {line}: beat {position:g} is beyond the "
            f"available range 0-{maximum}."
        )
    if position >= maximum - _EPS:
        return duration

    index = int(math.floor(position))
    amount = position - index
    start = beat_times[index]
    end = beat_times[index + 1] if index + 1 < len(beat_times) else duration
    return start + (end - start) * amount


def _position_to_seconds(position, time_unit, beat_times, duration, fps, line):
    if time_unit == "beats":
        return _beat_to_seconds(position, beat_times, duration, line)
    if time_unit == "seconds":
        return position
    return position / fps


def _resolve_duration(length, audio_duration, fps):
    if length <= 0:
        return audio_duration
    duration = length / fps
    if duration > audio_duration + _EPS:
        raise ValueError(
            f"Beat prompt schedule length {length:g} frames at {fps:g} FPS exceeds the "
            f"audio duration {audio_duration:g}s."
        )
    return min(duration, audio_duration)


def _resolve_schedule(
    sections,
    beat_times,
    duration,
    curve,
    time_unit="beats",
    fps=24.0,
    schedule_duration=None,
):
    limit = duration if schedule_duration is None else schedule_duration
    resolved = []
    for section in sections:
        start = _position_to_seconds(
            section["start_position"],
            time_unit,
            beat_times,
            duration,
            fps,
            section["line"],
        )
        end = _position_to_seconds(
            section["end_position"],
            time_unit,
            beat_times,
            duration,
            fps,
            section["line"],
        )
        if end <= start + _EPS:
            raise ValueError(
                f"Beat prompt schedule line {section['line']}: the selected range resolves "
                "to an empty time range."
            )
        if start >= limit - _EPS:
            continue
        end = min(end, limit)
        fade_in_end = _position_to_seconds(
            section["start_position"] + section["fade_in"],
            time_unit,
            beat_times,
            duration,
            fps,
            section["line"],
        )
        fade_out_start = _position_to_seconds(
            section["end_position"] - section["fade_out"],
            time_unit,
            beat_times,
            duration,
            fps,
            section["line"],
        )
        fade_in_end = min(end, fade_in_end)
        fade_out_start = min(end, max(start, fade_out_start))
        if section["crossfade"] > _EPS:
            if time_unit == "frames":
                crossfade_before = math.floor(section["crossfade"] / 2)
                crossfade_after = section["crossfade"] - crossfade_before
            else:
                crossfade_before = section["crossfade"] * 0.5
                crossfade_after = crossfade_before
            crossfade_start = _position_to_seconds(
                section["start_position"] - crossfade_before,
                time_unit,
                beat_times,
                duration,
                fps,
                section["line"],
            )
            crossfade_end = _position_to_seconds(
                section["start_position"] + crossfade_after,
                time_unit,
                beat_times,
                duration,
                fps,
                section["line"],
            )
            crossfade_start = max(0.0, crossfade_start)
            crossfade_end = min(limit, end, crossfade_end)
        else:
            crossfade_start = start
            crossfade_end = start
        resolved.append({
            **section,
            "start": start,
            "end": end,
            "fade_in_end": fade_in_end,
            "fade_out_start": fade_out_start,
            "crossfade_start": crossfade_start,
            "crossfade_end": crossfade_end,
            "curve": curve,
        })
    for previous, section in zip(resolved, resolved[1:]):
        if section["crossfade_end"] > section["crossfade_start"] + _EPS:
            previous["fade_out_start"] = previous["end"]
            section["fade_in_end"] = section["start"]
    return resolved


def _frame_sections(sections, fps, total_frames):
    frame_sections = []
    for section in sections:
        start_frame = min(total_frames - 1, max(0, round(section["start"] * fps)))
        end_frame = min(total_frames, max(start_frame + 1, round(section["end"] * fps)))
        fade_in_end = min(end_frame, max(start_frame, round(section["fade_in_end"] * fps)))
        fade_out_start = min(end_frame, max(start_frame, round(section["fade_out_start"] * fps)))
        crossfade_start = min(
            total_frames,
            max(0, round(section["crossfade_start"] * fps)),
        )
        crossfade_end = min(
            total_frames,
            max(crossfade_start, round(section["crossfade_end"] * fps)),
        )
        frame_section = {
            "line": section["line"],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "fade_in_frames": fade_in_end - start_frame,
            "fade_out_frames": end_frame - fade_out_start,
            "crossfade_start_frame": crossfade_start,
            "crossfade_end_frame": crossfade_end,
            "crossfade_frames": crossfade_end - crossfade_start,
            "prompt": section["prompt"],
            "curve": section["curve"],
        }
        if "render_group" in section:
            frame_section["render_group"] = section["render_group"]
        frame_sections.append(frame_section)
    return frame_sections


class FL_Audio_Beat_Prompt_Schedule(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_Audio_Beat_Prompt_Schedule",
            display_name="FL Audio Beat Prompt Schedule",
            category="🏵️Fill Nodes/Audio",
            description=(
                "Loads and trims audio, detects beats and drums, and turns prompt clips into "
                "a reusable diffusion-video schedule. A connected beat_positions input "
                "overrides the internally detected beat timing."
            ),
            inputs=[
                io.String.Input(
                    "beat_positions",
                    force_input=True,
                    optional=True,
                    tooltip=(
                        "Optional override from FL Audio BPM Analyzer. When connected, its exact "
                        "beat_times drive timing instead of this node's internal analysis."
                    ),
                ),
                io.String.Input(
                    "timeline",
                    multiline=True,
                    dynamic_prompts=True,
                    default=_DEFAULT_TIMELINE,
                    tooltip=(
                        "Ordered frame ranges followed by prompt text. Range ends are exclusive. "
                        "Older beat- and second-based schedules are converted by the sequencer."
                    ),
                ),
                io.Float.Input(
                    "default_fade_in",
                    display_name="default fade-in (frames)",
                    default=0.0,
                    min=0.0,
                    max=864000.0,
                    step=1.0,
                    tooltip=(
                        "Default prompt fade-in in frames. A header override takes priority."
                    ),
                ),
                io.Float.Input(
                    "default_fade_out",
                    display_name="default fade-out (frames)",
                    default=0.0,
                    min=0.0,
                    max=864000.0,
                    step=1.0,
                    tooltip=(
                        "Default prompt fade-out in frames. A header override takes priority."
                    ),
                ),
                io.Combo.Input(
                    "curve",
                    options=["linear", "cosine"],
                    default="cosine",
                    tooltip="Shape used for prompt fade-ins and fade-outs.",
                ),
                io.Combo.Input(
                    "time_unit",
                    options=["beats", "seconds", "frames"],
                    default="frames",
                    tooltip=(
                        "Internal source unit. New schedules use frames; legacy beat and second "
                        "schedules are converted by the sequencer."
                    ),
                ),
                io.Float.Input(
                    "fps",
                    display_name="FPS",
                    default=24.0,
                    min=1.0,
                    max=240.0,
                    step=0.001,
                    tooltip="Frames per second used to interpret frame-based positions and report frame counts.",
                ),
                io.Int.Input(
                    "sequence_duration",
                    display_name="length (frames)",
                    default=0,
                    min=0,
                    max=864000,
                    step=1,
                    tooltip=(
                        "Maximum schedule length in frames at the selected FPS. Zero uses the full "
                        "detected audio duration."
                    ),
                ),
                io.Combo.Input(
                    "audio_file",
                    display_name="audio",
                    options=[""] + available_audio_files(),
                    default="",
                    optional=True,
                    upload=io.UploadType.audio,
                    tooltip=(
                        "Upload or choose the source audio. The waveform and beat markers load "
                        "without queueing the workflow."
                    ),
                ),
                io.Int.Input(
                    "trim_start_frame",
                    display_name="trim start (frames)",
                    default=0,
                    min=0,
                    max=864000,
                    step=1,
                    tooltip="Source frame where the selected audio crop begins.",
                ),
                io.Boolean.Input(
                    "half_time",
                    display_name="half-time",
                    default=False,
                    tooltip="Use every other detected beat and report half the detected BPM.",
                ),
                io.Int.Input(
                    "beat_offset_ms",
                    display_name="beat offset (ms)",
                    default=0,
                    min=-1000,
                    max=1000,
                    step=1,
                    tooltip=(
                        "Backing value for the sequencer's live Beat offset control. It shifts only "
                        "the regular beat grid while audio-derived reference markers remain fixed."
                    ),
                ),
                io.Combo.Input(
                    "analysis_source",
                    display_name="transient source",
                    options=["mix", "drums", "vocals", "bass", "other"],
                    default="mix",
                    tooltip=(
                        "Choose the waveform and transient reference shown in the editor. Beat This "
                        "always analyzes the master mix; stem choices become available after separation."
                    ),
                ),
                io.Combo.Input(
                    "beat_grid_density",
                    display_name="beat grid density",
                    options=["every_2_beats", "every_beat", "half_beat"],
                    default="every_beat",
                    tooltip=(
                        "Backing value for the sequencer's Grid control. Every beat uses the "
                        "detected tempo; half-beat adds subdivisions."
                    ),
                ),
                io.String.Input(
                    "render_groups",
                    default="",
                    tooltip=(
                        "Sequencer-owned render grouping metadata. The popup editor manages this "
                        "automatically; empty keeps every prompt section independent."
                    ),
                ),
                io.String.Input(
                    "analysis_cache_key",
                    default="",
                    tooltip=(
                        "Sequencer-owned analysis cache reference used to restore a previously "
                        "selected local audio file after workflow widget migrations."
                    ),
                ),
            ],
            outputs=[
                FLPromptSchedule.Output(
                    display_name="prompt_schedule",
                    tooltip="Resolved second-based prompt schedule for compatible FL diffusion nodes.",
                ),
                io.Int.Output(
                    display_name="total_frames",
                    tooltip="Effective schedule duration converted to frames at the selected FPS.",
                ),
                io.Audio.Output(
                    display_name="audio",
                    tooltip="The selected, frame-aligned audio crop for downstream FL audio nodes.",
                ),
                io.Float.Output(
                    display_name="BPM",
                    tooltip="Detected musical tempo after applying the Half-time option.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        beat_positions=None,
        timeline=_DEFAULT_TIMELINE,
        default_fade_in=0.0,
        default_fade_out=0.0,
        curve="cosine",
        time_unit="frames",
        fps=24.0,
        sequence_duration=0,
        audio_file="",
        trim_start_frame=0,
        half_time=False,
        beat_offset_ms=0,
        analysis_source="mix",
        beat_grid_density="every_beat",
        render_groups="",
        analysis_cache_key="",
    ):
        internal_analysis = None
        cropped_audio = None
        if not audio_file and analysis_cache_key:
            audio_file = cached_analysis_audio_file(analysis_cache_key)
        if audio_file:
            internal_analysis, cropped_audio = analyze_audio_file(
                audio_file,
                fps,
                trim_start_frame,
                sequence_duration,
                half_time,
                beat_offset_ms,
                analysis_source,
                beat_grid_density,
                detect_beats=not bool(beat_positions),
            )
        if beat_positions:
            beat_payload = _parse_beat_payload(beat_positions)
            _load_beat_data(beat_payload)
            beat_payload = apply_beat_offset(
                beat_payload,
                fps,
                beat_offset_ms,
                beat_grid_density,
            )
            beat_positions = json.dumps(beat_payload, separators=(",", ":"))
            beat_data = _load_beat_data(beat_payload)
            if internal_analysis is not None:
                difference = abs(beat_data["audio_duration"] - internal_analysis["audio_duration"])
                if difference > max(_EPS, 1.0 / fps):
                    raise ValueError(
                        "Connected beat_positions duration does not match the selected audio crop. "
                        "Analyze the same crop or disconnect the override."
                    )
        elif internal_analysis is not None:
            beat_positions = json.dumps(internal_analysis, separators=(",", ":"))
            beat_data = _load_beat_data(beat_positions)
        else:
            raise ValueError("Choose an audio file or connect beat_positions.")

        beat_times = beat_data["beat_times"]
        audio_duration = beat_data["audio_duration"]
        if time_unit not in {"beats", "seconds", "frames"}:
            raise ValueError(f"Beat prompt schedule has an invalid time unit '{time_unit}'.")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Beat prompt schedule FPS must be greater than zero.")
        if not math.isfinite(sequence_duration) or sequence_duration < 0:
            raise ValueError("Beat prompt schedule length must be zero or greater.")
        if abs(sequence_duration - round(sequence_duration)) > _EPS:
            raise ValueError("Beat prompt schedule length must be a whole frame count.")
        sequence_duration = round(sequence_duration)
        duration = _resolve_duration(
            sequence_duration,
            audio_duration,
            fps,
        )
        parsed_sections = _apply_render_groups(
            _parse_schedule(
                timeline,
                default_fade_in,
                default_fade_out,
                time_unit,
            ),
            render_groups,
        )
        sections = _resolve_schedule(
            parsed_sections,
            beat_times,
            audio_duration,
            curve,
            time_unit,
            fps,
            duration,
        )
        total_frames = round(duration * fps)
        frame_sections = _frame_sections(sections, fps, total_frames)
        for section, frame_section in zip(sections, frame_sections):
            section.update(frame_section)
        schedule = {
            "type": "fl_prompt_schedule",
            "version": 2,
            "duration": duration,
            "audio_duration": audio_duration,
            "source_unit": time_unit,
            "fps": fps,
            "sections": sections,
        }
        ui_payload = {
            "bpm": beat_data["bpm"],
            "grid_bpm": beat_data["grid_bpm"],
            "base_grid_interval_seconds": beat_data["base_grid_interval_seconds"],
            "grid_interval_seconds": beat_data["grid_interval_seconds"],
            "beat_grid_density": beat_data["beat_grid_density"],
            "beat_times": beat_times,
            "base_beat_times": beat_data["base_beat_times"],
            "downbeat_times": beat_data["downbeat_times"],
            "base_downbeat_times": beat_data["base_downbeat_times"],
            "detected_beat_times": beat_data["detected_beat_times"],
            "base_detected_beat_times": beat_data["base_detected_beat_times"],
            "detected_downbeat_times": beat_data["detected_downbeat_times"],
            "base_detected_downbeat_times": beat_data["base_detected_downbeat_times"],
            "detected_beat_confidences": beat_data["detected_beat_confidences"],
            "base_detected_beat_confidences": beat_data["base_detected_beat_confidences"],
            "detected_downbeat_confidences": beat_data["detected_downbeat_confidences"],
            "base_detected_downbeat_confidences": beat_data["base_detected_downbeat_confidences"],
            "beat_offset_ms": int(round(beat_offset_ms)),
            "onset_times": (
                internal_analysis["onset_times"]
                if internal_analysis is not None
                else beat_data["onset_times"]
            ),
            "drum_times": (
                internal_analysis["drum_times"]
                if internal_analysis is not None
                else beat_data["drum_times"]
            ),
            "audio_duration": audio_duration,
            "source_duration": (
                internal_analysis["source_duration"]
                if internal_analysis is not None
                else audio_duration
            ),
            "source_start": (
                internal_analysis["source_start"]
                if internal_analysis is not None
                else 0.0
            ),
            "time_unit": time_unit,
            "source_unit": time_unit,
            "fps": fps,
            "total_frames": total_frames,
            "sections": sections,
            "frame_sections": frame_sections,
            "bpm_source": beat_data["bpm_source"],
            "analysis_source": (
                internal_analysis.get("analysis_source", analysis_source)
                if internal_analysis is not None
                else beat_data["analysis_source"]
            ),
            "beat_analysis_source": beat_data["beat_analysis_source"],
            "detector_version": beat_data["detector_version"],
            "detector": beat_data["detector"],
            "analysis_cache_hit": (
                bool(internal_analysis.get("analysis_cache_hit", False))
                if internal_analysis is not None
                else beat_data["analysis_cache_hit"]
            ),
        }
        if internal_analysis is not None:
            ui_payload["audio_file"] = internal_analysis.get("audio_file", audio_file)
            ui_payload["cache_key"] = internal_analysis.get("cache_key", "")
            source_analysis = internal_analysis.get("source_analysis")
            if isinstance(source_analysis, dict) and "beat_times" in source_analysis:
                ui_payload["source_analysis"] = source_analysis
        waveform = (
            internal_analysis["waveform_preview"]
            if internal_analysis is not None
            else beat_data["waveform_preview"]
        )
        if waveform is not None:
            ui_payload["waveform_preview"] = waveform
        return io.NodeOutput(
            schedule,
            total_frames,
            cropped_audio,
            float(beat_data["bpm"]),
            ui={"fl_prompt_sequencer": [ui_payload]},
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        audio_file="",
        analysis_cache_key="",
        beat_positions=None,
        **kwargs,
    ):
        if not audio_file and analysis_cache_key:
            audio_file = cached_analysis_audio_file(analysis_cache_key)
        if not audio_file:
            return None
        analysis_version = (
            f"audio-timeline-{ANALYSIS_VERSION}"
            if beat_positions
            else f"{DETECTOR_VERSION}:timeline-{ANALYSIS_VERSION}"
        )
        return f"{analysis_version}:{audio_file_hash(resolve_audio_path(audio_file))}"
