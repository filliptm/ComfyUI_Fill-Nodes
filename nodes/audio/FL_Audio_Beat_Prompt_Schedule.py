import json
import math
import re

from comfy_api.latest import io

from .audio_files import (
    audio_file_hash,
    available_audio_files,
    resolve_audio_path,
)
from .audio_timeline import analyze_audio_file


FLPromptSchedule = io.Custom("FL_PROMPT_SCHEDULE")
_HEADER = re.compile(
    r"^\s*\[\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)"
    r"(?:\s*\|\s*(.*?))?\s*\]\s*$"
)
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


def _load_beat_data(beat_positions):
    try:
        data = json.loads(beat_positions)
    except json.JSONDecodeError as error:
        raise ValueError(f"Beat positions is not valid JSON: {error.msg}.") from error
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
    return {
        "bpm": bpm,
        "beat_times": beat_times,
        "audio_duration": duration,
        "detected_beat_times": data.get("detected_beat_times", []),
        "onset_times": data.get("onset_times", []),
        "drum_times": data.get("drum_times", {}),
        "waveform_preview": (
            data.get("waveform_preview")
            if isinstance(data.get("waveform_preview"), dict)
            else None
        ),
    }


def _parse_options(options, line, default_fade_in, default_fade_out):
    values = {"fade_in": default_fade_in, "fade_out": default_fade_out}
    if not options:
        return values

    for option in options.split("|"):
        if "=" not in option:
            raise ValueError(
                f"Beat prompt schedule line {line}: options must use fade_in=value or fade_out=value."
            )
        name, value = (part.strip() for part in option.split("=", 1))
        if name not in values:
            raise ValueError(
                f"Beat prompt schedule line {line}: unknown option '{name}'. "
                "Use fade_in or fade_out."
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

        if line.lstrip().startswith("["):
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
        previous = section
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
        if end > limit + _EPS:
            raise ValueError(
                f"Beat prompt schedule line {section['line']}: section ends at {end:g}s, "
                f"beyond the sequence duration {limit:g}s."
            )
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
        resolved.append({
            **section,
            "start": start,
            "end": end,
            "fade_in_end": fade_in_end,
            "fade_out_start": fade_out_start,
            "curve": curve,
        })
    return resolved


def _frame_sections(sections, fps, total_frames):
    frame_sections = []
    for section in sections:
        start_frame = min(total_frames - 1, max(0, round(section["start"] * fps)))
        end_frame = min(total_frames, max(start_frame + 1, round(section["end"] * fps)))
        fade_in_end = min(end_frame, max(start_frame, round(section["fade_in_end"] * fps)))
        fade_out_start = min(end_frame, max(start_frame, round(section["fade_out_start"] * fps)))
        frame_sections.append({
            "line": section["line"],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "fade_in_frames": fade_in_end - start_frame,
            "fade_out_frames": end_frame - fade_out_start,
            "prompt": section["prompt"],
            "curve": section["curve"],
        })
    return frame_sections


def _format_time(seconds):
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:06.3f}"


def _preview(sections, time_unit="beats", fps=24.0):
    lines = []
    for section in sections:
        start_frame = round(section["start"] * fps)
        end_frame = round(section["end"] * fps)
        lines.append(
            f"[{time_unit} {section['start_position']:g} - {section['end_position']:g} | "
            f"{_format_time(section['start'])} - {_format_time(section['end'])} | "
            f"frames {start_frame} - {end_frame} @ {fps:g} fps | "
            f"fade_in={section['fade_in']:g} | fade_out={section['fade_out']:g}]\n"
            f"{section['prompt']}"
        )
    return "\n\n".join(lines)


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
                io.Combo.Input(
                    "bpm_method",
                    display_name="BPM method",
                    options=["beat_intervals", "onset_strength"],
                    default="beat_intervals",
                    tooltip="Choose median detected beat intervals or Librosa onset-strength tempo.",
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
                    tooltip="Shift detected beats earlier or later without moving the audio.",
                ),
                io.Combo.Input(
                    "analysis_source",
                    display_name="analysis source",
                    options=["mix", "drums", "vocals", "bass", "other"],
                    default="mix",
                    tooltip=(
                        "Analyze the full mix or an explicitly separated stem. Stem choices become "
                        "available after separation finishes."
                    ),
                ),
            ],
            outputs=[
                FLPromptSchedule.Output(
                    display_name="prompt_schedule",
                    tooltip="Resolved second-based prompt schedule for compatible FL diffusion nodes.",
                ),
                io.String.Output(
                    display_name="preview",
                    tooltip="Readable preview showing each source range and its resolved time and frame range.",
                ),
                io.Float.Output(
                    display_name="duration_seconds",
                    tooltip="Effective schedule duration in seconds.",
                ),
                io.Int.Output(
                    display_name="total_frames",
                    tooltip="Effective schedule duration converted to frames at the selected FPS.",
                ),
                io.Audio.Output(
                    display_name="audio",
                    tooltip="The selected, frame-aligned audio crop for downstream FL audio nodes.",
                ),
                io.String.Output(
                    display_name="beat_positions",
                    tooltip="Effective beat analysis JSON for FL audio-reactive nodes.",
                ),
                io.String.Output(
                    display_name="drum_times",
                    tooltip="Detected kick, snare, and hi-hat timestamps for FL audio-reactive nodes.",
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
        bpm_method="beat_intervals",
        half_time=False,
        beat_offset_ms=0,
        analysis_source="mix",
    ):
        internal_analysis = None
        cropped_audio = None
        if audio_file:
            internal_analysis, cropped_audio = analyze_audio_file(
                audio_file,
                fps,
                trim_start_frame,
                sequence_duration,
                bpm_method,
                half_time,
                beat_offset_ms,
                analysis_source,
            )
        if beat_positions:
            beat_data = _load_beat_data(beat_positions)
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
        sections = _resolve_schedule(
            _parse_schedule(
                timeline,
                default_fade_in,
                default_fade_out,
                time_unit,
            ),
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
            "version": 1,
            "duration": duration,
            "audio_duration": audio_duration,
            "source_unit": time_unit,
            "fps": fps,
            "sections": sections,
        }
        ui_payload = {
            "bpm": beat_data["bpm"],
            "beat_times": beat_times,
            "detected_beat_times": (
                internal_analysis["detected_beat_times"]
                if internal_analysis is not None
                else beat_data["detected_beat_times"]
            ),
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
        }
        waveform = (
            internal_analysis["waveform_preview"]
            if internal_analysis is not None
            else beat_data["waveform_preview"]
        )
        if waveform is not None:
            ui_payload["waveform_preview"] = waveform
        drum_times = ui_payload["drum_times"] or {
            "kick_times": [],
            "snare_times": [],
            "hihat_times": [],
            "duration": duration,
        }
        return io.NodeOutput(
            schedule,
            _preview(sections, time_unit, fps),
            duration,
            total_frames,
            cropped_audio,
            beat_positions,
            json.dumps(drum_times, separators=(",", ":")),
            ui={"fl_prompt_sequencer": [ui_payload]},
        )

    @classmethod
    def fingerprint_inputs(cls, audio_file="", **kwargs):
        if not audio_file:
            return None
        return audio_file_hash(resolve_audio_path(audio_file))
