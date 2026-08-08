import importlib.util
import json
import pathlib
import sys
import torch
import types
import unittest
from unittest import mock


AUDIO_NODE_PATH = pathlib.Path(__file__).parents[1] / "nodes" / "audio"
PACKAGE_NAME = "fl_fill_nodes_audio_tests"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(AUDIO_NODE_PATH)]
sys.modules[PACKAGE_NAME] = package
MODULE_PATH = AUDIO_NODE_PATH / "FL_Audio_Beat_Prompt_Schedule.py"
SPEC = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.FL_Audio_Beat_Prompt_Schedule",
    MODULE_PATH,
)
schedule = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = schedule
SPEC.loader.exec_module(schedule)


def beat_json():
    return json.dumps({
        "bpm": 120.0,
        "beat_times": [0.1, 0.6, 1.2, 1.9],
        "beat_frames": [4, 20, 40, 60],
        "num_beats": 4,
        "sample_rate": 48000,
        "audio_duration": 2.5,
    })


class BeatPromptScheduleTests(unittest.TestCase):
    def test_fractional_beats_use_exact_detected_intervals(self):
        beats, duration = schedule._load_beats(beat_json())
        sections = schedule._resolve_schedule(
            schedule._parse_schedule(
                "[0.5 - 2.5 | fade_in=0.5 | fade_out=0.5]\nCamera move.",
                0.0,
                0.0,
            ),
            beats,
            duration,
            "cosine",
        )

        self.assertAlmostEqual(sections[0]["start"], 0.35)
        self.assertAlmostEqual(sections[0]["end"], 1.55)
        self.assertAlmostEqual(sections[0]["fade_in_end"], 0.6)
        self.assertAlmostEqual(sections[0]["fade_out_start"], 1.2)

    def test_final_end_position_maps_to_audio_duration(self):
        beats, duration = schedule._load_beats(beat_json())
        self.assertEqual(schedule._beat_to_seconds(4.0, beats, duration, 1), 2.5)

    def test_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            schedule._parse_schedule(
                "[0 - 4]\nFirst.\n[3 - 4]\nSecond.",
                0.0,
                0.0,
            )

    def test_node_returns_schedule_frames_audio_and_bpm(self):
        output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
            beat_positions=beat_json(),
            timeline="[0 - 48 | fade_in=6]\nSubject turns.",
            default_fade_in=0.0,
            default_fade_out=0.0,
            curve="linear",
        ).result

        self.assertEqual(output[0]["type"], "fl_prompt_schedule")
        self.assertEqual(output[0]["version"], 2)
        self.assertEqual(output[0]["source_unit"], "frames")
        self.assertEqual(output[1], 60)
        self.assertIsNone(output[2])
        self.assertEqual(output[3], 120.0)

    def test_render_groups_are_attached_without_changing_timeline_syntax(self):
        node_output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
            beat_positions=beat_json(),
            timeline=(
                "[0 - 20]\nFirst.\n\n"
                "[20 - 40]\nSecond.\n\n"
                "[40 - 60]\nThird."
            ),
            default_fade_in=0.0,
            default_fade_out=0.0,
            curve="linear",
            time_unit="frames",
            fps=24.0,
            sequence_duration=60,
            render_groups=json.dumps({
                "version": 1,
                "section_groups": [1, 1, None],
            }),
        )
        output = node_output.result[0]

        self.assertEqual(output["sections"][0]["render_group"], 1)
        self.assertEqual(output["sections"][1]["render_group"], 1)
        self.assertNotIn("render_group", output["sections"][2])
        self.assertEqual(
            node_output.ui["fl_prompt_sequencer"][0]["frame_sections"][1]["render_group"],
            1,
        )

    def test_render_groups_require_matching_consecutive_touching_sections(self):
        sections = schedule._parse_schedule(
            "[0 - 10]\nFirst.\n[10 - 20]\nSecond.\n[20 - 30]\nThird.",
            0.0,
            0.0,
            "frames",
        )

        with self.assertRaisesRegex(ValueError, "one section_groups entry"):
            schedule._apply_render_groups(
                sections,
                {"version": 1, "section_groups": [1, 1]},
            )
        with self.assertRaisesRegex(ValueError, "consecutive"):
            schedule._apply_render_groups(
                sections,
                {"version": 1, "section_groups": [1, None, 1]},
            )

        sections = schedule._parse_schedule(
            "[0 - 10]\nFirst.\n[12 - 20]\nSecond.",
            0.0,
            0.0,
            "frames",
        )
        with self.assertRaisesRegex(ValueError, "touching"):
            schedule._apply_render_groups(
                sections,
                {"version": 1, "section_groups": [1, 1]},
            )

    def test_seconds_mode_uses_direct_positions(self):
        output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
            beat_positions=beat_json(),
            timeline="[0 - 1.5 | fade_in=0.25 | fade_out=0.5]\nCamera move.",
            default_fade_in=0.0,
            default_fade_out=0.0,
            curve="cosine",
            time_unit="seconds",
            fps=24.0,
            sequence_duration=0,
        )
        section = output.result[0]["sections"][0]

        self.assertEqual(section["start"], 0.0)
        self.assertEqual(section["end"], 1.5)
        self.assertEqual(section["fade_in_end"], 0.25)
        self.assertEqual(section["fade_out_start"], 1.0)
        self.assertEqual(output.result[0]["source_unit"], "seconds")

    def test_frames_mode_resolves_at_selected_fps(self):
        output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
            beat_positions=beat_json(),
            timeline="[0 - 48 | fade_in=6 | fade_out=6]\nCamera move.",
            default_fade_in=0.0,
            default_fade_out=0.0,
            curve="linear",
            time_unit="frames",
            fps=24.0,
            sequence_duration=48,
        )
        section = output.result[0]["sections"][0]

        self.assertEqual(section["start"], 0.0)
        self.assertEqual(section["end"], 2.0)
        self.assertEqual(section["fade_in_end"], 0.25)
        self.assertEqual(section["fade_out_start"], 1.75)
        self.assertEqual(output.result[1], 48)

    def test_frames_mode_rejects_fractional_frames(self):
        with self.assertRaisesRegex(ValueError, "whole frame"):
            schedule.FL_Audio_Beat_Prompt_Schedule.execute(
                beat_positions=beat_json(),
                timeline="[0 - 47.5]\nCamera move.",
                default_fade_in=0.0,
                default_fade_out=0.0,
                curve="linear",
                time_unit="frames",
                fps=24.0,
                sequence_duration=0,
            )

    def test_bracketed_prompt_lines_are_not_treated_as_headers(self):
        sections = schedule._parse_schedule(
            (
                "[0 - 24]\n"
                "[Subject and action]\n"
                "The rider looks over her shoulder.\n"
                "[Camera]\n"
                "Low tracking shot."
            ),
            0.0,
            0.0,
            "frames",
        )

        self.assertEqual(
            sections[0]["prompt"],
            (
                "[Subject and action]\n"
                "The rider looks over her shoulder.\n"
                "[Camera]\n"
                "Low tracking shot."
            ),
        )

    def test_malformed_numeric_header_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected a header"):
            schedule._parse_schedule(
                "[0 - 24]\nFirst prompt.\n[24 - later]\nSecond prompt.",
                0.0,
                0.0,
                "frames",
            )

    def test_crossfade_resolves_around_touching_frame_boundary(self):
        output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
            beat_positions=beat_json(),
            timeline=(
                "[0 - 24 | fade_out=6]\nFirst.\n"
                "[24 - 48 | fade_in=6 | crossfade=12]\nSecond."
            ),
            default_fade_in=0.0,
            default_fade_out=0.0,
            curve="cosine",
            time_unit="frames",
            fps=24.0,
            sequence_duration=48,
        )

        first, second = output.result[0]["sections"]
        self.assertEqual(first["fade_out_frames"], 0)
        self.assertEqual(second["fade_in_frames"], 0)
        self.assertEqual(second["crossfade_start_frame"], 18)
        self.assertEqual(second["crossfade_end_frame"], 30)
        self.assertEqual(second["crossfade_frames"], 12)
        self.assertEqual(second["crossfade_start"], 0.75)
        self.assertEqual(second["crossfade_end"], 1.25)

    def test_crossfade_requires_touching_previous_section(self):
        with self.assertRaisesRegex(ValueError, "touching previous"):
            schedule._parse_schedule(
                "[0 - 24]\nFirst.\n[30 - 48 | crossfade=8]\nSecond.",
                0.0,
                0.0,
                "frames",
            )

    def test_crossfade_cannot_exceed_shorter_adjacent_section(self):
        with self.assertRaisesRegex(ValueError, "shorter adjacent"):
            schedule._parse_schedule(
                "[0 - 8]\nFirst.\n[8 - 48 | crossfade=12]\nSecond.",
                0.0,
                0.0,
                "frames",
            )

    def test_crossfade_must_be_a_whole_frame(self):
        with self.assertRaisesRegex(ValueError, "whole frame"):
            schedule._parse_schedule(
                "[0 - 24]\nFirst.\n[24 - 48 | crossfade=5.5]\nSecond.",
                0.0,
                0.0,
                "frames",
            )

    def test_frame_length_crops_sections_at_its_end(self):
        output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
            beat_positions=beat_json(),
            timeline="[0 - 2.1]\nCamera move.",
            default_fade_in=0.0,
            default_fade_out=0.0,
            curve="linear",
            time_unit="seconds",
            fps=24.0,
            sequence_duration=48,
        )

        section = output.result[0]["sections"][0]
        self.assertEqual(section["start"], 0.0)
        self.assertEqual(section["end"], 2.0)
        self.assertEqual(section["end_frame"], 48)

    def test_frame_length_discards_sections_after_its_end(self):
        output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
            beat_positions=beat_json(),
            timeline="[49 - 59]\nCamera move.",
            default_fade_in=0.0,
            default_fade_out=0.0,
            curve="linear",
            time_unit="frames",
            fps=24.0,
            sequence_duration=48,
        )

        self.assertEqual(output.result[0]["sections"], [])

    def test_frame_length_preserves_visible_part_of_fade_out(self):
        output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
            beat_positions=beat_json(),
            timeline="[40 - 52 | fade_out=8]\nCamera move.",
            default_fade_in=0.0,
            default_fade_out=0.0,
            curve="linear",
            time_unit="frames",
            fps=24.0,
            sequence_duration=48,
        )

        section = output.result[0]["sections"][0]
        self.assertEqual(section["start_frame"], 40)
        self.assertEqual(section["end_frame"], 48)
        self.assertEqual(section["fade_out_frames"], 4)

    def test_length_is_always_a_frame_count(self):
        output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
            beat_positions=beat_json(),
            timeline="[0 - 1.5]\nCamera move.",
            default_fade_in=0.0,
            default_fade_out=0.0,
            curve="linear",
            time_unit="seconds",
            fps=24.0,
            sequence_duration=48,
        )

        self.assertEqual(output.result[1], 48)

    def test_ui_payload_contains_exact_beat_map(self):
        output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
            beat_positions=beat_json(),
            timeline="[0 - 48 | fade_in=6 | fade_out=6]\nCamera move.",
            default_fade_in=0.0,
            default_fade_out=0.0,
            curve="linear",
        )
        payload = output.ui["fl_prompt_sequencer"][0]

        self.assertEqual(payload["beat_times"], [0.1, 0.6, 1.2, 1.9])
        self.assertEqual(payload["audio_duration"], 2.5)
        self.assertEqual(payload["bpm"], 120.0)
        self.assertEqual(payload["source_unit"], "frames")
        self.assertEqual(payload["total_frames"], 60)
        self.assertEqual(payload["frame_sections"], [{
            "line": 1,
            "start_frame": 0,
            "end_frame": 48,
            "fade_in_frames": 6,
            "fade_out_frames": 6,
            "crossfade_start_frame": 0,
            "crossfade_end_frame": 0,
            "crossfade_frames": 0,
            "prompt": "Camera move.",
            "curve": "linear",
        }])

    def test_ui_payload_forwards_waveform_without_changing_schedule(self):
        beat_data = json.loads(beat_json())
        waveform_preview = {
            "version": 1,
            "duration": 2.5,
            "scale": 32767,
            "peaks": [-100, 200, -300, 400],
        }
        beat_data["waveform_preview"] = waveform_preview
        output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
            beat_positions=json.dumps(beat_data),
            timeline="[0 - 48]\nCamera move.",
            default_fade_in=0.0,
            default_fade_out=0.0,
            curve="linear",
        )

        self.assertEqual(
            output.ui["fl_prompt_sequencer"][0]["waveform_preview"],
            waveform_preview,
        )
        self.assertNotIn("waveform_preview", output.result[0])

    def test_legacy_beats_emit_frame_sections_for_editor_migration(self):
        output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
            beat_positions=beat_json(),
            timeline="[0 - 2 | fade_in=0.5 | fade_out=0.5]\nCamera move.",
            default_fade_in=0.0,
            default_fade_out=0.0,
            curve="cosine",
            time_unit="beats",
            fps=24.0,
        )
        section = output.ui["fl_prompt_sequencer"][0]["frame_sections"][0]

        self.assertEqual(section["start_frame"], 2)
        self.assertEqual(section["end_frame"], 29)
        self.assertEqual(section["fade_in_frames"], 6)
        self.assertEqual(section["fade_out_frames"], 7)

    def test_shared_resolved_boundary_maps_to_one_frame(self):
        output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
            beat_positions=beat_json(),
            timeline="[0 - 1.01]\nFirst.\n[1.01 - 2]\nSecond.",
            default_fade_in=0.0,
            default_fade_out=0.0,
            curve="linear",
            time_unit="seconds",
            fps=24.0,
        )
        frame_sections = output.ui["fl_prompt_sequencer"][0]["frame_sections"]

        self.assertEqual(frame_sections[0]["end_frame"], frame_sections[1]["start_frame"])

    def test_schema_exposes_native_tooltips(self):
        schema = schedule.FL_Audio_Beat_Prompt_Schedule.define_schema()
        inputs = {input.id: input for input in schema.inputs}
        self.assertTrue(schema.description)
        self.assertTrue(all(input.tooltip for input in schema.inputs))
        self.assertTrue(all(output.tooltip for output in schema.outputs))
        self.assertEqual(inputs["sequence_duration"].display_name, "length (frames)")
        self.assertEqual(inputs["default_fade_in"].display_name, "default fade-in (frames)")
        self.assertEqual(inputs["default_fade_out"].display_name, "default fade-out (frames)")

    def test_schema_exposes_compact_outputs(self):
        schema = schedule.FL_Audio_Beat_Prompt_Schedule.define_schema()

        self.assertEqual(
            [input.id for input in schema.inputs[:8]],
            [
                "beat_positions",
                "timeline",
                "default_fade_in",
                "default_fade_out",
                "curve",
                "time_unit",
                "fps",
                "sequence_duration",
            ],
        )
        self.assertEqual(
            [input.id for input in schema.inputs[8:]],
            [
                "audio_file",
                "trim_start_frame",
                "half_time",
                "beat_offset_ms",
                "analysis_source",
                "beat_grid_density",
                "render_groups",
                "analysis_cache_key",
            ],
        )
        self.assertEqual(
            [output.display_name for output in schema.outputs],
            [
                "prompt_schedule",
                "total_frames",
                "audio",
                "BPM",
            ],
        )

    def test_uploaded_audio_drives_schedule_audio_and_bpm_outputs(self):
        audio = {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000}
        analysis = {
            "bpm": 120.0,
            "beat_times": [0.0, 0.5],
            "downbeat_times": [0.0],
            "detected_beat_times": [0.05, 0.52],
            "detected_downbeat_times": [0.05],
            "detected_beat_confidences": [0.9, 0.8],
            "detected_downbeat_confidences": [0.95],
            "detector": {"name": "beat_this"},
            "analysis_source": "drums",
            "beat_analysis_source": "mix",
            "onset_times": [0.05, 0.25, 0.52],
            "audio_duration": 1.0,
            "source_duration": 2.0,
            "source_start": 0.5,
            "waveform_preview": {
                "version": 1,
                "duration": 1.0,
                "scale": 32767,
                "peaks": [-100, 100],
            },
            "drum_times": {
                "kick_times": [0.05],
                "snare_times": [0.52],
                "hihat_times": [],
                "duration": 1.0,
            },
        }
        with mock.patch.object(schedule, "analyze_audio_file", return_value=(analysis, audio)):
            output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
                beat_positions=None,
                timeline="[0 - 24]\nCamera pulse.",
                default_fade_in=0.0,
                default_fade_out=0.0,
                curve="linear",
                fps=24.0,
                sequence_duration=24,
                audio_file="song.wav",
                trim_start_frame=12,
            )

        self.assertIs(output.result[2], audio)
        self.assertEqual(output.result[3], 120.0)
        payload = output.ui["fl_prompt_sequencer"][0]
        self.assertEqual(payload["detected_beat_times"], [0.05, 0.52])
        self.assertEqual(payload["downbeat_times"], [0.0])
        self.assertEqual(payload["detected_downbeat_times"], [0.05])
        self.assertEqual(payload["detected_beat_confidences"], [0.9, 0.8])
        self.assertEqual(payload["detector"], {"name": "beat_this"})
        self.assertEqual(payload["analysis_source"], "drums")
        self.assertEqual(payload["beat_analysis_source"], "mix")
        self.assertEqual(payload["drum_times"]["kick_times"], [0.05])
        self.assertEqual(payload["source_start"], 0.5)

    def test_cached_analysis_restores_a_missing_audio_widget_value(self):
        audio = {"waveform": torch.zeros(1, 1, 24000), "sample_rate": 24000}
        analysis = {
            "bpm": 120.0,
            "beat_times": [0.0, 0.5],
            "detected_beat_times": [0.0, 0.5],
            "onset_times": [],
            "audio_duration": 1.0,
            "source_duration": 1.0,
            "source_start": 0.0,
            "waveform_preview": None,
            "drum_times": {},
            "audio_file": "song.wav",
            "cache_key": "a" * 64,
        }
        with (
            mock.patch.object(
                schedule,
                "cached_analysis_audio_file",
                return_value="song.wav",
            ) as restore,
            mock.patch.object(
                schedule,
                "analyze_audio_file",
                return_value=(analysis, audio),
            ) as analyze,
        ):
            output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
                beat_positions=None,
                timeline="[0 - 24]\nCamera pulse.",
                default_fade_in=0.0,
                default_fade_out=0.0,
                curve="linear",
                fps=24.0,
                sequence_duration=24,
                audio_file="",
                analysis_cache_key="a" * 64,
            )

        restore.assert_called_once_with("a" * 64)
        self.assertEqual(analyze.call_args.args[0], "song.wav")
        self.assertIs(output.result[2], audio)
        payload = output.ui["fl_prompt_sequencer"][0]
        self.assertEqual(payload["audio_file"], "song.wav")
        self.assertEqual(payload["cache_key"], "a" * 64)

    def test_external_beats_must_match_uploaded_crop(self):
        analysis = {
            "audio_duration": 1.0,
            "source_duration": 1.0,
            "source_start": 0.0,
            "drum_times": {},
            "waveform_preview": None,
        }
        audio = {"waveform": torch.zeros(1, 1, 48000), "sample_rate": 48000}
        with mock.patch.object(schedule, "analyze_audio_file", return_value=(analysis, audio)):
            with self.assertRaisesRegex(ValueError, "does not match"):
                schedule.FL_Audio_Beat_Prompt_Schedule.execute(
                    beat_positions=beat_json(),
                    timeline="[0 - 24]\nCamera pulse.",
                    default_fade_in=0.0,
                    default_fade_out=0.0,
                    curve="linear",
                    fps=24.0,
                    sequence_duration=24,
                    audio_file="song.wav",
                )

    def test_external_beats_skip_beat_this_and_keep_auxiliary_audio_analysis(self):
        external = json.loads(beat_json())
        external.update({
            "downbeat_times": [0.1, 1.2],
            "detected_beat_times": [0.1, 0.6, 1.2, 1.9],
            "detected_downbeat_times": [0.1, 1.2],
            "detected_beat_confidences": [0.9, 0.8, 0.85, 0.75],
            "detector": {"name": "external"},
        })
        auxiliary = {
            "audio_duration": 2.5,
            "source_duration": 2.5,
            "source_start": 0.0,
            "onset_times": [0.25],
            "drum_times": {"kick_times": [0.25]},
            "waveform_preview": None,
        }
        audio = {"waveform": torch.zeros(1, 1, 120000), "sample_rate": 48000}
        with mock.patch.object(
            schedule,
            "analyze_audio_file",
            return_value=(auxiliary, audio),
        ) as analyze:
            output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
                beat_positions=json.dumps(external),
                timeline="[0 - 60]\nCamera pulse.",
                default_fade_in=0.0,
                default_fade_out=0.0,
                curve="linear",
                fps=24.0,
                sequence_duration=60,
                audio_file="song.wav",
            )

        self.assertFalse(analyze.call_args.kwargs["detect_beats"])
        payload = output.ui["fl_prompt_sequencer"][0]
        self.assertEqual(payload["downbeat_times"], [0.1, 1.2])
        self.assertEqual(payload["base_detected_beat_confidences"], [0.9, 0.8, 0.85, 0.75])
        self.assertEqual(payload["detector"], {"name": "external"})
        self.assertEqual(payload["onset_times"], [0.25])

    def test_scheduler_offset_applies_to_external_beat_positions(self):
        output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
            beat_positions=beat_json(),
            timeline="[0 - 24]\nCamera pulse.",
            default_fade_in=0.0,
            default_fade_out=0.0,
            curve="linear",
            fps=24.0,
            sequence_duration=24,
            beat_offset_ms=100,
        )

        payload = output.ui["fl_prompt_sequencer"][0]
        self.assertEqual(payload["base_beat_times"], [0.1, 0.6, 1.2, 1.9])
        self.assertEqual(payload["beat_times"], [0.2, 0.7, 1.3, 2.0])
        self.assertEqual(payload["beat_offset_ms"], 100)
        self.assertEqual(payload["grid_interval_seconds"], 0.6)
        self.assertEqual(output.result[3], 120.0)

    def test_scheduler_density_controls_external_grid_and_output(self):
        output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
            beat_positions=beat_json(),
            timeline="[0 - 24]\nCamera pulse.",
            default_fade_in=0.0,
            default_fade_out=0.0,
            curve="linear",
            fps=24.0,
            sequence_duration=24,
            beat_grid_density="every_2_beats",
        )

        payload = output.ui["fl_prompt_sequencer"][0]
        self.assertEqual(payload["beat_times"], [0.1, 1.2])
        self.assertEqual(payload["base_beat_times"], [0.1, 0.6, 1.2, 1.9])
        self.assertEqual(payload["beat_grid_density"], "every_2_beats")
        self.assertEqual(payload["grid_bpm"], 50.0)
        self.assertEqual(output.result[3], 120.0)


if __name__ == "__main__":
    unittest.main()
