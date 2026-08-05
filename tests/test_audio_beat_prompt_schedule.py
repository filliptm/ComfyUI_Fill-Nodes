import importlib.util
import json
import pathlib
import unittest


MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "nodes"
    / "audio"
    / "FL_Audio_Beat_Prompt_Schedule.py"
)
SPEC = importlib.util.spec_from_file_location("fl_audio_beat_prompt_schedule", MODULE_PATH)
schedule = importlib.util.module_from_spec(SPEC)
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

    def test_node_returns_schedule_and_readable_preview(self):
        output = schedule.FL_Audio_Beat_Prompt_Schedule.execute(
            beat_positions=beat_json(),
            timeline="[0 - 48 | fade_in=6]\nSubject turns.",
            default_fade_in=0.0,
            default_fade_out=0.0,
            curve="linear",
        ).result

        self.assertEqual(output[0]["type"], "fl_prompt_schedule")
        self.assertEqual(output[0]["version"], 1)
        self.assertEqual(output[0]["source_unit"], "frames")
        self.assertIn("frames 0 - 48", output[1])
        self.assertIn("00:00.000 - 00:02.000", output[1])

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
        self.assertEqual(output.result[2], 2.0)
        self.assertEqual(output.result[3], 48)

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

    def test_frame_length_rejects_sections_past_its_end(self):
        with self.assertRaisesRegex(ValueError, "beyond the sequence duration"):
            schedule.FL_Audio_Beat_Prompt_Schedule.execute(
                beat_positions=beat_json(),
                timeline="[0 - 2.1]\nCamera move.",
                default_fade_in=0.0,
                default_fade_out=0.0,
                curve="linear",
                time_unit="seconds",
                fps=24.0,
                sequence_duration=48,
            )

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

        self.assertEqual(output.result[2], 2.0)
        self.assertEqual(output.result[3], 48)

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


if __name__ == "__main__":
    unittest.main()
