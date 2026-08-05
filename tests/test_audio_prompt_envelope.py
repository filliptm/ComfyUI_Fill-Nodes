import importlib.util
import json
import pathlib
import sys
import types
import unittest


AUDIO_PATH = pathlib.Path(__file__).parents[1] / "nodes" / "audio"
PACKAGE_NAME = "fl_audio_prompt_envelope_tests"
PACKAGE = types.ModuleType(PACKAGE_NAME)
PACKAGE.__path__ = [str(AUDIO_PATH)]
sys.modules.setdefault(PACKAGE_NAME, PACKAGE)
SPEC = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.FL_Audio_Prompt_Envelope",
    AUDIO_PATH / "FL_Audio_Prompt_Envelope.py",
)
prompt_envelope = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prompt_envelope
SPEC.loader.exec_module(prompt_envelope)
REACTIVE_SPEC = importlib.util.spec_from_file_location(
    "fl_audio_reactive_envelope_tests",
    AUDIO_PATH / "FL_Audio_Reactive_Envelope.py",
)
reactive_envelope = importlib.util.module_from_spec(REACTIVE_SPEC)
REACTIVE_SPEC.loader.exec_module(reactive_envelope)


def beat_json():
    return json.dumps({
        "bpm": 120.0,
        "beat_times": [0.0, 0.5, 1.0, 1.5],
        "audio_duration": 2.0,
    })


class BeatPromptEnvelopeTests(unittest.TestCase):
    def test_exact_beats_generate_full_duration_envelope(self):
        output = prompt_envelope.FL_Audio_Beat_Prompt_Envelope.execute(
            beat_positions=beat_json(),
            reactive_prompt="Pulse outward.",
            beat_stride=2,
            beat_phase=0,
            attack_beats=0.0,
            hold_beats=0.25,
            release_beats=0.5,
            floor_strength=0.0,
            peak_strength=3.0,
            curve="cosine",
            fps=24.0,
        ).result

        envelope = output[0]
        envelope_json = json.loads(output[1])
        self.assertEqual(envelope["type"], "fl_prompt_envelope")
        self.assertEqual(envelope["version"], 1)
        self.assertEqual(len(envelope["weights"]), 48)
        self.assertEqual(envelope_json["fps"], 24.0)
        self.assertEqual(envelope_json["duration"], 2.0)
        self.assertIn("2 selected hits", output[2])
        self.assertGreater(max(envelope["weights"]), 2.9)

    def test_overlapping_pulses_do_not_add_strength(self):
        values, _ = prompt_envelope._beat_envelope(
            beat_times=[0.0, 0.25, 0.5],
            duration=0.75,
            fps=24.0,
            beat_stride=1,
            beat_phase=0,
            attack_beats=0.0,
            hold_beats=0.5,
            release_beats=1.0,
            floor_strength=0.0,
            peak_strength=3.0,
            curve="linear",
        )

        self.assertLessEqual(max(values), 3.0)

    def test_phase_must_be_inside_stride(self):
        with self.assertRaisesRegex(ValueError, "smaller than beat stride"):
            prompt_envelope.FL_Audio_Beat_Prompt_Envelope.execute(
                beat_positions=beat_json(),
                reactive_prompt="Pulse.",
                beat_stride=2,
                beat_phase=2,
                attack_beats=0.0,
                hold_beats=0.25,
                release_beats=0.5,
                floor_strength=0.0,
                peak_strength=3.0,
                curve="cosine",
                fps=24.0,
            )


class EnvelopePromptTests(unittest.TestCase):
    def test_maps_existing_envelope_to_prompt_strength(self):
        source = json.dumps({
            "envelope": [0.0, 0.25, 0.5, 1.0],
            "fps": 20,
            "duration": 0.2,
        })
        output = prompt_envelope.FL_Audio_Envelope_Prompt.execute(
            envelope_json=source,
            reactive_prompt="Flash.",
            source_fps=24.0,
            threshold=0.25,
            response_gamma=1.0,
            floor_strength=0.0,
            peak_strength=3.0,
            invert=False,
        ).result

        envelope = output[0]
        self.assertEqual(envelope["fps"], 20.0)
        self.assertEqual(envelope["weights"][0], 0.0)
        self.assertEqual(envelope["weights"][1], 0.0)
        self.assertAlmostEqual(envelope["weights"][2], 1.0)
        self.assertEqual(envelope["weights"][3], 3.0)

    def test_legacy_envelope_uses_fallback_fps(self):
        values, fps, duration = prompt_envelope._load_envelope(
            json.dumps({"envelope": [0.0, 1.0, 0.0]}),
            30.0,
        )

        self.assertEqual(values, [0.0, 1.0, 0.0])
        self.assertEqual(fps, 30.0)
        self.assertEqual(duration, 0.1)

    def test_schema_exposes_native_tooltips(self):
        for node in (
            prompt_envelope.FL_Audio_Beat_Prompt_Envelope,
            prompt_envelope.FL_Audio_Envelope_Prompt,
        ):
            schema = node.define_schema()
            self.assertTrue(schema.description)
            self.assertTrue(all(input.tooltip for input in schema.inputs))
            self.assertTrue(all(output.tooltip for output in schema.outputs))


class ExistingReactiveEnvelopeTests(unittest.TestCase):
    def test_attack_peaks_on_hit_and_output_includes_timing_metadata(self):
        node = reactive_envelope.FL_Audio_Reactive_Envelope()
        values = node._generate_envelope(
            hit_times=[0.0],
            total_frames=4,
            fps=24,
            attack_frames=1,
            decay_frames=1,
            sustain_level=0.0,
            release_frames=1,
        )
        self.assertEqual(values[0], 1.0)

        drum_json = json.dumps({
            "kick_times": [0.0],
            "snare_times": [],
            "hihat_times": [],
            "duration": 0.11,
        })
        kick, _, _ = node.generate_envelopes(
            drum_times_json=drum_json,
            fps=24,
            kick_attack_frames=1,
            kick_decay_frames=1,
            kick_sustain_level=0.0,
            kick_release_frames=1,
        )
        data = json.loads(kick)
        self.assertEqual(data["fps"], 24)
        self.assertEqual(data["duration"], 0.11)
        self.assertEqual(data["total_frames"], 3)


if __name__ == "__main__":
    unittest.main()
