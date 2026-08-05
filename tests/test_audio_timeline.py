import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch


AUDIO_NODE_PATH = pathlib.Path(__file__).parents[1] / "nodes" / "audio"
PACKAGE_NAME = "fl_audio_timeline_tests"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(AUDIO_NODE_PATH)]
sys.modules[PACKAGE_NAME] = package
MODULE_PATH = AUDIO_NODE_PATH / "audio_timeline.py"
SPEC = importlib.util.spec_from_file_location(f"{PACKAGE_NAME}.audio_timeline", MODULE_PATH)
timeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = timeline
SPEC.loader.exec_module(timeline)


class AudioTimelineTests(unittest.TestCase):
    def test_beat_offset_shifts_only_beat_markers(self):
        analysis = {
            "audio_duration": 1.0,
            "beat_times": [0.1, 0.9],
            "detected_beat_times": [0.0, 0.95],
            "onset_times": [0.2],
            "drum_times": {"kick_times": [0.2]},
        }

        shifted = timeline.apply_beat_offset(analysis, fps=24.0, beat_offset_ms=200)

        self.assertEqual(shifted["base_beat_times"], [0.1, 0.9])
        self.assertEqual(shifted["beat_times"], [0.30000000000000004, 1.0])
        self.assertEqual(shifted["detected_beat_times"], [0.2, 1.0])
        self.assertEqual(shifted["beat_frames"], [7, 24])
        self.assertEqual(shifted["onset_times"], [0.2])
        self.assertEqual(shifted["drum_times"], {"kick_times": [0.2]})
        self.assertEqual(shifted["beat_offset_ms"], 200)

    def test_beat_offset_clamps_and_deduplicates_crop_boundaries(self):
        analysis = {
            "audio_duration": 1.0,
            "beat_times": [0.0, 0.1, 0.8],
            "detected_beat_times": [],
        }

        shifted = timeline.apply_beat_offset(analysis, fps=24.0, beat_offset_ms=-200)

        self.assertEqual(shifted["beat_times"], [0.0, 0.6000000000000001])
        self.assertEqual(shifted["base_beat_times"], [0.0, 0.1, 0.8])

    def test_crop_uses_video_frames_for_sample_boundaries(self):
        audio = {
            "waveform": torch.arange(0, 96000, dtype=torch.float32).reshape(1, 1, -1),
            "sample_rate": 48000,
        }

        cropped, metadata = timeline.crop_audio(
            audio,
            fps=24.0,
            trim_start_frame=12,
            length_frames=24,
        )

        self.assertEqual(cropped["waveform"].shape, (1, 1, 48000))
        self.assertEqual(cropped["waveform"][0, 0, 0], 24000)
        self.assertEqual(metadata["source_start"], 0.5)
        self.assertEqual(metadata["audio_duration"], 1.0)
        self.assertEqual(metadata["length_frames"], 24)

    def test_crop_rejects_ranges_past_source_end(self):
        audio = {
            "waveform": torch.zeros(1, 1, 48000),
            "sample_rate": 48000,
        }

        with self.assertRaisesRegex(ValueError, "exceeds"):
            timeline.crop_audio(
                audio,
                fps=24.0,
                trim_start_frame=12,
                length_frames=24,
            )

    def test_waveform_preview_keeps_minimum_and_maximum_pairs(self):
        preview = timeline.waveform_preview(
            torch.tensor([-1.0, 0.5, -0.25, 1.0]).numpy(),
            sample_rate=4,
        )

        self.assertEqual(preview["version"], 1)
        self.assertEqual(preview["duration"], 1.0)
        self.assertEqual(len(preview["peaks"]) % 2, 0)
        self.assertEqual(min(preview["peaks"]), -32767)
        self.assertEqual(max(preview["peaks"]), 32767)

    def test_stem_analysis_still_returns_the_master_audio_crop(self):
        master = {"waveform": torch.ones(1, 1, 48000), "sample_rate": 48000}
        stem = {"waveform": torch.full((1, 1, 48000), 2.0), "sample_rate": 48000}
        analysis = {
            "bpm": 120.0,
            "beat_times": [0.0, 0.5],
            "audio_duration": 1.0,
            "waveform_preview": {"version": 1, "duration": 1.0, "scale": 32767, "peaks": [0, 1]},
            "drum_times": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            cache_path = pathlib.Path(directory) / "analysis.json"
            with (
                mock.patch.object(timeline, "resolve_audio_path", return_value=pathlib.Path("song.wav")),
                mock.patch.object(timeline, "load_audio_file", return_value=(pathlib.Path("song.wav"), master)),
                mock.patch.object(timeline, "load_cached_stem", return_value=stem),
                mock.patch.object(timeline, "analysis_cache_key", return_value="key"),
                mock.patch.object(timeline, "_cache_path", return_value=cache_path),
                mock.patch.object(timeline, "analyze_audio", return_value=analysis) as analyze,
            ):
                _, cropped = timeline.analyze_audio_file(
                    "song.wav",
                    fps=24.0,
                    length_frames=24,
                    analysis_source="drums",
                )

        self.assertEqual(cropped["waveform"].mean(), 1.0)
        self.assertEqual(analyze.call_args.args[0]["waveform"].mean(), 2.0)

    def test_offset_changes_reuse_the_base_analysis_cache(self):
        master = {"waveform": torch.ones(1, 1, 48000), "sample_rate": 48000}
        analysis = {
            "bpm": 120.0,
            "beat_times": [0.1, 0.6],
            "detected_beat_times": [0.1, 0.6],
            "onset_times": [],
            "audio_duration": 1.0,
            "waveform_preview": {"version": 1, "duration": 1.0, "scale": 32767, "peaks": [0, 1]},
            "drum_times": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            cache_path = pathlib.Path(directory) / "analysis.json"
            with (
                mock.patch.object(timeline, "resolve_audio_path", return_value=pathlib.Path("song.wav")),
                mock.patch.object(timeline, "load_audio_file", return_value=(pathlib.Path("song.wav"), master)),
                mock.patch.object(timeline, "analysis_cache_key", return_value="key"),
                mock.patch.object(timeline, "_cache_path", return_value=cache_path),
                mock.patch.object(timeline, "analyze_audio", return_value=analysis) as analyze,
            ):
                first, _ = timeline.analyze_audio_file("song.wav", fps=24.0)
                shifted, _ = timeline.analyze_audio_file(
                    "song.wav",
                    fps=24.0,
                    beat_offset_ms=100,
                )

        self.assertEqual(analyze.call_count, 1)
        self.assertEqual(first["beat_times"], [0.1, 0.6])
        self.assertEqual(shifted["beat_times"], [0.2, 0.7])
        self.assertEqual(first["cache_key"], shifted["cache_key"])


if __name__ == "__main__":
    unittest.main()
