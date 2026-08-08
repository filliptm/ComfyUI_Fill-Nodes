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
    def test_beat_this_results_drive_bpm_grid_downbeats_and_confidence(self):
        audio = {
            "waveform": torch.zeros(1, 1, 200),
            "sample_rate": 100,
        }
        detected = {
            "beat_times": [0.1, 0.6, 1.1, 1.6],
            "downbeat_times": [0.1, 1.1],
            "beat_confidences": [0.9, 0.8, 0.7, 0.6],
            "downbeat_confidences": [0.95, 0.75],
            "detector": {"name": "Beat This", "checkpoint_sha256": "abc"},
        }
        with (
            mock.patch.object(timeline, "analyze_beat_this", return_value=detected),
            mock.patch.object(timeline.librosa.onset, "onset_strength", return_value=timeline.np.array([])),
            mock.patch.object(timeline.librosa.onset, "onset_detect", return_value=timeline.np.array([])),
            mock.patch.object(timeline.librosa, "frames_to_time", return_value=timeline.np.array([])),
            mock.patch.object(timeline, "_detect_drums", return_value={}),
            mock.patch.object(timeline, "waveform_preview", return_value={}),
        ):
            analysis = timeline.analyze_audio(audio, fps=24.0)

        self.assertAlmostEqual(analysis["bpm"], 120.0)
        self.assertEqual(analysis["bpm_source"], "beat_this_intervals_median")
        self.assertEqual(analysis["beat_times"], detected["beat_times"])
        self.assertEqual(analysis["downbeat_times"], detected["downbeat_times"])
        self.assertEqual(analysis["detected_beat_confidences"], detected["beat_confidences"])
        self.assertEqual(analysis["detector"], detected["detector"])

    def test_auxiliary_analysis_skips_beat_this(self):
        audio = {
            "waveform": torch.zeros(1, 1, 200),
            "sample_rate": 100,
        }
        with (
            mock.patch.object(timeline, "analyze_beat_this") as detector,
            mock.patch.object(timeline.librosa.onset, "onset_strength", return_value=timeline.np.array([])),
            mock.patch.object(timeline.librosa.onset, "onset_detect", return_value=timeline.np.array([])),
            mock.patch.object(timeline.librosa, "frames_to_time", return_value=timeline.np.array([])),
            mock.patch.object(timeline, "_detect_drums", return_value={}),
            mock.patch.object(timeline, "waveform_preview", return_value={}),
        ):
            analysis = timeline.analyze_audio(audio, fps=24.0, detect_beats=False)

        detector.assert_not_called()
        self.assertNotIn("beat_times", analysis)
        self.assertIn("onset_times", analysis)

    def test_beat_this_can_use_the_master_mix_with_stem_reference_analysis(self):
        stem = {"waveform": torch.zeros(1, 1, 200), "sample_rate": 100}
        mix = {"waveform": torch.ones(1, 1, 400), "sample_rate": 200}
        detected = {
            "beat_times": [0.1, 0.6, 1.1],
            "downbeat_times": [0.1],
            "beat_confidences": [0.9, 0.9, 0.9],
            "downbeat_confidences": [0.95],
            "detector": {"name": "beat_this"},
        }
        with (
            mock.patch.object(timeline, "analyze_beat_this", return_value=detected) as beat_this,
            mock.patch.object(timeline.librosa.onset, "onset_strength", return_value=timeline.np.array([])),
            mock.patch.object(timeline.librosa.onset, "onset_detect", return_value=timeline.np.array([])),
            mock.patch.object(timeline.librosa, "frames_to_time", return_value=timeline.np.array([])),
            mock.patch.object(timeline, "_detect_drums", return_value={}),
            mock.patch.object(timeline, "waveform_preview", return_value={}),
        ):
            timeline.analyze_audio(stem, fps=24.0, beat_audio=mix)

        self.assertEqual(beat_this.call_args.args[1], 200)
        timeline.np.testing.assert_array_equal(beat_this.call_args.args[0], timeline.np.ones(400))

    def test_cached_analysis_restores_an_existing_audio_file(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = pathlib.Path(directory) / "analysis.json"
            cache_path.write_text(
                '{"audio_file":"song.wav"}',
                encoding="utf-8",
            )
            with (
                mock.patch.object(timeline, "_cache_path", return_value=cache_path),
                mock.patch.object(
                    timeline,
                    "resolve_audio_path",
                    return_value=pathlib.Path("song.wav"),
                ) as resolve,
            ):
                filename = timeline.cached_analysis_audio_file("a" * 64)

        self.assertEqual(filename, "song.wav")
        resolve.assert_called_once_with("song.wav")

    def test_cached_analysis_rejects_invalid_keys_and_missing_audio(self):
        self.assertEqual(timeline.cached_analysis_audio_file("../analysis"), "")

        with tempfile.TemporaryDirectory() as directory:
            cache_path = pathlib.Path(directory) / "analysis.json"
            cache_path.write_text("{}", encoding="utf-8")
            with mock.patch.object(timeline, "_cache_path", return_value=cache_path):
                self.assertEqual(timeline.cached_analysis_audio_file("a" * 64), "")

    def test_beat_offset_shifts_only_the_regular_grid(self):
        analysis = {
            "audio_duration": 1.0,
            "beat_times": [0.1, 0.9],
            "detected_beat_times": [0.0, 0.95],
            "onset_times": [0.2],
            "drum_times": {"kick_times": [0.2]},
        }

        shifted = timeline.apply_beat_offset(analysis, fps=24.0, beat_offset_ms=200)

        self.assertEqual(shifted["base_beat_times"], [0.1, 0.9])
        self.assertEqual(shifted["beat_times"], [0.30000000000000004])
        self.assertEqual(shifted["detected_beat_times"], [0.0, 0.95])
        self.assertEqual(shifted["beat_frames"], [7])
        self.assertEqual(shifted["detected_beat_frames"], [0, 23])
        self.assertEqual(shifted["onset_times"], [0.2])
        self.assertEqual(shifted["drum_times"], {"kick_times": [0.2]})
        self.assertEqual(shifted["beat_offset_ms"], 200)
        self.assertEqual(shifted["grid_interval_seconds"], 0.8)

    def test_beat_offset_preserves_periodic_spacing_at_crop_boundaries(self):
        analysis = {
            "audio_duration": 1.0,
            "beat_times": [0.0, 0.4, 0.8],
            "detected_beat_times": [],
        }

        shifted = timeline.apply_beat_offset(analysis, fps=24.0, beat_offset_ms=-200)

        self.assertEqual(shifted["beat_times"], [0.2, 0.6000000000000001])
        self.assertEqual(shifted["base_beat_times"], [0.0, 0.4, 0.8])
        self.assertAlmostEqual(
            shifted["beat_times"][1] - shifted["beat_times"][0],
            0.4,
        )

    def test_beat_offset_keeps_the_grid_visible_after_all_native_beats_leave(self):
        analysis = {
            "audio_duration": 0.5,
            "bpm": 120.0,
            "beat_times": [0.1],
            "detected_beat_times": [0.1],
        }

        shifted = timeline.apply_beat_offset(analysis, fps=24.0, beat_offset_ms=1000)

        self.assertAlmostEqual(shifted["beat_times"][0], 0.1)
        self.assertEqual(shifted["detected_beat_times"], [0.1])

    def test_beat_grid_density_uses_native_beats_as_its_source(self):
        analysis = {
            "bpm": 120.0,
            "audio_duration": 2.0,
            "beat_times": [0.0, 0.5, 1.0, 1.5],
            "downbeat_times": [0.0, 1.0],
            "detected_beat_times": [0.05, 0.55],
            "detected_downbeat_times": [0.0, 1.0],
        }

        every_two = timeline.apply_beat_offset(
            analysis,
            fps=24.0,
            beat_grid_density="every_2_beats",
        )
        every_beat = timeline.apply_beat_offset(
            every_two,
            fps=24.0,
            beat_grid_density="every_beat",
        )
        subdivisions = timeline.apply_beat_offset(
            analysis,
            fps=24.0,
            beat_offset_ms=100,
            beat_grid_density="half_beat",
        )

        self.assertEqual(every_two["beat_times"], [0.0, 1.0])
        self.assertEqual(every_two["downbeat_times"], [0.0, 1.0])
        self.assertEqual(every_two["grid_bpm"], 60.0)
        self.assertEqual(every_beat["beat_times"], [0.0, 0.5, 1.0, 1.5])
        self.assertEqual(every_beat["downbeat_times"], [0.0, 1.0])
        self.assertEqual(every_beat["grid_bpm"], 120.0)
        self.assertEqual(
            subdivisions["beat_times"],
            [0.1, 0.35, 0.6, 0.85, 1.1, 1.35, 1.6],
        )
        self.assertEqual(subdivisions["detected_beat_times"], [0.05, 0.55])
        self.assertEqual(subdivisions["downbeat_times"], [0.1, 1.1])
        self.assertEqual(subdivisions["detected_downbeat_times"], [0.0, 1.0])
        self.assertEqual(subdivisions["grid_bpm"], 240.0)
        self.assertEqual(subdivisions["grid_interval_seconds"], 0.25)
        self.assertEqual(subdivisions["beat_grid_density"], "half_beat")

    def test_unknown_beat_grid_density_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown beat grid density"):
            timeline.apply_beat_offset(
                {"audio_duration": 1.0, "beat_times": [0.0]},
                fps=24.0,
                beat_grid_density="bars",
            )

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
        self.assertEqual(analyze.call_args.kwargs["beat_audio"]["waveform"].mean(), 1.0)

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
        self.assertFalse(first["analysis_cache_hit"])
        self.assertTrue(shifted["analysis_cache_hit"])

    def test_density_changes_reuse_the_base_analysis_cache(self):
        master = {"waveform": torch.ones(1, 1, 48000), "sample_rate": 48000}
        analysis = {
            "bpm": 120.0,
            "beat_times": [0.0, 0.25, 0.5, 0.75],
            "detected_beat_times": [],
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
                every_beat, _ = timeline.analyze_audio_file("song.wav", fps=24.0)
                every_two, _ = timeline.analyze_audio_file(
                    "song.wav",
                    fps=24.0,
                    beat_grid_density="every_2_beats",
                )

        self.assertEqual(analyze.call_count, 1)
        self.assertEqual(every_beat["beat_times"], [0.0, 0.25, 0.5, 0.75])
        self.assertEqual(every_two["beat_times"], [0.0, 0.5])
        self.assertEqual(every_beat["cache_key"], every_two["cache_key"])

    def test_beat_detection_mode_has_a_separate_versioned_cache_key(self):
        with mock.patch.object(timeline, "audio_file_hash", return_value="audio"):
            beats = timeline.analysis_cache_key(
                pathlib.Path("song.wav"), 24.0, 0, 0, False, "mix", True
            )
            auxiliary = timeline.analysis_cache_key(
                pathlib.Path("song.wav"), 24.0, 0, 0, False, "mix", False
            )

        self.assertNotEqual(beats, auxiliary)


if __name__ == "__main__":
    unittest.main()
