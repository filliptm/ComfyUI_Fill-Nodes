import importlib.util
import pathlib
import unittest

import numpy as np


MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "nodes"
    / "audio"
    / "FL_Audio_BPM_Analyzer.py"
)
SPEC = importlib.util.spec_from_file_location("fl_audio_bpm_analyzer", MODULE_PATH)
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


class WaveformPreviewTests(unittest.TestCase):
    def test_schema_has_one_bpm_policy(self):
        optional_inputs = analyzer.FL_Audio_BPM_Analyzer.INPUT_TYPES()["optional"]

        self.assertNotIn("bpm_method", optional_inputs)
        self.assertIn("half_time", optional_inputs)

    def test_preview_preserves_normalized_minimums_and_maximums(self):
        waveform = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)

        preview = analyzer._waveform_preview(waveform, 4)

        self.assertEqual(preview["version"], 1)
        self.assertEqual(preview["duration"], 1.0)
        self.assertEqual(preview["scale"], 32767)
        self.assertEqual(len(preview["peaks"]), 8)
        self.assertEqual(min(preview["peaks"]), -32767)
        self.assertEqual(max(preview["peaks"]), 32767)

    def test_preview_caps_bucket_count(self):
        waveform = np.linspace(-1.0, 1.0, 5000, dtype=np.float32)

        preview = analyzer._waveform_preview(waveform, 1)

        self.assertEqual(len(preview["peaks"]) // 2, analyzer._MAX_WAVEFORM_BUCKETS)

    def test_silent_preview_contains_zero_peaks(self):
        preview = analyzer._waveform_preview(np.zeros(48000, dtype=np.float32), 48000)

        self.assertEqual(set(preview["peaks"]), {0})


if __name__ == "__main__":
    unittest.main()
