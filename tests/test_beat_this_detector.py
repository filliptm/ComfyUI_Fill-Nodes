import hashlib
import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock

import torch


MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "nodes"
    / "audio"
    / "beat_this_detector.py"
)
SPEC = importlib.util.spec_from_file_location("fl_beat_this_detector", MODULE_PATH)
detector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(detector)


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        return (
            self.data[index : index + chunk_size]
            for index in range(0, len(self.data), chunk_size)
        )


class FakeModel:
    def __init__(self):
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self


class FakeTracker:
    instances = []

    def __init__(self, checkpoint_path, device, float16):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.float16 = float16
        self.model = FakeModel()
        self.instances.append(self)

    def signal2spect(self, waveform, sample_rate):
        return torch.zeros(100, 128)

    def spect2frames(self, spectrogram):
        beats = torch.full((100,), -4.0)
        downbeats = torch.full((100,), -4.0)
        beats[[10, 35, 60, 85]] = torch.tensor([2.0, 3.0, 4.0, 5.0])
        downbeats[10] = 6.0
        return beats, downbeats


class FakePostprocessor:
    def __init__(self, type):
        self.type = type

    def __call__(self, beat_logits, downbeat_logits):
        return [0.2, 0.7, 1.2, 1.7], [0.2]


class BeatThisDetectorTests(unittest.TestCase):
    def package_patches(self):
        return (
            mock.patch.object(detector, "_IMPORT_ERROR", ""),
            mock.patch.object(
                detector,
                "_package_version",
                return_value=detector.MODEL_PACKAGE_VERSION,
            ),
        )

    def test_missing_checkpoint_downloads_and_verifies_atomically(self):
        data = b"beat-this-checkpoint"
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "beat_this-final0.ckpt"
            import_patch, version_patch = self.package_patches()
            with (
                import_patch,
                version_patch,
                mock.patch.object(detector, "checkpoint_path", return_value=path),
                mock.patch.object(detector, "MODEL_SIZE", len(data)),
                mock.patch.object(
                    detector,
                    "MODEL_SHA256",
                    hashlib.sha256(data).hexdigest(),
                ),
                mock.patch.object(
                    detector.requests,
                    "get",
                    return_value=FakeResponse(data),
                ) as request,
            ):
                result = detector.ensure_checkpoint()

            self.assertEqual(result, path)
            self.assertEqual(path.read_bytes(), data)
            self.assertFalse(path.with_suffix(".ckpt.part").exists())
            request.assert_called_once()

    def test_verified_checkpoint_avoids_network_access(self):
        data = b"verified"
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "beat_this-final0.ckpt"
            path.write_bytes(data)
            import_patch, version_patch = self.package_patches()
            with (
                import_patch,
                version_patch,
                mock.patch.object(detector, "checkpoint_path", return_value=path),
                mock.patch.object(detector, "MODEL_SIZE", len(data)),
                mock.patch.object(
                    detector,
                    "MODEL_SHA256",
                    hashlib.sha256(data).hexdigest(),
                ),
                mock.patch.object(detector.requests, "get") as request,
            ):
                detector.ensure_checkpoint()

            request.assert_not_called()

    def test_bad_checksum_does_not_promote_partial_download(self):
        data = b"invalid"
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "beat_this-final0.ckpt"
            import_patch, version_patch = self.package_patches()
            with (
                import_patch,
                version_patch,
                mock.patch.object(detector, "checkpoint_path", return_value=path),
                mock.patch.object(detector, "MODEL_SIZE", len(data)),
                mock.patch.object(detector, "MODEL_SHA256", "0" * 64),
                mock.patch.object(
                    detector.requests,
                    "get",
                    return_value=FakeResponse(data),
                ),
            ):
                with self.assertRaisesRegex(detector.BeatThisError, "checksum mismatch"):
                    detector.ensure_checkpoint()

            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".ckpt.part").exists())

    def test_analysis_returns_downbeats_and_event_confidence(self):
        FakeTracker.instances.clear()
        with (
            mock.patch.object(detector, "ensure_checkpoint", return_value=pathlib.Path("model.ckpt")),
            mock.patch.object(detector, "Audio2Frames", FakeTracker),
            mock.patch.object(detector, "Postprocessor", FakePostprocessor),
            mock.patch.object(
                detector.comfy.model_management,
                "get_torch_device",
                return_value=torch.device("cpu"),
            ),
            mock.patch.object(detector.comfy.model_management, "free_memory") as free_memory,
            mock.patch.object(detector.comfy.model_management, "soft_empty_cache") as cleanup,
        ):
            result = detector.analyze_beats(torch.zeros(44100).numpy(), 44100)

        self.assertEqual(result["beat_times"], [0.2, 0.7, 1.2, 1.7])
        self.assertEqual(result["downbeat_times"], [0.2])
        self.assertGreater(result["beat_confidences"][-1], result["beat_confidences"][0])
        self.assertGreater(result["downbeat_confidences"][0], 0.99)
        self.assertEqual(result["detector"]["postprocessor"], "minimal")
        self.assertEqual(FakeTracker.instances[0].model.devices, ["cpu"])
        free_memory.assert_called_once_with(detector.MODEL_MEMORY_REQUIRED, torch.device("cpu"))
        cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
