import importlib.util
import json
import pathlib
import tempfile
import types
import unittest
from fractions import Fraction
from unittest import mock

import torch


MODULE_PATH = pathlib.Path(__file__).parents[1] / "nodes" / "video" / "FL_LoadVideo.py"
SPEC = importlib.util.spec_from_file_location("fl_load_video_tests", MODULE_PATH)
load_video = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(load_video)


def settings(**updates):
    configured = load_video.DEFAULT_LOAD_SETTINGS.copy()
    configured.update(updates)
    return configured


def probe(**updates):
    result = {
        "width": 1920,
        "height": 1080,
        "duration": 10.0,
        "frame_rate": 30.0,
        "frame_count": 300,
        "frame_count_estimated": False,
        "bit_depth": 8,
        "codec": "h264",
        "container": "mov,mp4",
        "has_audio": True,
        "size": 1000,
    }
    result.update(updates)
    return result


class LoadVideoSettingsTests(unittest.TestCase):
    def test_defaults_parse(self):
        parsed = load_video._parse_settings(load_video.DEFAULT_SETTINGS_JSON)

        self.assertEqual(parsed, load_video.DEFAULT_LOAD_SETTINGS)
        self.assertIsInstance(parsed["start_time"], float)
        self.assertIsInstance(parsed["target_fps"], float)

    def test_missing_fields_use_defaults(self):
        parsed = load_video._parse_settings('{"version":1,"frame_load_cap":81}')

        self.assertEqual(parsed["frame_load_cap"], 81)
        self.assertEqual(parsed["sample_mode"], "source")
        self.assertEqual(parsed["resize_mode"], "original")

    def test_invalid_json_and_version_fail(self):
        for value in ("{", "[]", "null"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_video._parse_settings(value)
        with self.assertRaisesRegex(ValueError, "version 2 is unsupported"):
            load_video._parse_settings('{"version":2}')

    def test_invalid_values_fail(self):
        cases = {
            "start_time": -1,
            "end_time": -1,
            "target_fps": 121,
            "sample_mode": "random",
            "select_every_nth": 0,
            "frame_load_cap": -1,
            "resize_mode": "stretch",
            "width": -1,
            "height": 20000,
            "include_audio": "yes",
        }
        for name, value in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_video._parse_settings(json.dumps(settings(**{name: value})))

    def test_trim_and_resize_combinations_are_validated(self):
        with self.assertRaisesRegex(ValueError, "greater than start_time"):
            load_video._parse_settings(json.dumps(settings(start_time=4, end_time=3)))
        with self.assertRaisesRegex(ValueError, "requires a width or height"):
            load_video._parse_settings(json.dumps(settings(resize_mode="fit")))
        with self.assertRaisesRegex(ValueError, "requires both width and height"):
            load_video._parse_settings(json.dumps(settings(resize_mode="crop", width=512)))


class LoadVideoPathTests(unittest.TestCase):
    def test_video_must_be_inside_input_directory(self):
        with tempfile.TemporaryDirectory() as input_directory, tempfile.TemporaryDirectory() as outside:
            inside = pathlib.Path(input_directory) / "clip.mp4"
            inside.touch()
            outside_file = pathlib.Path(outside) / "clip.mp4"
            outside_file.touch()

            with mock.patch.object(load_video.folder_paths, "get_input_directory", return_value=input_directory):
                self.assertEqual(load_video.resolve_video_path("clip.mp4"), inside.resolve())
                with self.assertRaisesRegex(ValueError, "inside the ComfyUI input"):
                    load_video.resolve_video_path(str(outside_file))

    def test_available_files_are_recursive_and_supported(self):
        with tempfile.TemporaryDirectory() as input_directory:
            root = pathlib.Path(input_directory)
            (root / "nested").mkdir()
            (root / "nested" / "clip.MOV").touch()
            (root / "notes.txt").touch()

            with mock.patch.object(load_video.folder_paths, "get_input_directory", return_value=input_directory):
                files = load_video.available_video_files()

        self.assertEqual(files, ["nested/clip.MOV"])


class LoadVideoPlanningTests(unittest.TestCase):
    def test_fit_and_crop_dimensions(self):
        fit = load_video._target_dimensions(1920, 1080, settings(resize_mode="fit", width=512, height=512))
        crop = load_video._target_dimensions(1920, 1080, settings(resize_mode="crop", width=512, height=512))

        self.assertEqual(fit, (512, 288))
        self.assertEqual(crop, (512, 512))

    def test_frame_cap_shortens_decode_window(self):
        plan = load_video.build_load_plan(
            probe(),
            settings(sample_mode="target_fps", target_fps=24, frame_load_cap=48),
        )

        self.assertEqual(plan["effective_fps"], 24)
        self.assertEqual(plan["estimated_output_frames"], 48)
        self.assertEqual(plan["decode_duration"], 2)
        self.assertEqual(plan["estimated_source_frames"], 60)

    def test_trim_is_clamped_to_source_duration(self):
        plan = load_video.build_load_plan(probe(duration=5), settings(start_time=2, end_time=8))

        self.assertEqual(plan["start_time"], 2)
        self.assertEqual(plan["end_time"], 5)
        self.assertEqual(plan["selected_duration"], 3)

    def test_start_beyond_end_fails(self):
        with self.assertRaisesRegex(ValueError, "beyond the end"):
            load_video.build_load_plan(probe(duration=5), settings(start_time=5))

    def test_memory_check_has_actionable_error(self):
        plan = {"estimated_peak_bytes": 900}
        memory = types.SimpleNamespace(available=1000)

        with (
            mock.patch.object(load_video.psutil, "virtual_memory", return_value=memory),
            self.assertRaisesRegex(RuntimeError, "Reduce the range"),
        ):
            load_video._check_memory(plan)


class LoadVideoProcessingTests(unittest.TestCase):
    def test_target_fps_selects_even_frames(self):
        images = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1, 1)

        sampled, fps = load_video._sample_images(
            images,
            30,
            settings(sample_mode="target_fps", target_fps=15),
        )

        self.assertEqual(fps, 15)
        self.assertEqual(sampled.flatten().tolist(), [0, 2, 4])

    def test_every_nth_and_cap_apply_in_order(self):
        images = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1)

        sampled, fps = load_video._sample_images(
            images,
            30,
            settings(sample_mode="every_nth", select_every_nth=3, frame_load_cap=2),
        )

        self.assertEqual(fps, 10)
        self.assertEqual(sampled.flatten().tolist(), [0, 3])

    def test_audio_is_trimmed_without_mutating_source(self):
        waveform = torch.arange(20, dtype=torch.float32).reshape(1, 1, 20)
        audio = {"waveform": waveform, "sample_rate": 10}

        trimmed = load_video._trim_audio(audio, 0.6, True)

        self.assertEqual(trimmed["waveform"].shape[-1], 6)
        self.assertIsNot(trimmed["waveform"], waveform)
        self.assertEqual(waveform.shape[-1], 20)
        self.assertIsNone(load_video._trim_audio(audio, 1, False))

    def test_resize_fit_and_crop(self):
        images = torch.rand((2, 6, 10, 3))

        fit = load_video._resize_images(images, settings(resize_mode="fit", width=5, height=5))
        crop = load_video._resize_images(images, settings(resize_mode="crop", width=4, height=4))

        self.assertEqual(fit.shape, (2, 3, 5, 3))
        self.assertEqual(crop.shape, (2, 4, 4, 3))


class LoadVideoExecutionTests(unittest.TestCase):
    def test_execution_returns_aligned_standard_outputs(self):
        with tempfile.TemporaryDirectory() as input_directory:
            path = pathlib.Path(input_directory) / "nested" / "clip.mp4"
            path.parent.mkdir()
            path.touch()
            images = torch.arange(6 * 2 * 4 * 3, dtype=torch.float32).reshape(6, 2, 4, 3)
            audio = {"waveform": torch.ones((1, 2, 48000)), "sample_rate": 48000}
            components = types.SimpleNamespace(
                images=images,
                audio=audio,
                frame_rate=Fraction(30, 1),
                metadata={"title": "test"},
            )
            source = mock.Mock()
            source.get_components.return_value = components
            native_video = mock.Mock()
            configured = settings(
                sample_mode="every_nth",
                select_every_nth=2,
                frame_load_cap=2,
                resize_mode="crop",
                width=2,
                height=2,
            )

            with (
                mock.patch.object(load_video.folder_paths, "get_input_directory", return_value=input_directory),
                mock.patch.object(load_video, "probe_video", return_value=probe(width=4, height=2, duration=0.2)),
                mock.patch.object(load_video, "_check_memory"),
                mock.patch.object(load_video.InputImpl, "VideoFromFile", return_value=source) as create_source,
                mock.patch.object(load_video.InputImpl, "VideoFromComponents", return_value=native_video) as create_video,
            ):
                result = load_video.FL_LoadVideo().load_video("nested/clip.mp4", json.dumps(configured))

        create_source.assert_called_once_with(str(path.resolve()), start_time=0, duration=4 / 30)
        loaded_images, loaded_audio, returned_video, fps, frame_count = result["result"]
        self.assertEqual(loaded_images.shape, (2, 2, 2, 3))
        self.assertEqual(loaded_audio["waveform"].shape[-1], 6400)
        self.assertIs(returned_video, native_video)
        self.assertEqual(fps, 15)
        self.assertEqual(frame_count, 2)

        video_components = create_video.call_args.args[0]
        self.assertIs(video_components.images, loaded_images)
        self.assertIs(video_components.audio, loaded_audio)
        self.assertEqual(float(video_components.frame_rate), 15)
        self.assertEqual(video_components.metadata, {"title": "test"})
        self.assertEqual(create_video.call_args.kwargs["bit_depth"], 8)

        preview = result["ui"]["fl_load_video"][0]
        self.assertEqual(preview["filename"], "clip.mp4")
        self.assertEqual(preview["subfolder"], "nested")
        self.assertEqual(preview["type"], "input")
        self.assertEqual(preview["loaded_frame_count"], 2)
        self.assertEqual(preview["loaded_fps"], 15)
        self.assertTrue(preview["has_audio"])

    def test_change_fingerprint_uses_metadata_not_file_hash(self):
        with tempfile.TemporaryDirectory() as input_directory:
            path = pathlib.Path(input_directory) / "clip.mp4"
            path.write_bytes(b"video")
            stat = path.stat()

            with mock.patch.object(load_video.folder_paths, "get_input_directory", return_value=input_directory):
                fingerprint = load_video.FL_LoadVideo.IS_CHANGED("clip.mp4")

        self.assertEqual(fingerprint, f"{stat.st_mtime_ns}:{stat.st_size}")


class LoadVideoFrontendTests(unittest.TestCase):
    def test_frontend_contains_the_full_settings_contract_and_media_flow(self):
        script = (pathlib.Path(__file__).parents[1] / "web" / "nodes" / "video" / "FL_LoadVideo.js").read_text(encoding="utf-8")

        for name in load_video.DEFAULT_LOAD_SETTINGS:
            with self.subTest(setting=name):
                self.assertIn(f"{name}:", script)
        for behavior in (
            'data-role="drop-zone"',
            'data-role="video"',
            'data-role="settings-menu"',
            'data-role="source-action"',
            'data-role="sample-value"',
            'data-role="audio-toggle"',
            'data-role="trim-timeline"',
            'data-role="trim-canvas"',
            'beginTrimPointer(event)',
            'renderTrimTimeline()',
            'this.updateSetting("start_time"',
            'this.updateSetting("end_time"',
            'data-role="trim-frame-label"',
            'selected frames',
            'syncFrameRange(name)',
            'effectiveFrameRate()',
            'this.settings.end_time = end >= bounds.duration',
            'flvl-range-row',
            'flvl-sampling-group',
            'flvl-output-group',
            'grid-template-rows: 29px minmax(0, 1fr) 40px 78px',
            'updateSourceAction(hasSource)',
            '"/upload/image"',
            "/fl/load-video/info?",
            'api.apiURL(`/view?',
            'message?.fl_load_video?.[0]',
            'MIN_NODE_WIDTH = 420',
            'MIN_NODE_HEIGHT = 440',
        ):
            with self.subTest(behavior=behavior):
                self.assertIn(behavior, script)

        menu_index = script.index('<div class="flvl-menu" data-role="settings-menu"')
        for visible_control in (
            'data-setting="resize_mode"',
            'data-setting="width"',
            'data-setting="height"',
            'data-setting="include_audio"',
        ):
            with self.subTest(visible_control=visible_control):
                self.assertLess(script.index(visible_control), menu_index)


if __name__ == "__main__":
    unittest.main()
