import importlib.util
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import torch


MODULE_PATH = pathlib.Path(__file__).parents[1] / "nodes" / "video" / "FL_VideoCombine.py"
SPEC = importlib.util.spec_from_file_location("fl_video_combine_tests", MODULE_PATH)
video_combine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(video_combine)


class VideoCombineSettingsTests(unittest.TestCase):
    def test_defaults_parse(self):
        settings = video_combine._parse_settings(video_combine.DEFAULT_SETTINGS_JSON)

        self.assertEqual(settings, video_combine.DEFAULT_RENDER_SETTINGS)
        self.assertIsInstance(settings["frame_rate"], float)
        self.assertIsInstance(settings["audio_gain_db"], float)

    def test_missing_fields_use_defaults(self):
        settings = video_combine._parse_settings('{"version":1,"frame_rate":30}')

        self.assertEqual(settings["frame_rate"], 30.0)
        self.assertEqual(settings["filename_prefix"], "FillVideo")
        self.assertEqual(settings["crf"], 19)
        self.assertEqual(settings["output_directory"], "")

    def test_invalid_json_and_non_object_fail(self):
        for value in ("{", "[]", "null"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                video_combine._parse_settings(value)

    def test_unsupported_version_fails(self):
        with self.assertRaisesRegex(ValueError, "version 2 is unsupported"):
            video_combine._parse_settings('{"version":2}')

    def test_invalid_settings_fail(self):
        cases = {
            "frame_rate": 0,
            "format": "webm",
            "codec": "hevc",
            "crf": 52,
            "bit_depth": 12,
            "audio_gain_db": 13,
            "output_directory": 42,
            "include_audio": "yes",
            "trim_video_to_audio": "yes",
            "save_output": 1,
            "save_metadata": None,
        }
        for name, value in cases.items():
            configured = video_combine.DEFAULT_RENDER_SETTINGS.copy()
            configured[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                video_combine._parse_settings(json.dumps(configured))

    def test_custom_output_directory_must_be_absolute(self):
        with self.assertRaisesRegex(ValueError, "must be an absolute path"):
            video_combine._parse_settings('{"version":1,"output_directory":"relative/videos"}')

    def test_custom_output_directory_is_normalized(self):
        directory = os.path.abspath(os.path.join("exports", "..", "video_exports"))
        settings = video_combine._parse_settings(json.dumps({
            "version": 1,
            "output_directory": directory,
        }))

        self.assertEqual(settings["output_directory"], os.path.normpath(directory))


class VideoCombineOutputDirectoryTests(unittest.TestCase):
    def test_custom_output_directory_is_created(self):
        with tempfile.TemporaryDirectory() as root:
            directory = os.path.join(root, "nested", "exports")
            settings = video_combine.DEFAULT_RENDER_SETTINGS.copy()
            settings["output_directory"] = directory

            output_directory, output_type = video_combine._output_directory(settings)

            self.assertEqual(output_directory, directory)
            self.assertEqual(output_type, "custom")
            self.assertTrue(os.path.isdir(directory))

    def test_preview_tokens_only_resolve_existing_registered_files(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            token = video_combine.register_preview_file(video.name)
            self.assertEqual(video_combine.preview_file_for_token(token), os.path.abspath(video.name))

        self.assertIsNone(video_combine.preview_file_for_token(token))
        self.assertIsNone(video_combine.preview_file_for_token("unknown"))


class VideoCombineImageTests(unittest.TestCase):
    def test_rgba_odd_dimensions_are_rgb_and_edge_padded(self):
        images = torch.arange(1 * 3 * 5 * 4, dtype=torch.float32).reshape(1, 3, 5, 4)

        prepared, source_width, source_height = video_combine._prepare_images(images)

        self.assertEqual((source_width, source_height), (5, 3))
        self.assertEqual(prepared.shape, (1, 4, 6, 3))
        torch.testing.assert_close(prepared[:, :3, :5], images[..., :3])
        torch.testing.assert_close(prepared[:, :3, 5], images[:, :, 4, :3])
        torch.testing.assert_close(prepared[:, 3], prepared[:, 2])
        self.assertEqual(images.shape, (1, 3, 5, 4))

    def test_even_rgb_input_is_reused(self):
        images = torch.zeros((2, 4, 6, 3))

        prepared, _, _ = video_combine._prepare_images(images)

        self.assertIs(prepared, images)

    def test_empty_and_invalid_channel_inputs_fail(self):
        for images in (torch.empty((0, 4, 4, 3)), torch.empty((1, 4, 4, 1))):
            with self.subTest(shape=tuple(images.shape)), self.assertRaises(ValueError):
                video_combine._prepare_images(images)


class VideoCombineAudioTests(unittest.TestCase):
    def test_excluded_audio_is_not_read(self):
        class UnreadableAudio:
            def __getitem__(self, key):
                raise AssertionError(f"Unexpected audio access: {key}")

        self.assertIsNone(video_combine._prepare_audio(UnreadableAudio(), False, 6))

    def test_zero_gain_reuses_audio(self):
        audio = {
            "waveform": torch.ones((1, 2, 8)),
            "sample_rate": 48000,
        }

        prepared = video_combine._prepare_audio(audio, True, 0)

        self.assertIs(prepared, audio)

    def test_gain_changes_copy_without_mutating_input(self):
        waveform = torch.ones((1, 1, 8))
        audio = {"waveform": waveform, "sample_rate": 44100}

        prepared = video_combine._prepare_audio(audio, True, -6)

        self.assertIsNot(prepared, audio)
        self.assertEqual(prepared["sample_rate"], 44100)
        torch.testing.assert_close(prepared["waveform"], waveform * (10 ** (-6 / 20)))
        torch.testing.assert_close(audio["waveform"], waveform)

    def test_unsupported_channels_fail(self):
        audio = {
            "waveform": torch.zeros((1, 4, 8)),
            "sample_rate": 48000,
        }

        with self.assertRaisesRegex(ValueError, "mono, stereo, or 5.1"):
            video_combine._prepare_audio(audio, True, 0)

    def test_audio_duration_trims_extra_video_frames(self):
        images = torch.zeros((10, 4, 6, 3))
        audio = {
            "waveform": torch.zeros((1, 2, 5)),
            "sample_rate": 24,
        }

        trimmed = video_combine._trim_images_to_audio(images, audio, 24, True)

        self.assertEqual(trimmed.shape[0], 5)
        self.assertEqual(trimmed.data_ptr(), images.data_ptr())

    def test_audio_trim_is_inactive_when_disabled_or_audio_is_longer(self):
        images = torch.zeros((10, 4, 6, 3))
        short_audio = {
            "waveform": torch.zeros((1, 2, 5)),
            "sample_rate": 24,
        }
        long_audio = {
            "waveform": torch.zeros((1, 2, 20)),
            "sample_rate": 24,
        }

        self.assertIs(video_combine._trim_images_to_audio(images, short_audio, 24, False), images)
        self.assertIs(video_combine._trim_images_to_audio(images, None, 24, True), images)
        self.assertIs(video_combine._trim_images_to_audio(images, long_audio, 24, True), images)


class VideoCombineExecutionTests(unittest.TestCase):
    def test_execution_builds_native_video_and_returns_preview(self):
        settings = video_combine.DEFAULT_RENDER_SETTINGS.copy()
        settings.update({
            "filename_prefix": "clips/test",
            "frame_rate": 12,
            "crf": 23,
            "bit_depth": 10,
            "audio_gain_db": -6,
            "save_output": False,
        })
        images = torch.ones((3, 5, 7, 4))
        audio = {
            "waveform": torch.ones((1, 2, 48000)),
            "sample_rate": 48000,
        }
        native_video = mock.Mock()

        with (
            mock.patch.object(video_combine.folder_paths, "get_temp_directory", return_value="D:\\temp"),
            mock.patch.object(
                video_combine.folder_paths,
                "get_save_image_path",
                return_value=("D:\\temp\\clips", "test", 7, "clips", "clips/test"),
            ) as save_path,
            mock.patch.object(video_combine.InputImpl, "VideoFromComponents", return_value=native_video) as create_video,
            mock.patch.object(video_combine.args, "disable_metadata", False),
        ):
            result = video_combine.FL_VideoCombine().combine_video(
                images,
                json.dumps(settings),
                audio=audio,
                prompt={"1": {"class_type": "Test"}},
                extra_pnginfo={"workflow": {"nodes": []}},
            )

        save_path.assert_called_once_with("clips/test", "D:\\temp", 8, 6)
        components = create_video.call_args.args[0]
        self.assertEqual(components.images.shape, (3, 6, 8, 3))
        self.assertEqual(float(components.frame_rate), 12.0)
        self.assertEqual(components.audio["sample_rate"], 48000)
        torch.testing.assert_close(components.audio["waveform"], audio["waveform"] * (10 ** (-6 / 20)))
        self.assertEqual(create_video.call_args.kwargs["bit_depth"], 10)

        output_path = os.path.join("D:\\temp\\clips", "test_00007_.mp4")
        native_video.save_to.assert_called_once_with(
            output_path,
            format=video_combine.Types.VideoContainer.MP4,
            codec=video_combine.Types.VideoCodec.H264,
            metadata={
                "workflow": {"nodes": []},
                "prompt": {"1": {"class_type": "Test"}},
            },
            crf=23,
        )
        self.assertEqual(result["result"], (output_path,))
        preview = result["ui"]["fl_video_combine"][0]
        self.assertEqual(preview["filename"], "test_00007_.mp4")
        self.assertEqual(preview["type"], "temp")
        self.assertEqual(preview["frame_count"], 3)
        self.assertEqual(preview["duration"], 0.25)
        self.assertEqual((preview["source_width"], preview["source_height"]), (7, 5))
        self.assertEqual((preview["encoded_width"], preview["encoded_height"]), (8, 6))
        self.assertTrue(preview["has_audio"])

    def test_custom_directory_overrides_default_destination_and_uses_token_preview(self):
        with tempfile.TemporaryDirectory() as output_directory:
            settings = video_combine.DEFAULT_RENDER_SETTINGS.copy()
            settings.update({
                "filename_prefix": "custom",
                "output_directory": output_directory,
                "save_output": False,
            })
            images = torch.ones((2, 4, 6, 3))
            native_video = mock.Mock()
            output_path = os.path.join(output_directory, "custom_00003_.mp4")

            with (
                mock.patch.object(video_combine.folder_paths, "get_output_directory") as default_output,
                mock.patch.object(video_combine.folder_paths, "get_temp_directory") as temp_output,
                mock.patch.object(
                    video_combine.folder_paths,
                    "get_save_image_path",
                    return_value=(output_directory, "custom", 3, "", "custom"),
                ) as save_path,
                mock.patch.object(video_combine.InputImpl, "VideoFromComponents", return_value=native_video),
                mock.patch.object(video_combine, "register_preview_file", return_value="preview-token") as register_preview,
            ):
                result = video_combine.FL_VideoCombine().combine_video(images, json.dumps(settings))

            default_output.assert_not_called()
            temp_output.assert_not_called()
            save_path.assert_called_once_with("custom", output_directory, 6, 4)
            register_preview.assert_called_once_with(output_path)
            self.assertEqual(result["result"], (output_path,))
            preview = result["ui"]["fl_video_combine"][0]
            self.assertEqual(preview["type"], "custom")
            self.assertEqual(preview["preview_url"], "/fl/video-combine/preview/preview-token")

    def test_disabled_metadata_returns_none(self):
        with mock.patch.object(video_combine.args, "disable_metadata", False):
            metadata = video_combine._build_metadata({"prompt": True}, {"workflow": True}, False)

        self.assertIsNone(metadata)

    def test_global_metadata_disable_wins(self):
        with mock.patch.object(video_combine.args, "disable_metadata", True):
            metadata = video_combine._build_metadata({"prompt": True}, {"workflow": True}, True)

        self.assertIsNone(metadata)


if __name__ == "__main__":
    unittest.main()
