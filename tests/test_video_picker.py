import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import torch


MODULE_PATH = pathlib.Path(__file__).parents[1] / "nodes" / "video" / "FL_VideoPicker.py"


class FakeRoutes:
    def get(self, _path):
        return lambda function: function

    def post(self, _path):
        return lambda function: function


fake_server = types.ModuleType("server")
fake_server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=FakeRoutes(), send_sync=lambda *_args: None))
original_server = sys.modules.get("server")
sys.modules["server"] = fake_server
try:
    spec = importlib.util.spec_from_file_location("fl_video_picker_tests", MODULE_PATH)
    video_picker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(video_picker)
finally:
    if original_server is None:
        del sys.modules["server"]
    else:
        sys.modules["server"] = original_server


class VideoPickerTests(unittest.TestCase):
    def test_grid_is_row_major_and_retains_every_frame(self):
        images = torch.arange(2 * 4 * 6 * 3, dtype=torch.float32).reshape(2, 4, 6, 3)

        videos = video_picker.split_video_grid(images, 2, 2)

        self.assertEqual([tuple(video.shape) for video in videos], [(2, 2, 3, 3)] * 4)
        torch.testing.assert_close(videos[0], images[:, :2, :3])
        torch.testing.assert_close(videos[1], images[:, :2, 3:])
        torch.testing.assert_close(videos[2], images[:, 2:, :3])
        torch.testing.assert_close(videos[3], images[:, 2:, 3:])

    def test_uneven_dimensions_are_partitioned_without_dropping_pixels(self):
        images = torch.randn((3, 5, 7, 3))

        videos = video_picker.split_video_grid(images, 2, 2)

        self.assertEqual([video.shape[1:3] for video in videos], [
            torch.Size((2, 4)),
            torch.Size((2, 3)),
            torch.Size((3, 4)),
            torch.Size((3, 3)),
        ])

    def test_selected_outputs_are_independent_copies_and_form_a_list(self):
        images = torch.randn((2, 4, 6, 3))
        picker = video_picker.FL_VideoPicker()
        def choose(_name, payload):
            session = video_picker._pending_video_selections[payload["session_id"]]
            session["selection"] = [1, 3]
            session["event"].set()

        with (
            mock.patch.object(video_picker, "_write_preview"),
            mock.patch.object(video_picker, "_remove_previews"),
            mock.patch.object(video_picker.folder_paths, "get_temp_directory", return_value="D:\\temp"),
            mock.patch.object(video_picker.os, "makedirs"),
            mock.patch.object(video_picker.PromptServer.instance, "send_sync", side_effect=choose),
        ):
            selected, info, indexes = picker.select_videos(images, 2, 2, 24.0, 300)

        self.assertEqual(len(selected), 2)
        self.assertEqual(indexes, "1,3")
        self.assertIn("Selected 2 of 4", info)
        source = video_picker.split_video_grid(images, 2, 2)
        torch.testing.assert_close(selected[0], source[1])
        self.assertNotEqual(selected[0].data_ptr(), source[1].data_ptr())

    def test_schema_marks_video_output_as_a_list(self):
        self.assertEqual(video_picker.FL_VideoPicker.OUTPUT_IS_LIST, (True, False, False))
        self.assertEqual(video_picker.FL_VideoPicker.RETURN_NAMES[0], "selected_videos")

    def test_frontend_has_synchronized_playback_and_multi_select_controls(self):
        script = (pathlib.Path(__file__).parents[1] / "web" / "nodes" / "video" / "FL_VideoPicker.js").read_text(encoding="utf-8")
        for behavior in (
            'api.addEventListener("fl_video_picker_show"',
            "togglePlayback()",
            "Math.abs(video.currentTime - leader.currentTime) > 0.08",
            '"Select all"',
            '"Invert"',
            "Keep ${count} selected",
            'api.fetchApi("/fl_video_picker/select"',
            "video.removeAttribute(\"src\")",
        ):
            with self.subTest(behavior=behavior):
                self.assertIn(behavior, script)


if __name__ == "__main__":
    unittest.main()
