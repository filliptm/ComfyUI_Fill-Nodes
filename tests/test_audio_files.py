import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).parents[1] / "nodes" / "audio" / "audio_files.py"
SPEC = importlib.util.spec_from_file_location("fl_audio_files_tests", MODULE_PATH)
audio_files = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audio_files)


class AudioFilesTests(unittest.TestCase):
    def test_audio_library_recursively_lists_supported_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            nested = root / "album" / "stems"
            nested.mkdir(parents=True)
            (root / "song.wav").write_bytes(b"wave")
            (nested / "drums.mp3").write_bytes(b"mp3")
            (nested / "notes.txt").write_text("ignore", encoding="utf-8")

            with mock.patch.object(
                audio_files.folder_paths,
                "get_input_directory",
                return_value=directory,
            ):
                files = audio_files.available_audio_files()
                entries = audio_files.audio_library_entries()

        self.assertEqual(files, ["album/stems/drums.mp3", "song.wav"])
        self.assertEqual(
            [(entry["path"], entry["folder"]) for entry in entries],
            [("album/stems/drums.mp3", "album/stems"), ("song.wav", "")],
        )
        self.assertTrue(all(entry["size"] > 0 for entry in entries))


if __name__ == "__main__":
    unittest.main()
