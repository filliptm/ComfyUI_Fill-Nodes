import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch


MODULE_PATH = pathlib.Path(__file__).parents[1] / "nodes" / "image" / "FL_LoadImage.py"
SPEC = importlib.util.spec_from_file_location("fl_load_image_tests", MODULE_PATH)
load_image = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(load_image)


def settings(**updates):
    configured = load_image.DEFAULT_LOAD_SETTINGS.copy()
    configured.update(updates)
    return configured


class LoadImageSettingsTests(unittest.TestCase):
    def test_defaults_and_missing_fields_parse(self):
        self.assertEqual(load_image._parse_settings(load_image.DEFAULT_SETTINGS_JSON), load_image.DEFAULT_LOAD_SETTINGS)
        self.assertEqual(
            load_image._parse_settings('{"version":1,"resize_mode":"crop","width":64,"height":32}'),
            settings(resize_mode="crop", width=64, height=32),
        )

    def test_invalid_json_version_and_values_fail(self):
        for value in ("{", "[]", "null"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_image._parse_settings(value)
        with self.assertRaisesRegex(ValueError, "version 2 is unsupported"):
            load_image._parse_settings('{"version":2}')

        cases = {
            "resize_mode": "stretch",
            "width": -1,
            "height": 20000,
        }
        for name, value in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_image._parse_settings(json.dumps(settings(**{name: value})))

    def test_resize_requirements_are_validated(self):
        with self.assertRaisesRegex(ValueError, "requires a width or height"):
            load_image._parse_settings(json.dumps(settings(resize_mode="fit")))
        with self.assertRaisesRegex(ValueError, "requires both width and height"):
            load_image._parse_settings(json.dumps(settings(resize_mode="crop", width=512)))

    def test_connected_dimensions_override_gui_values(self):
        configured = json.dumps(settings(resize_mode="fit", width=640, height=480))

        effective = load_image._parse_settings(configured, width_override=320, height_override=0)

        self.assertEqual((effective["width"], effective["height"]), (320, 0))
        with self.assertRaisesRegex(ValueError, "requires both width and height"):
            load_image._parse_settings(
                json.dumps(settings(resize_mode="crop", width=640, height=480)),
                width_override=0,
            )

    def test_override_inputs_are_optional_connection_sockets(self):
        with tempfile.TemporaryDirectory() as input_directory:
            with mock.patch.object(load_image.folder_paths, "get_input_directory", return_value=input_directory):
                optional = load_image.FL_LoadImage.INPUT_TYPES()["optional"]

        for name in ("width_override", "height_override"):
            with self.subTest(name=name):
                self.assertEqual(optional[name][0], "INT")
                self.assertTrue(optional[name][1]["forceInput"])


class LoadImagePathTests(unittest.TestCase):
    def test_input_image_must_stay_inside_input_directory(self):
        with tempfile.TemporaryDirectory() as input_directory, tempfile.TemporaryDirectory() as outside:
            root = pathlib.Path(input_directory)
            nested = root / "nested" / "image.png"
            nested.parent.mkdir()
            nested.touch()
            outside_image = pathlib.Path(outside) / "image.png"
            outside_image.touch()

            with mock.patch.object(load_image.folder_paths, "get_input_directory", return_value=input_directory):
                self.assertEqual(load_image._resolve_input_image("nested/image.png"), nested.resolve())
                with self.assertRaisesRegex(ValueError, "inside the ComfyUI input"):
                    load_image._resolve_input_image(str(outside_image))

    def test_legacy_image_must_stay_inside_its_root(self):
        with tempfile.TemporaryDirectory() as root_directory, tempfile.TemporaryDirectory() as outside:
            root = pathlib.Path(root_directory)
            nested = root / "nested" / "image.png"
            nested.parent.mkdir()
            nested.touch()
            outside_image = pathlib.Path(outside) / "image.png"
            outside_image.touch()

            self.assertEqual(load_image._resolve_legacy_image(root_directory, "nested/image.png"), nested.resolve())
            with self.assertRaisesRegex(ValueError, "inside its root"):
                load_image._resolve_legacy_image(root_directory, str(outside_image))

    def test_available_files_are_recursive_supported_and_sorted(self):
        with tempfile.TemporaryDirectory() as input_directory:
            root = pathlib.Path(input_directory)
            (root / "nested").mkdir()
            (root / "nested" / "B.WEBP").touch()
            (root / "a.png").touch()
            (root / "notes.txt").touch()

            with mock.patch.object(load_image.folder_paths, "get_input_directory", return_value=input_directory):
                files = load_image.available_image_files()

        self.assertEqual(files, ["a.png", "nested/B.WEBP"])

    def test_no_source_has_an_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "Choose an image"):
            load_image.resolve_image_path(".")


class LoadImageProcessingTests(unittest.TestCase):
    def test_fit_and_crop_dimensions(self):
        fit = load_image._target_dimensions(1920, 1080, settings(resize_mode="fit", width=512, height=512))
        crop = load_image._target_dimensions(1920, 1080, settings(resize_mode="crop", width=512, height=512))

        self.assertEqual(fit, (512, 288))
        self.assertEqual(crop, (512, 512))

    def test_resize_preserves_or_changes_shape(self):
        image = torch.rand((1, 6, 10, 3))

        self.assertIs(load_image._resize_image(image, settings()), image)
        fit = load_image._resize_image(image, settings(resize_mode="fit", width=5, height=5))
        crop = load_image._resize_image(image, settings(resize_mode="crop", width=4, height=4))

        self.assertEqual(fit.shape, (1, 3, 5, 3))
        self.assertEqual(crop.shape, (1, 4, 4, 3))

    def test_alpha_is_composited_over_white(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "alpha.png"
            source = Image.new("RGBA", (2, 1))
            source.putdata([(255, 0, 0, 0), (255, 0, 0, 255)])
            source.save(path)

            loaded, width, height, has_alpha = load_image._load_image(path)

        self.assertEqual((width, height), (2, 1))
        self.assertTrue(has_alpha)
        np.testing.assert_allclose(loaded[0, 0, 0].numpy(), [1, 1, 1])
        np.testing.assert_allclose(loaded[0, 0, 1].numpy(), [1, 0, 0])

    def test_grayscale_is_converted_to_rgb_and_exif_is_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "gray.png"
            Image.new("L", (3, 2), 64).save(path)

            with mock.patch.object(load_image.ImageOps, "exif_transpose", wraps=load_image.ImageOps.exif_transpose) as transpose:
                loaded, width, height, has_alpha = load_image._load_image(path)

        transpose.assert_called_once()
        self.assertEqual((width, height), (3, 2))
        self.assertFalse(has_alpha)
        self.assertEqual(loaded.shape, (1, 2, 3, 3))
        torch.testing.assert_close(loaded[..., 0], loaded[..., 1])
        torch.testing.assert_close(loaded[..., 1], loaded[..., 2])


class LoadImageExecutionTests(unittest.TestCase):
    def test_execution_returns_path_and_preview_metadata(self):
        with tempfile.TemporaryDirectory() as input_directory:
            path = pathlib.Path(input_directory) / "nested" / "image.png"
            path.parent.mkdir()
            Image.new("RGBA", (8, 4), (0, 0, 0, 128)).save(path)
            configured = settings(resize_mode="crop", width=3, height=2)

            with mock.patch.object(load_image.folder_paths, "get_input_directory", return_value=input_directory):
                result = load_image.FL_LoadImage().browse_files(
                    ".",
                    image="nested/image.png",
                    load_settings=json.dumps(configured),
                )

        loaded, returned_path = result["result"]
        self.assertEqual(loaded.shape, (1, 2, 3, 3))
        self.assertEqual(returned_path, str(path.resolve()))
        preview = result["ui"]["fl_load_image"][0]
        self.assertEqual(preview["filename"], "image.png")
        self.assertEqual(preview["subfolder"], "nested")
        self.assertEqual(preview["type"], "input")
        self.assertEqual((preview["source_width"], preview["source_height"]), (8, 4))
        self.assertEqual((preview["loaded_width"], preview["loaded_height"]), (3, 2))
        self.assertEqual(preview["resize_mode"], "crop")
        self.assertEqual((preview["requested_width"], preview["requested_height"]), (3, 2))
        self.assertTrue(preview["source_has_alpha"])

    def test_execution_uses_connected_dimensions_over_gui_dimensions(self):
        with tempfile.TemporaryDirectory() as input_directory:
            path = pathlib.Path(input_directory) / "image.png"
            Image.new("RGB", (8, 4), "blue").save(path)
            configured = settings(resize_mode="crop", width=7, height=6)

            with mock.patch.object(load_image.folder_paths, "get_input_directory", return_value=input_directory):
                result = load_image.FL_LoadImage().browse_files(
                    ".",
                    image="image.png",
                    load_settings=json.dumps(configured),
                    width_override=3,
                    height_override=2,
                )

        self.assertEqual(result["result"][0].shape, (1, 2, 3, 3))
        preview = result["ui"]["fl_load_image"][0]
        self.assertEqual((preview["loaded_width"], preview["loaded_height"]), (3, 2))
        self.assertEqual((preview["requested_width"], preview["requested_height"]), (3, 2))

    def test_legacy_execution_remains_supported(self):
        with tempfile.TemporaryDirectory() as root_directory, tempfile.TemporaryDirectory() as input_directory:
            path = pathlib.Path(root_directory) / "image.jpg"
            Image.new("RGB", (2, 2), "blue").save(path)

            with mock.patch.object(load_image.folder_paths, "get_input_directory", return_value=input_directory):
                result = load_image.FL_LoadImage().browse_files(root_directory, selected_file=str(path))

        self.assertEqual(result["result"][1], str(path.resolve()))
        self.assertEqual(result["ui"]["fl_load_image"][0]["type"], "legacy")

    def test_change_fingerprint_uses_file_metadata(self):
        with tempfile.TemporaryDirectory() as input_directory:
            path = pathlib.Path(input_directory) / "image.png"
            path.write_bytes(b"image")
            stat = path.stat()

            with mock.patch.object(load_image.folder_paths, "get_input_directory", return_value=input_directory):
                fingerprint = load_image.FL_LoadImage.IS_CHANGED(".", image="image.png")

        self.assertEqual(fingerprint, f"{stat.st_mtime_ns}:{stat.st_size}")


class LoadImageFrontendTests(unittest.TestCase):
    def test_frontend_contains_source_preview_resize_and_lifecycle_contract(self):
        script = (pathlib.Path(__file__).parents[1] / "web" / "nodes" / "image" / "FL_LoadImage.js").read_text(encoding="utf-8")

        for name in load_image.DEFAULT_LOAD_SETTINGS:
            with self.subTest(setting=name):
                self.assertIn(f"{name}:", script)
        for behavior in (
            'data-role="drop-zone"',
            'data-role="image"',
            'data-role="settings-menu"',
            'data-action="replace"',
            'data-setting="resize_mode"',
            'data-setting="width"',
            'data-setting="height"',
            'body.append("image", file)',
            'api.fetchApi("/upload/image"',
            'api.apiURL(`/view?',
            "migrateLegacySource()",
            'input.name === `${name}_override`',
            "handleConnectionsChanged()",
            "connectedOverrideValue(name)",
            "requested_${name}",
            "applyPreviewGeometry()",
            'data-role="image-stage"',
            'data-resize-mode="crop"',
            'container.className = "flli-container"',
            "object-position: center center",
            "grid-template-columns: minmax(0, .55fr)",
            'message?.fl_load_image?.[0]',
            "URL.revokeObjectURL",
            "MIN_NODE_WIDTH = 240",
            "MIN_NODE_HEIGHT = 230",
        ):
            with self.subTest(behavior=behavior):
                self.assertIn(behavior, script)

    def test_backend_no_longer_registers_arbitrary_file_browser_routes(self):
        script = MODULE_PATH.read_text(encoding="utf-8")

        self.assertNotIn("PromptServer", script)
        self.assertNotIn("/fl_file_browser/", script)
        self.assertNotIn("get_directory_structure", script)
        self.assertNotIn("get_thumbnail", script)


if __name__ == "__main__":
    unittest.main()
