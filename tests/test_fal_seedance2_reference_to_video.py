import importlib.util
import io
import pathlib
import unittest
import wave
from unittest import mock

import fal_client
import requests
import torch


MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "nodes"
    / "ai"
    / "FL_Fal_Seedance2_ReferenceToVideo.py"
)
SPEC = importlib.util.spec_from_file_location("fl_fal_seedance2_tests", MODULE_PATH)
seedance2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seedance2)


class FakeVideo:
    def __init__(self, duration=3.0, dimensions=(640, 640), data=b"mp4"):
        self.duration = duration
        self.dimensions = dimensions
        self.data = data
        self.saved = None

    def get_duration(self):
        return self.duration

    def get_dimensions(self):
        return self.dimensions

    def save_to(self, buffer, format, codec):
        self.saved = (format, codec)
        buffer.write(self.data)


class FakeResponse:
    headers = {"content-length": "10"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield b"fake-video"


class FakeHandle:
    request_id = "request-123"

    def __init__(self):
        self.statuses = [
            fal_client.Queued(position=2),
            fal_client.InProgress(logs=[{"message": "Rendering"}]),
            fal_client.Completed(logs=[], metrics={}),
        ]
        self.cancelled = False

    def status(self, with_logs=False):
        return self.statuses.pop(0)

    def get(self):
        return {
            "video": {"url": "https://example.test/video.mp4"},
            "seed": 42,
        }

    def cancel(self):
        self.cancelled = True


class FakeClient:
    def __init__(self):
        self.uploads = []
        self.endpoint = None
        self.arguments = None
        self.handle = FakeHandle()

    def upload(self, data, content_type, file_name=None):
        self.uploads.append((content_type, file_name, data))
        return f"https://example.test/{file_name}"

    def submit(self, endpoint, arguments):
        self.endpoint = endpoint
        self.arguments = arguments
        return self.handle


class ReferenceValidationTests(unittest.TestCase):
    def test_references_are_ordered_by_socket_number(self):
        references = {
            "image_3": "third",
            "ignored": "nope",
            "image_1": "first",
            "image_2": None,
        }
        self.assertEqual(
            seedance2._ordered_references(references, "image"),
            ["first", "third"],
        )

    def test_prompt_tags_must_point_to_connected_references(self):
        seedance2._validate_reference_tags(
            "Use @Image2, @video1, and @Audio1.",
            image_count=2,
            video_count=1,
            audio_count=1,
        )
        with self.assertRaisesRegex(ValueError, "@Image3"):
            seedance2._validate_reference_tags(
                "Use @Image3.",
                image_count=2,
                video_count=0,
                audio_count=0,
            )

    def test_audio_only_and_total_reference_limits_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            seedance2._validate_reference_counts([], [], [object()])
        with self.assertRaisesRegex(ValueError, "12 reference files"):
            seedance2._validate_reference_counts(
                [object()] * 9,
                [object()] * 3,
                [object()],
            )

    def test_prompt_only_is_valid(self):
        seedance2._validate_reference_counts([], [], [])
        seedance2._validate_reference_tags("A prompt without tags.", 0, 0, 0)


class MediaPreparationTests(unittest.TestCase):
    def test_image_is_encoded_as_png(self):
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        data = seedance2._image_to_png(image, 1)
        self.assertTrue(data.startswith(b"\x89PNG"))

    def test_image_batch_is_rejected_to_keep_reference_numbering_stable(self):
        image = torch.zeros((2, 8, 8, 3), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "exactly one image"):
            seedance2._image_to_png(image, 1)

    def test_video_is_encoded_as_h264_mp4_and_validates_pixel_area(self):
        video = FakeVideo()
        data, duration = seedance2._video_to_mp4(video, 1)
        self.assertEqual(data, b"mp4")
        self.assertEqual(duration, 3.0)
        self.assertEqual(
            video.saved,
            (seedance2.Types.VideoContainer.MP4, seedance2.Types.VideoCodec.H264),
        )

        with self.assertRaisesRegex(ValueError, "640x640"):
            seedance2._video_to_mp4(FakeVideo(dimensions=(320, 320)), 1)

    def test_audio_is_encoded_as_pcm16_wav(self):
        audio = {
            "waveform": torch.zeros((1, 2, 16000), dtype=torch.float32),
            "sample_rate": 16000,
        }
        data, duration = seedance2._audio_to_wav(audio, 1)
        self.assertEqual(duration, 1.0)
        with wave.open(io.BytesIO(data), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 2)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 16000)

    def test_combined_video_duration_is_validated(self):
        with mock.patch.object(seedance2, "_emit"), mock.patch.object(
            seedance2, "_check_interrupted"
        ):
            with self.assertRaisesRegex(ValueError, "2 to 15 seconds"):
                seedance2._prepare_references([], [FakeVideo(duration=1.0)], [], "7")


class RequestTests(unittest.TestCase):
    def test_empty_optional_fields_are_omitted(self):
        arguments = seedance2._build_arguments(
            "Prompt",
            "720p",
            "auto",
            "auto",
            True,
            "standard",
            " ",
            [],
            [],
            [],
        )
        self.assertNotIn("end_user_id", arguments)
        self.assertNotIn("image_urls", arguments)
        self.assertNotIn("video_urls", arguments)
        self.assertNotIn("audio_urls", arguments)

    def test_full_execution_uses_request_local_key_and_returns_native_video(self):
        client = FakeClient()
        events = []

        with (
            mock.patch.object(seedance2.fal_client, "SyncClient", return_value=client) as client_type,
            mock.patch.object(seedance2.requests, "get", return_value=FakeResponse()),
            mock.patch.object(seedance2.time, "sleep"),
            mock.patch.object(
                seedance2,
                "_emit",
                side_effect=lambda node_id, phase, **details: events.append((phase, details)),
            ),
        ):
            output = seedance2.FL_Fal_Seedance2_ReferenceToVideo.execute(
                prompt="An octopus plays football.",
                reference_images={},
                reference_videos={},
                reference_audios={},
                resolution="720p",
                duration="auto",
                aspect_ratio="auto",
                generate_audio=True,
                bitrate_mode="standard",
                end_user_id="",
                fal_api_key="secret-key",
            )

        client_type.assert_called_once_with(key="secret-key")
        self.assertEqual(client.endpoint, seedance2.ENDPOINT)
        self.assertNotIn("image_urls", client.arguments)
        self.assertEqual(output.result[1:], ("https://example.test/video.mp4", 42, "request-123"))
        self.assertIsInstance(output.result[0], seedance2.InputImpl.VideoFromFile)
        self.assertEqual(
            [phase for phase, _ in events],
            ["preparing", "queued", "queued", "generating", "downloading", "complete"],
        )
        self.assertNotIn("secret-key", repr(events))

    def test_interruption_cancels_the_existing_request(self):
        class Interrupted(BaseException):
            pass

        handle = FakeHandle()
        with (
            mock.patch.object(
                seedance2.comfy.model_management,
                "processing_interrupted",
                return_value=True,
            ),
            mock.patch.object(
                seedance2.comfy.model_management,
                "throw_exception_if_processing_interrupted",
                side_effect=Interrupted,
            ),
            mock.patch.object(seedance2, "_emit"),
        ):
            with self.assertRaises(Interrupted):
                seedance2._check_interrupted(handle, "3")
        self.assertTrue(handle.cancelled)

    def test_status_poll_retries_the_same_request(self):
        class RetryHandle(FakeHandle):
            def __init__(self):
                super().__init__()
                self.statuses = [
                    requests.ConnectionError("temporary"),
                    fal_client.InProgress(logs=None),
                    fal_client.Completed(logs=[], metrics={}),
                ]

            def status(self, with_logs=False):
                status = self.statuses.pop(0)
                if isinstance(status, Exception):
                    raise status
                return status

        handle = RetryHandle()
        events = []
        with (
            mock.patch.object(seedance2.time, "sleep"),
            mock.patch.object(seedance2, "_check_interrupted"),
            mock.patch.object(
                seedance2,
                "_emit",
                side_effect=lambda node_id, phase, **details: events.append((phase, details)),
            ),
        ):
            seedance2._poll_request(handle, "3")

        self.assertFalse(handle.cancelled)
        self.assertEqual([phase for phase, _ in events], ["generating", "generating"])
        self.assertIn("retrying", events[0][1]["log"])

    def test_api_key_is_redacted_from_error_text(self):
        self.assertEqual(
            seedance2._safe_error(ValueError("bad secret-key value"), "secret-key"),
            "bad *** value",
        )


class NodeContractTests(unittest.TestCase):
    def test_schema_exposes_three_zero_minimum_autogrow_groups(self):
        schema = seedance2.FL_Fal_Seedance2_ReferenceToVideo.GET_SCHEMA()
        inputs = {item.id: item for item in schema.inputs}

        self.assertEqual(inputs["reference_images"].template.min, 0)
        self.assertEqual(inputs["reference_images"].template.names[-1], "image_9")
        self.assertEqual(inputs["reference_videos"].template.min, 0)
        self.assertEqual(inputs["reference_videos"].template.names[-1], "video_3")
        self.assertEqual(inputs["reference_audios"].template.min, 0)
        self.assertEqual(inputs["reference_audios"].template.names[-1], "audio_3")
        self.assertEqual(
            seedance2.FL_Fal_Seedance2_ReferenceToVideo.RETURN_TYPES,
            ["VIDEO", "STRING", "INT", "STRING"],
        )

    def test_frontend_has_node_scoped_progress_and_result_controls(self):
        script = (
            pathlib.Path(__file__).parents[1]
            / "web"
            / "nodes"
            / "ai"
            / "FL_Fal_Seedance2_ReferenceToVideo.js"
        ).read_text(encoding="utf-8")

        for behavior in (
            'const NODE_CLASS = "FL_Fal_Seedance2_ReferenceToVideo"',
            'serialize: false',
            'data-role="chips"',
            'data-role="progress"',
            'data-role="video"',
            'data-action="copy-url"',
            'data-action="copy-request"',
            'api.addEventListener(EVENT_NAME',
            'prompt.callback?.(prompt.value)',
        ):
            with self.subTest(behavior=behavior):
                self.assertIn(behavior, script)


if __name__ == "__main__":
    unittest.main()
