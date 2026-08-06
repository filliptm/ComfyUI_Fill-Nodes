import importlib.util
import pathlib
import unittest
from unittest import mock

import torch

import comfy.nested_tensor


ROOT = pathlib.Path(__file__).parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timeline = load_module(
    "fl_minimax_h3_beat_shot_planner",
    "nodes/prompting/FL_MiniMaxH3PromptTimeline.py",
)
sampler = load_module(
    "fl_minimax_h3_beat_ksampler",
    "nodes/ksamplers/FL_MiniMaxH3BeatKSampler.py",
)
assembler = load_module(
    "fl_minimax_h3_shot_assembler",
    "nodes/video/FL_MiniMaxH3ShotAssembler.py",
)


def schedule(boundaries=(0, 54, 109, 163, 217)):
    sections = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
        sections.append({
            "line": index,
            "start": start / 24,
            "end": end / 24,
            "fade_in_end": start / 24,
            "fade_out_start": end / 24,
            "crossfade_start": start / 24,
            "crossfade_end": start / 24,
            "start_frame": start,
            "end_frame": end,
            "prompt": f"Shot {index}.",
            "curve": "cosine",
        })
    return {
        "type": "fl_prompt_schedule",
        "version": 2,
        "duration": boundaries[-1] / 24,
        "audio_duration": boundaries[-1] / 24,
        "source_unit": "frames",
        "fps": 24.0,
        "sections": sections,
    }


class FakeClip:
    def tokenize(self, prompt, **kwargs):
        return {"prompt": prompt, **kwargs}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros((1, 1, 1)), {"prompt": tokens["prompt"]}]]


class FakeVideoVAE:
    def encode(self, images):
        return torch.zeros((1, 24, 2, 2, 2))


class FakeAudioVAE:
    audio_sample_rate = 24000

    def __init__(self):
        self.encoded_samples = []

    def encode(self, audio):
        self.encoded_samples.append(audio.shape[1])
        latent_t = max(1, round(audio.shape[1] / self.audio_sample_rate * 40))
        return torch.zeros((1, 32, 2, latent_t))


class ShotPlannerTests(unittest.TestCase):
    def test_current_timeline_becomes_four_independent_h3_shots(self):
        audio_vae = FakeAudioVAE()
        audio = {
            "waveform": torch.zeros((1, 2, 217000)),
            "sample_rate": 24000,
        }

        plan = timeline.FL_MiniMaxH3BeatShotPlanner.execute(
            clip=FakeClip(),
            vae=FakeVideoVAE(),
            audio_vae=audio_vae,
            prompt_schedule=schedule(),
            timeline_audio=audio,
            global_prompt="Same character.",
            width=64,
            height=64,
            affect_audio="video only",
            ref_image_size="match",
        ).result[0]

        self.assertEqual(plan["total_frames"], 217)
        self.assertEqual(plan["total_render_frames"], 224)
        self.assertEqual(
            [shot["authored_frames"] for shot in plan["shots"]],
            [54, 55, 54, 54],
        )
        self.assertEqual(
            [shot["render_frames"] for shot in plan["shots"]],
            [56, 56, 56, 56],
        )
        self.assertEqual(audio_vae.encoded_samples, [54000, 55000, 54000, 54000])
        for shot in plan["shots"]:
            self.assertIsInstance(
                shot["latent"]["samples"],
                comfy.nested_tensor.NestedTensor,
            )
            self.assertEqual(len(shot["latent"]["samples"].unbind()), 2)

    def test_crossfade_is_rejected_for_independent_shots(self):
        value = schedule((0, 54, 109))
        value["sections"][1]["crossfade_start"] = 50 / 24
        value["sections"][1]["crossfade_end"] = 58 / 24

        with self.assertRaisesRegex(ValueError, "Remove the crossfade"):
            timeline._independent_shot_sections(value)

    def test_gap_is_rejected_instead_of_silently_changing_duration(self):
        value = schedule((0, 54, 109))
        value["sections"][1]["start"] = 55 / 24
        value["sections"][1]["fade_in_end"] = 55 / 24
        value["sections"][1]["start_frame"] = 55
        value["sections"][1]["crossfade_start"] = 55 / 24
        value["sections"][1]["crossfade_end"] = 55 / 24

        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            timeline._independent_shot_sections(value)

    def test_audio_slice_is_a_copy_with_exact_frame_boundaries(self):
        waveform = torch.arange(12000, dtype=torch.float32).reshape(1, 1, -1)
        shot = timeline._shot_audio(
            {"waveform": waveform, "sample_rate": 24000},
            5,
            10,
        )

        self.assertEqual(shot["waveform"].shape[-1], 5000)
        self.assertEqual(shot["waveform"][0, 0, 0], waveform[0, 0, 5000])
        shot["waveform"].zero_()
        self.assertNotEqual(waveform[0, 0, 5000], 0)


class BeatKSamplerTests(unittest.TestCase):
    def test_samples_every_shot_with_incremented_seeds(self):
        calls = []
        shots = [
            {
                "index": index,
                "start_frame": index * 5,
                "end_frame": (index + 1) * 5,
                "authored_frames": 5,
                "render_frames": 5,
                "conditioning": f"conditioning-{index}",
                "latent": {"samples": f"latent-{index}"},
            }
            for index in range(3)
        ]
        plan = {
            "type": "minimax_h3_beat_shot_plan",
            "version": 1,
            "fps": 24,
            "total_frames": 15,
            "shots": shots,
        }

        def sample(*args, **kwargs):
            calls.append((args, kwargs))
            return ({"samples": f"sampled-{len(calls)}"},)

        with mock.patch.object(sampler.nodes, "common_ksampler", side_effect=sample):
            output = sampler.FL_MiniMaxH3BeatKSampler.execute(
                model=object(),
                shot_plan=plan,
                seed=100,
                seed_mode="increment",
                steps=20,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
            ).result[0]

        self.assertEqual([shot["seed"] for shot in output["shots"]], [100, 101, 102])
        self.assertEqual(len(calls), 3)
        for index, (args, kwargs) in enumerate(calls):
            self.assertEqual(args[6], f"conditioning-{index}")
            self.assertEqual(args[7], f"conditioning-{index}")
            self.assertEqual(kwargs["denoise"], 1.0)

    def test_sampling_error_names_the_shot_and_frame_range(self):
        plan = {
            "type": "minimax_h3_beat_shot_plan",
            "version": 1,
            "shots": [{
                "start_frame": 10,
                "end_frame": 20,
                "conditioning": [],
                "latent": {"samples": "latent"},
            }],
        }
        with mock.patch.object(
            sampler.nodes,
            "common_ksampler",
            side_effect=RuntimeError("model failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "frames 10-19"):
                sampler.FL_MiniMaxH3BeatKSampler.execute(
                    object(),
                    plan,
                    0,
                    "fixed",
                    1,
                    1.0,
                    "euler",
                    "normal",
                    1.0,
                )


class FakeDecodeVAE:
    def __init__(self):
        self.calls = 0

    def decode(self, video):
        self.calls += 1
        latent_t = video.shape[2]
        frame_count = 5 if latent_t == 2 else ((latent_t - 2) // 5) * 17 + 5
        return torch.full((1, frame_count, 4, 6, 3), float(self.calls))


def sampled_shot(authored_frames, latent_t):
    video = torch.zeros((1, 24, latent_t, 4, 6))
    audio = torch.zeros((1, 32, 2, 10))
    return {
        "authored_frames": authored_frames,
        "latent": {
            "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        },
    }


class ShotAssemblerTests(unittest.TestCase):
    def test_decodes_independently_trims_padding_and_assembles_pixels(self):
        value = {
            "type": "minimax_h3_sampled_shots",
            "version": 1,
            "total_frames": 109,
            "shots": [
                sampled_shot(54, 17),
                sampled_shot(55, 17),
            ],
        }
        vae = FakeDecodeVAE()

        images = assembler.FL_MiniMaxH3ShotAssembler.execute(value, vae).result[0]

        self.assertEqual(vae.calls, 2)
        self.assertEqual(images.shape, (109, 4, 6, 3))
        self.assertTrue(torch.all(images[:54] == 1))
        self.assertTrue(torch.all(images[54:] == 2))

    def test_rejects_non_nested_latent(self):
        value = {
            "type": "minimax_h3_sampled_shots",
            "version": 1,
            "total_frames": 5,
            "shots": [{
                "authored_frames": 5,
                "latent": {"samples": torch.zeros((1, 4, 2, 2))},
            }],
        }

        with self.assertRaisesRegex(TypeError, "not a nested H3 latent"):
            assembler.FL_MiniMaxH3ShotAssembler.execute(value, FakeDecodeVAE())


if __name__ == "__main__":
    unittest.main()
