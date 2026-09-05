import copy
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import torch

import comfy.latent_formats
import comfy.model_patcher
import comfy.nested_tensor


PACKAGE = "fl_context_sampler_tests"
ROOT = pathlib.Path(__file__).parents[1]
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "nodes" / "ksamplers")]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(
    f"{PACKAGE}.FL_KsamplerContextWindow", ROOT / "nodes" / "ksamplers" / "FL_KsamplerContextWindow.py"
)
sampler = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sampler
spec.loader.exec_module(sampler)
Node = sampler.FL_KsamplerContextWindow


def model_for(latent_format):
    model = torch.nn.Module()
    model.latent_format = latent_format
    return comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))


AUTO_CASES = [
    (comfy.latent_formats.Wan21(), 81, ("wan", 21, 7, 81, 28)),
    (comfy.latent_formats.Wan22(), 82, ("wan", 21, 7, 81, 28)),
    (comfy.latent_formats.LTXV(), 81, ("ltx", 11, 3, 81, 24)),
    (comfy.latent_formats.LTXAV(), 96, ("ltx", 12, 3, 89, 24)),
    (comfy.latent_formats.MiniMaxH3AV(), 124, ("minimax_h3", 37, 8, 124, 29)),
    (comfy.latent_formats.MiniMaxH3AV(), 81, ("minimax_h3", 22, 8, 73, 29)),
]


def sample_args(model, latent):
    return dict(model=model, positive=[], negative=[], latent_image=latent, seed=7, steps=2,
                cfg=1.0, sampler_name="euler", scheduler="simple", denoise=1.0,
                context_length=22, context_overlap=5, context_schedule="standard_static",
                context_stride=1, fuse_method="pyramid", temporal_unit="auto")


class ContextWindowSamplerTests(unittest.TestCase):
    def test_auto_resolution(self):
        for latent_format, length, expected in AUTO_CASES:
            with self.subTest(profile=expected[0], length=length):
                settings = Node._resolve_context(model_for(latent_format), length, 30, "auto", "standard_static")
                assert tuple(settings[key] for key in ("profile", "latent_length", "latent_overlap", "video_frames", "overlap_frames")) == expected

    def test_zero_overlap_and_batched_ignored_overlap(self):
        for profile in ("wan", "ltx", "minimax_h3", "latent_frames", "video_frames_4n_plus_1"):
            with self.subTest(profile=profile):
                assert Node._resolve_context(None, 81, 0, profile, "standard_static")["latent_overlap"] == 0
                assert Node._resolve_context(None, 81, 9999, profile, "batched")["latent_overlap"] == 0

    def test_old_modes_keep_their_overlap_conversion(self):
        legacy = Node._resolve_context(None, 81, 30, "video_frames_4n_plus_1", "standard_static")
        raw = Node._resolve_context(None, 21, 8, "latent_frames", "standard_static")
        assert (legacy["latent_length"], legacy["latent_overlap"]) == (21, 8)
        assert (raw["latent_length"], raw["latent_overlap"]) == (21, 8)

    def test_unknown_auto_and_invalid_lengths_fail(self):
        with self.assertRaisesRegex(ValueError, "Auto does not recognize"):
            Node._resolve_context(model_for(comfy.latent_formats.SD15()), 81, 30, "auto", "standard_static")
        with self.assertRaisesRegex(ValueError, "at least 5"):
            Node._resolve_context(None, 4, 0, "minimax_h3", "standard_static")
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            Node._resolve_context(None, 81, -1, "wan", "standard_static")
        with self.assertRaisesRegex(ValueError, "smaller"):
            Node._validate_context(11, 11)

    def test_h3_overlap_bound_for_every_token_phase(self):
        spans = (1, 4, 4, 4, 4)
        for requested in range(80):
            resolved = Node._resolve_context(None, 124, requested, "minimax_h3", "standard_static")
            count = resolved["latent_overlap"]
            assert max(sum(spans[(phase + i) % 5] for i in range(count)) for phase in range(5)) <= requested
            assert max(sum(spans[(phase + i) % 5] for i in range(count + 1)) for phase in range(5)) > requested

    def test_sampling_preserves_native_pair_metadata_and_model_options(self):
        model = model_for(comfy.latent_formats.MiniMaxH3AV())
        original_options = copy.deepcopy(model.model_options)
        latent, _ = sampler.nodes_minimax_h3._empty_av_latent(32, 32, 56)
        latent["fl_h3_shot"] = {"authored_frames": 53, "render_frames": 56}
        latent["noise_mask"] = comfy.nested_tensor.NestedTensor([torch.ones_like(t) for t in latent["samples"].unbind()])
        metadata = latent["fl_h3_shot"].copy()
        events = []

        def sample(cloned, *args, **kwargs):
            assert cloned is not model
            assert args[7] is latent
            handler = cloned.model_options["context_handler"]
            assert (handler.context_length, handler.context_overlap) == (7, 1)
            assert cloned.get_wrappers(sampler.comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING, "ContextWindows_prepare_sampling")
            return (latent.copy(),)

        with mock.patch.object(sampler, "common_ksampler", side_effect=sample), mock.patch.object(
            sampler.FLSafeIndexListContextHandler, "_send_event", side_effect=events.append
        ):
            output = Node().sample(**sample_args(model, latent), unique_id="subgraph:7")
        assert output[0] is model
        assert output[3]["samples"] is latent["samples"]
        assert output[3]["noise_mask"] is latent["noise_mask"]
        assert output[3]["fl_h3_shot"] == metadata
        assert model.model_options == original_options
        assert events[0]["profile"] == "minimax_h3"
        assert events[-1]["status"] == "done"
        assert all(event["node"] == "subgraph:7" for event in events)

    def test_h3_rejects_bad_pair_and_unverified_window_modes_before_sampling(self):
        model = model_for(comfy.latent_formats.MiniMaxH3AV())
        latent, _ = sampler.nodes_minimax_h3._empty_av_latent(32, 32, 56)
        with mock.patch.object(sampler, "common_ksampler") as sample:
            for override, message in [({"context_schedule": "standard_uniform"}, "standard_static"),
                                      ({"freenoise": True}, "FreeNoise"),
                                      ({"temporal_dim": 3}, "temporal_dim=2")]:
                with self.assertRaisesRegex(ValueError, message):
                    Node().sample(**{**sample_args(model, latent), **override})
            sample.assert_not_called()
        video, audio = latent["samples"].unbind()
        with self.assertRaisesRegex(ValueError, "durations do not match"):
            Node._validate_h3(comfy.nested_tensor.NestedTensor((video, audio[..., :-1])), video.shape[2])

    def test_schema_keeps_existing_widget_and_output_order(self):
        required = list(Node.INPUT_TYPES()["required"])
        assert required == ["model", "positive", "negative", "latent_image", "seed", "steps", "cfg", "sampler_name",
                            "scheduler", "denoise", "context_length", "context_overlap", "context_schedule", "context_stride",
                            "fuse_method", "temporal_unit", "closed_loop", "freenoise", "causal_window_fix", "temporal_dim",
                            "cond_retain_index_list", "split_conds_to_windows"]
        assert Node.RETURN_NAMES == ("model", "positive", "negative", "latent", "vae", "image", "debug_info")
        assert Node.INPUT_TYPES()["required"]["temporal_unit"][1]["default"] == "auto"


if __name__ == "__main__":
    unittest.main()
