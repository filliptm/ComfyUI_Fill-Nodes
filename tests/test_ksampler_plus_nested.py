import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import torch

import comfy.nested_tensor


ROOT = pathlib.Path(__file__).parents[1]
PACKAGE = "fl_ksampler_plus_tests"


def load_sampler(module_name):
    package = sys.modules.get(PACKAGE)
    if package is None:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(ROOT / "nodes" / "ksamplers")]
        sys.modules[PACKAGE] = package

    path = ROOT / "nodes" / "ksamplers" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Model:
    def get_model_object(self, name):
        raise KeyError(name)


class KSamplerPlusNestedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = (
            load_sampler("FL_KsamplerPlus"),
            load_sampler("FL_KsamplerPlusV2"),
        )

    def test_samples_audio_once_and_preserves_it_across_video_tiles(self):
        for module in self.modules:
            with self.subTest(module=module.__name__):
                calls = []

                def sample_tile(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise):
                    video, audio = latent["samples"].unbind()
                    calls.append(latent)
                    if len(calls) == 1:
                        self.assertNotIn("noise_mask", latent)
                        audio = torch.full_like(audio, 7)
                    else:
                        torch.testing.assert_close(audio, torch.full_like(audio, 7))
                        _, audio_mask = latent["noise_mask"].unbind()
                        torch.testing.assert_close(audio_mask, torch.zeros_like(audio_mask))
                    return ({"samples": comfy.nested_tensor.NestedTensor((torch.full_like(video, len(calls)), audio))},)

                video = torch.zeros((1, 24, 1, 4, 4))
                audio = torch.zeros((1, 32, 2, 8))
                latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
                sampler = getattr(module, module.__name__.rsplit(".", 1)[-1])()
                arguments = {
                    "model": Model(),
                    "positive": [],
                    "negative": [],
                    "seed": 1,
                    "steps": 1,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "input_type": "latent",
                    "x_slices": 2,
                    "y_slices": 1,
                    "overlap": 0.0,
                    "batch_size": 1,
                    "use_sliced_conditioning": False,
                    "latent_image": latent,
                }
                if module.__name__.endswith("V2"):
                    arguments.update(conditioning_strength=1.0, debug_mode=False)

                with (
                    mock.patch.object(module.comfy.model_management, "get_torch_device", return_value=torch.device("cpu")),
                    mock.patch.object(module, "common_ksampler", side_effect=sample_tile),
                ):
                    result = sampler.sample(**arguments)

                output_video, output_audio = result[3]["samples"].unbind()
                self.assertEqual(len(calls), 2)
                torch.testing.assert_close(output_video[..., :2], torch.ones_like(output_video[..., :2]))
                torch.testing.assert_close(output_video[..., 2:], torch.full_like(output_video[..., 2:], 2))
                torch.testing.assert_close(output_audio, torch.full_like(output_audio, 7))
                torch.testing.assert_close(audio, torch.zeros_like(audio))


if __name__ == "__main__":
    unittest.main()
