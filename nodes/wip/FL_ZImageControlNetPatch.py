"""Z-Image ControlNet patch loader for experimental Fill Nodes workflows."""

import torch

import folder_paths
import comfy.ldm.lumina.controlnet
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
from comfy_extras.nodes_model_patch import z_image_convert


class FL_ZImageControlNetPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": (folder_paths.get_filename_list("model_patches"),),
                "auto_config": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "n_control_layers": ("INT", {"default": 6, "min": 1, "max": 30, "step": 1}),
                "additional_in_dim": ("INT", {"default": 17, "min": 0, "max": 64, "step": 1}),
                "refiner_control": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL_PATCH",)
    FUNCTION = "load_model_patch"
    CATEGORY = "🏵️Fill Nodes/WIP"
    EXPERIMENTAL = True

    def load_model_patch(
        self,
        name,
        auto_config=True,
        n_control_layers=6,
        additional_in_dim=17,
        refiner_control=False,
    ):
        model_patch_path = folder_paths.get_full_path_or_raise("model_patches", name)
        sd = comfy.utils.load_torch_file(model_patch_path, safe_load=True)
        if "control_all_x_embedder.2-1.weight" not in sd:
            raise ValueError(
                f"{name} does not look like a Z-Image ControlNet patch: "
                "missing control_all_x_embedder.2-1.weight"
            )

        dtype = comfy.utils.weight_dtype(sd)
        sd = z_image_convert(sd)
        config = self._config_from_state_dict(
            sd,
            auto_config=auto_config,
            n_control_layers=n_control_layers,
            additional_in_dim=additional_in_dim,
            refiner_control=refiner_control,
        )

        model = comfy.ldm.lumina.controlnet.ZImage_Control(
            device=comfy.model_management.unet_offload_device(),
            dtype=dtype,
            operations=comfy.ops.manual_cast,
            **config,
        )
        model_patcher = comfy.model_patcher.CoreModelPatcher(
            model,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
        )
        model.load_state_dict(sd, assign=model_patcher.is_dynamic())
        return (model_patcher,)

    def _config_from_state_dict(
        self,
        sd,
        auto_config,
        n_control_layers,
        additional_in_dim,
        refiner_control,
    ):
        if not auto_config:
            return {
                "n_control_layers": n_control_layers,
                "additional_in_dim": additional_in_dim,
                "refiner_control": refiner_control,
            }

        config = {
            "n_control_layers": self._count_control_layers(sd),
            "additional_in_dim": self._infer_additional_in_dim(sd),
            "refiner_control": False,
        }

        # Match ComfyUI's built-in handling for the 3-layer and 15-layer Fun
        # ControlNet variants, while allowing our 6-layer outpaint variant.
        if "control_layers.4.adaLN_modulation.0.weight" not in sd:
            config["n_control_layers"] = 3
            config["additional_in_dim"] = 17
            config["refiner_control"] = True

        if "control_layers.14.adaLN_modulation.0.weight" in sd:
            config["n_control_layers"] = 15
            config["additional_in_dim"] = 17
            config["refiner_control"] = True
            ref_weight = sd.get("control_noise_refiner.0.after_proj.weight", None)
            if ref_weight is not None and torch.count_nonzero(ref_weight) == 0:
                config["broken"] = True

        return config

    def _infer_additional_in_dim(self, sd):
        embed_weight = sd["control_all_x_embedder.2-1.weight"]
        patch_size = 2
        f_patch_size = 1
        control_in_dim = 16
        total_in_dim = embed_weight.shape[1] // (patch_size * patch_size * f_patch_size)
        return total_in_dim - control_in_dim

    def _count_control_layers(self, sd):
        layer_ids = set()
        prefix = "control_layers."
        for key in sd:
            if key.startswith(prefix):
                rest = key[len(prefix):]
                layer_id = rest.split(".", 1)[0]
                if layer_id.isdigit():
                    layer_ids.add(int(layer_id))
        return max(layer_ids) + 1 if layer_ids else 6


NODE_CLASS_MAPPINGS = {
    "FL_ZImageControlNetPatch": FL_ZImageControlNetPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ZImageControlNetPatch": "FL Z-Image ControlNet Patch Loader",
}
