from dataclasses import dataclass
import torch
import torch.nn as nn
from comfy.model_patcher import ModelPatcher
from typing import Union

T = torch.Tensor


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d


class StyleAlignedArgs:
    def __init__(self, share_attn: str) -> None:
        self.adain_keys = "k" in share_attn
        self.adain_values = "v" in share_attn
        self.adain_queries = "q" in share_attn

    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = True


def calc_mean_std(feat, eps: float = 1e-5) -> "tuple[T, T]":
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def expand_first(feat: T, scale=1.0) -> T:
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.0) -> T:
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def enhanced_adain(feat: T, style_feat: T) -> T:
    if style_feat is None:
        return feat
    feat_mean, feat_std = calc_mean_std(feat)
    style_mean, style_std = calc_mean_std(style_feat)

    # Enhanced AdaIN with a learnable scaling factor
    scaling_factor = torch.nn.Parameter(torch.ones_like(feat_std))
    feat_normalized = (feat - feat_mean) / feat_std
    return scaling_factor * (feat_normalized * style_std + style_mean)


class EnhancedSharedAttentionProcessor:
    def __init__(self, args: StyleAlignedArgs, scale: float):
        self.args = args
        self.scale = scale

    def __call__(self, q, k, v, extra_options):
        style_feat = extra_options.get('style_feat', None)
        if self.args.adain_queries and style_feat is not None:
            q = enhanced_adain(q, style_feat)
        if self.args.adain_keys and style_feat is not None:
            k = enhanced_adain(k, style_feat)
        if self.args.adain_values and style_feat is not None:
            v = enhanced_adain(v, style_feat)
        if self.args.share_attention:
            k = concat_first(k, -2, scale=self.scale)
            v = concat_first(v, -2)
        return q, k, v


def get_norm_layers(
        layer: nn.Module,
        norm_layers_: "dict[str, list[Union[nn.GroupNorm, nn.LayerNorm]]]",
        share_layer_norm: bool,
        share_group_norm: bool,
):
    if isinstance(layer, nn.LayerNorm) and share_layer_norm:
        norm_layers_["layer"].append(layer)
    if isinstance(layer, nn.GroupNorm) and share_group_norm:
        norm_layers_["group"].append(layer)
    else:
        for child_layer in layer.children():
            get_norm_layers(
                child_layer, norm_layers_, share_layer_norm, share_group_norm
            )


def register_norm_forward(
        norm_layer: Union[nn.GroupNorm, nn.LayerNorm],
) -> Union[nn.GroupNorm, nn.LayerNorm]:
    if not hasattr(norm_layer, "orig_forward"):
        setattr(norm_layer, "orig_forward", norm_layer.forward)
    orig_forward = norm_layer.orig_forward

    def forward_(hidden_states: T, *args, **kwargs) -> T:
        style_feat = kwargs.get('style_feat', None)
        n = hidden_states.shape[-2]
        hidden_states = concat_first(hidden_states, dim=-2)
        hidden_states = enhanced_adain(hidden_states, style_feat)
        hidden_states = orig_forward(hidden_states)
        return hidden_states[..., :n, :]

    norm_layer.forward = forward_
    return norm_layer


def register_shared_norm(
        model: ModelPatcher,
        share_group_norm: bool = True,
        share_layer_norm: bool = True,
):
    norm_layers = {"group": [], "layer": []}
    get_norm_layers(model.model, norm_layers, share_layer_norm, share_group_norm)
    print(
        f"Patching {len(norm_layers['group'])} group norms, {len(norm_layers['layer'])} layer norms."
    )
    return [register_norm_forward(layer) for layer in norm_layers["group"]] + [
        register_norm_forward(layer) for layer in norm_layers["layer"]
    ]


SHARE_NORM_OPTIONS = ["both", "group", "layer", "disabled"]
SHARE_ATTN_OPTIONS = ["q+k", "q+k+v", "disabled"]


class FL_BatchAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "share_norm": (SHARE_NORM_OPTIONS,),
                "share_attn": (SHARE_ATTN_OPTIONS,),
                "scale": ("FLOAT", {"default": 1, "min": -2, "max": 2, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "🏵️Fill Nodes/experiments"

    def patch(
            self,
            model: ModelPatcher,
            share_norm: str,
            share_attn: str,
            scale: float,
    ):
        m = model.clone()
        share_group_norm = share_norm in ["group", "both"]
        share_layer_norm = share_norm in ["layer", "both"]
        register_shared_norm(model, share_group_norm, share_layer_norm)
        args = StyleAlignedArgs(share_attn)
        m.set_model_attn1_patch(EnhancedSharedAttentionProcessor(args, scale))
        return (m,)


def consistency_loss(batch_images: T) -> T:
    """Calculate consistency loss to penalize differences within the batch."""
    mean_image = batch_images.mean(dim=0, keepdim=True)
    loss = ((batch_images - mean_image) ** 2).mean()
    return loss


class ConsistencyEnforcedFLBatchAlign(FL_BatchAlign):
    def patch(
            self,
            model: ModelPatcher,
            share_norm: str,
            share_attn: str,
            scale: float,
    ):
        m = super().patch(model, share_norm, share_attn, scale)
        # Apply consistency loss (example, in practice this should be integrated into the training loop)
        batch_images = get_batch_images()  # Assume this function retrieves batch images
        loss = consistency_loss(batch_images)
        # Here you would typically backpropagate this loss if in a training loop
        return (m, loss)