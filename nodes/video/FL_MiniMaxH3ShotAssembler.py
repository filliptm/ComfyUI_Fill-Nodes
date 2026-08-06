import logging

import torch

import comfy.nested_tensor
import comfy.utils
from comfy_api.latest import io


H3SampledShots = io.Custom("FL_H3_SAMPLED_SHOTS")


def _sampled_shots(value):
    if not isinstance(value, dict) or value.get("type") != "minimax_h3_sampled_shots":
        raise TypeError("FL MiniMax H3 Shot Assembler received invalid sampled shots.")
    if value.get("version") != 1:
        raise ValueError("FL MiniMax H3 Shot Assembler supports sampled shot version 1.")
    shots = value.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("FL MiniMax H3 Shot Assembler requires at least one sampled shot.")
    return shots


def _video_latent(shot, position):
    latent = shot.get("latent")
    if not isinstance(latent, dict):
        raise ValueError(f"FL MiniMax H3 Shot Assembler shot {position} has no latent.")
    samples = latent.get("samples")
    if not isinstance(samples, comfy.nested_tensor.NestedTensor):
        raise TypeError(
            f"FL MiniMax H3 Shot Assembler shot {position} is not a nested H3 latent."
        )
    tensors = samples.unbind()
    if len(tensors) != 2:
        raise ValueError(
            f"FL MiniMax H3 Shot Assembler shot {position} must contain video and audio latents."
        )
    video = tensors[0]
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
        raise ValueError(
            f"FL MiniMax H3 Shot Assembler shot {position} has an invalid video latent."
        )
    return video


class FL_MiniMaxH3ShotAssembler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3ShotAssembler",
            display_name="FL MiniMax H3 Shot Assembler",
            category="🏵️Fill Nodes/Video",
            description=(
                "Decodes MiniMax H3 shots separately, removes each shot's H3 padding, "
                "and assembles the authored frames in pixel space."
            ),
            inputs=[
                H3SampledShots.Input(
                    "sampled_shots",
                    tooltip="Independent sampled shots from FL MiniMax H3 Beat KSampler.",
                ),
                io.Vae.Input(
                    "vae",
                    tooltip="MiniMax H3 video VAE used to decode every shot independently.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="Final frame batch with hard cuts at the authored beat boundaries.",
                )
            ],
        )

    @classmethod
    def execute(cls, sampled_shots, vae):
        shots = _sampled_shots(sampled_shots)
        total_frames = sampled_shots.get("total_frames")
        if not isinstance(total_frames, int) or total_frames <= 0:
            raise ValueError("FL MiniMax H3 Shot Assembler received an invalid total frame count.")

        output = None
        cursor = 0
        progress = comfy.utils.ProgressBar(len(shots))
        for position, shot in enumerate(shots, 1):
            authored_frames = shot.get("authored_frames")
            if not isinstance(authored_frames, int) or authored_frames <= 0:
                raise ValueError(
                    f"FL MiniMax H3 Shot Assembler shot {position} has an invalid authored length."
                )
            images = vae.decode(_video_latent(shot, position))
            if images.ndim == 5:
                images = images.reshape(
                    -1,
                    images.shape[-3],
                    images.shape[-2],
                    images.shape[-1],
                )
            if images.ndim != 4 or images.shape[0] < authored_frames:
                raise ValueError(
                    f"FL MiniMax H3 Shot Assembler decoded too few frames for shot {position}."
                )
            images = images[:authored_frames]
            if output is None:
                output = torch.empty(
                    (total_frames, *images.shape[1:]),
                    dtype=images.dtype,
                    device=images.device,
                )
            elif images.shape[1:] != output.shape[1:]:
                raise ValueError(
                    f"FL MiniMax H3 Shot Assembler shot {position} decoded at a different resolution."
                )
            if cursor + authored_frames > total_frames:
                raise ValueError("FL MiniMax H3 Shot Assembler decoded more than the planned duration.")
            output[cursor:cursor + authored_frames].copy_(images)
            cursor += authored_frames
            progress.update(1)

        if cursor != total_frames:
            raise ValueError(
                f"FL MiniMax H3 Shot Assembler produced {cursor} frames; expected {total_frames}."
            )
        logging.info(
            "FL MiniMax H3 shot assembler: decoded %d independent shots into %d frames.",
            len(shots),
            total_frames,
        )
        return io.NodeOutput(output)
