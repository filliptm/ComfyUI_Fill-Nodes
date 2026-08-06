import logging

import comfy.samplers
import comfy.utils
import nodes
from comfy_api.latest import io


H3ShotPlan = io.Custom("FL_H3_SHOT_PLAN")
H3SampledShots = io.Custom("FL_H3_SAMPLED_SHOTS")
_MAX_SEED = 0xffffffffffffffff


def _validate_shot_plan(plan):
    if not isinstance(plan, dict) or plan.get("type") != "minimax_h3_beat_shot_plan":
        raise TypeError("FL MiniMax H3 Beat KSampler received an invalid shot plan.")
    if plan.get("version") != 1:
        raise ValueError("FL MiniMax H3 Beat KSampler supports shot plan version 1.")
    shots = plan.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("FL MiniMax H3 Beat KSampler requires at least one planned shot.")
    for index, shot in enumerate(shots, 1):
        if not isinstance(shot, dict) or "latent" not in shot or "conditioning" not in shot:
            raise ValueError(f"FL MiniMax H3 Beat KSampler shot {index} is incomplete.")
    return shots


class FL_MiniMaxH3BeatKSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3BeatKSampler",
            display_name="FL MiniMax H3 Beat KSampler",
            category="🏵️Fill Nodes/Ksamplers",
            description=(
                "Samples every MiniMax H3 beat-planned shot independently so temporal attention "
                "cannot smear content across scene cuts."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="MiniMax H3 model used for every independent shot.",
                ),
                H3ShotPlan.Input(
                    "shot_plan",
                    tooltip="Independent shot plan from FL MiniMax H3 Beat Shot Planner.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=_MAX_SEED,
                    control_after_generate=True,
                    tooltip="Base noise seed for the first shot.",
                ),
                io.Combo.Input(
                    "seed_mode",
                    options=["increment", "fixed"],
                    default="increment",
                    tooltip=(
                        "increment adds the shot index to the base seed; fixed reuses the same seed "
                        "for every shot."
                    ),
                ),
                io.Int.Input(
                    "steps",
                    default=20,
                    min=1,
                    max=10000,
                    tooltip="Sampling steps performed independently for every shot.",
                ),
                io.Float.Input(
                    "cfg",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    tooltip="MiniMax H3 guidance scale. The normal H3 workflow uses CFG 1.",
                ),
                io.Combo.Input(
                    "sampler_name",
                    options=comfy.samplers.KSampler.SAMPLERS,
                    tooltip="ComfyUI sampler used for every shot.",
                ),
                io.Combo.Input(
                    "scheduler",
                    options=comfy.samplers.KSampler.SCHEDULERS,
                    tooltip="ComfyUI scheduler used for every shot.",
                ),
                io.Float.Input(
                    "denoise",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Denoise strength applied independently to every planned latent.",
                ),
            ],
            outputs=[
                H3SampledShots.Output(
                    display_name="sampled_shots",
                    tooltip="Independently sampled nested H3 shots, ready for separate VAE decoding.",
                )
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        shot_plan,
        seed,
        seed_mode,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
    ):
        shots = _validate_shot_plan(shot_plan)
        sampled = []
        progress = comfy.utils.ProgressBar(len(shots))

        for position, shot in enumerate(shots):
            shot_seed = seed if seed_mode == "fixed" else (seed + position) & _MAX_SEED
            logging.info(
                "FL MiniMax H3 beat sampler: shot %d/%d, frames %d-%d, seed %d.",
                position + 1,
                len(shots),
                shot["start_frame"],
                shot["end_frame"] - 1,
                shot_seed,
            )
            try:
                latent = nodes.common_ksampler(
                    model,
                    shot_seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    shot["conditioning"],
                    shot["conditioning"],
                    shot["latent"],
                    denoise=denoise,
                )[0]
            except Exception as error:
                raise RuntimeError(
                    "FL MiniMax H3 Beat KSampler failed on "
                    f"shot {position + 1}/{len(shots)} "
                    f"(frames {shot['start_frame']}-{shot['end_frame'] - 1})."
                ) from error
            sampled.append({**shot, "latent": latent, "seed": shot_seed})
            progress.update(1)

        return io.NodeOutput({
            **shot_plan,
            "type": "minimax_h3_sampled_shots",
            "shots": sampled,
        })
