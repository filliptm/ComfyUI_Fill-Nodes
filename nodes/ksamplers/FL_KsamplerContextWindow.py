import logging
import time

import torch

import comfy.context_windows
import comfy.latent_formats
import comfy.nested_tensor
import comfy.samplers
from comfy_execution.utils import get_executing_context
from comfy_extras import nodes_minimax_h3
from nodes import VAEEncode, common_ksampler
from server import PromptServer

from ._vae_helpers import safe_vae_decode


CONTEXT_SCHEDULES = [
    comfy.context_windows.ContextSchedules.STATIC_STANDARD,
    comfy.context_windows.ContextSchedules.UNIFORM_STANDARD,
    comfy.context_windows.ContextSchedules.UNIFORM_LOOPED,
    comfy.context_windows.ContextSchedules.BATCHED,
]

FUSE_METHODS = comfy.context_windows.ContextFuseMethods.LIST_STATIC

TEMPORAL_UNITS = [
    "auto",
    "wan",
    "ltx",
    "minimax_h3",
    "video_frames_4n_plus_1",
    "latent_frames",
]


class FLSafeIndexListContextHandler(comfy.context_windows.IndexListContextHandler):
    def __init__(self, *args, node_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_id = str(node_id) if node_id is not None else None
        self._total_steps = 1
        self._last_total_windows = 1
        self._last_progress_value = 0
        self._last_event_at = 0.0

    def set_step(self, timestep: torch.Tensor, model_options: dict[str]):
        sample_sigmas = model_options.get("transformer_options", {}).get("sample_sigmas")
        if sample_sigmas is None:
            return
        self._total_steps = max(1, int(sample_sigmas.numel()) - 1)
        current_timestep = timestep[0].to(device=sample_sigmas.device, dtype=sample_sigmas.dtype)
        mask = torch.isclose(sample_sigmas, current_timestep, rtol=0.0001)
        matches = torch.nonzero(mask)
        if torch.numel(matches) == 0:
            return
        self._step = int(matches[0].item())

    def get_context_windows(self, model, x_in: torch.Tensor, model_options: dict):
        context_windows = super().get_context_windows(model, x_in, model_options)
        self._last_total_windows = max(1, len(context_windows))
        return context_windows

    def combine_context_window_results(
        self,
        x_in: torch.Tensor,
        sub_conds_out,
        sub_conds,
        window,
        window_idx: int,
        total_windows: int,
        timestep: torch.Tensor,
        conds_final,
        counts_final,
        biases_final,
    ):
        result = super().combine_context_window_results(
            x_in,
            sub_conds_out,
            sub_conds,
            window,
            window_idx,
            total_windows,
            timestep,
            conds_final,
            counts_final,
            biases_final,
        )
        self._emit_progress(window_idx, max(1, total_windows), window)
        return result

    def emit_done(self):
        if self.node_id is None:
            return
        max_value = max(1, self._total_steps * self._last_total_windows)
        self._send_event(
            {
                "node": self.node_id,
                "status": "done",
                "value": max_value,
                "max": max_value,
                "step": self._total_steps,
                "total_steps": self._total_steps,
                "window_index": self._last_total_windows,
                "total_windows": self._last_total_windows,
                "window": [],
            }
        )

    def emit_settings(self, settings):
        if self.node_id is not None:
            self._send_event({"node": self.node_id, "status": "resolved", **settings})

    def _emit_progress(self, window_idx: int, total_windows: int, window):
        if self.node_id is None:
            return

        max_value = max(1, self._total_steps * total_windows)
        raw_value = min(max_value, self._step * total_windows + window_idx + 1)
        value = max(self._last_progress_value, raw_value)
        self._last_progress_value = value

        now = time.monotonic()
        if now - self._last_event_at < 0.25 and value < max_value:
            return
        self._last_event_at = now

        self._send_event(
            {
                "node": self.node_id,
                "status": "running",
                "value": value,
                "max": max_value,
                "step": min(self._step + 1, self._total_steps),
                "total_steps": self._total_steps,
                "window_index": window_idx + 1,
                "total_windows": total_windows,
                "window": list(getattr(window, "index_list", [])),
            }
        )

    @staticmethod
    def _send_event(payload: dict):
        server = getattr(PromptServer, "instance", None)
        if server is not None:
            server.send_sync("fl_context_window_progress", payload)


class FL_KsamplerContextWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "context_length": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 1, "tooltip": "Requested video frames per window in Auto mode. Snaps down to the model's frame grid; does not change the clip duration."}),
                "context_overlap": ("INT", {"default": 30, "min": 0, "max": 10000, "tooltip": "Requested shared video frames. Snaps down to latent boundaries. Ignored by the batched schedule."}),
                "context_schedule": (CONTEXT_SCHEDULES, {"default": comfy.context_windows.ContextSchedules.STATIC_STANDARD}),
                "context_stride": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "fuse_method": (FUSE_METHODS, {"default": comfy.context_windows.ContextFuseMethods.PYRAMID}),
                "temporal_unit": (TEMPORAL_UNITS, {"default": "auto", "tooltip": "Auto reads the connected model. Explicit profiles use video frames; latent_frames uses raw latent positions. video_frames_4n_plus_1 preserves the original conversion for saved workflows."}),
                "closed_loop": ("BOOLEAN", {"default": False}),
                "freenoise": ("BOOLEAN", {"default": False}),
                "causal_window_fix": ("BOOLEAN", {"default": True}),
                "temporal_dim": ("INT", {"default": 2, "min": 0, "max": 5}),
                "cond_retain_index_list": ("STRING", {"default": "", "multiline": False}),
                "split_conds_to_windows": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "IMAGE", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "vae", "image", "debug_info")
    FUNCTION = "sample"
    CATEGORY = "🏵️Fill Nodes/Ksamplers"

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        context_length,
        context_overlap,
        context_schedule,
        context_stride,
        fuse_method,
        temporal_unit,
        vae=None,
        image=None,
        closed_loop=False,
        freenoise=False,
        causal_window_fix=True,
        temporal_dim=2,
        cond_retain_index_list="",
        split_conds_to_windows=False,
        unique_id=None,
    ):
        try:
            node_id = unique_id
            if node_id is None:
                context = get_executing_context()
                if context is not None:
                    node_id = context.node_id

            if image is not None:
                if vae is None:
                    raise ValueError("FL_KsamplerContextWindow: image input requires a VAE.")
                latent_image = VAEEncode().encode(vae, image)[0]

            if latent_image is None or "samples" not in latent_image:
                raise ValueError("FL_KsamplerContextWindow: latent_image must contain samples.")

            samples = latent_image["samples"]
            if isinstance(samples, comfy.nested_tensor.NestedTensor):
                primary_samples = samples.unbind()[0]
            elif isinstance(samples, torch.Tensor):
                primary_samples = samples
            else:
                raise ValueError("FL_KsamplerContextWindow: latent samples must be a tensor or nested tensor.")

            if temporal_dim >= primary_samples.ndim:
                raise ValueError(
                    "FL_KsamplerContextWindow: temporal_dim is outside the latent sample shape. "
                    f"Got temporal_dim={temporal_dim}, shape={tuple(primary_samples.shape)}."
                )

            if primary_samples.ndim == 4 and temporal_dim != 0:
                raise ValueError(
                    "FL_KsamplerContextWindow: expected a 5D video latent [B, C, T, H, W]. "
                    "For 4D latents, set temporal_dim=0 only if you intentionally want to window over batch."
                )

            settings = self._resolve_context(model, context_length, context_overlap, temporal_unit, context_schedule)
            latent_context_length = settings["latent_length"]
            latent_context_overlap = settings["latent_overlap"]
            self._validate_context(latent_context_length, latent_context_overlap)
            if temporal_unit not in ("latent_frames", "video_frames_4n_plus_1") and temporal_dim != 2:
                raise ValueError("FL_KsamplerContextWindow: model-aware video windows use temporal_dim=2. Use latent_frames for another axis.")
            total_temporal = primary_samples.shape[temporal_dim]
            context_active = total_temporal > latent_context_length
            if settings["profile"] == "minimax_h3":
                self._validate_h3(samples, total_temporal)
                if context_active and context_schedule != comfy.context_windows.ContextSchedules.STATIC_STANDARD:
                    raise ValueError("FL_KsamplerContextWindow: MiniMax H3 model-aware windows currently require standard_static.")
                if freenoise:
                    raise ValueError("FL_KsamplerContextWindow: disable FreeNoise for MiniMax H3; independent audio shuffling is not supported.")
            settings.update({
                "requested_length": context_length,
                "requested_overlap": context_overlap,
                "total_latents": total_temporal,
                "context_active": context_active,
                "anchor_latents": int(causal_window_fix and context_active),
                "schedule": context_schedule,
            })

            context_model = model.clone()
            context_model.model_options["context_handler"] = FLSafeIndexListContextHandler(
                context_schedule=comfy.context_windows.get_matching_context_schedule(context_schedule),
                fuse_method=comfy.context_windows.get_matching_fuse_method(fuse_method),
                context_length=latent_context_length,
                context_overlap=latent_context_overlap,
                context_stride=context_stride,
                closed_loop=closed_loop,
                dim=temporal_dim,
                freenoise=freenoise,
                cond_retain_index_list=cond_retain_index_list,
                split_conds_to_windows=split_conds_to_windows,
                causal_window_fix=causal_window_fix,
                node_id=node_id,
            )
            context_handler = context_model.model_options["context_handler"]
            context_handler.emit_settings(settings)
            comfy.context_windows.create_prepare_sampling_wrapper(context_model)
            if freenoise:
                comfy.context_windows.create_sampler_sample_wrapper(context_model)

            sampled = common_ksampler(
                context_model,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise,
            )[0]
            context_handler.emit_done()

            output_image = None
            if vae is not None:
                output_image = safe_vae_decode(vae, sampled, node_name="FL_KsamplerContextWindow")

            debug_info = self._debug_info(
                samples=primary_samples,
                temporal_dim=temporal_dim,
                context_length=context_length,
                context_overlap=context_overlap,
                latent_context_length=latent_context_length,
                latent_context_overlap=latent_context_overlap,
                context_schedule=context_schedule,
                context_stride=context_stride,
                closed_loop=closed_loop,
                fuse_method=fuse_method,
                freenoise=freenoise,
                causal_window_fix=causal_window_fix,
                temporal_unit=temporal_unit,
            )
            debug_info += (
                f"\n- resolved_model_profile: {settings['profile']}"
                f"\n- effective_video_frames: {settings['video_frames']}"
                f"\n- effective_overlap_frames: {settings['overlap_frames']}"
                f"\n- additional_anchor_latents: {settings['anchor_latents']}"
            )

            return (model, positive, negative, sampled, vae, output_image, debug_info)

        except Exception as e:
            logging.error(f"Error in FL_KsamplerContextWindow: {e}")
            raise

    @staticmethod
    def _resolve_context(model, context_length, context_overlap, temporal_unit, context_schedule):
        length, overlap = int(context_length), int(context_overlap)
        if length < 1 or overlap < 0:
            raise ValueError("FL_KsamplerContextWindow: window length must be positive and overlap cannot be negative.")
        profile = temporal_unit
        if profile == "auto":
            latent_format = model.get_model_object("latent_format")
            if isinstance(latent_format, comfy.latent_formats.MiniMaxH3Video):
                profile = "minimax_h3"
            elif isinstance(latent_format, comfy.latent_formats.LTXV):
                profile = "ltx"
            elif isinstance(latent_format, comfy.latent_formats.Wan21):
                profile = "wan"
            else:
                raise ValueError("FL_KsamplerContextWindow: Auto does not recognize this model. Select a frame profile or latent_frames in Advanced.")

        video_frames = overlap_frames = None
        if profile in ("latent_frames", "video_frames_4n_plus_1"):
            latent_length, latent_overlap = FL_KsamplerContextWindow._convert_context_units(length, overlap, profile)
            if profile == "video_frames_4n_plus_1":
                video_frames = (latent_length - 1) * 4 + 1
                overlap_frames = (latent_overlap - 1) * 4 + 1 if latent_overlap else 0
        elif profile in ("wan", "ltx"):
            ratio = 4 if profile == "wan" else 8
            latent_length = (length - 1) // ratio + 1
            latent_overlap = overlap // ratio
            video_frames = (latent_length - 1) * ratio + 1
            overlap_frames = latent_overlap * ratio
        elif profile == "minimax_h3":
            if length < 5:
                raise ValueError("FL_KsamplerContextWindow: MiniMax H3 requires a window of at least 5 video frames.")
            video_frames = nodes_minimax_h3.align_frame_count(length)
            if video_frames > length:
                video_frames -= 17
            latent_length = nodes_minimax_h3.video_latent_t(video_frames)
            # Bound the shared span at any phase of H3's repeating (1, 4, 4, 4, 4) token durations.
            cycles, remainder = divmod(overlap, 17)
            latent_overlap = cycles * 5 + remainder // 4
            overlap_frames = cycles * 17 + (remainder // 4) * 4
        else:
            raise ValueError(f"FL_KsamplerContextWindow: unknown temporal_unit '{temporal_unit}'.")

        if context_schedule == comfy.context_windows.ContextSchedules.BATCHED:
            latent_overlap = overlap_frames = 0
        return {
            "profile": profile,
            "latent_length": latent_length,
            "latent_overlap": latent_overlap,
            "video_frames": video_frames,
            "overlap_frames": overlap_frames,
        }

    @staticmethod
    def _validate_h3(samples, video_t):
        if not isinstance(samples, comfy.nested_tensor.NestedTensor) or len(samples.unbind()) != 2:
            raise ValueError("FL_KsamplerContextWindow: MiniMax H3 requires its native video/audio latent pair.")
        video, audio = samples.unbind()
        if video.ndim != 5 or video.shape[1] != 24 or audio.ndim != 4 or audio.shape[1:3] != (32, 2):
            raise ValueError("FL_KsamplerContextWindow: invalid MiniMax H3 video/audio latent shapes.")
        if video_t < 2 or (video_t - 2) % 5:
            raise ValueError("FL_KsamplerContextWindow: MiniMax H3 clip latents must have 5n+2 video positions. Align the length in the latent creator or H3 planner.")
        frame_count = (video_t - 2) // 5 * 17 + 5
        if audio.shape[-1] != round(frame_count / nodes_minimax_h3.FPS * nodes_minimax_h3.AUDIO_LATENT_FPS):
            raise ValueError("FL_KsamplerContextWindow: MiniMax H3 audio and video durations do not match. Recreate the paired latent with the H3 planner.")

    @staticmethod
    def _convert_context_units(context_length, context_overlap, temporal_unit):
        if temporal_unit == "latent_frames":
            return int(context_length), int(context_overlap)
        if temporal_unit == "video_frames_4n_plus_1":
            latent_length = max(((int(context_length) - 1) // 4) + 1, 1)
            latent_overlap = max(((int(context_overlap) - 1) // 4) + 1, 0) if context_overlap > 0 else 0
            return latent_length, latent_overlap
        raise ValueError(f"FL_KsamplerContextWindow: unknown temporal_unit '{temporal_unit}'.")

    @staticmethod
    def _validate_context(context_length, context_overlap):
        if context_length < 1:
            raise ValueError("FL_KsamplerContextWindow: context_length must be at least 1.")
        if context_overlap < 0:
            raise ValueError("FL_KsamplerContextWindow: context_overlap cannot be negative.")
        if context_overlap >= context_length:
            raise ValueError("FL_KsamplerContextWindow: context_overlap must be smaller than context_length.")

    @staticmethod
    def _debug_info(
        samples,
        temporal_dim,
        context_length,
        context_overlap,
        latent_context_length,
        latent_context_overlap,
        context_schedule,
        context_stride,
        closed_loop,
        fuse_method,
        freenoise,
        causal_window_fix,
        temporal_unit,
    ):
        total_temporal = samples.shape[temporal_dim]
        context_active = total_temporal > latent_context_length
        return (
            "FL Context Window KSampler\n"
            f"- latent_shape: {tuple(samples.shape)}\n"
            f"- temporal_dim: {temporal_dim}\n"
            f"- total_temporal_length: {total_temporal}\n"
            f"- temporal_unit: {temporal_unit}\n"
            f"- requested_context_length: {context_length}\n"
            f"- requested_context_overlap: {context_overlap}\n"
            f"- effective_latent_context_length: {latent_context_length}\n"
            f"- effective_latent_context_overlap: {latent_context_overlap}\n"
            f"- context_schedule: {context_schedule}\n"
            f"- context_stride: {context_stride}\n"
            f"- closed_loop: {closed_loop}\n"
            f"- fuse_method: {fuse_method}\n"
            f"- freenoise: {freenoise}\n"
            f"- causal_window_fix: {causal_window_fix}\n"
            f"- context_active: {context_active}"
        )
