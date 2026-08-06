import copy
import importlib.util
import pathlib
import unittest

import torch

import comfy.nested_tensor


MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "nodes"
    / "prompting"
    / "FL_MiniMaxH3PromptTimeline.py"
)
SPEC = importlib.util.spec_from_file_location("fl_minimax_h3_prompt_timeline", MODULE_PATH)
timeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timeline)


def conditioning(value):
    return [[torch.full((1, 2, 3), value), {}]]


def latent(height, width):
    video = torch.zeros((1, 24, 7, height, width))
    audio = torch.zeros((1, 32, 2, 37))
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


class TimelineParserTests(unittest.TestCase):
    def test_seconds_and_multiline_prompts(self):
        sections = timeline._parse_timeline(
            """
[00:00.000 - 00:02.000]
First line.
Second line.

[00:02.000 - 00:04.500]
Second section.
""",
            "seconds",
            120.0,
        )

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0]["start"], 0.0)
        self.assertEqual(sections[1]["end"], 4.5)
        self.assertEqual(sections[0]["prompt"], "First line.\nSecond line.")

    def test_frame_and_beat_positions(self):
        frames = timeline._parse_timeline("[0 - 48]\nFrames.", "frames", 120.0)
        beats = timeline._parse_timeline("[0 - 8]\nBeats.", "beats", 120.0)

        self.assertEqual(frames[0]["end"], 2.0)
        self.assertEqual(beats[0]["end"], 4.0)

    def test_text_before_header_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must follow a time header"):
            timeline._parse_timeline("orphan prompt", "seconds", 120.0)

    def test_overlap_is_rejected(self):
        sections = timeline._parse_timeline(
            "[0 - 2]\nFirst.\n[1 - 3]\nSecond.",
            "seconds",
            120.0,
        )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            timeline._resolve_sections(sections, 4.0, "strict", "hard", 0)

    def test_duration_policies(self):
        sections = timeline._parse_timeline("[0 - 4]\nSection.", "seconds", 120.0)

        with self.assertRaisesRegex(ValueError, "aligned H3 latent"):
            timeline._resolve_sections(sections, 2.0, "strict", "hard", 0)
        clamped = timeline._resolve_sections(sections, 2.0, "clamp", "hard", 0)
        fitted = timeline._resolve_sections(sections, 2.0, "fit", "hard", 0)

        self.assertEqual(clamped[0]["end"], 2.0)
        self.assertEqual(fitted[0]["end"], 2.0)


class TimelineMaskTests(unittest.TestCase):
    def setUp(self):
        self.duration = 22 / 24
        self.sections = [
            {"start": 0.0, "end": 5 / 24, "prompt": "First."},
            {"start": 5 / 24, "end": self.duration, "prompt": "Second."},
        ]

    def test_video_weights_follow_h3_temporal_grid(self):
        video, audio = timeline._temporal_weights(
            self.sections,
            video_t=7,
            audio_t=37,
            transition_mode="hard",
            transition_frames=0,
            affect_audio="video and audio",
        )

        self.assertEqual(video[0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(video[1], [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        for first, second in zip(audio[0], audio[1]):
            self.assertAlmostEqual(first + second, 1.0)

    def test_cosine_transition_keeps_unit_coverage(self):
        video, _ = timeline._temporal_weights(
            self.sections,
            video_t=7,
            audio_t=37,
            transition_mode="cosine",
            transition_frames=4,
            affect_audio="video only",
        )

        for first, second in zip(video[0], video[1]):
            self.assertAlmostEqual(first + second, 1.0)

    def test_video_only_leaves_audio_for_global_conditioning(self):
        _, audio = timeline._temporal_weights(
            self.sections,
            video_t=7,
            audio_t=37,
            transition_mode="hard",
            transition_frames=0,
            affect_audio="video only",
        )

        self.assertTrue(all(weight == 0.0 for section in audio for weight in section))

    def test_schedule_fades_move_between_global_and_scheduled_prompt(self):
        sections = [{
            "start": 0.0,
            "end": 1.0,
            "fade_in_end": 0.25,
            "fade_out_start": 0.75,
            "curve": "linear",
            "prompt": "Transform.",
        }]

        self.assertAlmostEqual(
            timeline._weights_at_time(sections, 0.125, "hard", 0.0)[0],
            0.5,
        )
        self.assertEqual(
            timeline._weights_at_time(sections, 0.5, "hard", 0.0)[0],
            1.0,
        )
        self.assertAlmostEqual(
            timeline._weights_at_time(sections, 0.875, "hard", 0.0)[0],
            0.5,
        )

    def test_schedule_crossfade_blends_adjacent_prompts_with_unit_coverage(self):
        sections = [
            {
                "start": 0.0,
                "end": 1.0,
                "fade_in_end": 0.0,
                "fade_out_start": 1.0,
                "crossfade_start": 0.0,
                "crossfade_end": 0.0,
                "curve": "cosine",
                "prompt": "First.",
            },
            {
                "start": 1.0,
                "end": 2.0,
                "fade_in_end": 1.0,
                "fade_out_start": 2.0,
                "crossfade_start": 0.75,
                "crossfade_end": 1.25,
                "curve": "cosine",
                "prompt": "Second.",
            },
        ]

        for seconds in (0.75, 0.875, 1.0, 1.125, 1.249):
            first, second = timeline._weights_at_time(sections, seconds, "hard", 0.0)
            self.assertAlmostEqual(first + second, 1.0)
        self.assertEqual(timeline._weights_at_time(sections, 0.75, "hard", 0.0), [1.0, 0.0])
        midpoint = timeline._weights_at_time(sections, 1.0, "hard", 0.0)
        self.assertAlmostEqual(midpoint[0], 0.5)
        self.assertAlmostEqual(midpoint[1], 0.5)

    def test_schedule_crossfade_can_cover_video_and_audio_tokens(self):
        sections = [
            {
                "start": 0.0,
                "end": 5 / 24,
                "fade_in_end": 0.0,
                "fade_out_start": 5 / 24,
                "crossfade_start": 0.0,
                "crossfade_end": 0.0,
                "curve": "linear",
                "prompt": "First.",
            },
            {
                "start": 5 / 24,
                "end": self.duration,
                "fade_in_end": 5 / 24,
                "fade_out_start": self.duration,
                "crossfade_start": 3 / 24,
                "crossfade_end": 7 / 24,
                "curve": "linear",
                "prompt": "Second.",
            },
        ]

        video, audio = timeline._temporal_weights(
            sections,
            video_t=7,
            audio_t=37,
            transition_mode="hard",
            transition_frames=0,
            affect_audio="video and audio",
        )

        for first, second in zip(video[0], video[1]):
            self.assertAlmostEqual(first + second, 1.0)
        for first, second in zip(audio[0], audio[1]):
            self.assertAlmostEqual(first + second, 1.0)

    def test_repeated_prompt_crossfade_keeps_full_group_weight(self):
        merged = timeline._merge_section_weights(
            [
                [1.0, 0.75, 0.5, 0.25, 0.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            ],
            [0, 1],
        )

        self.assertEqual(merged, [1.0] * 5)

    def test_prompt_envelope_is_averaged_on_h3_temporal_grid(self):
        envelope = {
            "prompt": "Pulse.",
            "weights": [0.0, 4.0, 0.0, 0.0, 0.0],
            "fps": 24.0,
            "duration": 5 / 24,
        }
        video, audio = timeline._envelope_temporal_weights(
            envelope,
            video_t=2,
            audio_t=8,
            affect_audio="video only",
        )

        self.assertEqual(video[0], 0.0)
        self.assertEqual(video[1], 1.0)
        self.assertTrue(all(weight == 0.0 for weight in audio))

    def test_prompt_envelope_can_affect_audio_tokens(self):
        envelope = {
            "prompt": "Pulse.",
            "weights": [3.0] * 24,
            "fps": 24.0,
            "duration": 1.0,
        }
        _, audio = timeline._envelope_temporal_weights(
            envelope,
            video_t=7,
            audio_t=37,
            affect_audio="video and audio",
        )

        self.assertTrue(all(weight == 3.0 for weight in audio))

    def test_apply_rebuilds_masks_for_spatial_resolution(self):
        timeline_object = {
            "type": "minimax_h3_prompt_timeline",
            "video_t": 7,
            "audio_t": 37,
            "sections": self.sections,
            "conditioning_groups": [
                {"conditioning": conditioning(1.0), "section_indices": [0]},
                {"conditioning": conditioning(2.0), "section_indices": [1]},
            ],
            "global_conditioning": conditioning(0.0),
            "transition_mode": "hard",
            "transition_frames": 0,
            "affect_audio": "video and audio",
        }

        small = timeline._apply_timeline(timeline_object, latent(4, 6))
        large = timeline._apply_timeline(timeline_object, latent(8, 12))

        expected_small = 24 * 7 * 4 * 6 + 32 * 2 * 37
        expected_large = 24 * 7 * 8 * 12 + 32 * 2 * 37
        self.assertEqual(small[0][1]["mask"].shape, (1, expected_small))
        self.assertEqual(large[0][1]["mask"].shape, (1, expected_large))
        self.assertTrue(small[-1][1]["default"])
        self.assertTrue(large[-1][1]["default"])

    def test_apply_rejects_temporal_shape_changes(self):
        timeline_object = {
            "type": "minimax_h3_prompt_timeline",
            "video_t": 2,
            "audio_t": 8,
            "sections": [],
            "conditioning_groups": [],
            "global_conditioning": conditioning(0.0),
            "transition_mode": "hard",
            "transition_frames": 0,
            "affect_audio": "video only",
        }

        with self.assertRaisesRegex(ValueError, "same video and audio duration"):
            timeline._apply_timeline(timeline_object, latent(4, 6))

    def test_apply_adds_reactive_mask_without_timeline_sections(self):
        timeline_object = {
            "type": "minimax_h3_prompt_timeline",
            "video_t": 7,
            "audio_t": 37,
            "sections": [],
            "conditioning_groups": [],
            "prompt_envelopes": [{
                "prompt": "Pulse.",
                "weights": [0.0, 3.0] + [0.0] * 20,
                "fps": 24.0,
                "duration": 22 / 24,
            }],
            "prompt_envelope_groups": [{
                "conditioning": conditioning(3.0),
                "envelope_indices": [0],
            }],
            "global_conditioning": conditioning(0.0),
            "transition_mode": "hard",
            "transition_frames": 0,
            "affect_audio": "video only",
        }

        applied = timeline._apply_timeline(timeline_object, latent(4, 6))

        self.assertEqual(len(applied), 2)
        self.assertGreater(applied[0][1]["mask"].max().item(), 0.0)
        self.assertTrue(applied[-1][1]["default"])


class H3BoundaryAlignmentTests(unittest.TestCase):
    @staticmethod
    def section(start_frame, end_frame, prompt):
        start = start_frame / 24
        end = end_frame / 24
        return {
            "start": start,
            "end": end,
            "fade_in_end": start,
            "fade_out_start": end,
            "crossfade_start": start,
            "crossfade_end": start,
            "curve": "linear",
            "prompt": prompt,
        }

    def test_video_token_edges_match_h3_frame_grid(self):
        edges = timeline._video_token_edges(67)

        self.assertEqual(edges[:7], [0, 1, 5, 9, 13, 17, 18])
        self.assertEqual(edges[-1], 226)

    def test_hard_boundaries_align_without_mutating_authored_sections(self):
        sections = [
            self.section(0, 54, "First."),
            self.section(54, 109, "Second."),
            self.section(109, 163, "Third."),
            self.section(163, 217, "Fourth."),
        ]
        authored = copy.deepcopy(sections)

        aligned, adjustments, extended = timeline._align_h3_sections(
            sections,
            authored_frame_count=217,
            frame_count=226,
            video_t=67,
            transition_mode="hard",
        )

        self.assertEqual(sections, authored)
        self.assertEqual(
            adjustments,
            [
                {"authored_frame": 54, "aligned_frame": 56, "offset_frames": 2},
                {"authored_frame": 109, "aligned_frame": 111, "offset_frames": 2},
                {"authored_frame": 163, "aligned_frame": 162, "offset_frames": -1},
            ],
        )
        self.assertEqual(
            [round(section["start"] * 24) for section in aligned],
            [0, 56, 111, 162],
        )
        self.assertEqual(
            [round(section["end"] * 24) for section in aligned],
            [56, 111, 162, 226],
        )
        self.assertTrue(extended)

        video, _ = timeline._temporal_weights(
            aligned,
            video_t=67,
            audio_t=377,
            transition_mode="hard",
            transition_frames=0,
            affect_audio="video only",
        )
        for token_weights in zip(*video):
            self.assertEqual(sum(token_weights), 1.0)
            self.assertTrue(all(weight in (0.0, 1.0) for weight in token_weights))

    def test_crossfade_boundary_is_preserved(self):
        sections = [
            self.section(0, 54, "First."),
            self.section(54, 109, "Second."),
        ]
        sections[1]["crossfade_start"] = 27 / 24
        sections[1]["crossfade_end"] = 81 / 24

        aligned, adjustments, _ = timeline._align_h3_sections(
            sections,
            authored_frame_count=109,
            frame_count=124,
            video_t=37,
            transition_mode="hard",
        )

        self.assertEqual(round(aligned[0]["end"] * 24), 54)
        self.assertEqual(round(aligned[1]["start"] * 24), 54)
        self.assertEqual(round(aligned[1]["crossfade_start"] * 24), 27)
        self.assertEqual(round(aligned[1]["crossfade_end"] * 24), 81)
        self.assertEqual(adjustments, [])

    def test_fade_lengths_move_with_aligned_boundary(self):
        sections = [
            self.section(0, 54, "First."),
            self.section(54, 109, "Second."),
        ]
        sections[0]["fade_out_start"] = 50 / 24
        sections[1]["fade_in_end"] = 60 / 24

        aligned, _, _ = timeline._align_h3_sections(
            sections,
            authored_frame_count=109,
            frame_count=124,
            video_t=37,
            transition_mode="hard",
        )

        self.assertEqual(round(aligned[0]["fade_out_start"] * 24), 52)
        self.assertEqual(round(aligned[1]["fade_in_end"] * 24), 62)

    def test_intentional_tail_is_not_extended(self):
        sections = [self.section(0, 209, "First.")]

        aligned, _, extended = timeline._align_h3_sections(
            sections,
            authored_frame_count=217,
            frame_count=226,
            video_t=67,
            transition_mode="hard",
        )

        self.assertEqual(round(aligned[0]["end"] * 24), 209)
        self.assertFalse(extended)

    def test_soft_transition_keeps_authored_boundary(self):
        sections = [
            self.section(0, 54, "First."),
            self.section(54, 109, "Second."),
        ]

        aligned, adjustments, _ = timeline._align_h3_sections(
            sections,
            authored_frame_count=109,
            frame_count=111,
            video_t=33,
            transition_mode="cosine",
        )

        self.assertEqual(adjustments, [])
        self.assertEqual(round(aligned[0]["end"] * 24), 54)
        self.assertEqual(round(aligned[1]["start"] * 24), 54)


class TimelineNodeTests(unittest.TestCase):
    def test_schema_exposes_native_tooltips(self):
        for node in (
            timeline.FL_MiniMaxH3PromptTimeline,
            timeline.FL_MiniMaxH3ApplyTimeline,
        ):
            schema = node.define_schema()
            self.assertTrue(schema.description)
            self.assertTrue(all(input.tooltip for input in schema.inputs))
            self.assertTrue(all(output.tooltip for output in schema.outputs))

    def test_node_builds_native_h3_outputs_without_references(self):
        class Clip:
            def tokenize(self, prompt, **kwargs):
                return prompt

            def encode_from_tokens_scheduled(self, tokens):
                return [[torch.zeros((1, 2, 3)), {"prompt": tokens}]]

        output = timeline.FL_MiniMaxH3PromptTimeline.execute(
            clip=Clip(),
            vae=None,
            audio_vae=None,
            global_prompt="same subject",
            timeline=(
                "[0 - 0.1]\nturns toward camera\n"
                "[0.1 - 0.2]\nturns toward camera"
            ),
            width=32,
            height=32,
            length=5,
            time_unit="seconds",
            bpm=120.0,
            transition_mode="hard",
            transition_frames=0,
            affect_audio="video only",
            duration_policy="strict",
            ref_image_size="match",
        ).result

        video, audio = output[1]["samples"].unbind()
        self.assertEqual(video.shape, (1, 24, 2, 2, 2))
        self.assertEqual(audio.shape, (1, 32, 2, 8))
        self.assertEqual(output[3]["type"], "minimax_h3_prompt_timeline")
        self.assertEqual(len(output[3]["conditioning_groups"]), 1)
        self.assertEqual(output[3]["authored_frame_count"], 5)
        self.assertEqual(output[3]["padding_frames"], 0)
        self.assertEqual(
            output[3]["boundary_adjustments"],
            [{"authored_frame": 2, "aligned_frame": 1, "offset_frames": -1}],
        )
        self.assertIn("mask", output[0][0][1])
        self.assertTrue(output[0][-1][1]["default"])

    def test_external_schedule_overrides_manual_timeline(self):
        class Clip:
            def tokenize(self, prompt, **kwargs):
                return prompt

            def encode_from_tokens_scheduled(self, tokens):
                return [[torch.zeros((1, 2, 3)), {"prompt": tokens}]]

        schedule = {
            "type": "fl_prompt_schedule",
            "version": 1,
            "duration": 0.2,
            "sections": [{
                "line": 1,
                "start": 0.0,
                "end": 0.2,
                "fade_in_end": 0.05,
                "fade_out_start": 0.15,
                "curve": "cosine",
                "prompt": "Beat-timed action.",
            }],
        }
        output = timeline.FL_MiniMaxH3PromptTimeline.execute(
            clip=Clip(),
            vae=None,
            audio_vae=None,
            global_prompt="same subject",
            timeline="not a valid manual timeline",
            prompt_schedule=schedule,
            width=32,
            height=32,
            length=5,
            time_unit="seconds",
            bpm=120.0,
            transition_mode="hard",
            transition_frames=0,
            affect_audio="video and audio",
            duration_policy="strict",
            ref_image_size="match",
        ).result

        self.assertEqual(output[3]["sections"][0]["prompt"], "Beat-timed action.")
        self.assertEqual(output[3]["transition_frames"], 0)

    def test_external_schedule_version_two_keeps_crossfade_boundaries(self):
        schedule = {
            "type": "fl_prompt_schedule",
            "version": 2,
            "duration": 2.0,
            "sections": [
                {
                    "line": 1,
                    "start": 0.0,
                    "end": 1.0,
                    "fade_in_end": 0.0,
                    "fade_out_start": 1.0,
                    "crossfade_start": 0.0,
                    "crossfade_end": 0.0,
                    "curve": "cosine",
                    "prompt": "First.",
                },
                {
                    "line": 3,
                    "start": 1.0,
                    "end": 2.0,
                    "fade_in_end": 1.0,
                    "fade_out_start": 2.0,
                    "crossfade_start": 0.75,
                    "crossfade_end": 1.25,
                    "curve": "cosine",
                    "prompt": "Second.",
                },
            ],
        }

        sections = timeline._schedule_sections(schedule)

        self.assertEqual(sections[1]["crossfade_start"], 0.75)
        self.assertEqual(sections[1]["crossfade_end"], 1.25)

    def test_node_accepts_multiple_prompt_envelopes_and_deduplicates_prompts(self):
        class Clip:
            def tokenize(self, prompt, **kwargs):
                return prompt

            def encode_from_tokens_scheduled(self, tokens):
                return [[torch.zeros((1, 2, 3)), {"prompt": tokens}]]

        envelope = {
            "type": "fl_prompt_envelope",
            "version": 1,
            "duration": 0.2,
            "fps": 24.0,
            "prompt": "Pulse outward.",
            "weights": [0.0, 3.0, 0.0, 0.0, 0.0],
        }
        output = timeline.FL_MiniMaxH3PromptTimeline.execute(
            clip=Clip(),
            vae=None,
            audio_vae=None,
            global_prompt="same subject",
            timeline="",
            prompt_envelopes={
                "prompt_envelope_0": envelope,
                "prompt_envelope_1": envelope,
            },
            width=32,
            height=32,
            length=5,
            time_unit="seconds",
            bpm=120.0,
            transition_mode="hard",
            transition_frames=0,
            affect_audio="video only",
            duration_policy="strict",
            ref_image_size="match",
        ).result

        self.assertEqual(len(output[3]["prompt_envelopes"]), 2)
        self.assertEqual(len(output[3]["prompt_envelope_groups"]), 1)
        self.assertIn("Audio-reactive accents:", output[2][0][1]["prompt"])


if __name__ == "__main__":
    unittest.main()
