# FL_Audio_Reactive_Envelope: Generate per-frame envelopes from drum detections
import torch
import numpy as np
import json
from typing import Tuple, Dict, Any


class FL_Audio_Reactive_Envelope:
    """
    A ComfyUI node for generating per-frame control envelopes from drum detections.
    Creates ADSR envelopes for kicks, snares, and hi-hats across the entire song.
    """

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("kick_envelope_json", "snare_envelope_json", "hihat_envelope_json")
    FUNCTION = "generate_envelopes"
    CATEGORY = "🏵️Fill Nodes/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "drum_times_json": ("STRING", {"description": "Drum times JSON from FL_Audio_Drum_Detector"}),
                "fps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "description": "Frames per second"
                }),
            },
            "optional": {
                # Kick envelope settings
                "kick_attack_frames": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 30,
                    "step": 1,
                    "description": "Kick attack duration in frames"
                }),
                "kick_decay_frames": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "description": "Kick decay duration in frames"
                }),
                "kick_sustain_level": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "description": "Kick sustain level (0-1)"
                }),
                "kick_release_frames": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "description": "Kick release duration in frames"
                }),
                # Snare envelope settings
                "snare_attack_frames": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 30,
                    "step": 1,
                    "description": "Snare attack duration in frames"
                }),
                "snare_decay_frames": ("INT", {
                    "default": 6,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "description": "Snare decay duration in frames"
                }),
                "snare_sustain_level": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "description": "Snare sustain level (0-1)"
                }),
                "snare_release_frames": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "description": "Snare release duration in frames"
                }),
                # Hi-hat envelope settings
                "hihat_attack_frames": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 30,
                    "step": 1,
                    "description": "Hi-hat attack duration in frames"
                }),
                "hihat_decay_frames": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "description": "Hi-hat decay duration in frames"
                }),
                "hihat_sustain_level": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "description": "Hi-hat sustain level (0-1)"
                }),
                "hihat_release_frames": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "description": "Hi-hat release duration in frames"
                }),
            }
        }

    def generate_envelopes(
        self,
        drum_times_json: str,
        fps: int = 30,
        # Kick params
        kick_attack_frames: int = 1,
        kick_decay_frames: int = 8,
        kick_sustain_level: float = 0.2,
        kick_release_frames: int = 10,
        # Snare params
        snare_attack_frames: int = 1,
        snare_decay_frames: int = 6,
        snare_sustain_level: float = 0.1,
        snare_release_frames: int = 8,
        # Hi-hat params
        hihat_attack_frames: int = 1,
        hihat_decay_frames: int = 4,
        hihat_sustain_level: float = 0.0,
        hihat_release_frames: int = 5,
    ) -> Tuple[str, str, str]:
        """
        Generate per-frame envelopes from drum detections

        Args:
            drum_times_json: JSON string from FL_Audio_Drum_Detector
            fps: Frames per second
            (Additional ADSR parameters for each drum type)

        Returns:
            Tuple of (kick_envelope_json, snare_envelope_json, hihat_envelope_json) as JSON strings
        """
        print(f"\n{'='*60}")
        print(f"[FL Audio Reactive Envelope] DEBUG: Function called")
        print(f"[FL Audio Reactive Envelope] DEBUG: FPS = {fps}")
        print(f"{'='*60}\n")

        try:
            # Parse drum times JSON
            drum_data = json.loads(drum_times_json)
            kick_times = drum_data['kick_times']
            snare_times = drum_data['snare_times']
            hihat_times = drum_data['hihat_times']
            duration = drum_data['duration']

            total_frames = int(duration * fps)

            print(f"[FL Audio Reactive Envelope] DEBUG: Duration = {duration:.2f}s")
            print(f"[FL Audio Reactive Envelope] DEBUG: Total frames = {total_frames}")
            print(f"[FL Audio Reactive Envelope] DEBUG: Kicks = {len(kick_times)}")
            print(f"[FL Audio Reactive Envelope] DEBUG: Snares = {len(snare_times)}")
            print(f"[FL Audio Reactive Envelope] DEBUG: Hi-hats = {len(hihat_times)}")

            # Generate envelopes
            kick_envelope = self._generate_envelope(
                kick_times, total_frames, fps,
                kick_attack_frames, kick_decay_frames, kick_sustain_level, kick_release_frames
            )

            snare_envelope = self._generate_envelope(
                snare_times, total_frames, fps,
                snare_attack_frames, snare_decay_frames, snare_sustain_level, snare_release_frames
            )

            hihat_envelope = self._generate_envelope(
                hihat_times, total_frames, fps,
                hihat_attack_frames, hihat_decay_frames, hihat_sustain_level, hihat_release_frames
            )

            print(f"\n{'='*60}")
            print(f"[FL Audio Reactive Envelope] Envelope generation complete!")
            print(f"[FL Audio Reactive Envelope] Kick envelope: {len(kick_envelope)} frames")
            print(f"[FL Audio Reactive Envelope] Snare envelope: {len(snare_envelope)} frames")
            print(f"[FL Audio Reactive Envelope] Hi-hat envelope: {len(hihat_envelope)} frames")
            print(f"{'='*60}\n")

            # Convert to JSON strings
            kick_json = json.dumps({"envelope": kick_envelope, "total_frames": len(kick_envelope)})
            snare_json = json.dumps({"envelope": snare_envelope, "total_frames": len(snare_envelope)})
            hihat_json = json.dumps({"envelope": hihat_envelope, "total_frames": len(hihat_envelope)})

            return (kick_json, snare_json, hihat_json)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL Audio Reactive Envelope] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return ("{}", "{}", "{}")

    def _generate_envelope(
        self,
        hit_times: list,
        total_frames: int,
        fps: int,
        attack_frames: int,
        decay_frames: int,
        sustain_level: float,
        release_frames: int
    ) -> list:
        """Generate ADSR envelope for a drum type"""
        envelope = [0.0] * total_frames

        for hit_time in hit_times:
            hit_frame = int(hit_time * fps)

            if hit_frame >= total_frames:
                continue

            # Attack phase
            for i in range(attack_frames):
                frame_idx = hit_frame + i
                if frame_idx >= total_frames:
                    break
                # Linear attack to 1.0
                envelope[frame_idx] = max(envelope[frame_idx], i / max(attack_frames, 1))

            # Decay phase
            for i in range(decay_frames):
                frame_idx = hit_frame + attack_frames + i
                if frame_idx >= total_frames:
                    break
                # Linear decay to sustain level
                decay_value = 1.0 - (i / decay_frames) * (1.0 - sustain_level)
                envelope[frame_idx] = max(envelope[frame_idx], decay_value)

            # Sustain phase (hold at sustain level)
            sustain_frame = hit_frame + attack_frames + decay_frames
            if sustain_frame < total_frames:
                envelope[sustain_frame] = max(envelope[sustain_frame], sustain_level)

            # Release phase
            for i in range(release_frames):
                frame_idx = hit_frame + attack_frames + decay_frames + i
                if frame_idx >= total_frames:
                    break
                # Linear release to 0.0
                release_value = sustain_level * (1.0 - i / release_frames)
                envelope[frame_idx] = max(envelope[frame_idx], release_value)

        return envelope
