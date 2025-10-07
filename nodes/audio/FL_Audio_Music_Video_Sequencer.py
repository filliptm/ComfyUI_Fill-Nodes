# FL_Audio_Music_Video_Sequencer: Generate complete shot sequence for music videos
import torch
import numpy as np
import json
from typing import Tuple, Dict, Any, List


class FL_Audio_Music_Video_Sequencer:
    """
    A ComfyUI node for generating complete music video shot sequences.
    Takes beat positions and a pattern, outputs a full edit list for the entire song.
    """

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("sequence_json", "total_shots")
    FUNCTION = "generate_sequence"
    CATEGORY = "🏵️Fill Nodes/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"description": "Input audio tensor"}),
                "beat_positions": ("STRING", {"description": "Beat positions JSON from FL_BPM_Analyzer"}),
                "pattern_A": ("STRING", {
                    "default": "4,4,8,4,4,8",
                    "multiline": False,
                    "description": "Pattern A: Comma-separated beat counts (e.g., '4,4,8')"
                }),
            },
            "optional": {
                "pattern_B": ("STRING", {
                    "default": "2,2,2,2",
                    "multiline": False,
                    "description": "Pattern B: Comma-separated beat counts"
                }),
                "pattern_C": ("STRING", {
                    "default": "8,8",
                    "multiline": False,
                    "description": "Pattern C: Comma-separated beat counts"
                }),
                "pattern_D": ("STRING", {
                    "default": "16",
                    "multiline": False,
                    "description": "Pattern D: Comma-separated beat counts"
                }),
                "pattern_sequence": ("STRING", {
                    "default": "A",
                    "multiline": False,
                    "description": "Pattern orchestration (e.g., 'A,A,B,B,C,A,A,D'). Letters reference patterns A-D."
                }),
                "fps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "description": "Frames per second for video output"
                }),
                "repeat_pattern": ("BOOLEAN", {
                    "default": True,
                    "description": "Loop pattern sequence to cover entire song"
                }),
                "max_shots": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "description": "Maximum number of shots to generate"
                }),
            }
        }

    def generate_sequence(
        self,
        audio: Dict[str, Any],
        beat_positions: str,
        pattern_A: str,
        pattern_B: str = "2,2,2,2",
        pattern_C: str = "8,8",
        pattern_D: str = "16",
        pattern_sequence: str = "A",
        fps: int = 30,
        repeat_pattern: bool = True,
        max_shots: int = 1000
    ) -> Tuple[str, int]:
        """
        Generate complete shot sequence for music video

        Args:
            audio: Input audio tensor dict with 'waveform' and 'sample_rate'
            beat_positions: JSON string with beat positions from analyzer
            pattern_A: Pattern A - Comma-separated beat counts per shot (e.g., "4,4,8")
            pattern_B: Pattern B - Comma-separated beat counts per shot
            pattern_C: Pattern C - Comma-separated beat counts per shot
            pattern_D: Pattern D - Comma-separated beat counts per shot
            pattern_sequence: Orchestration sequence (e.g., "A,A,B,B,C,A,A,D")
            fps: Frames per second for frame count calculation
            repeat_pattern: Loop pattern sequence to cover entire song
            max_shots: Maximum number of shots to generate

        Returns:
            Tuple of (sequence_json, total_shots)
        """
        print(f"\n{'='*60}")
        print(f"[FL Music Video Sequencer] DEBUG: Function called")
        print(f"[FL Music Video Sequencer] DEBUG: Pattern A = {pattern_A}")
        print(f"[FL Music Video Sequencer] DEBUG: Pattern B = {pattern_B}")
        print(f"[FL Music Video Sequencer] DEBUG: Pattern C = {pattern_C}")
        print(f"[FL Music Video Sequencer] DEBUG: Pattern D = {pattern_D}")
        print(f"[FL Music Video Sequencer] DEBUG: Pattern Sequence = {pattern_sequence}")
        print(f"[FL Music Video Sequencer] DEBUG: FPS = {fps}")
        print(f"[FL Music Video Sequencer] DEBUG: Repeat pattern = {repeat_pattern}")
        print(f"[FL Music Video Sequencer] DEBUG: Max shots = {max_shots}")
        print(f"{'='*60}\n")

        try:
            # Parse beat positions JSON
            try:
                beat_data = json.loads(beat_positions)
                beat_times = np.array(beat_data['beat_times'])
                sample_rate = beat_data['sample_rate']
                total_beats = beat_data['num_beats']
                bpm = beat_data['bpm']

                print(f"[FL Music Video Sequencer] DEBUG: Loaded {total_beats} beat positions")
                print(f"[FL Music Video Sequencer] DEBUG: BPM = {bpm:.2f}")
                print(f"[FL Music Video Sequencer] DEBUG: Sample rate = {sample_rate}")

            except (json.JSONDecodeError, KeyError) as e:
                error_msg = f"Error parsing beat positions JSON: {e}"
                print(f"[FL Music Video Sequencer] ERROR: {error_msg}")
                return "{}", 0

            # Parse all patterns
            try:
                patterns = {
                    'A': [int(x.strip()) for x in pattern_A.split(',') if x.strip()],
                    'B': [int(x.strip()) for x in pattern_B.split(',') if x.strip()],
                    'C': [int(x.strip()) for x in pattern_C.split(',') if x.strip()],
                    'D': [int(x.strip()) for x in pattern_D.split(',') if x.strip()],
                }

                # Validate all patterns are non-empty
                for letter, pattern_beats in patterns.items():
                    if not pattern_beats:
                        print(f"[FL Music Video Sequencer] ERROR: Pattern {letter} is empty")
                        return "{}", 0

                print(f"[FL Music Video Sequencer] DEBUG: Pattern A = {patterns['A']}")
                print(f"[FL Music Video Sequencer] DEBUG: Pattern B = {patterns['B']}")
                print(f"[FL Music Video Sequencer] DEBUG: Pattern C = {patterns['C']}")
                print(f"[FL Music Video Sequencer] DEBUG: Pattern D = {patterns['D']}")

            except ValueError as e:
                error_msg = f"Error parsing patterns: {e}"
                print(f"[FL Music Video Sequencer] ERROR: {error_msg}")
                return "{}", 0

            # Parse pattern sequence
            try:
                sequence_letters = [x.strip().upper() for x in pattern_sequence.split(',') if x.strip()]
                if not sequence_letters:
                    print(f"[FL Music Video Sequencer] ERROR: Empty pattern sequence")
                    return "{}", 0

                # Validate sequence letters are valid (A, B, C, or D)
                for letter in sequence_letters:
                    if letter not in patterns:
                        print(f"[FL Music Video Sequencer] ERROR: Invalid pattern letter '{letter}' in sequence. Use A, B, C, or D.")
                        return "{}", 0

                print(f"[FL Music Video Sequencer] DEBUG: Pattern sequence = {sequence_letters}")

            except Exception as e:
                error_msg = f"Error parsing pattern sequence '{pattern_sequence}': {e}"
                print(f"[FL Music Video Sequencer] ERROR: {error_msg}")
                return "{}", 0

            # Get waveform for audio segments
            waveform = audio['waveform']

            # Generate shots
            shots = []
            current_beat = 0
            current_frame = 0
            sequence_index = 0  # Index in the pattern sequence (A, B, C, D)
            pattern_index = 0   # Index within the current pattern's beats
            shot_id = 0

            while current_beat < total_beats and shot_id < max_shots:
                # Get current pattern letter from sequence
                current_pattern_letter = sequence_letters[sequence_index]
                current_pattern_beats = patterns[current_pattern_letter]

                # Get beat count for this shot from the current pattern
                beat_count = current_pattern_beats[pattern_index]

                # Calculate end beat
                end_beat = current_beat + beat_count

                # Check if we've run out of beats
                if current_beat >= total_beats:
                    break

                # Adjust if end_beat exceeds total beats
                if end_beat > total_beats:
                    if not repeat_pattern:
                        # Last shot - use remaining beats
                        end_beat = total_beats
                        beat_count = end_beat - current_beat
                    else:
                        # Pattern repeating but not enough beats left
                        break

                # Calculate time boundaries
                segment_start_time = beat_times[current_beat]

                if end_beat < total_beats:
                    segment_end_time = beat_times[end_beat]
                else:
                    # Last segment - extend by median interval
                    if len(beat_times) > 1:
                        median_interval = np.median(np.diff(beat_times))
                        segment_end_time = beat_times[end_beat - 1] + median_interval
                    else:
                        segment_end_time = beat_times[end_beat - 1]

                # Calculate frame boundaries (drift-free)
                end_frame = round(segment_end_time * fps)
                frame_count = end_frame - current_frame

                # Calculate sample boundaries
                segment_start_sample = int(segment_start_time * sample_rate)
                segment_end_sample = int(segment_end_time * sample_rate)

                # Create shot data
                shot = {
                    "shot_id": shot_id,
                    "start_beat": int(current_beat),
                    "end_beat": int(end_beat),
                    "beat_count": int(beat_count),
                    "start_time": float(segment_start_time),
                    "end_time": float(segment_end_time),
                    "duration": float(segment_end_time - segment_start_time),
                    "start_frame": int(current_frame),
                    "end_frame": int(end_frame),
                    "frame_count": int(frame_count),
                    "start_sample": int(segment_start_sample),
                    "end_sample": int(segment_end_sample),
                    "pattern_letter": current_pattern_letter,
                    "pattern_index": int(pattern_index),
                    "sequence_index": int(sequence_index)
                }

                shots.append(shot)

                print(f"[FL Music Video Sequencer] Shot {shot_id}: Pattern {current_pattern_letter}[{pattern_index}], beats {current_beat}-{end_beat} ({beat_count} beats), frames {current_frame}-{end_frame} ({frame_count} frames), {segment_start_time:.2f}s-{segment_end_time:.2f}s")

                # Move to next shot
                current_beat = end_beat
                current_frame = end_frame
                shot_id += 1

                # Advance pattern index within current pattern
                pattern_index += 1

                # Check if we've completed the current pattern
                if pattern_index >= len(current_pattern_beats):
                    # Move to next pattern in sequence
                    pattern_index = 0
                    sequence_index += 1

                    # Check if we've completed the entire sequence
                    if sequence_index >= len(sequence_letters):
                        if repeat_pattern:
                            # Loop back to start of sequence
                            sequence_index = 0
                        else:
                            # Stay on last pattern in sequence
                            sequence_index = len(sequence_letters) - 1

            # Create sequence data
            sequence_data = {
                "metadata": {
                    "bpm": float(bpm),
                    "total_beats": int(total_beats),
                    "total_shots": len(shots),
                    "fps": int(fps),
                    "pattern_A": pattern_A,
                    "pattern_B": pattern_B,
                    "pattern_C": pattern_C,
                    "pattern_D": pattern_D,
                    "pattern_sequence": pattern_sequence,
                    "patterns": patterns,
                    "sequence_letters": sequence_letters,
                    "repeat_pattern": repeat_pattern,
                    "sample_rate": int(sample_rate),
                    "total_duration": float(beat_times[-1]) if len(beat_times) > 0 else 0.0,
                    "total_frames": int(current_frame)
                },
                "shots": shots
            }

            sequence_json = json.dumps(sequence_data, indent=2)

            print(f"\n{'='*60}")
            print(f"[FL Music Video Sequencer] Sequence generation complete!")
            print(f"[FL Music Video Sequencer] Total shots: {len(shots)}")
            print(f"[FL Music Video Sequencer] Total beats covered: {current_beat}/{total_beats}")
            print(f"[FL Music Video Sequencer] Total frames: {current_frame}")
            print(f"[FL Music Video Sequencer] Total duration: {beat_times[-1]:.2f}s" if len(beat_times) > 0 else "N/A")
            print(f"{'='*60}\n")

            return sequence_json, len(shots)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL Music Video Sequencer] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return "{}", 0
