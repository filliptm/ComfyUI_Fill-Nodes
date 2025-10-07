# FL_Audio_Segment_Extractor: Extract audio segments based on beat positions
import torch
import numpy as np
import json
from typing import Tuple, Dict, Any


class FL_Audio_Segment_Extractor:
    """
    A ComfyUI node for extracting audio segments based on pre-analyzed beat positions.
    Takes beat positions from FL_Audio_BPM_Analyzer and extracts specific beat ranges.
    """

    RETURN_TYPES = ("AUDIO", "INT", "INT")
    RETURN_NAMES = ("audio_segment", "frame_count", "end_frame")
    FUNCTION = "extract_segment"
    CATEGORY = "🏵️Fill Nodes/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"description": "Input audio tensor"}),
                "beat_positions": ("STRING", {"description": "Beat positions JSON from FL_BPM_Analyzer"}),
                "start_beat": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "description": "Starting beat index (0-based)"
                }),
                "beat_count": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "description": "Number of beats to extract"
                }),
            },
            "optional": {
                "fps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "description": "Frames per second for video output"
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000000,
                    "step": 1,
                    "description": "Starting frame number (for drift-free chaining)"
                }),
            }
        }

    def extract_segment(self, audio: Dict[str, Any], beat_positions: str, start_beat: int, beat_count: int, fps: int = 30, start_frame: int = 0) -> Tuple[Dict[str, Any], int, int]:
        """
        Extract audio segment based on beat positions

        Args:
            audio: Input audio tensor dict with 'waveform' and 'sample_rate'
            beat_positions: JSON string with beat positions from analyzer
            start_beat: Starting beat index (0-based)
            beat_count: Number of beats to extract
            fps: Frames per second for frame count calculation
            start_frame: Starting frame number (for drift-free chaining)

        Returns:
            Tuple of (audio_segment, frame_count, end_frame)
        """
        print(f"\n{'='*60}")
        print(f"[FL Audio Segment] DEBUG: Function called")
        print(f"[FL Audio Segment] DEBUG: Start beat = {start_beat}")
        print(f"[FL Audio Segment] DEBUG: Beat count = {beat_count}")
        print(f"[FL Audio Segment] DEBUG: FPS = {fps}")
        print(f"[FL Audio Segment] DEBUG: Start frame = {start_frame}")
        print(f"{'='*60}\n")

        try:
            # Parse beat positions JSON
            try:
                beat_data = json.loads(beat_positions)
                beat_times = np.array(beat_data['beat_times'])
                sample_rate = beat_data['sample_rate']
                total_beats = beat_data['num_beats']
                bpm = beat_data['bpm']

                print(f"[FL Audio Segment] DEBUG: Loaded {total_beats} beat positions")
                print(f"[FL Audio Segment] DEBUG: BPM = {bpm:.2f}")
                print(f"[FL Audio Segment] DEBUG: Sample rate = {sample_rate}")

            except (json.JSONDecodeError, KeyError) as e:
                error_msg = f"Error parsing beat positions JSON: {e}"
                print(f"[FL Audio Segment] ERROR: {error_msg}")
                return audio, 0, 0

            # Validate beat range
            end_beat = start_beat + beat_count
            if start_beat >= total_beats:
                print(f"[FL Audio Segment] ERROR: start_beat ({start_beat}) >= total beats ({total_beats})")
                return audio, 0, start_frame

            if end_beat > total_beats:
                print(f"[FL Audio Segment] WARNING: end_beat ({end_beat}) > total beats ({total_beats}), adjusting...")
                end_beat = total_beats
                beat_count = end_beat - start_beat

            # Get waveform
            waveform = audio['waveform']

            print(f"[FL Audio Segment] DEBUG: Extracting beats {start_beat} to {end_beat-1}")
            print(f"[FL Audio Segment] DEBUG: Beat times: {beat_times[start_beat:end_beat].tolist()}")

            # Calculate segment boundaries
            # Start at the specified beat
            segment_start_time = beat_times[start_beat]

            # End at the next beat after our range (for clean transitions)
            if end_beat < total_beats:
                segment_end_time = beat_times[end_beat]
                print(f"[FL Audio Segment] DEBUG: Using next beat as end point")
            else:
                # Last segment - extend a bit past the last beat
                if len(beat_times) > 1:
                    avg_interval = np.median(np.diff(beat_times))
                    segment_end_time = beat_times[end_beat - 1] + avg_interval
                    print(f"[FL Audio Segment] DEBUG: Last segment - adding median interval")
                else:
                    segment_end_time = beat_times[end_beat - 1]

            print(f"[FL Audio Segment] DEBUG: Segment time range: {segment_start_time:.3f}s to {segment_end_time:.3f}s")
            print(f"[FL Audio Segment] DEBUG: Segment duration: {segment_end_time - segment_start_time:.3f}s")

            # Convert to sample indices
            segment_start_sample = int(segment_start_time * sample_rate)
            segment_end_sample = int(segment_end_time * sample_rate)

            print(f"[FL Audio Segment] DEBUG: Sample range: {segment_start_sample} to {segment_end_sample}")

            # Extract segment from waveform
            if isinstance(waveform, torch.Tensor):
                if len(waveform.shape) == 3:  # [batch, channels, samples]
                    segment_waveform = waveform[:, :, segment_start_sample:segment_end_sample]
                elif len(waveform.shape) == 2:  # [channels, samples]
                    segment_waveform = waveform[:, segment_start_sample:segment_end_sample]
                else:  # [samples]
                    segment_waveform = waveform[segment_start_sample:segment_end_sample]
            else:
                segment_waveform = waveform

            # Create segment audio dict
            segment_audio = {
                'waveform': segment_waveform,
                'sample_rate': sample_rate
            }

            # Calculate frame count using cumulative time to prevent drift
            # Convert segment end time to absolute frame position
            end_frame = round(segment_end_time * fps)

            # Calculate frame count as difference from start frame
            frame_count = end_frame - start_frame

            print(f"[FL Audio Segment] DEBUG: Segment start time: {segment_start_time:.3f}s")
            print(f"[FL Audio Segment] DEBUG: Segment end time: {segment_end_time:.3f}s")
            print(f"[FL Audio Segment] DEBUG: Start frame: {start_frame}")
            print(f"[FL Audio Segment] DEBUG: End frame: {end_frame}")
            print(f"[FL Audio Segment] DEBUG: Frame count: {frame_count}")

            print(f"\n{'='*60}")
            print(f"[FL Audio Segment] Extraction complete!")
            print(f"[FL Audio Segment] Beats: {start_beat} to {end_beat-1}")
            print(f"[FL Audio Segment] Duration: {segment_end_time - segment_start_time:.3f}s")
            print(f"[FL Audio Segment] Frames: {start_frame} to {end_frame} ({frame_count} frames @ {fps} FPS)")
            print(f"{'='*60}\n")

            return segment_audio, frame_count, end_frame

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL Audio Segment] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return audio, 0, 0
