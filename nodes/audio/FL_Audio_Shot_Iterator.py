# FL_Audio_Shot_Iterator: Extract individual shot data from sequence JSON
import json
from typing import Tuple


class FL_Audio_Shot_Iterator:
    """
    A ComfyUI node for extracting individual shot data from a music video sequence.
    Takes sequence JSON and shot index, outputs frame count and shot details.
    """

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT", "BOOLEAN", "INT")
    RETURN_NAMES = ("frame_count", "start_frame", "end_frame", "start_beat", "beat_count", "start_time", "duration", "is_last_shot", "total_frames")
    FUNCTION = "get_shot"
    CATEGORY = "🏵️Fill Nodes/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence_json": ("STRING", {"description": "Sequence JSON from FL_Music_Video_Sequencer"}),
                "shot_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "description": "Shot index (0-based)"
                }),
            }
        }

    def get_shot(self, sequence_json: str, shot_index: int) -> Tuple[int, int, int, int, int, float, float, bool, int]:
        """
        Extract shot data from sequence JSON

        Args:
            sequence_json: JSON string from FL_Music_Video_Sequencer
            shot_index: Shot index (0-based)

        Returns:
            Tuple of (frame_count, start_frame, end_frame, start_beat, beat_count, start_time, duration, is_last_shot, total_frames)
        """
        print(f"\n{'='*60}")
        print(f"[FL Shot Iterator] DEBUG: Function called")
        print(f"[FL Shot Iterator] DEBUG: Shot index = {shot_index}")
        print(f"{'='*60}\n")

        try:
            # Parse sequence JSON
            try:
                sequence_data = json.loads(sequence_json)
                shots = sequence_data['shots']
                metadata = sequence_data['metadata']
                total_shots = len(shots)

                print(f"[FL Shot Iterator] DEBUG: Total shots in sequence = {total_shots}")

            except (json.JSONDecodeError, KeyError) as e:
                error_msg = f"Error parsing sequence JSON: {e}"
                print(f"[FL Shot Iterator] ERROR: {error_msg}")
                return 0, 0, 0, 0, 0, 0.0, 0.0, False, 0

            # Validate shot index
            if shot_index < 0 or shot_index >= total_shots:
                print(f"[FL Shot Iterator] ERROR: shot_index ({shot_index}) out of range (0-{total_shots-1})")
                return 0, 0, 0, 0, 0, 0.0, 0.0, False, 0

            # Get shot data
            shot = shots[shot_index]

            frame_count = shot['frame_count']
            start_frame = shot['start_frame']
            end_frame = shot['end_frame']
            start_beat = shot['start_beat']
            beat_count = shot['beat_count']
            start_time = shot['start_time']
            duration = shot['duration']
            is_last_shot = (shot_index == total_shots - 1)

            # Get total frames from metadata
            total_frames = metadata.get('total_frames', 0)

            print(f"\n{'='*60}")
            print(f"[FL Shot Iterator] Shot {shot_index} data:")
            print(f"[FL Shot Iterator]   Beats: {start_beat} to {shot['end_beat']} ({beat_count} beats)")
            print(f"[FL Shot Iterator]   Time: {start_time:.3f}s to {shot['end_time']:.3f}s ({duration:.3f}s)")
            print(f"[FL Shot Iterator]   Frames: {start_frame} to {end_frame} ({frame_count} frames)")
            print(f"[FL Shot Iterator]   Is last shot: {is_last_shot}")
            print(f"[FL Shot Iterator]   Total frames in sequence: {total_frames}")
            print(f"{'='*60}\n")

            return frame_count, start_frame, end_frame, start_beat, beat_count, start_time, duration, is_last_shot, total_frames

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL Shot Iterator] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return 0, 0, 0, 0, 0, 0.0, 0.0, False, 0
