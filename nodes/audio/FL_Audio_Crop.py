# FL_Audio_Crop: Crop/trim audio to specific start and end times
import torch
from typing import Tuple, Dict, Any


class FL_Audio_Crop:
    """
    A ComfyUI node for cropping (trimming) audio to a specific start and end time.
    """

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "crop_audio"
    CATEGORY = "🏵️Fill Nodes/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"description": "Input audio tensor"}),
                "start_time": ("STRING", {
                    "default": "0:00",
                    "description": "Start time (MM:SS or seconds)"
                }),
                "end_time": ("STRING", {
                    "default": "1:00",
                    "description": "End time (MM:SS or seconds)"
                }),
            }
        }

    def crop_audio(
        self,
        audio: Dict[str, Any],
        start_time: str = "0:00",
        end_time: str = "1:00"
    ) -> Tuple[Dict[str, Any]]:
        """
        Crop audio to specific start and end times

        Args:
            audio: Input audio tensor dict with 'waveform' and 'sample_rate'
            start_time: Start time as "MM:SS" or seconds
            end_time: End time as "MM:SS" or seconds

        Returns:
            Tuple containing cropped audio dict
        """
        print(f"\n{'='*60}")
        print(f"[FL Audio Crop] DEBUG: Function called")
        print(f"[FL Audio Crop] DEBUG: Start time = {start_time}")
        print(f"[FL Audio Crop] DEBUG: End time = {end_time}")
        print(f"{'='*60}\n")

        try:
            waveform = audio['waveform']
            sample_rate = audio['sample_rate']

            # If no ":" in input, assume user is specifying seconds
            if ":" not in start_time:
                start_time = f"00:{start_time}"
            if ":" not in end_time:
                end_time = f"00:{end_time}"

            # Parse start time
            start_seconds = 60 * int(start_time.split(":")[0]) + int(start_time.split(":")[1])
            start_frame = start_seconds * sample_rate

            # Parse end time
            end_seconds = 60 * int(end_time.split(":")[0]) + int(end_time.split(":")[1])
            end_frame = end_seconds * sample_rate

            # Clamp to valid range
            total_frames = waveform.shape[-1]
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(0, min(end_frame, total_frames - 1))

            if start_frame >= end_frame:
                raise ValueError(
                    f"FL Audio Crop: Start time ({start_time}) must be less than end time ({end_time}) "
                    f"and be within the audio length."
                )

            # Crop waveform
            cropped_waveform = waveform[..., start_frame:end_frame]

            cropped_audio = {
                'waveform': cropped_waveform,
                'sample_rate': sample_rate
            }

            duration = (end_frame - start_frame) / sample_rate
            print(f"\n{'='*60}")
            print(f"[FL Audio Crop] Crop complete!")
            print(f"[FL Audio Crop] Start: {start_time} ({start_frame} frames)")
            print(f"[FL Audio Crop] End: {end_time} ({end_frame} frames)")
            print(f"[FL Audio Crop] Duration: {duration:.2f}s")
            print(f"{'='*60}\n")

            return (cropped_audio,)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL Audio Crop] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return (audio,)
