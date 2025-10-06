# FL_Audio_Drum_Detector: Detect kicks, snares, and hi-hats from audio
import torch
import numpy as np
import json
from typing import Tuple, Dict, Any


class FL_Audio_Drum_Detector:
    """
    A ComfyUI node for detecting drum elements (kicks, snares, hi-hats) from audio.
    Uses onset detection with frequency band analysis to classify drum types.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("drum_times_json",)
    FUNCTION = "detect_drums"
    CATEGORY = "🏵️Fill Nodes/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"description": "Input audio tensor (preferably drums stem)"}),
            },
            "optional": {
                "kick_sensitivity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "description": "Kick detection sensitivity (lower = more sensitive)"
                }),
                "snare_sensitivity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "description": "Snare detection sensitivity (lower = more sensitive)"
                }),
                "hihat_sensitivity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "description": "Hi-hat detection sensitivity (lower = more sensitive)"
                }),
            }
        }

    def detect_drums(
        self,
        audio: Dict[str, Any],
        kick_sensitivity: float = 0.5,
        snare_sensitivity: float = 0.5,
        hihat_sensitivity: float = 0.5
    ) -> Tuple[str]:
        """
        Detect drum elements from audio

        Args:
            audio: Input audio tensor dict with 'waveform' and 'sample_rate'
            kick_sensitivity: Kick detection threshold (0-1)
            snare_sensitivity: Snare detection threshold (0-1)
            hihat_sensitivity: Hi-hat detection threshold (0-1)

        Returns:
            Tuple containing JSON string with drum timestamps
        """
        print(f"\n{'='*60}")
        print(f"[FL Audio Drum Detector] DEBUG: Function called")
        print(f"[FL Audio Drum Detector] DEBUG: Kick sensitivity = {kick_sensitivity}")
        print(f"[FL Audio Drum Detector] DEBUG: Snare sensitivity = {snare_sensitivity}")
        print(f"[FL Audio Drum Detector] DEBUG: Hi-hat sensitivity = {hihat_sensitivity}")
        print(f"{'='*60}\n")

        try:
            import librosa

            waveform = audio['waveform']
            sample_rate = audio['sample_rate']

            # Convert to numpy and handle shape
            if isinstance(waveform, torch.Tensor):
                waveform_np = waveform.cpu().numpy()
            else:
                waveform_np = np.array(waveform)

            # Convert to mono if needed
            if len(waveform_np.shape) == 3:
                waveform_np = waveform_np[0]
            if len(waveform_np.shape) == 2 and waveform_np.shape[0] > 1:
                waveform_np = waveform_np.mean(axis=0)
            elif len(waveform_np.shape) == 2:
                waveform_np = waveform_np[0]

            print(f"[FL Audio Drum Detector] DEBUG: Waveform shape = {waveform_np.shape}")
            print(f"[FL Audio Drum Detector] DEBUG: Sample rate = {sample_rate}")

            # Detect onsets
            print(f"[FL Audio Drum Detector] Detecting onsets...")
            onset_env = librosa.onset.onset_strength(y=waveform_np, sr=sample_rate)
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=sample_rate,
                units='frames'
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)

            print(f"[FL Audio Drum Detector] DEBUG: Total onsets detected = {len(onset_times)}")

            # Frequency band analysis for classification
            # More precise frequency ranges for drum separation
            kick_times = []
            snare_times = []
            hihat_times = []

            # Compute STFT for frequency analysis
            stft = librosa.stft(waveform_np)
            stft_mag = np.abs(stft)

            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=sample_rate)

            # Define more specific frequency ranges for each drum type
            # Kicks: Very low frequencies (30-120 Hz) with peak around 60-80 Hz
            kick_low_mask = (freqs >= 30) & (freqs <= 120)
            kick_mid_mask = (freqs >= 120) & (freqs <= 300)  # Kick transient/attack

            # Snares: Mid frequencies (150-400 Hz fundamental) + high noise (4000-10000 Hz)
            snare_fundamental_mask = (freqs >= 150) & (freqs <= 400)
            snare_noise_mask = (freqs >= 4000) & (freqs <= 10000)

            # Hi-hats: High frequencies (6000+ Hz) with very little low end
            hihat_mask = (freqs >= 6000)
            low_reject_mask = (freqs <= 200)  # Hihats should have minimal low freq

            print(f"[FL Audio Drum Detector] Classifying drum types...")

            for onset_time, onset_frame in zip(onset_times, onset_frames):
                if onset_frame >= stft_mag.shape[1]:
                    continue

                # Get spectral magnitude at onset time
                spectrum = stft_mag[:, onset_frame]

                # Calculate energy in each band (use sum for better detection)
                kick_low_energy = np.sum(spectrum[kick_low_mask]) if np.any(kick_low_mask) else 0
                kick_mid_energy = np.sum(spectrum[kick_mid_mask]) if np.any(kick_mid_mask) else 0

                snare_fundamental_energy = np.sum(spectrum[snare_fundamental_mask]) if np.any(snare_fundamental_mask) else 0
                snare_noise_energy = np.sum(spectrum[snare_noise_mask]) if np.any(snare_noise_mask) else 0

                hihat_energy = np.sum(spectrum[hihat_mask]) if np.any(hihat_mask) else 0
                low_energy = np.sum(spectrum[low_reject_mask]) if np.any(low_reject_mask) else 0

                total_energy = np.sum(spectrum) + 1e-10

                # Calculate ratios
                kick_ratio = (kick_low_energy + kick_mid_energy) / total_energy
                snare_ratio = (snare_fundamental_energy + snare_noise_energy) / total_energy
                hihat_ratio = hihat_energy / total_energy
                low_ratio = low_energy / total_energy

                # More strict classification with mutual exclusion
                classified = False

                # Kicks: Strong low end, weak high end
                if (kick_low_energy > 0 and
                    kick_ratio > (1 - kick_sensitivity) * 0.3 and
                    low_ratio > 0.25 and
                    hihat_ratio < 0.4):  # Reject if too much high freq
                    kick_times.append(float(onset_time))
                    classified = True

                # Snares: Mid fundamental + high noise, moderate low end
                elif (not classified and
                      snare_fundamental_energy > 0 and
                      snare_noise_energy > 0 and
                      snare_ratio > (1 - snare_sensitivity) * 0.3 and
                      low_ratio < 0.6):  # Not too much low like kick
                    snare_times.append(float(onset_time))
                    classified = True

                # Hi-hats: Dominant high frequencies, minimal low end
                elif (not classified and
                      hihat_energy > 0 and
                      hihat_ratio > (1 - hihat_sensitivity) * 0.4 and
                      low_ratio < 0.2):  # Very little low frequency
                    hihat_times.append(float(onset_time))
                    classified = True

            # Create output JSON
            drum_data = {
                "kick_times": kick_times,
                "snare_times": snare_times,
                "hihat_times": hihat_times,
                "sample_rate": int(sample_rate),
                "duration": float(len(waveform_np) / sample_rate),
                "total_kicks": len(kick_times),
                "total_snares": len(snare_times),
                "total_hihats": len(hihat_times)
            }

            drum_times_json = json.dumps(drum_data, indent=2)

            print(f"\n{'='*60}")
            print(f"[FL Audio Drum Detector] Detection complete!")
            print(f"[FL Audio Drum Detector] Kicks detected: {len(kick_times)}")
            print(f"[FL Audio Drum Detector] Snares detected: {len(snare_times)}")
            print(f"[FL Audio Drum Detector] Hi-hats detected: {len(hihat_times)}")
            print(f"[FL Audio Drum Detector] Duration: {drum_data['duration']:.2f}s")
            print(f"{'='*60}\n")

            return (drum_times_json,)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL Audio Drum Detector] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return ("{}",)
