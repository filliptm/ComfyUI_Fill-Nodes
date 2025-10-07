# FL_Audio_BPM_Analyzer: Analyze audio and detect all beats
import torch
import numpy as np
from typing import Tuple, Dict, Any


class FL_Audio_BPM_Analyzer:
    """
    A ComfyUI node for BPM and beat detection using Librosa.
    Analyzes the entire audio once and outputs beat positions for use in segmentation.
    """

    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING", "IMAGE")
    RETURN_NAMES = ("audio", "bpm", "beat_positions", "visualization")
    FUNCTION = "analyze_beats"
    CATEGORY = "🏵️Fill Nodes/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"description": "Input audio tensor"}),
            },
            "optional": {
                "bpm_method": (["beat_intervals", "onset_strength"], {
                    "default": "beat_intervals",
                    "description": "BPM calculation: beat_intervals (accurate) or onset_strength (librosa default)"
                }),
                "half_time": ("BOOLEAN", {
                    "default": False,
                    "description": "Half the detected BPM (for songs detected at double-time)"
                }),
                "beat_offset_ms": ("INT", {
                    "default": 0,
                    "min": -1000,
                    "max": 1000,
                    "step": 1,
                    "description": "Shift all beat positions (in milliseconds, positive or negative)"
                }),
            }
        }

    def analyze_beats(self, audio: Dict[str, Any], bpm_method: str = "beat_intervals", half_time: bool = False, beat_offset_ms: int = 0) -> Tuple[Dict[str, Any], float, str, torch.Tensor]:
        """
        Analyze audio and detect all beats

        Args:
            audio: Input audio tensor dict with 'waveform' and 'sample_rate'
            bpm_method: BPM calculation method ("beat_intervals" or "onset_strength")
            half_time: Half the detected BPM (for songs detected at double-time)
            beat_offset_ms: Offset in milliseconds to shift all beat positions (positive or negative)

        Returns:
            Tuple of (audio_passthrough, bpm_value, beat_positions_json, visualization_image)
        """
        print(f"\n{'='*60}")
        print(f"[FL BPM Analyzer] DEBUG: Function called")
        print(f"[FL BPM Analyzer] DEBUG: BPM method = {bpm_method}")
        print(f"[FL BPM Analyzer] DEBUG: Audio input type = {type(audio)}")
        print(f"{'='*60}\n")

        try:
            # Import librosa
            try:
                import librosa
                print(f"[FL BPM Analyzer] DEBUG: Successfully imported librosa (version: {librosa.__version__})")
            except ImportError as import_err:
                error_msg = f"Error: librosa library not installed. Install with: pip install librosa standard-aifc standard-sunau. Details: {import_err}"
                print(f"[FL BPM Analyzer] DEBUG: Import failed - {error_msg}")
                return audio, 0.0, "[]", torch.zeros((1, 512, 512, 3), dtype=torch.float32)

            # Validate audio input
            if audio is None:
                print(f"[FL BPM Analyzer] DEBUG: Audio is None!")
                return audio, 0.0, "[]", torch.zeros((1, 512, 512, 3), dtype=torch.float32)

            print(f"[FL BPM Analyzer] DEBUG: Audio dict keys: {audio.keys()}")

            waveform = audio['waveform']
            sample_rate = audio['sample_rate']

            print(f"[FL BPM Analyzer] DEBUG: Waveform shape = {waveform.shape}")
            print(f"[FL BPM Analyzer] DEBUG: Waveform dtype = {waveform.dtype}")
            print(f"[FL BPM Analyzer] DEBUG: Sample rate = {sample_rate}")

            # Convert waveform to numpy array and ensure proper format
            print(f"[FL BPM Analyzer] DEBUG: Processing waveform for Librosa...")
            if isinstance(waveform, torch.Tensor):
                waveform_np = waveform.cpu().numpy()
            else:
                waveform_np = np.array(waveform)

            # Librosa expects mono audio as 1D array
            if len(waveform_np.shape) == 3:  # [batch, channels, samples]
                waveform_np = waveform_np.squeeze(0)  # Remove batch dimension
                print(f"[FL BPM Analyzer] DEBUG: Squeezed batch dimension")

            if len(waveform_np.shape) == 2:  # [channels, samples]
                # Convert stereo to mono by averaging channels
                waveform_np = np.mean(waveform_np, axis=0)
                print(f"[FL BPM Analyzer] DEBUG: Converted stereo to mono")

            # Ensure float32 format
            if waveform_np.dtype != np.float32:
                waveform_np = waveform_np.astype(np.float32)
                print(f"[FL BPM Analyzer] DEBUG: Converted to float32")

            print(f"[FL BPM Analyzer] DEBUG: Final waveform shape = {waveform_np.shape}")
            print(f"[FL BPM Analyzer] DEBUG: Final waveform dtype = {waveform_np.dtype}")

            # Detect tempo and beats using librosa
            print(f"[FL BPM Analyzer] Running Librosa beat tracking...")

            tempo, beat_frames = librosa.beat.beat_track(
                y=waveform_np,
                sr=sample_rate,
                units='frames'
            )

            # Handle tempo output (can be array or scalar)
            if isinstance(tempo, np.ndarray):
                if len(tempo) > 0:
                    onset_strength_bpm = float(tempo[0])
                else:
                    onset_strength_bpm = 0.0
            else:
                onset_strength_bpm = float(tempo)

            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)

            print(f"[FL BPM Analyzer] DEBUG: Number of beats detected (raw) = {len(beat_times)}")
            print(f"[FL BPM Analyzer] DEBUG: Beat times (first 10) = {beat_times[:10].tolist()}")

            # Calculate beat intervals and BPM from actual beats
            if len(beat_times) > 1:
                beat_intervals = np.diff(beat_times)
                avg_interval = np.mean(beat_intervals)
                median_interval = np.median(beat_intervals)

                # Calculate BPM from beat intervals
                beat_interval_bpm = 60.0 / avg_interval
                median_bpm = 60.0 / median_interval

                print(f"[FL BPM Analyzer] DEBUG: Onset strength BPM = {onset_strength_bpm:.2f}")
                print(f"[FL BPM Analyzer] DEBUG: Beat interval BPM (median) = {median_bpm:.2f}")

                # Choose BPM based on method
                if bpm_method == "beat_intervals":
                    bpm = median_bpm
                    bpm_source = "beat_intervals_median"
                else:
                    bpm = onset_strength_bpm
                    bpm_source = "onset_strength"

                print(f"[FL BPM Analyzer] DEBUG: Using {bpm_source}: {bpm:.2f} BPM")
            else:
                bpm = onset_strength_bpm
                bpm_source = "onset_strength"
                beat_intervals = np.array([])
                median_interval = 60.0 / bpm if bpm > 0 else 0.5

            # Apply half-time if requested
            if half_time:
                print(f"[FL BPM Analyzer] DEBUG: Half-time enabled - doubling beat intervals")
                bpm = bpm / 2.0
                median_interval = median_interval * 2.0
                # Keep every other beat
                beat_times = beat_times[::2]
                beat_frames = beat_frames[::2]
                print(f"[FL BPM Analyzer] DEBUG: Half-time BPM = {bpm:.2f}")
                print(f"[FL BPM Analyzer] DEBUG: Half-time beats = {len(beat_times)}")

            # Fill in missing beats to cover entire song duration
            audio_duration = len(waveform_np) / sample_rate
            print(f"[FL BPM Analyzer] DEBUG: Audio duration = {audio_duration:.2f}s")

            if len(beat_times) > 1:
                # Calculate the beat interval to use for filling
                fill_interval = median_interval

                print(f"[FL BPM Analyzer] DEBUG: Using {fill_interval:.3f}s interval for beat filling")

                # Fill backwards from first beat to start of song
                filled_beats = list(beat_times)
                current_time = beat_times[0] - fill_interval
                while current_time > 0:
                    filled_beats.insert(0, current_time)
                    current_time -= fill_interval

                print(f"[FL BPM Analyzer] DEBUG: Added {len(filled_beats) - len(beat_times)} beats before first detected beat")

                # Fill forward from last beat to end of song
                current_time = beat_times[-1] + fill_interval
                while current_time < audio_duration:
                    filled_beats.append(current_time)
                    current_time += fill_interval

                print(f"[FL BPM Analyzer] DEBUG: Added {len(filled_beats) - len(beat_times)} beats after last detected beat (total)")

                # Fill gaps in the middle if any large gaps exist
                filled_beats_array = np.array(filled_beats)
                filled_beats_sorted = np.sort(filled_beats_array)

                # Check for gaps larger than 2x the median interval
                gaps = np.diff(filled_beats_sorted)
                large_gaps = gaps > (fill_interval * 1.5)

                if np.any(large_gaps):
                    print(f"[FL BPM Analyzer] DEBUG: Found {np.sum(large_gaps)} large gaps, filling...")
                    final_beats = []
                    for i, beat in enumerate(filled_beats_sorted[:-1]):
                        final_beats.append(beat)
                        if large_gaps[i]:
                            # Fill this gap
                            gap_start = beat
                            gap_end = filled_beats_sorted[i + 1]
                            num_beats_needed = int((gap_end - gap_start) / fill_interval)
                            for j in range(1, num_beats_needed):
                                final_beats.append(gap_start + j * fill_interval)
                    final_beats.append(filled_beats_sorted[-1])
                    beat_times = np.array(final_beats)
                else:
                    beat_times = filled_beats_sorted

                print(f"[FL BPM Analyzer] DEBUG: Total beats after filling = {len(beat_times)}")
                print(f"[FL BPM Analyzer] DEBUG: Coverage: {beat_times[0]:.2f}s to {beat_times[-1]:.2f}s (audio ends at {audio_duration:.2f}s)")
            else:
                print(f"[FL BPM Analyzer] WARNING: Not enough beats detected to fill entire song")

            # Apply beat offset if specified
            if beat_offset_ms != 0:
                beat_offset_seconds = beat_offset_ms / 1000.0
                beat_times = beat_times + beat_offset_seconds
                print(f"[FL BPM Analyzer] DEBUG: Applied {beat_offset_ms}ms ({beat_offset_seconds:.3f}s) offset to all beats")
                # Clamp to valid range [0, audio_duration]
                beat_times = np.clip(beat_times, 0, audio_duration)

            # Format beat positions as JSON
            import json
            beat_positions_dict = {
                "bpm": float(bpm),
                "bpm_source": bpm_source,
                "beat_times": beat_times.tolist(),
                "beat_frames": beat_frames.tolist(),
                "num_beats": len(beat_times),
                "sample_rate": int(sample_rate),
                "audio_duration": float(len(waveform_np) / sample_rate)
            }

            beat_positions_json = json.dumps(beat_positions_dict, indent=2)
            print(f"[FL BPM Analyzer] DEBUG: Beat positions JSON length = {len(beat_positions_json)} chars")

            # Generate matplotlib visualization
            print(f"[FL BPM Analyzer] DEBUG: Generating visualization...")
            try:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')
                from io import BytesIO
                from PIL import Image

                # Calculate audio duration and figure width
                audio_duration = len(waveform_np) / sample_rate
                fig_width = max(20, min(100, audio_duration * 0.5))

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 8))

                # Compute onset strength for visualization
                onset_env = librosa.onset.onset_strength(y=waveform_np, sr=sample_rate)
                times = librosa.times_like(onset_env, sr=sample_rate)

                # Plot 1: Waveform with beat markers
                ax1.plot(np.arange(len(waveform_np)) / sample_rate, waveform_np, alpha=0.6, linewidth=0.5, label='Waveform')
                ax1.vlines(beat_times, -1, 1, color='r', alpha=0.8, linestyle='--', linewidth=2, label='Detected Beats')
                ax1.set_xlim(0, audio_duration)
                ax1.set_xlabel('Time (seconds)')
                ax1.set_ylabel('Amplitude')
                ax1.set_title(f'Beat Detection: {bpm:.2f} BPM - {len(beat_times)} beats detected ({audio_duration:.1f}s total)')
                ax1.legend(loc='upper right')
                ax1.grid(True, alpha=0.3)

                # Plot 2: Onset strength envelope with beat markers
                ax2.plot(times, onset_env, label='Onset Strength', color='blue', linewidth=1.5)
                ax2.vlines(beat_times, 0, onset_env.max(), color='r', alpha=0.8, linestyle='--', linewidth=2, label='Detected Beats')
                ax2.set_xlim(0, audio_duration)
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel('Onset Strength')
                ax2.set_title('Onset Strength Envelope')
                ax2.legend(loc='upper right')
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()

                # Convert to image tensor
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                pil_image = Image.open(buf)
                img_array = np.array(pil_image.convert('RGB')).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0)

                plt.close()
                buf.close()

                print(f"[FL BPM Analyzer] Visualization generated: {img_tensor.shape}")

            except Exception as viz_error:
                print(f"[FL BPM Analyzer] WARNING: Could not generate visualization: {viz_error}")
                img_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)

            print(f"\n{'='*60}")
            print(f"[FL BPM Analyzer] Analysis complete!")
            print(f"[FL BPM Analyzer] BPM: {bpm:.2f}")
            print(f"[FL BPM Analyzer] Total beats: {len(beat_times)}")
            print(f"[FL BPM Analyzer] Audio duration: {audio_duration:.2f}s")
            print(f"{'='*60}\n")

            return audio, float(bpm), beat_positions_json, img_tensor

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL BPM Analyzer] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return audio, 0.0, "[]", torch.zeros((1, 512, 512, 3), dtype=torch.float32)
