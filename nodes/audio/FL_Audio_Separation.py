# FL_Audio_Separation: Separate audio into stems (bass, drums, other, vocals)
import torch
from typing import Tuple, Dict, Any


class FL_Audio_Separation:
    """
    A ComfyUI node for separating audio into four sources: bass, drums, other, and vocals.
    Uses the Hybrid Demucs model from torchaudio.
    """

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("bass", "drums", "other", "vocals")
    FUNCTION = "separate_audio"
    CATEGORY = "🏵️Fill Nodes/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"description": "Input audio tensor"}),
            },
            "optional": {
                "chunk_length": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 60.0,
                    "step": 0.1,
                    "description": "Length of each processing chunk in seconds (longer = more memory)"
                }),
                "chunk_overlap": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.05,
                    "description": "Overlap between chunks in seconds (higher = smoother)"
                }),
                "chunk_fade_shape": (["linear", "half_sine", "logarithmic", "exponential"], {
                    "default": "linear",
                    "description": "Fade shape for chunk overlaps"
                }),
            }
        }

    def separate_audio(
        self,
        audio: Dict[str, Any],
        chunk_length: float = 10.0,
        chunk_overlap: float = 0.1,
        chunk_fade_shape: str = "linear"
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Separate audio into bass, drums, other, and vocals

        Args:
            audio: Input audio tensor dict with 'waveform' and 'sample_rate'
            chunk_length: Length of each processing chunk in seconds
            chunk_overlap: Overlap between chunks in seconds
            chunk_fade_shape: Fade shape for chunk overlaps

        Returns:
            Tuple of (bass_audio, drums_audio, other_audio, vocals_audio)
        """
        print(f"\n{'='*60}")
        print(f"[FL Audio Separation] DEBUG: Function called")
        print(f"[FL Audio Separation] DEBUG: Chunk length = {chunk_length}s")
        print(f"[FL Audio Separation] DEBUG: Chunk overlap = {chunk_overlap}s")
        print(f"[FL Audio Separation] DEBUG: Fade shape = {chunk_fade_shape}")
        print(f"{'='*60}\n")

        try:
            # Import required libraries
            from torchaudio.transforms import Fade, Resample
            from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
            import comfy.model_management

            waveform = audio['waveform']
            input_sample_rate = audio['sample_rate']

            # Get device
            device = comfy.model_management.get_torch_device()
            waveform = waveform.squeeze(0).to(device)

            print(f"[FL Audio Separation] DEBUG: Input waveform shape = {waveform.shape}")
            print(f"[FL Audio Separation] DEBUG: Input sample rate = {input_sample_rate}")
            print(f"[FL Audio Separation] DEBUG: Device = {device}")

            # Load Demucs model
            print(f"[FL Audio Separation] Loading Demucs model...")
            bundle = HDEMUCS_HIGH_MUSDB_PLUS
            model = bundle.get_model().to(device)
            model_sample_rate = bundle.sample_rate

            # Ensure stereo
            waveform = self._ensure_stereo(waveform)

            # Resample if needed
            if input_sample_rate != model_sample_rate:
                print(f"[FL Audio Separation] Resampling from {input_sample_rate}Hz to {model_sample_rate}Hz")
                resample = Resample(input_sample_rate, model_sample_rate).to(device)
                waveform = resample(waveform)

            # Normalize
            ref = waveform.mean(0)
            waveform = (waveform - ref.mean()) / ref.std()

            # Separate sources
            print(f"[FL Audio Separation] Separating sources...")
            sources = self._separate_sources(
                model,
                waveform[None],
                model_sample_rate,
                segment=chunk_length,
                overlap=chunk_overlap,
                device=device,
                chunk_fade_shape=chunk_fade_shape
            )[0]

            # Denormalize
            sources = sources * ref.std() + ref.mean()

            # Convert to dict
            sources_list = model.sources
            sources_dict = dict(zip(sources_list, list(sources)))

            # Output in order: bass, drums, other, vocals
            output_order = ["bass", "drums", "other", "vocals"]
            outputs = []
            for source_name in output_order:
                if source_name not in sources_dict:
                    raise ValueError(f"Missing source {source_name} in the output")

                output_audio = {
                    'waveform': sources_dict[source_name].cpu().unsqueeze(0),
                    'sample_rate': model_sample_rate
                }
                outputs.append(output_audio)
                print(f"[FL Audio Separation] {source_name.capitalize()}: {output_audio['waveform'].shape}")

            print(f"\n{'='*60}")
            print(f"[FL Audio Separation] Separation complete!")
            print(f"{'='*60}\n")

            return tuple(outputs)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL Audio Separation] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            # Return original audio for all outputs on error
            return (audio, audio, audio, audio)

    def _ensure_stereo(self, waveform: torch.Tensor) -> torch.Tensor:
        """Ensure waveform is stereo"""
        if waveform.ndim not in (2, 3):
            raise ValueError("Audio must have 2 or 3 dimensions")

        is_batched = waveform.ndim == 3
        channels_dim = 1 if is_batched else 0

        # Already stereo
        if waveform.shape[channels_dim] == 2:
            return waveform

        # Mono - duplicate channels
        elif waveform.shape[channels_dim] == 1:
            return waveform.repeat(1, 2, 1) if is_batched else waveform.repeat(2, 1)

        # Multi-channel - downmix to stereo
        waveform = waveform.narrow(channels_dim, 0, 2).mean(dim=channels_dim, keepdim=True)
        return waveform.repeat(1, 2, 1) if is_batched else waveform.repeat(2, 1)

    def _separate_sources(
        self,
        model: torch.nn.Module,
        mix: torch.Tensor,
        sample_rate: int,
        segment: float = 10.0,
        overlap: float = 0.1,
        device: torch.device = None,
        chunk_fade_shape: str = "linear"
    ) -> torch.Tensor:
        """
        Apply model to mixture using chunking with fade and overlap.
        Based on: https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html
        """
        from torchaudio.transforms import Fade

        if device is None:
            device = mix.device
        else:
            device = torch.device(device)

        batch, channels, length = mix.shape

        chunk_len = int(sample_rate * segment * (1 + overlap))
        start = 0
        end = chunk_len
        overlap_frames = overlap * sample_rate
        fade = Fade(
            fade_in_len=0,
            fade_out_len=int(overlap_frames),
            fade_shape=chunk_fade_shape
        )

        final = torch.zeros(batch, len(model.sources), channels, length, device=device)

        while start < length - overlap_frames:
            chunk = mix[:, :, start:end]
            with torch.no_grad():
                out = model.forward(chunk)
            out = fade(out)
            final[:, :, :, start:end] += out

            if start == 0:
                fade.fade_in_len = int(overlap_frames)
                start += int(chunk_len - overlap_frames)
            else:
                start += chunk_len
            end += chunk_len
            if end >= length:
                fade.fade_out_len = 0

        return final
