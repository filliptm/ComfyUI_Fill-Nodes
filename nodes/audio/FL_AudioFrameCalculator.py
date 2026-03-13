import math


class FL_AudioFrameCalculator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frame_count",)
    FUNCTION = "calculate"
    CATEGORY = "🏵️Fill Nodes/Audio"

    def calculate(self, audio, fps):
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        num_samples = waveform.shape[-1]
        duration = num_samples / sample_rate
        frame_count = math.ceil(duration * fps)
        return (frame_count,)
