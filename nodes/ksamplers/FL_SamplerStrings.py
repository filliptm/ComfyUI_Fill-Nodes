import comfy.samplers

class FL_SamplerStrings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                sampler: ("BOOLEAN", {"default": False}) for sampler in comfy.samplers.KSampler.SAMPLERS
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_sampler_string"
    CATEGORY = "🏵️Fill Nodes/Ksamplers"

    def generate_sampler_string(self, **kwargs):
        selected_samplers = [key for key, value in kwargs.items() if value]
        return (",".join(selected_samplers),)
