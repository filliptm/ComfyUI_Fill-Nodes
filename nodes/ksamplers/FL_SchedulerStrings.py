import comfy.samplers

class FL_SchedulerStrings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                scheduler: ("BOOLEAN", {"default": False}) for scheduler in comfy.samplers.KSampler.SCHEDULERS
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_scheduler_string"
    CATEGORY = "🏵️Fill Nodes/Ksamplers"

    def generate_scheduler_string(self, **kwargs):
        selected_schedulers = [key for key, value in kwargs.items() if value]
        return (",".join(selected_schedulers),)
