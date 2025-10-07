class FL_SchedulerStrings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "normal": ("BOOLEAN", {"default": False}),
                "karras": ("BOOLEAN", {"default": False}),
                "exponential": ("BOOLEAN", {"default": False}),
                "sgm_uniform": ("BOOLEAN", {"default": False}),
                "simple": ("BOOLEAN", {"default": False}),
                "ddim_uniform": ("BOOLEAN", {"default": False}),
                "beta": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_scheduler_string"
    CATEGORY = "🏵️Fill Nodes/Ksamplers"

    def generate_scheduler_string(self, **kwargs):
        selected_schedulers = [key for key, value in kwargs.items() if value]
        return (",".join(selected_schedulers),)