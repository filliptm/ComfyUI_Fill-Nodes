class FL_JS:
    def __init__(self):
        pass

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "event": ((
                    "None",
                    "before_queued",
                    "after_queued",
                    "status",
                    "progress",
                    "executing",
                    "executed",
                    "execution_start",
                    "execution_success",
                    "execution_error",
                    "execution_cached",
                    "b_preview",
                ),),
                "javascript": ("STRING", {"default": "", "multiline": True}),
            }
        }
    
    CATEGORY = "🏵️Fill Nodes/Utility"
    FUNCTION = "exec"
    RETURN_TYPES = ()
    DESCRIPTION = "Execute JavaScript code when specific events are triggered"

    def exec(self, **kwargs):
        return ()