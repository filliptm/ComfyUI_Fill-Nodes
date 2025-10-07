class FL_SamplerStrings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "euler": ("BOOLEAN", {"default": False}),
                "euler_cfg_pp": ("BOOLEAN", {"default": False}),
                "euler_ancestral": ("BOOLEAN", {"default": False}),
                "euler_ancestral_cfg_pp": ("BOOLEAN", {"default": False}),
                "heun": ("BOOLEAN", {"default": False}),
                "heunpp2": ("BOOLEAN", {"default": False}),
                "dpm_2": ("BOOLEAN", {"default": False}),
                "dpm_2_ancestral": ("BOOLEAN", {"default": False}),
                "lms": ("BOOLEAN", {"default": False}),
                "dpm_fast": ("BOOLEAN", {"default": False}),
                "dpm_adaptive": ("BOOLEAN", {"default": False}),
                "dpmpp_2s_ancestral": ("BOOLEAN", {"default": False}),
                "dpmpp_sde": ("BOOLEAN", {"default": False}),
                "dpmpp_sde_gpu": ("BOOLEAN", {"default": False}),
                "dpmpp_2m": ("BOOLEAN", {"default": False}),
                "dpmpp_2m_sde": ("BOOLEAN", {"default": False}),
                "dpmpp_2m_sde_gpu": ("BOOLEAN", {"default": False}),
                "dpmpp_3m_sde": ("BOOLEAN", {"default": False}),
                "dpmpp_3m_sde_gpu": ("BOOLEAN", {"default": False}),
                "ddpm": ("BOOLEAN", {"default": False}),
                "lcm": ("BOOLEAN", {"default": False}),
                "ipndm": ("BOOLEAN", {"default": False}),
                "ipndm_v": ("BOOLEAN", {"default": False}),
                "deis": ("BOOLEAN", {"default": False}),
                "ddim": ("BOOLEAN", {"default": False}),
                "uni_pc": ("BOOLEAN", {"default": False}),
                "uni_pc_bh2": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_sampler_string"
    CATEGORY = "🏵️Fill Nodes/Ksamplers"

    def generate_sampler_string(self, **kwargs):
        selected_samplers = [key for key, value in kwargs.items() if value]
        return (",".join(selected_samplers),)