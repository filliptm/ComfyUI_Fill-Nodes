import torch

class FL_ImageAddNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff, 
                    "control_after_generate": True, 
                    "tooltip": "The random seed used for creating the noise."
                }),
                "strength": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_noise"
    CATEGORY = "🏵️Fill Nodes/Image"

    def add_noise(self, image, seed, strength):
        generator = torch.manual_seed(seed)
        s = torch.clip((image + strength * torch.randn(image.size(), generator=generator, device="cpu").to(image)), min=0.0, max=1.0)
        return (s,)