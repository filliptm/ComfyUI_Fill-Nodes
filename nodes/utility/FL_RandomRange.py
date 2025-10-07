import random
import torch

class FL_RandomNumber:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "min_value": ("FLOAT", {"default": 0.0, "min": -1000000.0, "max": 1000000.0, "step": 0.1}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -1000000.0, "max": 1000000.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 1000000}),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT")
    FUNCTION = "generate_random_number"
    CATEGORY = "🏵️Fill Nodes/Utility"

    def generate_random_number(self, min_value=0.0, max_value=1.0, seed=0):
        if min_value > max_value:
            raise ValueError("min_value should be less than or equal to max_value")

        # Generate a random seed if seed is 0
        if seed == 0:
            seed = random.randint(1, 1000000)

        # Set the random seed for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)

        # Generate a random float value within the specified range
        random_float = min_value + (max_value - min_value) * random.random()

        # Generate a random integer value within the specified range
        random_int = int(min_value + (max_value - min_value) * random.random())

        return (random_int, random_float)