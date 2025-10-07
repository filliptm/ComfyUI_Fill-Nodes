import random
import hashlib


class FL_MadLibGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {"multiline": True}),
                "delimiter_1": ("STRING", {"default": "(1)"}),
                "delimiter_2": ("STRING", {"default": "(2)"}),
                "delimiter_3": ("STRING", {"default": "(3)"}),
                "delimiter_4": ("STRING", {"default": "(4)"}),
                "delimiter_5": ("STRING", {"default": "(5)"}),
                "word_list_1": ("STRING", {}),
                "word_list_2": ("STRING", {}),
                "word_list_3": ("STRING", {}),
                "word_list_4": ("STRING", {}),
                "word_list_5": ("STRING", {}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_madlib"
    CATEGORY = "🏵️Fill Nodes/Prompting"

    def generate_madlib(self, template, delimiter_1, delimiter_2, delimiter_3, delimiter_4, delimiter_5,
                        word_list_1, word_list_2, word_list_3, word_list_4, word_list_5, seed=-1):

        # Custom seeding function
        def get_seed(base_seed, list_index, word_list):
            # Combine base seed, list index, and word list content for unique seeding
            seed_material = f"{base_seed}_{list_index}_{word_list}"
            return int(hashlib.md5(seed_material.encode()).hexdigest(), 16)

        # Prepare word lists and their individual RNGs
        word_lists = [
            (delimiter_1, word_list_1.split('\n')),
            (delimiter_2, word_list_2.split('\n')),
            (delimiter_3, word_list_3.split('\n')),
            (delimiter_4, word_list_4.split('\n')),
            (delimiter_5, word_list_5.split('\n')),
        ]
        word_lists = [(delim, [w.strip() for w in words if w.strip()]) for delim, words in word_lists]

        # Create separate RNGs for each word list
        rngs = [random.Random(get_seed(seed, i, ''.join(words))) for i, (_, words) in enumerate(word_lists)]

        # Function to get a random word from a specific list
        def get_random_word(list_index):
            delimiter, words = word_lists[list_index]
            if not words:
                return f"[NO WORDS FOR {delimiter}]"
            return rngs[list_index].choice(words)

        # Generate madlib
        result = template
        for i, (delimiter, _) in enumerate(word_lists):
            while delimiter in result:
                result = result.replace(delimiter, get_random_word(i), 1)

        return (result,)