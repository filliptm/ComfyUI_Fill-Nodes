import os
import random
from PIL import Image
import torch
import numpy as np

class FL_NFTGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "dummy_seed": ("INT", {"default": 0, "min": 0, "max": 1000000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "generate_nft"
    CATEGORY = "🏵️Fill Nodes"

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def generate_nft(self, folder_path, dummy_seed):
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")

        image_files = [f for f in os.listdir(folder_path) if not f.lower().endswith("-mask.png") and not f.lower().endswith("-mask.jpg") and not f.lower().endswith("-mask.jpeg")]
        if not image_files:
            raise ValueError(f"No image files found in the folder: {folder_path}")

        # Extract rarity percentages from image filenames
        rarities = []
        for image_file in image_files:
            if "-" in image_file:
                rarity_str = image_file.split("-")[1].split("per")[0]
                rarity = int(rarity_str)
                rarities.append(rarity)
            else:
                raise ValueError(f"Invalid image filename format: {image_file}")

        # Calculate cumulative probabilities
        total_rarity = sum(rarities)
        probabilities = [rarity / total_rarity for rarity in rarities]
        cumulative_probabilities = [sum(probabilities[:i+1]) for i in range(len(probabilities))]

        # Generate a random number between 0 and 1 using the dummy seed
        random.seed(dummy_seed)
        random_number = random.random()

        # Find the index of the selected image based on the random number and cumulative probabilities
        selected_index = None
        for i, prob in enumerate(cumulative_probabilities):
            if random_number <= prob:
                selected_index = i
                break

        if selected_index is None:
            raise ValueError("Failed to select an image based on rarity.")

        # Get the selected image and its corresponding mask
        selected_image_file = image_files[selected_index]
        selected_image_path = os.path.join(folder_path, selected_image_file)
        selected_image = Image.open(selected_image_path)

        # Get the file extension of the selected image
        _, extension = os.path.splitext(selected_image_file)

        # Generate the mask filename based on the selected image filename
        mask_file = selected_image_file.rsplit(".", 1)[0] + "-mask" + extension
        mask_path = os.path.join(folder_path, mask_file)

        if os.path.exists(mask_path):
            mask_image = Image.open(mask_path)
        else:
            # Create a blank mask image if the corresponding mask is not found
            mask_image = Image.new("RGB", selected_image.size, (0, 0, 0))

        selected_image_tensor = torch.from_numpy(np.array(selected_image).astype(np.float32) / 255.0).unsqueeze(0)
        mask_image_tensor = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)

        return (selected_image_tensor, mask_image_tensor)