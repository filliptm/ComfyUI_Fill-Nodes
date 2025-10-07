import torch
import cv2
import numpy as np


class FL_SeparateMaskComponents:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK_MAPPING")
    FUNCTION = "separate"
    CATEGORY = "🏵️Fill Nodes/Utility"

    def separate(self, mask):
        device = mask.device

        # Ensure mask is in the correct format (B, H, W, C)
        if mask.dim() == 3:
            mask = mask.unsqueeze(-1)

        B, H, W, C = mask.shape

        all_component_masks = []
        all_mappings = []

        for b in range(B):
            # Convert to numpy and ensure it's a single-channel image
            mask_np = mask[b].squeeze().cpu().numpy()
            if mask_np.ndim == 3:
                mask_np = mask_np.mean(axis=-1)  # Average across channels if multi-channel

            # Threshold the mask
            mask_np = (mask_np > 0).astype(np.uint8)

            # Use OpenCV for connected component labeling
            num_labels, labels = cv2.connectedComponents(mask_np)

            for i in range(1, num_labels):  # Skip background (label 0)
                component_mask = (labels == i)
                component_tensor = torch.from_numpy(component_mask).to(device).unsqueeze(-1).expand(-1, -1, C)
                all_component_masks.append(component_tensor * mask[b])
                all_mappings.append(b)

        if all_component_masks:
            result = torch.stack(all_component_masks)
            mappings = torch.tensor(all_mappings, device=device)
        else:
            # Handle case where no components were found
            result = torch.zeros((0, H, W, C), device=device)
            mappings = torch.zeros(0, dtype=torch.long, device=device)

        return (result, mappings)