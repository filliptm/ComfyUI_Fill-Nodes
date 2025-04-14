import torch
import numpy as np
from comfy.utils import ProgressBar


class FL_BlackFrameReject:
    """
    A node that filters out images from a batch based on the percentage of black pixels.
    Images with a black pixel percentage above the threshold are rejected.
    """

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("filtered_images", "kept_count", "rejected_count",)
    FUNCTION = "reject_black_frames"
    CATEGORY = "🏵️Fill Nodes/Image"

    DESCRIPTION = """
    FL_BlackFrameReject analyzes each image in a batch to detect the percentage of black pixels.
    Images with black pixel percentage exceeding the specified threshold are removed from the batch.
    Returns the filtered batch of images and counts of kept and rejected images.
    Useful for removing black frames, fades to black, or images with excessive dark regions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 95.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                        "description": "Percentage of black pixels needed to reject an image (0-100)"}),
                "black_level": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                                          "description": "Pixel brightness threshold to consider a pixel as black (0.0-0.5)"})
            },
            "optional": {
                "channels": (["all", "average", "luminance", "rgb"], {"default": "luminance",
                                                                      "description": "Channel(s) to analyze for black pixels"}),
                "invert": ("BOOLEAN", {"default": False,
                                       "description": "If True, reject frames with LESS than threshold % of black pixels"}),
                "show_progress": ("BOOLEAN", {"default": True,
                                              "description": "Show progress bar during processing"})
            }
        }

    def reject_black_frames(self, images: torch.Tensor, threshold: float, black_level: float,
                            channels: str = "luminance", invert: bool = False,
                            show_progress: bool = True) -> tuple:
        """
        Filters out images from a batch based on the percentage of black pixels.

        Args:
            images: Batch of images as a tensor (B, H, W, C)
            threshold: Percentage threshold for black pixels to reject an image (0-100)
            black_level: Maximum pixel value to consider as black (0.0-0.5)
            channels: Which channels to analyze for black pixels
            invert: If True, reject frames with less than threshold % of black pixels
            show_progress: Show progress bar during processing

        Returns:
            Tuple of (filtered_images, kept_count, rejected_count)
        """
        batch_size = images.shape[0]

        # Early return if batch is empty
        if batch_size == 0:
            return images, 0, 0

        # Initialize progress bar if enabled
        if show_progress:
            pbar = ProgressBar(batch_size)
            print(f"[FL_BlackFrameReject] Analyzing {batch_size} images for black pixel content...")

        # Compute black pixel percentages for each image
        black_percentages = []
        keep_indices = []

        for i in range(batch_size):
            img = images[i]

            # Different methods to calculate black pixels based on selected channels
            if channels == "all":
                # Check if all RGB channels are below black_level
                black_mask = torch.all(img <= black_level, dim=2)
            elif channels == "average":
                # Use average of all channels
                black_mask = torch.mean(img, dim=2) <= black_level
            elif channels == "luminance":
                # Use luminance formula (perceived brightness)
                # Y = 0.2126*R + 0.7152*G + 0.0722*B
                luminance = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
                black_mask = luminance <= black_level
            else:  # "rgb"
                # Check if any RGB channel is below black_level
                black_mask = torch.any(img <= black_level, dim=2)

            # Calculate percentage of black pixels
            black_percentage = torch.mean(black_mask.float()) * 100
            black_percentages.append(black_percentage.item())

            # Determine if image should be kept based on threshold and invert flag
            keep_image = black_percentage < threshold if not invert else black_percentage >= threshold

            if keep_image:
                keep_indices.append(i)

            # Update progress bar
            if show_progress:
                pbar.update_absolute(i + 1)

        # Create filtered batch
        if len(keep_indices) > 0:
            filtered_images = images[keep_indices]
        else:
            # Return empty tensor with correct dimensions if all images are rejected
            filtered_images = torch.zeros((0,) + images.shape[1:], dtype=images.dtype, device=images.device)

        # Calculate stats
        kept_count = len(keep_indices)
        rejected_count = batch_size - kept_count

        # Print summary
        print(f"[FL_BlackFrameReject] Kept {kept_count}/{batch_size} images, rejected {rejected_count} images")

        if kept_count > 0:
            print(
                f"[FL_BlackFrameReject] Black pixel range in kept images: {min([black_percentages[i] for i in keep_indices]):.2f}% - {max([black_percentages[i] for i in keep_indices]):.2f}%")

        return filtered_images, kept_count, rejected_count