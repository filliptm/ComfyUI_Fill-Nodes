import torch
import torch.nn.functional as F
import numpy as np


class FL_VideoCropMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "mask": ("IMAGE",),
                "output_width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "output_height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "smoothing_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "CROP_DATA")
    RETURN_NAMES = ("cropped_video", "mask", "original_video", "crop_data")
    FUNCTION = "crop_video"
    CATEGORY = "🏵️Fill Nodes/Utility"

    def crop_video(self, video: torch.Tensor, mask: torch.Tensor, output_width: int, output_height: int, padding: int,
                   smoothing_factor: float):
        batch_size, height, width, channels = video.shape

        cropped_video = []
        cropped_masks = []
        crop_data_list = []

        prev_center_x, prev_center_y = None, None
        prev_crop_width, prev_crop_height = None, None

        for i in range(batch_size):
            frame = video[i]
            frame_mask = mask[i]

            # Find the bounding box of the mask
            mask_binary = (frame_mask.sum(dim=-1) > 0).float()
            y_indices, x_indices = torch.where(mask_binary > 0)

            if len(y_indices) == 0 or len(x_indices) == 0:
                # If no mask is found, use the previous crop or the center of the frame
                if prev_center_x is None:
                    center_y, center_x = height // 2, width // 2
                    crop_width, crop_height = width, height
                else:
                    center_y, center_x = prev_center_y, prev_center_x
                    crop_width, crop_height = prev_crop_width, prev_crop_height
            else:
                top, bottom = y_indices.min().item(), y_indices.max().item()
                left, right = x_indices.min().item(), x_indices.max().item()

                center_y = (top + bottom) // 2
                center_x = (left + right) // 2

                crop_width = right - left + 2 * padding
                crop_height = bottom - top + 2 * padding

            # Apply smoothing to the center position and crop size
            if prev_center_x is not None:
                center_x = int(smoothing_factor * center_x + (1 - smoothing_factor) * prev_center_x)
                center_y = int(smoothing_factor * center_y + (1 - smoothing_factor) * prev_center_y)
                crop_width = int(smoothing_factor * crop_width + (1 - smoothing_factor) * prev_crop_width)
                crop_height = int(smoothing_factor * crop_height + (1 - smoothing_factor) * prev_crop_height)

            prev_center_x, prev_center_y = center_x, center_y
            prev_crop_width, prev_crop_height = crop_width, crop_height

            # Calculate the aspect ratio of the output and the crop
            output_aspect_ratio = output_width / output_height
            crop_aspect_ratio = crop_width / crop_height

            # Adjust crop size to fit the output aspect ratio without distortion
            if crop_aspect_ratio > output_aspect_ratio:
                # Crop is wider, adjust height
                crop_height = int(crop_width / output_aspect_ratio)
            else:
                # Crop is taller, adjust width
                crop_width = int(crop_height * output_aspect_ratio)

            # Ensure the crop stays within the frame
            top = max(0, center_y - crop_height // 2)
            bottom = min(height, top + crop_height)
            left = max(0, center_x - crop_width // 2)
            right = min(width, left + crop_width)

            # Adjust if the crop goes out of bounds
            if top == 0:
                bottom = crop_height
            if bottom == height:
                top = height - crop_height
            if left == 0:
                right = crop_width
            if right == width:
                left = width - crop_width

            # Crop the video and mask
            cropped_frame = frame[top:bottom, left:right, :]
            cropped_frame_mask = frame_mask[top:bottom, left:right, :]

            # Resize the cropped video and mask to the desired output size
            cropped_frame = F.interpolate(cropped_frame.unsqueeze(0).permute(0, 3, 1, 2),
                                          size=(output_height, output_width), mode='bilinear',
                                          align_corners=False).squeeze(0).permute(1, 2, 0)
            cropped_frame_mask = F.interpolate(cropped_frame_mask.unsqueeze(0).permute(0, 3, 1, 2),
                                               size=(output_height, output_width), mode='nearest').squeeze(0).permute(1,
                                                                                                                      2,
                                                                                                                      0)

            cropped_video.append(cropped_frame)
            cropped_masks.append(cropped_frame_mask)

            # Create crop data
            crop_data = {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "original_height": height,
                "original_width": width,
                "output_height": output_height,
                "output_width": output_width,
            }
            crop_data_list.append(crop_data)

        cropped_video = torch.stack(cropped_video)
        cropped_masks = torch.stack(cropped_masks)

        return (cropped_video, cropped_masks, video, crop_data_list)


class FL_VideoRecompose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_video": ("IMAGE",),
                "cropped_video": ("IMAGE",),
                "crop_data": ("CROP_DATA",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_video",)
    FUNCTION = "replace_crop"
    CATEGORY = "🏵️Fill Nodes/experiments"

    def replace_crop(self, original_video: torch.Tensor, cropped_video: torch.Tensor, crop_data: list):
        batch_size, height, width, channels = original_video.shape

        output_video = []

        for i in range(batch_size):
            frame = original_video[i]
            cropped_frame = cropped_video[i]
            frame_crop_data = crop_data[i]

            # Resize the cropped video back to its original size
            resized_crop = F.interpolate(
                cropped_frame.unsqueeze(0).permute(0, 3, 1, 2),
                size=(
                frame_crop_data["bottom"] - frame_crop_data["top"], frame_crop_data["right"] - frame_crop_data["left"]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)

            # Create a copy of the original frame
            output_frame = frame.clone()

            # Replace the cropped area in the original frame
            output_frame[frame_crop_data["top"]:frame_crop_data["bottom"],
            frame_crop_data["left"]:frame_crop_data["right"], :] = resized_crop

            output_video.append(output_frame)

        output_video = torch.stack(output_video)

        return (output_video,)