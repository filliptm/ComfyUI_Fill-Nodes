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

            crop_width = max(1, crop_width)
            crop_height = max(1, crop_height)

            # Scale oversized crops back inside the source frame while preserving aspect ratio.
            crop_scale = min(width / crop_width, height / crop_height, 1.0)
            crop_width = max(1, int(crop_width * crop_scale))
            crop_height = max(1, int(crop_height * crop_scale))

            # Ensure the crop stays within the frame
            top = min(max(0, center_y - crop_height // 2), height - crop_height)
            bottom = top + crop_height
            left = min(max(0, center_x - crop_width // 2), width - crop_width)
            right = left + crop_width

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

    @staticmethod
    def _fit_crop_channels(base_region: torch.Tensor, crop: torch.Tensor) -> torch.Tensor:
        base_channels = base_region.shape[-1]
        crop_channels = crop.shape[-1]

        if crop_channels == base_channels:
            return crop

        if crop_channels == 4 and base_channels >= 3:
            alpha = crop[..., 3:4].clamp(0.0, 1.0)
            fitted = base_region.clone()
            fitted[..., :3] = crop[..., :3] * alpha + base_region[..., :3] * (1.0 - alpha)
            if base_channels > 3:
                shared_extra_channels = min(base_channels, crop_channels) - 3
                fitted[..., 3:3 + shared_extra_channels] = crop[..., 3:3 + shared_extra_channels]
            return fitted

        fitted = base_region.clone()
        shared_channels = min(base_channels, crop_channels)
        fitted[..., :shared_channels] = crop[..., :shared_channels]
        return fitted

    def replace_crop(self, original_video: torch.Tensor, cropped_video: torch.Tensor, crop_data: list):
        batch_size, height, width, channels = original_video.shape

        output_video = []

        for i in range(batch_size):
            frame = original_video[i]
            cropped_frame = cropped_video[i]
            frame_crop_data = crop_data[i]

            top = max(0, min(height, int(frame_crop_data["top"])))
            bottom = max(top, min(height, int(frame_crop_data["bottom"])))
            left = max(0, min(width, int(frame_crop_data["left"])))
            right = max(left, min(width, int(frame_crop_data["right"])))

            if bottom <= top or right <= left:
                output_video.append(frame.clone())
                continue

            # Resize the cropped video back to its original size
            resized_crop = F.interpolate(
                cropped_frame.unsqueeze(0).permute(0, 3, 1, 2),
                size=(bottom - top, right - left),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)

            # Create a copy of the original frame
            output_frame = frame.clone()

            target_region = output_frame[
                top:bottom,
                left:right,
                :
            ]
            resized_crop = self._fit_crop_channels(target_region, resized_crop)

            # Replace the cropped area in the original frame
            output_frame[top:bottom, left:right, :] = resized_crop

            output_video.append(output_frame)

        output_video = torch.stack(output_video)

        return (output_video,)
