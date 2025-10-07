import torch
import torch.nn.functional as F

class FL_VideoCrossfade:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_a": ("IMAGE",),
                "images_b": ("IMAGE",),
                "crossfade_frames": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "add", "subtract"], {
                    "default": "normal"
                }),
                "target_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8
                }),
                "target_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crossfade_videos"
    CATEGORY = "🏵️Fill Nodes/Video"

    def resize_batch(self, images, target_width, target_height):
        """Resize a batch of images to target dimensions"""
        if images.shape[2] == target_height and images.shape[3] == target_width:
            return images
        
        # Permute to [batch, channels, height, width] for F.interpolate
        images = images.permute(0, 3, 1, 2)
        images = F.interpolate(images, size=(target_height, target_width), mode='bilinear', align_corners=False)
        # Permute back to [batch, height, width, channels]
        images = images.permute(0, 2, 3, 1)
        
        return images

    def apply_blend_mode(self, img_a, img_b, alpha, blend_mode):
        """Apply different blending modes"""
        if blend_mode == "normal":
            return img_a * (1 - alpha) + img_b * alpha
        elif blend_mode == "multiply":
            blended = img_a * img_b
            return img_a * (1 - alpha) + blended * alpha
        elif blend_mode == "screen":
            blended = 1 - (1 - img_a) * (1 - img_b)
            return img_a * (1 - alpha) + blended * alpha
        elif blend_mode == "overlay":
            mask = img_a < 0.5
            blended = torch.where(mask, 2 * img_a * img_b, 1 - 2 * (1 - img_a) * (1 - img_b))
            return img_a * (1 - alpha) + blended * alpha
        elif blend_mode == "soft_light":
            mask = img_b < 0.5
            blended = torch.where(mask, 
                                img_a - (1 - 2 * img_b) * img_a * (1 - img_a),
                                img_a + (2 * img_b - 1) * (torch.sqrt(torch.clamp(img_a, 0.0001, 1.0)) - img_a))
            return img_a * (1 - alpha) + blended * alpha
        elif blend_mode == "add":
            blended = torch.clamp(img_a + img_b, 0, 1)
            return img_a * (1 - alpha) + blended * alpha
        elif blend_mode == "subtract":
            blended = torch.clamp(img_a - img_b, 0, 1)
            return img_a * (1 - alpha) + blended * alpha
        else:
            return img_a * (1 - alpha) + img_b * alpha

    def crossfade_videos(self, images_a, images_b, crossfade_frames, blend_mode, target_width, target_height):
        # Get batch sizes
        batch_a = images_a.shape[0]
        batch_b = images_b.shape[0]
        
        # Validate crossfade frames
        max_crossfade = min(batch_a, batch_b)
        if crossfade_frames > max_crossfade:
            print(f"[FL_VideoCrossfade] Warning: Crossfade frames ({crossfade_frames}) reduced to {max_crossfade} to fit batch sizes.")
            crossfade_frames = max_crossfade
        
        # Resize both batches to target resolution
        print(f"[FL_VideoCrossfade] Resizing videos to {target_width}x{target_height}")
        images_a_resized = self.resize_batch(images_a, target_width, target_height)
        images_b_resized = self.resize_batch(images_b, target_width, target_height)
        
        # Calculate output length: A + B - crossfade_frames
        output_length = batch_a + batch_b - crossfade_frames
        output_images = []
        
        for i in range(output_length):
            if i < batch_a - crossfade_frames:
                # Pure sequence A (before crossfade)
                output_images.append(images_a_resized[i])
                
            elif i < batch_a:
                # Crossfade region
                a_idx = i
                b_idx = i - (batch_a - crossfade_frames)
                
                # Calculate blend ratio (0 to 1)
                blend_progress = (i - (batch_a - crossfade_frames)) / crossfade_frames
                alpha = torch.tensor(blend_progress, dtype=torch.float32)
                
                # Get images to blend
                img_a = images_a_resized[a_idx]
                img_b = images_b_resized[b_idx]
                
                # Apply blend mode
                blended = self.apply_blend_mode(img_a, img_b, alpha, blend_mode)
                output_images.append(blended)
                
            else:
                # Pure sequence B (after crossfade)
                b_idx = i - (batch_a - crossfade_frames)
                output_images.append(images_b_resized[b_idx])
        
        # Stack all images into final tensor
        result = torch.stack(output_images, dim=0)
        
        print(f"[FL_VideoCrossfade] Crossfaded {batch_a} + {batch_b} frames with {crossfade_frames} frame crossfade.")
        print(f"[FL_VideoCrossfade] Output: {result.shape[0]} frames at {target_width}x{target_height} using '{blend_mode}' blend mode.")
        
        return (result,)