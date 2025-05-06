import torch
import math
from nodes import common_ksampler, VAEDecode, VAEEncode
import comfy.samplers
import comfy.utils
import logging
import numpy as np
from PIL import Image
import torch.nn.functional as F
import latent_preview


class FL_KsamplerPlusV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "input_type": (["latent", "image"],),
                "x_slices": ("INT", {"default": 2, "min": 1, "max": 8}),
                "y_slices": ("INT", {"default": 2, "min": 1, "max": 8}),
                "overlap": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.9, "step": 0.01}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "use_sliced_conditioning": ("BOOLEAN", {"default": True}),
                "conditioning_strength": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 5.0, "step": 0.1}),
                "debug_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "latent_image": ("LATENT",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "IMAGE", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "vae", "image", "debug_info")
    FUNCTION = "sample"
    CATEGORY = "🏵️Fill Nodes/Ksamplers"

    @staticmethod
    def crop_tensor(tensor, region):
        x1, y1, x2, y2 = region
        return tensor[:, :, y1:y2, x1:x2]

    @staticmethod
    def resize_tensor(tensor, size, mode="nearest-exact"):
        return F.interpolate(tensor, size=size, mode=mode)

    @staticmethod
    def resize_region(region, init_size, resize_size):
        x1, y1, x2, y2 = region
        init_width, init_height = init_size
        resize_width, resize_height = resize_size
        x1 = math.floor(x1 * resize_width / init_width)
        x2 = math.ceil(x2 * resize_width / init_width)
        y1 = math.floor(y1 * resize_height / init_height)
        y2 = math.ceil(y2 * resize_height / init_height)
        return (x1, y1, x2, y2)

    @classmethod
    def crop_controlnet(cls, cond_dict, region, init_size, canvas_size, tile_size):
        """Crop ControlNet conditioning to match a specific region"""
        if "control" not in cond_dict:
            return
        c = cond_dict["control"]
        controlnet = c.copy()
        cond_dict["control"] = controlnet
        while c is not None:
            hint = controlnet.cond_hint_original
            resized_crop = cls.resize_region(region, canvas_size, hint.shape[2:])
            hint = cls.crop_tensor(hint, resized_crop)
            hint = cls.resize_tensor(hint, tile_size)
            controlnet.cond_hint_original = hint
            c = c.previous_controlnet
            controlnet.set_previous_controlnet(c.copy() if c is not None else None)
            controlnet = controlnet.previous_controlnet

    @classmethod
    def split_tensor_spatially(cls, tensor, region, init_size):
        """Split a tensor spatially based on the region"""
        # This is a simplified approach - in reality, we would need to know
        # how the tensor corresponds to spatial dimensions
        
        # Calculate the relative position of this region within the full image
        x1, y1, x2, y2 = region
        full_w, full_h = init_size
        
        # Calculate normalized coordinates (0-1) for this region
        region_center_x = (x1 + x2) / (2 * full_w)
        region_center_y = (y1 + y2) / (2 * full_h)
        region_width = (x2 - x1) / full_w
        region_height = (y2 - y1) / full_h
        
        # For tensors that don't have obvious spatial dimensions,
        # we'll use a simple approach: scale the tensor based on the region size
        # This assumes the tensor somehow encodes the full image
        region_scale = (region_width * region_height) / (full_w * full_h)
        
        # Scale the tensor - this is a very simplified approach
        # In a real implementation, we would need to know how the tensor
        # corresponds to spatial dimensions
        return tensor * region_scale

    @classmethod
    def crop_cond(cls, cond, region, init_size, canvas_size, tile_size, split_all_conditioning=True):
        """Enhanced conditioning cropping that completely splits all conditioning between slices"""
        cropped = []
        for emb, x in cond:
            # Create a deep copy of the conditioning dictionary
            cond_dict = x.copy()
            
            # Handle ControlNet conditioning
            cls.crop_controlnet(cond_dict, region, init_size, canvas_size, tile_size)
            
            if split_all_conditioning:
                # Split all tensor-based conditioning between slices
                for key, value in cond_dict.items():
                    if isinstance(value, torch.Tensor):
                        # Split the tensor spatially
                        cond_dict[key] = cls.split_tensor_spatially(value, region, init_size)
                
                # Add region information to the conditioning
                cond_dict["region_info"] = {
                    "region": region,
                    "init_size": init_size,
                    "canvas_size": canvas_size,
                    "tile_size": tile_size
                }
            
            cropped.append([emb, cond_dict])
        return cropped

    @staticmethod
    def adjust_conditioning_strength(cond, strength_factor):
        """Apply a strength factor to conditioning embeddings while preserving other data"""
        adjusted = []
        for emb, x in cond:
            # Adjust the embedding tensor by the strength factor
            adjusted_emb = emb * strength_factor
            
            # Create a copy of the conditioning dictionary to avoid modifying the original
            adjusted_dict = x.copy()
            
            # If there's a guidance value, we might want to adjust it as well
            if "guidance" in adjusted_dict:
                # Optionally adjust guidance - commented out for now as it might not be necessary
                # adjusted_dict["guidance"] = adjusted_dict["guidance"] * strength_factor
                pass
                
            adjusted.append([adjusted_emb, adjusted_dict])
        return adjusted

    @staticmethod
    def create_blend_mask(height, width, overlap_h, overlap_w, is_top, is_left, is_bottom, is_right, device):
        mask = torch.ones((height, width), device=device)

        if overlap_h > 0:
            x = torch.linspace(0, math.pi, overlap_h, device=device)
            cos = 0.5 * (1 - torch.cos(x))
            if not is_top:
                mask[:overlap_h, :] *= cos[:, None]
            if not is_bottom:
                mask[-overlap_h:, :] *= cos.flip(0)[:, None]
        if overlap_w > 0:
            x = torch.linspace(0, math.pi, overlap_w, device=device)
            cos = 0.5 * (1 - torch.cos(x))
            if not is_left:
                mask[:, :overlap_w] *= cos[None, :]
            if not is_right:
                mask[:, -overlap_w:] *= cos.flip(0)[None, :]

        return mask

    def sample(self, model, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise, input_type,
               x_slices, y_slices, overlap, batch_size, use_sliced_conditioning, conditioning_strength, 
               debug_mode, latent_image=None, image=None, vae=None):
        try:
            device = comfy.model_management.get_torch_device()

            if input_type == "image" and image is not None and vae is not None:
                latent_image = VAEEncode().encode(vae, image)[0]
            elif input_type == "latent" and latent_image is None:
                raise ValueError("Latent image is required when input type is set to latent")
            elif input_type == "image" and (image is None or vae is None):
                raise ValueError("Both image and VAE are required when input type is set to image")

            # We no longer force batch_size to 1 for sliced conditioning
            # This allows for more efficient processing

            b, c, h, w = latent_image["samples"].shape
            base_slice_height = h // y_slices
            base_slice_width = w // x_slices
            overlap_height = int(base_slice_height * overlap)
            overlap_width = int(base_slice_width * overlap)

            samples = None  # We'll initialize this later when we know the correct number of channels

            # We're splitting all conditioning between slices

            def process_slice(y, x):
                y_start = max(0, y * base_slice_height - overlap_height)
                y_end = min(h, (y + 1) * base_slice_height + overlap_height)
                x_start = max(0, x * base_slice_width - overlap_width)
                x_end = min(w, (x + 1) * base_slice_width + overlap_width)

                section = latent_image["samples"][:, :, y_start:y_end, x_start:x_end].to(device=device)

                if use_sliced_conditioning:
                    region = (x_start * 8, y_start * 8, x_end * 8, y_end * 8)
                    init_size = (w * 8, h * 8)
                    canvas_size = init_size
                    tile_size = ((x_end - x_start) * 8, (y_end - y_start) * 8)
                    strength_factor = conditioning_strength  # User-configurable strength factor
                    cropped_positive = self.adjust_conditioning_strength(
                        self.crop_cond(positive, region, init_size, canvas_size, tile_size, split_all_conditioning=True), 
                        strength_factor)
                    cropped_negative = self.adjust_conditioning_strength(
                        self.crop_cond(negative, region, init_size, canvas_size, tile_size, split_all_conditioning=True), 
                        strength_factor)
                else:
                    cropped_positive = positive
                    cropped_negative = negative

                return section, y_start, y_end, x_start, x_end, cropped_positive, cropped_negative

            total_slices = x_slices * y_slices
            all_slices = [(y, x) for y in range(y_slices) for x in range(x_slices)]

            for i in range(0, total_slices, batch_size):
                batch_slices = all_slices[i:min(i + batch_size, total_slices)]
                batch_sections = [process_slice(y, x) for y, x in batch_slices]

                batch_latents = torch.cat([section for section, _, _, _, _, _, _ in batch_sections], dim=0)

                if use_sliced_conditioning:
                    # Handle multiple slices in a batch with sliced conditioning
                    batch_positives = [section[5] for section in batch_sections]
                    batch_negatives = [section[6] for section in batch_sections]
                    
                    # Combine the conditionings for batch processing
                    batch_positive = []
                    for pos_cond in batch_positives:
                        batch_positive.extend(pos_cond)
                    
                    batch_negative = []
                    for neg_cond in batch_negatives:
                        batch_negative.extend(neg_cond)
                else:
                    batch_positive = positive * len(batch_sections)
                    batch_negative = negative * len(batch_sections)

                processed_batch = common_ksampler(model, seed + i, steps, cfg, sampler_name, scheduler,
                                                  batch_positive, batch_negative,
                                                  {"samples": batch_latents}, denoise=denoise)[0]

                processed_sections = torch.split(processed_batch["samples"], b, dim=0)

                # Initialize samples tensor if it hasn't been initialized yet
                if samples is None:
                    processed_channels = processed_sections[0].shape[1]
                    samples = torch.zeros((b, processed_channels, h, w), device=device)

                for (_, y_start, y_end, x_start, x_end, _, _), processed_section in zip(batch_sections,
                                                                                        processed_sections):
                    is_top = y_start == 0
                    is_left = x_start == 0
                    is_bottom = y_end == h
                    is_right = x_end == w
                    blend_mask = self.create_blend_mask(y_end - y_start, x_end - x_start,
                                                        overlap_height, overlap_width,
                                                        is_top, is_left, is_bottom, is_right,
                                                        device)

                    blend_mask = blend_mask.unsqueeze(0).unsqueeze(0).expand_as(processed_section)

                    processed_section = processed_section.to(device=device)
                    blend_mask = blend_mask.to(device=device)

                    samples[:, :, y_start:y_end, x_start:x_end] = (
                            samples[:, :, y_start:y_end, x_start:x_end] * (1 - blend_mask) +
                            processed_section * blend_mask
                    )

                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            output_image = None
            if vae is not None:
                vae_decoder = VAEDecode()
                output_image = vae_decoder.decode(vae, {"samples": samples})[0]

            # Prepare debug info if debug mode is enabled
            debug_info = ""
            if debug_mode:
                debug_info = "KSamplerPlus V2 Debug Info:\n"
                debug_info += f"- Input shape: {latent_image['samples'].shape}\n"
                debug_info += f"- Slices: {x_slices}x{y_slices} with {overlap*100:.1f}% overlap\n"
                debug_info += f"- Conditioning strength: {conditioning_strength}\n"
                debug_info += f"- Batch size: {batch_size}\n"
                debug_info += f"- Splitting all conditioning: {True}\n"
                
                # Add info about the conditioning
                if len(positive) > 0:
                    emb, cond_dict = positive[0]
                    debug_info += f"- Positive conditioning shape: {emb.shape}\n"
                    debug_info += "- Conditioning keys: " + ", ".join(cond_dict.keys()) + "\n"
                    
                    if "pooled_output" in cond_dict:
                        debug_info += f"- Pooled output shape: {cond_dict['pooled_output'].shape}\n"
                        debug_info += "- Splitting pooled output between slices\n"
                    
                    if "control" in cond_dict:
                        debug_info += "- ControlNet detected and processed\n"

            return (model, positive, negative, {"samples": samples}, vae, output_image, debug_info)

        except Exception as e:
            logging.error(f"Error in FL_KsamplerPlusV2: {str(e)}")
            raise