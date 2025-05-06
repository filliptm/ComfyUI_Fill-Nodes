import torch
import os
import re
import json
import logging
from collections import OrderedDict
import numpy as np
import glob
import pickle
import struct
import zipfile
import io

# Try to import safetensors if available
try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

class FL_ModelInspector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_path": ("STRING", {"default": "", "multiline": False}),
                "include_state_dict": ("BOOLEAN", {"default": False}),
                "include_layer_details": ("BOOLEAN", {"default": True}),
                "show_all_keys": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "MODEL")
    RETURN_NAMES = ("model_info", "model")
    FUNCTION = "inspect_model"
    CATEGORY = "🏵️Fill Nodes/Utility"

    def count_parameters(self, model_dict):
        """Count the total number of parameters in the model"""
        return sum(np.prod(v.shape) for k, v in model_dict.items() if isinstance(v, torch.Tensor))

    def get_layer_shapes(self, model_dict):
        """Extract shapes of all layers"""
        shapes = {}
        for key, value in model_dict.items():
            if isinstance(value, torch.Tensor):
                shapes[key] = list(value.shape)
        return shapes

    def guess_input_shape(self, model_dict, layer_shapes):
        """Try to infer the input shape from the first Conv2D layer"""
        # Look for conv layers that might be the first layer
        for key in model_dict.keys():
            if re.search(r'conv.*\.weight', key) or re.search(r'down.*conv.*\.weight', key):
                shape = layer_shapes.get(key)
                if shape and len(shape) == 4:  # Conv2D weights are typically [out_channels, in_channels, kernel_h, kernel_w]
                    in_channels = shape[1]
                    # Common input sizes for image models
                    return f"({in_channels}, H, W) → Likely a {in_channels}-channel input"
        return "Could not determine input shape"

    def analyze_architecture(self, model_keys):
        """Analyze model architecture based on key patterns"""
        architecture_info = []
        
        # Check for common model architectures
        if any("unet" in key.lower() for key in model_keys):
            architecture_info.append("U-Net architecture detected")
        
        if any("transformer" in key.lower() for key in model_keys):
            architecture_info.append("Transformer components detected")
            
        if any("attention" in key.lower() for key in model_keys):
            architecture_info.append("Attention mechanisms detected")
            
        if any("resnet" in key.lower() or "residual" in key.lower() for key in model_keys):
            architecture_info.append("ResNet/Residual connections detected")
            
        if any("encoder" in key.lower() and "decoder" in key.lower() for key in model_keys):
            architecture_info.append("Encoder-Decoder architecture detected")
            
        if any("embedding" in key.lower() for key in model_keys):
            architecture_info.append("Embedding layers detected")
            
        if any("norm" in key.lower() for key in model_keys):
            architecture_info.append("Normalization layers detected")
            
        # Count layer types
        conv_count = sum(1 for key in model_keys if "conv" in key.lower())
        linear_count = sum(1 for key in model_keys if "linear" in key.lower() or "fc" in key.lower())
        
        if conv_count > 0:
            architecture_info.append(f"Contains {conv_count} convolutional layers/components")
        if linear_count > 0:
            architecture_info.append(f"Contains {linear_count} linear/fully-connected layers")
            
        return architecture_info

    def inspect_model(self, ckpt_path, include_state_dict, include_layer_details, show_all_keys):
        try:
            # Validate path
            if not os.path.exists(ckpt_path):
                # Try to find the file with a similar name
                directory = os.path.dirname(ckpt_path) or "."
                filename = os.path.basename(ckpt_path)
                similar_files = glob.glob(f"{directory}/*{filename}*")
                
                if similar_files:
                    suggestion = f"File not found at {ckpt_path}. Did you mean one of these?\n"
                    for file in similar_files[:5]:  # Limit to 5 suggestions
                        suggestion += f"- {file}\n"
                    return suggestion, None
                else:
                    return f"Error: File not found at {ckpt_path}", None
                
            # Load the checkpoint
            logging.info(f"Loading checkpoint from {ckpt_path}")
            load_info = []
            checkpoint = None
            
            # Get file info
            file_size = os.path.getsize(ckpt_path) / (1024 * 1024)  # Size in MB
            file_extension = os.path.splitext(ckpt_path)[1].lower()
            load_info.append(f"File size: {file_size:.2f} MB")
            load_info.append(f"File extension: {file_extension}")
            
            # Try loading with safetensors first if it's a .safetensors file
            if file_extension == ".safetensors" and SAFETENSORS_AVAILABLE:
                try:
                    checkpoint = safetensors.torch.load_file(ckpt_path, device="cpu")
                    load_info.append("Checkpoint loaded successfully with safetensors")
                except Exception as e:
                    load_info.append(f"Failed to load with safetensors: {str(e)}")
            
            # Try standard PyTorch loading if not loaded yet
            if checkpoint is None:
                try:
                    checkpoint = torch.load(ckpt_path, map_location="cpu")
                    load_info.append("Checkpoint loaded successfully with torch.load")
                except Exception as e:
                    error_msg = f"Failed to load with torch.load: {str(e)}"
                    load_info.append(error_msg)
                    
                    # Try alternative loading methods
                    if "storages" in str(e):
                        load_info.append("Detected 'storages' error, trying alternative loading methods...")
                        
                        # Method 1: Try loading with pickle
                        try:
                            with open(ckpt_path, 'rb') as f:
                                checkpoint = pickle.load(f)
                            load_info.append("Checkpoint loaded successfully with pickle")
                        except Exception as pickle_e:
                            load_info.append(f"Failed to load with pickle: {str(pickle_e)}")
                        
                        # Method 2: Try loading with torch.load and different map_location
                        if checkpoint is None:
                            try:
                                checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
                                load_info.append("Checkpoint loaded successfully with custom storage mapping")
                            except Exception as torch_e:
                                load_info.append(f"Failed with custom storage mapping: {str(torch_e)}")
                        
                        # Method 3: Try loading with torch.serialization directly
                        if checkpoint is None:
                            try:
                                with open(ckpt_path, 'rb') as f:
                                    checkpoint = torch._utils._rebuild_tensor_v2(
                                        torch.storage._load_from_bytes(pickle.load(f)),
                                        0, (), None)
                                load_info.append("Checkpoint loaded successfully with torch serialization")
                            except Exception as ser_e:
                                load_info.append(f"Failed with torch serialization: {str(ser_e)}")
                        
                        # Method 4: Try to read as a zip file (many .ckpt files are zip archives)
                        if checkpoint is None:
                            try:
                                with zipfile.ZipFile(ckpt_path, 'r') as z:
                                    file_list = z.namelist()
                                    load_info.append(f"File appears to be a zip archive with {len(file_list)} files")
                                    
                                    # Look for data.pkl or similar files
                                    data_files = [f for f in file_list if f.endswith('.pkl') or f.endswith('.pt') or f.endswith('.bin')]
                                    if data_files:
                                        for data_file in data_files:
                                            try:
                                                with z.open(data_file) as f:
                                                    data = pickle.load(io.BytesIO(f.read()))
                                                    if isinstance(data, dict):
                                                        checkpoint = data
                                                        load_info.append(f"Successfully loaded data from {data_file} in zip archive")
                                                        break
                                            except:
                                                pass
                            except Exception as zip_e:
                                load_info.append(f"Failed to read as zip archive: {str(zip_e)}")
            
            # If all loading methods failed, create a synthetic checkpoint based on file characteristics
            if checkpoint is None:
                # Create a minimal synthetic checkpoint for analysis
                checkpoint = {"__synthetic__": True}
                state_dict = {}
                
                # Add some basic synthetic data based on file size
                estimated_params = file_size * 1024 * 1024 / 4  # Rough estimate: 4 bytes per parameter (float32)
                
                # Create a more realistic synthetic model based on file size
                if file_size > 1000:  # If > 1GB
                    # Likely a diffusion model (SD 1.x style)
                    # Calculate dimensions to match estimated parameter count
                    # For SD 1.5, about 65% of params are in the middle block
                    middle_dim = int(np.sqrt(estimated_params * 0.65 / (3*3)))
                    middle_dim = max(middle_dim, 1280)  # Ensure reasonable minimum
                    
                    # Create a realistic structure with proper dimensions
                    state_dict["model.diffusion_model.input_blocks.0.0.weight"] = torch.zeros((320, 4, 3, 3))
                    state_dict["model.diffusion_model.middle_block.0.weight"] = torch.zeros((middle_dim, middle_dim, 3, 3))
                    state_dict["model.diffusion_model.output_blocks.0.0.weight"] = torch.zeros((320, 320, 3, 3))
                    
                    # Add more typical SD model components
                    state_dict["model.diffusion_model.input_blocks.1.0.weight"] = torch.zeros((320, 320, 3, 3))
                    state_dict["model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"] = torch.zeros((320, 320))
                    state_dict["model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight"] = torch.zeros((320, 320))
                    state_dict["model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_v.weight"] = torch.zeros((320, 320))
                    state_dict["model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.weight"] = torch.zeros((1280, 320))
                    state_dict["first_stage_model.encoder.down.0.block.0.norm1.weight"] = torch.zeros((128,))
                    state_dict["first_stage_model.decoder.up.0.block.0.norm1.weight"] = torch.zeros((128,))
                    state_dict["cond_stage_model.transformer.text_model.embeddings.position_embedding.weight"] = torch.zeros((77, 768))
                    state_dict["cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"] = torch.zeros((768, 768))
                    
                elif file_size > 500:  # If > 500MB
                    # Medium sized model (LoRA or smaller diffusion model)
                    state_dict["encoder.blocks.0.weight"] = torch.zeros((768, 768, 3, 3))
                    state_dict["decoder.blocks.0.weight"] = torch.zeros((768, 768, 3, 3))
                    state_dict["transformer.blocks.0.attn.q_proj.weight"] = torch.zeros((1024, 1024))
                    state_dict["transformer.blocks.0.attn.k_proj.weight"] = torch.zeros((1024, 1024))
                    state_dict["transformer.blocks.0.attn.v_proj.weight"] = torch.zeros((1024, 1024))
                    state_dict["transformer.blocks.0.mlp.fc1.weight"] = torch.zeros((4096, 1024))
                    state_dict["transformer.blocks.0.mlp.fc2.weight"] = torch.zeros((1024, 4096))
                else:
                    # Smaller model
                    state_dict["backbone.layer1.0.conv1.weight"] = torch.zeros((64, 3, 3, 3))
                    state_dict["backbone.layer1.0.conv2.weight"] = torch.zeros((64, 64, 3, 3))
                    state_dict["backbone.layer2.0.conv1.weight"] = torch.zeros((128, 64, 3, 3))
                    state_dict["backbone.layer2.0.conv2.weight"] = torch.zeros((128, 128, 3, 3))
                    state_dict["backbone.layer3.0.conv1.weight"] = torch.zeros((256, 128, 3, 3))
                    state_dict["backbone.layer3.0.conv2.weight"] = torch.zeros((256, 256, 3, 3))
                    state_dict["backbone.layer4.0.conv1.weight"] = torch.zeros((512, 256, 3, 3))
                    state_dict["backbone.layer4.0.conv2.weight"] = torch.zeros((512, 512, 3, 3))
                    state_dict["fc.weight"] = torch.zeros((1000, 512))
                
                checkpoint["state_dict"] = state_dict
                
                load_info.append("Created synthetic model information based on file characteristics")
                load_info.append(f"Estimated parameters (based on file size): ~{estimated_params:,.0f}")
                
                # Add warning about failed loading
                load_info.append("WARNING: All loading methods failed. Using file characteristics to estimate model structure.")
            
            # Determine the structure of the checkpoint
            if isinstance(checkpoint, dict):
                if "__synthetic__" in checkpoint:
                    # We're using a synthetic checkpoint
                    state_dict = checkpoint.get("state_dict", {})
                    load_info.append("Using synthetic state dictionary based on file characteristics")
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                    load_info.append("Found 'state_dict' key in checkpoint")
                else:
                    # Try to find a key that might contain the state dict
                    potential_state_dicts = [k for k, v in checkpoint.items() 
                                           if isinstance(v, dict) or isinstance(v, OrderedDict)]
                    
                    if potential_state_dicts:
                        state_dict = checkpoint[potential_state_dicts[0]]
                        load_info.append(f"Using '{potential_state_dicts[0]}' as state_dict")
                    else:
                        # Assume the checkpoint itself is the state dict
                        state_dict = checkpoint
                        load_info.append("Using entire checkpoint as state_dict")
            else:
                return "Error: Checkpoint is not a dictionary", None
            
            # Extract model information
            model_keys = list(state_dict.keys())
            layer_shapes = self.get_layer_shapes(state_dict)
            num_params = self.count_parameters(state_dict)
            input_shape_guess = self.guess_input_shape(state_dict, layer_shapes)
            architecture_analysis = self.analyze_architecture(model_keys)
            
            # Format the output
            info_sections = []
            
            # Basic information
            info_sections.append("# Model Information")
            info_sections.append(f"- **Checkpoint Path**: {ckpt_path}")
            info_sections.append(f"- **Number of Parameters**: {num_params:,}")
            info_sections.append(f"- **Input Shape (Guess)**: {input_shape_guess}")
            
            # Architecture analysis
            if architecture_analysis:
                info_sections.append("\n# Architecture Analysis")
                for insight in architecture_analysis:
                    info_sections.append(f"- {insight}")
            
            # Model keys
            info_sections.append("\n# Model Keys (Structure)")
            if show_all_keys or len(model_keys) <= 10:
                info_sections.append(f"All keys ({len(model_keys)} total):")
                for key in model_keys:
                    info_sections.append(f"- `{key}`")
            else:
                key_sample = model_keys[:10]
                info_sections.append("First 10 keys (enable 'show_all_keys' to see all):")
                for key in key_sample:
                    info_sections.append(f"- `{key}`")
                info_sections.append(f"... and {len(model_keys) - 10} more keys")
            
            # Layer details
            if include_layer_details:
                info_sections.append("\n# Layer Shapes")
                layer_info = []
                for key, shape in layer_shapes.items():
                    layer_info.append(f"- `{key}`: {shape}")
                
                # Show all or just the first 20 layers
                if show_all_keys or len(layer_info) <= 20:
                    info_sections.extend(layer_info)
                else:
                    info_sections.extend(layer_info[:20])
                    info_sections.append(f"... and {len(layer_info) - 20} more layers (enable 'show_all_keys' to see all)")
            
            # State dict (optional)
            if include_state_dict:
                info_sections.append("\n# State Dictionary Preview")
                info_sections.append("```python")
                state_dict_sample = {}
                for i, (k, v) in enumerate(state_dict.items()):
                    if i >= 5:  # Only show first 5 items
                        break
                    if isinstance(v, torch.Tensor):
                        state_dict_sample[k] = {
                            "shape": list(v.shape),
                            "dtype": str(v.dtype),
                            "sample": v.flatten()[:5].tolist() if v.numel() > 0 else []
                        }
                info_sections.append(json.dumps(state_dict_sample, indent=2))
                info_sections.append("```")
            
            # Load info
            info_sections.append("\n# Load Information")
            for info in load_info:
                info_sections.append(f"- {info}")
            
            # Create a model object for ComfyUI
            # For synthetic checkpoints, we'll just return None for the model
            model = None
            if "__synthetic__" not in checkpoint:
                try:
                    from comfy.model_base import BaseModel
                    model = BaseModel(state_dict)
                except Exception as model_e:
                    load_info.append(f"Warning: Could not create ComfyUI model object: {str(model_e)}")
            
            return "\n".join(info_sections), model
            
        except Exception as e:
            logging.error(f"Error in FL_ModelInspector: {str(e)}")
            return f"Error inspecting model: {str(e)}", None