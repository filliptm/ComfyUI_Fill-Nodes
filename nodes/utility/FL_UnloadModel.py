import comfy.model_management as model_management
import gc
import torch
import time

# Note: This doesn't work with reroute for some reason?
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

class FL_UnloadModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": (any, )}, # For passthrough
            "optional": {"model": (any, )},
        }
    
    RETURN_TYPES = (any, )
    FUNCTION = "route"
    CATEGORY = "🏵️Fill Nodes/Utility"
    
    def route(self, **kwargs):
        print("FL Unload Model:")
        loaded_models = model_management.loaded_models()
        if kwargs.get("model") in loaded_models:
            print(" - Model found in memory, unloading...")
            loaded_models.remove(kwargs.get("model"))
        else:
            # Just delete it, I need the VRAM!
            model = kwargs.get("model")
            if type(model) == dict:
                keys = [(key, type(value).__name__) for key, value in model.items()]
                for key, model_type in keys:
                    if key == 'model':
                        print(f"Unloading model of type {model_type}")
                        del model[key]
                        # Emptying the cache after this should free the memory.
        model_management.free_memory(1e30, model_management.get_torch_device(), loaded_models)
        model_management.soft_empty_cache(True)
        try:
            print(" - Clearing Cache...")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except:
            print("   - Unable to clear cache")
        
        return (list(kwargs.values()))

class FL_UnloadAllModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": (any, )},
        }
    
    RETURN_TYPES = (any, )
    FUNCTION = "route"
    CATEGORY = "🏵️Fill Nodes/Utility"
    
    def route(self, **kwargs):
        print("FL Unload All Models:")
        print(" - Unloading all models...")
        model_management.unload_all_models()
        model_management.soft_empty_cache(True)
        try:
            print(" - Clearing Cache...")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except:
            print("   - Unable to clear cache")
        
        return (list(kwargs.values()))