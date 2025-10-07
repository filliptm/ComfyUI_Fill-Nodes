# custom_nodes/FL_SystemCheck.py

import sys
import os
import platform
import psutil
import importlib
import json
from server import PromptServer
from aiohttp import web

class FL_SystemCheck:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "run_check"
    OUTPUT_NODE = True
    CATEGORY = "🏵️Fill Nodes/Utility"

    def run_check(self):
        return (True,)

def gather_system_info():
    def get_gpu_info():
        try:
            import torch
            return f"CUDA available: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CUDA not available"
        except ImportError:
            return "PyTorch not installed"

    def check_library_version(library):
        try:
            module = importlib.import_module(library)
            return module.__version__
        except ImportError:
            return "Not installed"

    def get_env_var(var):
        return os.environ.get(var, 'Not set')

    info = {
        "Python version": sys.version.split()[0],
        "Operating System": f"{platform.system()} {platform.release()}",
        "CPU": platform.processor() or "Unable to determine",
        "RAM": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
        "GPU": get_gpu_info(),
        "PyTorch": check_library_version('torch'),
        "torchvision": check_library_version('torchvision'),
        "xformers": check_library_version('xformers'),
        "numpy": check_library_version('numpy'),
        "Pillow": check_library_version('pillow'),
        "OpenCV": check_library_version('cv2'),
        "transformers": check_library_version('transformers'),
        "diffusers": check_library_version('diffusers'),
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["CUDA version"] = torch.version.cuda
    except:
        info["CUDA version"] = "Unable to determine"

    for var in ['PYTHONPATH', 'CUDA_HOME', 'LD_LIBRARY_PATH']:
        info[f"Env: {var}"] = get_env_var(var)

    return info

@PromptServer.instance.routes.get("/fl_system_info")
async def system_info(request):
    return web.json_response(gather_system_info())