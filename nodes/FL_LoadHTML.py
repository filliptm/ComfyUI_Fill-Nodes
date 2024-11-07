import os
from server import PromptServer
from aiohttp import web
import base64


class FL_LoadHTML:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "load_html"
    OUTPUT_NODE = True
    CATEGORY = "🏵️Fill Nodes/HTML"

    def load_html(self):
        return {}


# Add this to your server setup
@PromptServer.instance.routes.post("/fl_load_images")
async def load_images(request):
    data = await request.json()
    node_id = data.get("node_id")

    # Replace this with your actual image loading logic
    image_folder = "path/to/your/image/folder"
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    images = []
    for image_file in image_files:
        with open(os.path.join(image_folder, image_file), "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
            images.append(f"data:image/png;base64,{image_data}")

    PromptServer.instance.send_sync("fl_image_sequence", {"node_id": node_id, "images": images})

    return web.Response(text="Images loaded")