# FL_ImagePicker: Interactive image selector with modal UI
import os
import io
import base64
import threading
import time
import uuid
import torch
import numpy as np
from PIL import Image
from aiohttp import web
from server import PromptServer
import execution

# Global storage for pending selections
pending_selections = {}


class InterruptProcessing(Exception):
    """Exception to interrupt ComfyUI processing."""
    pass


class FL_ImagePicker:
    """
    An interactive image selector node that pauses execution and displays
    a modal UI for the user to select which images to keep from a batch.
    """

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("selected_images", "selection_info")
    FUNCTION = "select_images"
    CATEGORY = "🏵️Fill Nodes/Image"

    THUMBNAIL_MAX_SIZE = 512  # Max dimension for thumbnails sent to frontend

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "timeout_seconds": ("INT", {"default": 300, "min": 30, "max": 3600, "step": 10}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    def _create_thumbnail(self, pil_img, max_size):
        """Create a thumbnail that fits within max_size while preserving aspect ratio."""
        width, height = pil_img.size

        # Only resize if larger than max_size
        if width <= max_size and height <= max_size:
            return pil_img

        # Calculate new size preserving aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        # Use LANCZOS for high-quality downscaling
        return pil_img.resize((new_width, new_height), Image.LANCZOS)

    def select_images(self, images, timeout_seconds, unique_id):
        batch_size = images.shape[0]

        if batch_size == 0:
            return (images, "No images in batch")

        # Generate a session ID for this selection
        session_id = f"{unique_id}_{uuid.uuid4().hex[:8]}"

        # Convert images to thumbnails for sending to frontend (faster loading)
        # Original full-res images are kept in pending_selections for output
        image_data = []
        print(f"[FL_ImagePicker] Generating thumbnails for {batch_size} images...")
        for i in range(batch_size):
            img_tensor = images[i]
            # Convert tensor to PIL
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # Store original dimensions before thumbnailing
            orig_width, orig_height = pil_img.size

            # Create thumbnail for frontend display
            thumbnail = self._create_thumbnail(pil_img, self.THUMBNAIL_MAX_SIZE)

            # Convert thumbnail to base64 (using JPEG for smaller size)
            buffered = io.BytesIO()
            thumbnail.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            image_data.append({
                "index": i,
                "data": f"data:image/jpeg;base64,{img_base64}",
                "width": orig_width,   # Report original dimensions
                "height": orig_height  # so user knows actual size
            })

        # Create event to block on
        event = threading.Event()
        pending_selections[session_id] = {
            "event": event,
            "selection": None,
            "cancelled": False,
            "images": images
        }

        print(f"[FL_ImagePicker] Session created: {session_id}")
        print(f"[FL_ImagePicker] Active sessions after create: {list(pending_selections.keys())}")

        # Send message to frontend to show selector
        PromptServer.instance.send_sync("fl_image_picker_show", {
            "session_id": session_id,
            "images": image_data,
            "batch_size": batch_size,
            "timeout_seconds": timeout_seconds
        })

        print(f"[FL_ImagePicker] Waiting for user selection... (timeout: {timeout_seconds}s)")

        # Block until user responds or timeout
        event_set = event.wait(timeout=timeout_seconds)

        # Get result
        result = pending_selections.get(session_id, {})
        selection = result.get("selection")
        cancelled = result.get("cancelled", False)

        # Clean up
        if session_id in pending_selections:
            del pending_selections[session_id]

        if not event_set:
            # Timeout - cancel the execution
            print(f"[FL_ImagePicker] Timeout reached. Cancelling execution...")
            nodes = execution.nodes
            if hasattr(nodes, 'interrupt_processing'):
                nodes.interrupt_processing()
            raise InterruptProcessing("Image selection timed out")

        if cancelled:
            # User cancelled - interrupt the execution
            print(f"[FL_ImagePicker] Selection cancelled by user. Interrupting execution...")
            # Set the interrupt flag in ComfyUI's execution module
            nodes = execution.nodes
            if hasattr(nodes, 'interrupt_processing'):
                nodes.interrupt_processing()
            # Raise an exception to stop this node's execution
            raise InterruptProcessing("Image selection was cancelled by user")

        if selection is None or len(selection) == 0:
            # No selection - return all
            print(f"[FL_ImagePicker] No images selected. Returning all {batch_size} images.")
            return (images, f"No selection: returned all {batch_size} images")

        # Filter to selected indices
        selected_indices = sorted(selection)
        selected_images = images[selected_indices]

        info = f"Selected {len(selected_indices)} of {batch_size} images: indices {selected_indices}"
        print(f"[FL_ImagePicker] {info}")

        return (selected_images, info)


# API endpoint to receive selection from frontend
@PromptServer.instance.routes.post("/fl_image_picker/select")
async def receive_selection(request):
    try:
        data = await request.json()
        session_id = data.get("session_id")
        selection = data.get("selection", [])  # List of indices
        cancelled = data.get("cancelled", False)

        if session_id in pending_selections:
            pending_selections[session_id]["selection"] = selection
            pending_selections[session_id]["cancelled"] = cancelled
            pending_selections[session_id]["event"].set()  # Unblock the waiting thread
            return web.json_response({"status": "ok", "received": len(selection)})
        else:
            return web.json_response({"status": "error", "message": "Session not found"}, status=404)
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)


# API endpoint to check if a session is still active (for frontend to detect if node was interrupted)
@PromptServer.instance.routes.get("/fl_image_picker/status/{session_id}")
async def check_status(request):
    session_id = request.match_info.get("session_id")
    if session_id in pending_selections:
        return web.json_response({"status": "active"})
    else:
        return web.json_response({"status": "inactive"})


# API endpoint to fetch full-res image on demand for preview
@PromptServer.instance.routes.get("/fl_image_picker/full_image/{session_id}/{index}")
async def get_full_image(request):
    try:
        session_id = request.match_info.get("session_id")
        index = int(request.match_info.get("index"))

        print(f"[FL_ImagePicker] Full image request: session={session_id}, index={index}")
        print(f"[FL_ImagePicker] Active sessions: {list(pending_selections.keys())}")

        if session_id not in pending_selections:
            return web.json_response({"status": "error", "message": f"Session not found. Active: {list(pending_selections.keys())}"}, status=404)

        images = pending_selections[session_id].get("images")
        if images is None:
            return web.json_response({"status": "error", "message": "Images not found"}, status=404)

        batch_size = images.shape[0]
        if index < 0 or index >= batch_size:
            return web.json_response({"status": "error", "message": f"Invalid index {index}"}, status=400)

        # Convert the requested image to full-res base64
        img_tensor = images[index]
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        # Convert to PNG for full quality
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return web.json_response({
            "status": "ok",
            "index": index,
            "data": f"data:image/png;base64,{img_base64}",
            "width": pil_img.width,
            "height": pil_img.height
        })
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)
