import torch

class FL_VideoBatchSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frames_per_batch": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "output_count": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 20,
                    "step": 1
                }),
            }
        }

    # Define maximum possible outputs (20 as per max value in INPUT_TYPES)
    RETURN_TYPES = tuple(["IMAGE"] * 20)
    RETURN_NAMES = tuple([f"batch_{i+1}" for i in range(20)])
    FUNCTION = "split_video_batches"
    CATEGORY = "🏵️Fill Nodes/Video"

    def split_video_batches(self, images, frames_per_batch, output_count):
        # Get the number of frames in the batch
        total_frames = images.shape[0]
        
        if total_frames == 0:
            raise ValueError("[FL_VideoBatchSplitter] Error: Input batch is empty.")
        
        # Calculate expected total frames based on output count and frames per batch
        expected_total_frames = frames_per_batch * output_count
        
        if total_frames != expected_total_frames:
            raise ValueError(
                f"[FL_VideoBatchSplitter] Error: Frame count mismatch. "
                f"Expected {expected_total_frames} frames ({frames_per_batch} frames × {output_count} outputs), "
                f"but got {total_frames} frames. "
                f"Please adjust your input to have exactly {expected_total_frames} frames or "
                f"change the frames_per_batch/output_count values."
            )
        
        # Split the batch into exact batches
        batches = []
        for i in range(output_count):
            start_idx = i * frames_per_batch
            end_idx = start_idx + frames_per_batch
            batch = images[start_idx:end_idx]
            batches.append(batch)
        
        # Pad with None for unused outputs (ComfyUI expects exactly 20 outputs)
        while len(batches) < 20:
            batches.append(None)
        
        print(f"[FL_VideoBatchSplitter] Split {total_frames} frames into {output_count} batches of {frames_per_batch} frames each.")
        
        return tuple(batches)