import os
import cv2
import numpy as np
import torch
from comfy.utils import ProgressBar
import glob
from typing import List, Tuple, Optional

def load_video_frames_and_crop(video_path: str, target_frame_count: int, device: str) -> Optional[torch.Tensor]:
    """Loads frames from a video, crops to target_frame_count from the middle, and returns as a tensor."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[FL_SceneCadenceCompile] Error: Could not open video file {video_path}")
            return None

        video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if video_total_frames == 0 or frame_width == 0 or frame_height == 0:
            print(f"[FL_SceneCadenceCompile] Error: Video file {video_path} has zero frames or dimensions.")
            cap.release()
            return None

        frames_to_process = []

        if video_total_frames == target_frame_count:
            start_idx = 0
            end_idx = video_total_frames
        elif video_total_frames > target_frame_count:
            drop_total = video_total_frames - target_frame_count
            drop_start = drop_total // 2
            # drop_end = drop_total - drop_start # Not needed directly, end_idx calculated from start_idx
            start_idx = drop_start
            end_idx = start_idx + target_frame_count
        else: # video_total_frames < target_frame_count
            print(f"[FL_SceneCadenceCompile] Error: Video {video_path} has {video_total_frames} frames, but cadence requires {target_frame_count}. Skipping.")
            cap.release()
            return None
        
        current_frame_pos = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx) # Seek to start_idx
        current_frame_pos = start_idx

        while current_frame_pos < end_idx:
            ret, frame = cap.read()
            if not ret:
                print(f"[FL_SceneCadenceCompile] Warning: Could not read frame {current_frame_pos} from {video_path} (expected up to {end_idx-1}).")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_to_process.append(frame_rgb)
            current_frame_pos += 1
        
        cap.release()

        if not frames_to_process:
            print(f"[FL_SceneCadenceCompile] Error: No frames processed for {video_path} with target {target_frame_count} frames.")
            return None
        
        # Ensure all frames collected match the target count after attempting to read
        if len(frames_to_process) != target_frame_count:
            print(f"[FL_SceneCadenceCompile] Warning: Expected {target_frame_count} frames from {video_path} after cropping, but got {len(frames_to_process)}. This might be due to issues reading frames.")
            # Decide if this is a hard error or if we proceed with what we have. For now, let's make it an error if not exact.
            if len(frames_to_process) < target_frame_count:
                 print(f"[FL_SceneCadenceCompile] Error: Not enough frames collected for {video_path}. Skipping.")
                 return None


        np_frames = np.stack(frames_to_process, axis=0) # (count, H, W, C)
        tensor_frames = torch.from_numpy(np_frames).float() / 255.0
        return tensor_frames.to(device)

    except Exception as e:
        print(f"[FL_SceneCadenceCompile] Exception processing video {video_path}: {e}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return None

class FL_VideoCadenceCompile:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "compile_from_cadence"
    CATEGORY = "🏵️Fill Nodes/Video"

    DESCRIPTION = """
    Takes a cadence list (frame counts) and a directory of video files.
    Crops each video from the middle to match its cadence frame count.
    Concatenates all processed frames into a single image batch.
    Handles errors like count mismatches or dimension inconsistencies.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cadence_list_str": ("STRING", {"default": "24,48,24", "multiline": False}),
                "video_directory": ("STRING", {"default": "./input/videos_for_cadence"}),
                "use_gpu_acceleration": ("BOOLEAN", {"default": True}),
            },
        }

    def compile_from_cadence(self, cadence_list_str: str, video_directory: str, use_gpu_acceleration: bool) -> Tuple[Optional[torch.Tensor]]:
        try:
            cadences = [int(c.strip()) for c in cadence_list_str.split(',') if c.strip()]
            if not cadences:
                print("[FL_SceneCadenceCompile] Error: Cadence list is empty or invalid.")
                return (None,)
        except ValueError:
            print("[FL_SceneCadenceCompile] Error: Cadence list contains non-integer values.")
            return (None,)

        if not os.path.isdir(video_directory):
            print(f"[FL_SceneCadenceCompile] Error: Video directory '{video_directory}' not found.")
            return (None,)

        video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(video_directory, ext)))
        
        video_files.sort() # Ensure consistent order

        if not video_files:
            print(f"[FL_SceneCadenceCompile] Error: No video files found in '{video_directory}'.")
            return (None,)

        if len(cadences) != len(video_files):
            print(f"[FL_SceneCadenceCompile] Error: Mismatch between number of cadences ({len(cadences)}) and video files ({len(video_files)}).")
            return (None,)

        device = 'cuda' if use_gpu_acceleration and torch.cuda.is_available() else 'cpu'
        print(f"[FL_SceneCadenceCompile] Using device: {device}")

        all_processed_frames_tensors: List[torch.Tensor] = []
        target_height, target_width = -1, -1
        
        pbar = ProgressBar(len(video_files))
        print(f"[FL_SceneCadenceCompile] Processing {len(video_files)} videos...")

        for i, (video_path, target_frames) in enumerate(zip(video_files, cadences)):
            print(f"[FL_SceneCadenceCompile] Processing '{os.path.basename(video_path)}' with target {target_frames} frames.")
            
            processed_video_tensor = load_video_frames_and_crop(video_path, target_frames, device)

            if processed_video_tensor is None:
                print(f"[FL_SceneCadenceCompile] Error: Failed to process video {video_path}. Aborting.")
                return (None,) # Hard stop on first video processing error

            current_frames, h, w, c = processed_video_tensor.shape
            
            if target_height == -1 and target_width == -1: # First successfully processed video sets the dimensions
                target_height, target_width = h, w
                print(f"[FL_SceneCadenceCompile] Batch dimensions set to: {target_height}x{target_width}")
            elif h != target_height or w != target_width:
                print(f"[FL_SceneCadenceCompile] Error: Dimension mismatch in video {video_path}. Expected {target_height}x{target_width}, got {h}x{w}. Aborting.")
                return (None,)
            
            all_processed_frames_tensors.append(processed_video_tensor)
            pbar.update(1)

        if not all_processed_frames_tensors:
            print("[FL_SceneCadenceCompile] Error: No video frames were successfully processed.")
            return (None,)

        try:
            final_batch = torch.cat(all_processed_frames_tensors, dim=0) # Concatenate along the batch dimension (count)
            print(f"[FL_SceneCadenceCompile] Successfully compiled. Final batch shape: {final_batch.shape}")
            return (final_batch,)
        except Exception as e:
            print(f"[FL_SceneCadenceCompile] Error during final concatenation: {e}")
            return (None,)