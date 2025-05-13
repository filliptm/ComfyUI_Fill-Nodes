import os
import cv2
import numpy as np
import torch
from comfy.utils import ProgressBar, common_upscale
from typing import List, Tuple

class FL_VideoCadence:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("scene_lengths_str",)
    FUNCTION = "get_scene_cadence"
    CATEGORY = "🏵️Fill Nodes/Video"

    DESCRIPTION = """
    Analyzes a batch of images (video frames) to detect scene changes and outputs
    a comma-separated string of frame counts for each detected scene.
    FPS input is for user context when setting frame-based parameters like min_scene_length.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1, "description": "FPS of the input image sequence (for user context)"}),
                "threshold": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 100.0, "step": 0.1,
                                     "description": "Threshold for scene change detection (higher = less sensitive)"}),
                "min_scene_length": ("INT", {"default": 12, "min": 2, "max": 1000, "step": 1,
                                           "description": "Minimum number of frames required per scene segment"}),
                "detection_method": (["intensity", "histogram", "hybrid"], {"default": "hybrid",
                                           "description": "Method used for scene detection - hybrid is often a good balance"}),
                "downsample_detection": ("BOOLEAN", {"default": True,
                                      "description": "Downsample frames for faster detection"}),
                "use_gpu_acceleration": ("BOOLEAN", {"default": True,
                                      "description": "Use GPU acceleration for detection when possible"})
            },
        }

    def get_scene_cadence(self, images: torch.Tensor, fps: int,
                         threshold: float, min_scene_length: int,
                         detection_method: str = "hybrid",
                         downsample_detection: bool = True, use_gpu_acceleration: bool = True) -> Tuple[str]:
        
        print(f"[FL_SceneCadence] Starting scene cadence analysis with method: {detection_method}")
        
        batch_size, height, width, channels = images.shape
        print(f"[FL_SceneCadence] Analyzing {batch_size} frames ({width}x{height}). FPS context: {fps}")
        
        use_gpu = use_gpu_acceleration and torch.cuda.is_available()
        device = 'cuda' if use_gpu else 'cpu'
        
        detection_images = images
        if downsample_detection and (width > 640 or height > 640):
            scale = 640 / max(width, height)
            detection_width = int(width * scale)
            detection_height = int(height * scale)
            print(f"[FL_SceneCadence] Downsampling to {detection_width}x{detection_height} for detection...")
            
            pbar_ds = ProgressBar(batch_size)
            images_bchw = images.permute(0, 3, 1, 2)
            max_batch_chunk = 64 if use_gpu else 16
            detection_images_list = []
            
            for i in range(0, batch_size, max_batch_chunk):
                end_idx = min(i + max_batch_chunk, batch_size)
                batch_chunk = images_bchw[i:end_idx]
                if use_gpu:
                    batch_chunk = batch_chunk.to(device)
                
                resized_chunk = common_upscale(batch_chunk, detection_width, detection_height, "bilinear", "disabled")
                
                if use_gpu:
                    resized_chunk = resized_chunk.cpu()
                
                detection_images_list.append(resized_chunk.permute(0, 2, 3, 1))
                pbar_ds.update_absolute(end_idx)
            
            detection_images = torch.cat(detection_images_list, dim=0)
            print(f"[FL_SceneCadence] Downsampling complete")

        if use_gpu:
            detection_np = (detection_images * 255).cpu().numpy().astype(np.uint8)
        else:
            detection_np = (detection_images * 255).cpu().numpy().astype(np.uint8)
        
        scene_boundaries = self._detect_scenes_optimized(
            detection_np, threshold, min_scene_length, detection_method, use_gpu=use_gpu
        )
        
        scene_lengths = []
        if len(scene_boundaries) > 1:
            for i in range(len(scene_boundaries) - 1):
                start_frame_idx = scene_boundaries[i]
                end_frame_idx = scene_boundaries[i+1] # This is the start of the next scene (exclusive for current)
                length = end_frame_idx - start_frame_idx
                scene_lengths.append(length)
        
        scene_lengths_str = ",".join(map(str, scene_lengths))
        print(f"[FL_SceneCadence] Detected scene lengths (frames): {scene_lengths_str}")
        
        return (scene_lengths_str,)

    def _detect_scenes_optimized(self, np_images: np.ndarray, threshold: float, min_scene_length: int,
                               detection_method: str, use_gpu: bool = False) -> List[int]:
        batch_size = np_images.shape[0]
        if batch_size == 0:
            return []
        if batch_size == 1: # Single frame is a single scene of that frame's length
            return [0, 1]


        scene_boundaries = [0]
        
        if detection_method == "intensity":
            adjusted_threshold = threshold * 0.8
            detection_func = self._detect_by_intensity
        elif detection_method == "histogram":
            adjusted_threshold = threshold * 3.0 
            detection_func = self._detect_by_histogram
        else:  # hybrid
            adjusted_threshold = threshold
            detection_func = self._detect_hybrid
        
        pbar = ProgressBar(batch_size - 1)
        print(f"[FL_SceneCadence] Processing {batch_size} frames for scene detection...")
        
        gray_frames = None
        if detection_method in ["intensity", "hybrid"]:
            gray_frames = np.zeros((batch_size, np_images.shape[1], np_images.shape[2]), dtype=np.uint8)
            batch_chunk_size = 100
            for i in range(0, batch_size, batch_chunk_size):
                end_idx = min(i + batch_chunk_size, batch_size)
                for j in range(i, end_idx):
                    gray_frames[j] = cv2.cvtColor(np_images[j], cv2.COLOR_RGB2GRAY)
        
        for i in range(1, batch_size):
            is_scene_cut = detection_func(np_images, gray_frames, i, i-1, adjusted_threshold)
            
            if is_scene_cut and (i - scene_boundaries[-1]) >= min_scene_length:
                scene_boundaries.append(i)
            
            pbar.update_absolute(i) # Iterate up to batch_size-1 comparisons
        
        # Ensure the final boundary is batch_size to correctly calculate length of the last scene
        if scene_boundaries[-1] != batch_size:
            # If the last detected boundary was too close to batch_size to satisfy min_scene_length for a new scene,
            # or if no cuts made it to the end, ensure the last scene extends to batch_size.
            # Check if adding batch_size creates a valid final scene
            if (batch_size - scene_boundaries[-1]) >= min_scene_length:
                 scene_boundaries.append(batch_size)
            elif len(scene_boundaries) > 1 : # if there was at least one cut
                 # The last segment is too short, merge it with the previous one by replacing last boundary
                 scene_boundaries[-1] = batch_size
            else: # No cuts at all, or first scene too short to make it to batch_size
                 scene_boundaries = [0, batch_size]


        # Filter out scenes that became too short after adjustments or due to boundary conditions
        # This step is crucial if min_scene_length is strictly enforced for all segments.
        # The previous logic for appending boundaries already considers min_scene_length for *new* cuts.
        # This re-validates all segments.
        if len(scene_boundaries) > 1:
            valid_scene_boundaries = [scene_boundaries[0]]
            for i in range(len(scene_boundaries) -1):
                current_start = valid_scene_boundaries[-1]
                potential_end = scene_boundaries[i+1]
                if (potential_end - current_start) >= min_scene_length:
                    valid_scene_boundaries.append(potential_end)
                elif i == len(scene_boundaries) - 2: # Last potential segment
                    # If last segment is too short, extend the previous one
                    if len(valid_scene_boundaries) > 1: # Ensure there is a previous one to extend
                         valid_scene_boundaries[-1] = potential_end
                    # else: # This means even the first scene is too short, handled by initial check or becomes [0, batch_size]
            scene_boundaries = valid_scene_boundaries
            
            # Ensure the very last boundary is batch_size if any valid scenes exist
            if scene_boundaries[-1] != batch_size and len(scene_boundaries) > 1:
                scene_boundaries[-1] = batch_size # Force last boundary to be total frames if it was adjusted
            elif len(scene_boundaries) == 1 and scene_boundaries[0] == 0 : # Only [0] left, means all scenes too short
                 scene_boundaries.append(batch_size) # Make it one scene of total length if all else fails

        if not scene_boundaries or scene_boundaries == [0]: # Handle edge case of no frames or no valid scenes
            if batch_size > 0: scene_boundaries = [0, batch_size]
            else: return []


        num_scenes = len(scene_boundaries) - 1
        print(f"[FL_SceneCadence] Detected {num_scenes} scenes with boundaries: {scene_boundaries}")
        
        return scene_boundaries

    def _get_intensity_diff(self, gray_frames, curr_idx, prev_idx):
        curr_gray = gray_frames[curr_idx]
        prev_gray = gray_frames[prev_idx]
        frame_diff = cv2.absdiff(curr_gray, prev_gray)
        return np.mean(frame_diff)

    def _detect_by_intensity(self, frames, gray_frames, curr_idx, prev_idx, threshold):
        return self._get_intensity_diff(gray_frames, curr_idx, prev_idx) > threshold
    
    def _detect_by_histogram(self, frames, _, curr_idx, prev_idx, threshold):
        curr_frame = frames[curr_idx]
        prev_frame = frames[prev_idx]
        
        h_bins, s_bins = 16, 16
        histSize = [h_bins, s_bins]
        ranges = [0, 180, 0, 256]
        
        curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2HSV)
        prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2HSV)
        
        curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, histSize, ranges)
        prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, histSize, ranges)
        
        cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)
        
        hist_diff = cv2.compareHist(curr_hist, prev_hist, cv2.HISTCMP_CHISQR)
        return hist_diff > threshold
    
    def _detect_hybrid(self, frames, gray_frames, curr_idx, prev_idx, threshold):
        intensity_diff_value = self._get_intensity_diff(gray_frames, curr_idx, prev_idx)
        
        if intensity_diff_value < threshold * 0.3: return False
        if intensity_diff_value > threshold * 3.0: return True
            
        curr_gray = gray_frames[curr_idx]
        prev_gray = gray_frames[prev_idx]
        
        sobelx = cv2.Sobel(curr_gray, cv2.CV_64F, 1, 0, ksize=3) - cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(curr_gray, cv2.CV_64F, 0, 1, ksize=3) - cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_diff = np.mean(np.sqrt(sobelx**2 + sobely**2))
        
        combined_score = (intensity_diff_value * 0.6) + (edge_diff * 0.4)
        return combined_score > threshold * 1.2