import os
import cv2
import numpy as np
import torch
from PIL import Image
import tempfile
from pathlib import Path
import uuid
import shutil
from typing import List, Tuple, Dict, Any, Optional
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from comfy.utils import ProgressBar, common_upscale

class FL_VideoCut:
    """
    A node that detects scene cuts in a batch of images (video frames) and outputs
    the segmented clips as MP4 files to a specified folder.
    """
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("video_paths", "output_folder",)
    FUNCTION = "detect_scene_cuts"
    CATEGORY = "🏵️Fill Nodes/Video"
    
    DESCRIPTION = """
    FL_SceneCut analyzes a batch of images representing video frames to detect scene cuts.
    It uses optimized intensity thresholding and histogram comparison to identify significant changes
    between frames that likely represent scene transitions. The node saves each detected scene
    as an individual MP4 file in the specified output folder and returns the paths to these files.
    You can adjust sensitivity parameters to fine-tune detection for different types of content.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_folder": ("STRING", {"default": "./output/scene_cuts"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "threshold": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 100.0, "step": 0.1,
                             "description": "Threshold for scene change detection (higher = less sensitive)"}),
                "min_scene_length": ("INT", {"default": 12, "min": 2, "max": 1000, "step": 1,
                                   "description": "Minimum number of frames required per scene segment (frames between scene boundaries)"}),
                "output_quality": ("INT", {"default": 85, "min": 1, "max": 100, "step": 1,
                                 "description": "Video quality for output clips (1-100)"}),
                "detection_method": (["intensity", "histogram", "hybrid"], {"default": "hybrid",
                                   "description": "Method used for scene detection - hybrid is faster and more accurate"}),
                "max_workers": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1,
                             "description": "Maximum number of parallel workers for video creation"}),
                "downsample_detection": ("BOOLEAN", {"default": True,
                              "description": "Downsample frames for faster detection"}),
                "use_gpu_acceleration": ("BOOLEAN", {"default": True,
                              "description": "Use GPU acceleration for detection when possible"})
            },
        }
    
    def detect_scene_cuts(self, images: torch.Tensor, output_folder: str, fps: float,
                         threshold: float, min_scene_length: int, output_quality: int,
                         detection_method: str = "hybrid", max_workers: int = 4,
                         downsample_detection: bool = True, use_gpu_acceleration: bool = True) -> Tuple[str, str]:
        """
        Detect scene cuts in a batch of images and save each scene as an MP4 file.
        
        Args:
            images: Batch of images as a tensor (B, H, W, C)
            output_folder: Path to save the output video clips
            fps: Frames per second for the output videos
            threshold: Threshold for scene change detection
            min_scene_length: Minimum number of frames per scene
            output_quality: Quality of output videos (1-100)
            detection_method: Method used for scene detection
            max_workers: Maximum number of parallel workers for video creation
            
        Returns:
            Tuple of (comma-separated list of video paths, output folder path)
        """
        print(f"[FL_SceneCut] Starting scene detection with method: {detection_method}")
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get batch size and dimensions
        batch_size, height, width, channels = images.shape
        
        print(f"[FL_SceneCut] Detecting scenes in {batch_size} frames ({width}x{height})...")
        
        # Keep images on GPU if available and requested
        use_gpu = use_gpu_acceleration and torch.cuda.is_available()
        device = 'cuda' if use_gpu else 'cpu'
        
        # Downsampling for faster detection (detection doesn't need full resolution)
        if downsample_detection and (width > 640 or height > 640):
            # Calculate scale factor to reduce to max 640px on larger dimension
            scale = 640 / max(width, height)
            detection_width = int(width * scale)
            detection_height = int(height * scale)
            print(f"[FL_SceneCut] Downsampling to {detection_width}x{detection_height} for detection...")
            
            # Initialize progress bar for downsampling
            pbar = ProgressBar(batch_size)
            pbar.update(0)
            print(f"[FL_SceneCut] Downsampling {batch_size} frames...")
            
            # Use common_upscale for efficient downsampling, which works on both CPU and GPU
            # Convert images to BCHW format for common_upscale
            images_bchw = images.permute(0, 3, 1, 2)
            
            # Process in smaller batches to avoid memory issues and provide progress updates
            max_batch_chunk = 64 if use_gpu else 16  # Smaller batch size for CPU
            detection_images_list = []
            
            for i in range(0, batch_size, max_batch_chunk):
                end_idx = min(i + max_batch_chunk, batch_size)
                
                # Get current batch chunk
                batch_chunk = images_bchw[i:end_idx]
                
                # Use common_upscale for efficient resizing (works on both CPU and GPU)
                if use_gpu:
                    batch_chunk = batch_chunk.to(device)
                
                # Perform the downsampling
                resized_chunk = common_upscale(
                    batch_chunk,
                    detection_width,
                    detection_height,
                    "lanczos",  # High quality downsampling
                    "disabled"
                )
                
                if use_gpu:
                    resized_chunk = resized_chunk.cpu()
                
                # Convert back to BHWC and add to result list
                resized_chunk = resized_chunk.permute(0, 2, 3, 1)
                detection_images_list.append(resized_chunk)
                
                # Update progress
                pbar.update_absolute(end_idx)
            
            # Combine processed batches
            detection_images = torch.cat(detection_images_list, dim=0)
            print(f"[FL_SceneCut] Downsampling complete")
        else:
            detection_images = images
            
        # Convert to numpy for detection if needed (some CV operations require numpy)
        if use_gpu:
            # Process in batches to reduce memory pressure and improve efficiency
            np_images = (images * 255).cpu().numpy().astype(np.uint8)  # Original full-res for video creation
            detection_np = (detection_images * 255).cpu().numpy().astype(np.uint8)
        else:
            np_images = (images * 255).cpu().numpy().astype(np.uint8)
            detection_np = (detection_images * 255).cpu().numpy().astype(np.uint8) if downsample_detection else np_images
        
        # Detect scene cuts - optimized algorithm based on detection_method
        scene_boundaries = self._detect_scenes_optimized(
            detection_np, threshold, min_scene_length, detection_method, use_gpu=use_gpu
        )
        
        # Create MP4 files for each scene with parallel processing
        video_paths = self._create_scene_videos_parallel(
            np_images, scene_boundaries, output_folder, fps, output_quality, max_workers
        )
        
        return ",".join(video_paths), output_folder
    
    def _detect_scenes_optimized(self, np_images: np.ndarray, threshold: float, min_scene_length: int,
                               detection_method: str, use_gpu: bool = False) -> List[int]:
        """
        Optimized scene detection algorithm based on specified method.
        
        Args:
            np_images: Batch of images as numpy array
            threshold: Detection threshold
            min_scene_length: Minimum number of frames per scene
            detection_method: Method used for detection
            use_gpu: Whether to use GPU acceleration for supported operations
            
        Returns:
            List of frame indices where scene cuts occur
        """
        batch_size = np_images.shape[0]
        scene_boundaries = [0]  # Start of first scene
        
        # Adjust thresholds based on method for better sensitivity
        if detection_method == "intensity":
            # Intensity method is fastest
            adjusted_threshold = threshold * 0.8
            detection_func = self._detect_by_intensity
        elif detection_method == "histogram":
            # Histogram is more accurate but slower
            adjusted_threshold = threshold * 3.0
            detection_func = self._detect_by_histogram
        else:  # hybrid
            # Hybrid uses a weighted combination for best results
            adjusted_threshold = threshold
            detection_func = self._detect_hybrid
        
        pbar = ProgressBar(batch_size - 1)
        print(f"[FL_SceneCut] Processing {batch_size} frames for scene detection...")
        
        # Pre-compute grayscale frames for faster processing if using intensity method
        gray_frames = None
        if detection_method in ["intensity", "hybrid"]:
            print("[FL_SceneCut] Pre-computing grayscale frames...")
            
            # Batch process grayscale conversion for better performance
            batch_size = np_images.shape[0]
            gray_frames = np.zeros((batch_size, np_images.shape[1], np_images.shape[2]), dtype=np.uint8)
            
            # Process in smaller batches to avoid memory issues
            batch_chunk_size = 100  # Process 100 frames at a time
            for i in range(0, batch_size, batch_chunk_size):
                end_idx = min(i + batch_chunk_size, batch_size)
                # Process a chunk of frames at once using OpenCV's batch processing
                for j in range(i, end_idx):
                    gray_frames[j] = cv2.cvtColor(np_images[j], cv2.COLOR_RGB2GRAY)
        
        # Process frames in pairs for scene detection
        for i in range(1, batch_size):
            # Use the appropriate detection function
            is_scene_cut = detection_func(np_images, gray_frames, i, i-1, adjusted_threshold)
            
            # If a scene cut is detected and minimum scene length is satisfied
            if is_scene_cut and (i - scene_boundaries[-1]) >= min_scene_length:
                # Add boundary at frame i (this will be the start of next scene, end of current scene)
                # NOTE: The boundary is EXCLUSIVE for the previous scene and INCLUSIVE for the next scene
                scene_boundaries.append(i)
                print(f"[FL_SceneCut] Scene cut detected at frame {i} (last frame of previous scene is {i-1}, first frame of next scene is {i})")
            
            pbar.update_absolute(i - 1)
        
        # Add the end of the last scene if not already added
        if scene_boundaries[-1] != batch_size - 1:
            scene_boundaries.append(batch_size - 1)
        
        num_scenes = len(scene_boundaries) - 1
        print(f"[FL_SceneCut] Detected {num_scenes} scenes")
        
        # Validate scene lengths
        valid_scenes = []
        invalid_count = 0
        for i in range(len(scene_boundaries) - 1):
            start = scene_boundaries[i]
            end = scene_boundaries[i + 1]
            # Scene length is (end - start) since end is exclusive
            if end - start >= min_scene_length:
                valid_scenes.append(scene_boundaries[i])
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"[FL_SceneCut] Removed {invalid_count} scenes that were too short")
        
        # Add the last boundary
        valid_scenes.append(scene_boundaries[-1])
        
        return valid_scenes
    
    def _detect_by_intensity(self, frames, gray_frames, curr_idx, prev_idx, threshold):
        """Fast intensity-based detection"""
        curr_gray = gray_frames[curr_idx]
        prev_gray = gray_frames[prev_idx]
        
        # Calculate absolute difference and mean
        frame_diff = cv2.absdiff(curr_gray, prev_gray)
        mean_diff = np.mean(frame_diff)
        
        return mean_diff > threshold
    
    def _detect_by_histogram(self, frames, _, curr_idx, prev_idx, threshold):
        """More accurate histogram-based detection"""
        curr_frame = frames[curr_idx]
        prev_frame = frames[prev_idx]
        
        # Calculate histograms with optimized bin sizes
        h_bins = 16  # Reduced for speed
        s_bins = 16
        histSize = [h_bins, s_bins]
        ranges = [0, 180, 0, 256]  # Hue is [0, 180], saturation is [0, 256]
        
        # Convert to HSV color space for better color comparison
        curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2HSV)
        prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2HSV)
        
        # Calculate histograms focusing on Hue and Saturation channels
        curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, histSize, ranges)
        prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, histSize, ranges)
        
        # Normalize histograms
        cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)
        
        # Compare histograms using Chi-Square method (good for color differences)
        hist_diff = cv2.compareHist(curr_hist, prev_hist, cv2.HISTCMP_CHISQR)
        
        return hist_diff > threshold
    
    def _detect_hybrid(self, frames, gray_frames, curr_idx, prev_idx, threshold):
        """Hybrid method combining intensity and edge detection for best results"""
        # Get intensity difference (fast pre-filter)
        # Use a lower threshold for this initial check (making it a pre-filter)
        intensity_diff_value = self._get_intensity_diff(gray_frames, curr_idx, prev_idx)
        
        # Fast early rejection - if intensity difference is very low
        if intensity_diff_value < threshold * 0.3:
            return False
        
        # Fast early acceptance - if intensity difference is very high
        if intensity_diff_value > threshold * 3.0:
            return True
            
        # Get edge-based difference (more robust but more expensive)
        curr_gray = gray_frames[curr_idx]
        prev_gray = gray_frames[prev_idx]
        
        # Use faster edge detection method - Sobel instead of Canny
        # Calculate x and y gradients
        sobelx = cv2.Sobel(curr_gray, cv2.CV_64F, 1, 0, ksize=3) - cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(curr_gray, cv2.CV_64F, 0, 1, ksize=3) - cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate the magnitude
        edge_diff = np.mean(np.sqrt(sobelx**2 + sobely**2))
        
        # Weighted decision based on both methods
        combined_score = (intensity_diff_value * 0.6) + (edge_diff * 0.4)
        return combined_score > threshold * 1.2
        
    def _get_intensity_diff(self, gray_frames, curr_idx, prev_idx):
        """Helper function to get intensity difference between two frames"""
        curr_gray = gray_frames[curr_idx]
        prev_gray = gray_frames[prev_idx]
        
        # Calculate absolute difference and mean - faster than full detection method
        frame_diff = cv2.absdiff(curr_gray, prev_gray)
        return np.mean(frame_diff)
    
    def _create_scene_videos_parallel(self, np_images: np.ndarray, scene_boundaries: List[int],
                                    output_folder: str, fps: float, output_quality: int,
                                    max_workers: int) -> List[str]:
        """
        Create MP4 files for each detected scene using parallel processing.
        Uses direct pipe to ffmpeg for dramatically improved performance.
        
        Args:
            np_images: Batch of images as numpy array
            scene_boundaries: List of frame indices where scene cuts occur
            output_folder: Path to save the output videos
            fps: Frames per second for the output videos
            output_quality: Quality of output videos
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of paths to the created video files
        """
        video_paths = []
        
        # Prepare scene data
        num_scenes = len(scene_boundaries) - 1
        scenes = []
        
        print(f"[FL_SceneCut] Preparing to create {num_scenes} video clips...")
        
        for scene_idx in range(num_scenes):
            start_frame = scene_boundaries[scene_idx]
            end_frame = scene_boundaries[scene_idx + 1]
            
            # Skip scenes that are too short
            if end_frame - start_frame < 3:  # At least 3 frames needed for a valid video
                continue
            
            scene_filename = os.path.join(output_folder, f"scene_{scene_idx:03d}_{uuid.uuid4().hex[:8]}.mp4")
            
            scenes.append({
                'idx': scene_idx,
                'start': start_frame,
                'end': end_frame,
                'filename': scene_filename,
                'frame_count': end_frame - start_frame  # Adjusted for exclusive end boundary
            })
        
        # Function to process one scene with direct ffmpeg pipe (much faster)
        def process_scene(scene):
            try:
                # Get frame dimensions
                height, width = np_images[0].shape[:2]
                
                # Calculate CRF (quality) - lower is better quality
                crf = str(int((100 - output_quality) / 4) + 1)
                
                # Set up ffmpeg command for piped input
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{width}x{height}",
                    "-pix_fmt", "rgb24",
                    "-r", str(fps),
                    "-i", "pipe:",  # Read from stdin
                    "-c:v", "libx264",
                    "-crf", crf,
                    "-preset", "veryfast",  # Balance between speed and compression
                    "-tune", "film",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-loglevel", "error",
                    scene['filename']
                ]
                
                # Start ffmpeg process with pipe
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Write frames directly to ffmpeg stdin
                # Important: scene['end'] is EXCLUSIVE - it's the index of the first frame of the next scene
                # This ensures no frames from the next scene get included in the current scene's video
                for i in range(scene['start'], scene['end']):
                    # Write raw bytes to ffmpeg
                    process.stdin.write(np_images[i].tobytes())
                
                # Close the pipe and wait for ffmpeg to finish
                process.stdin.close()
                process.wait()
                
                # Check if successful
                if process.returncode != 0:
                    stderr = process.stderr.read().decode('utf-8')
                    raise RuntimeError(f"FFmpeg error: {stderr}")
                
                return {'success': True, 'idx': scene['idx'], 'filename': scene['filename']}
                
            except Exception as e:
                print(f"[FL_SceneCut] Error processing scene {scene['idx']}: {str(e)}")
                return {'success': False, 'idx': scene['idx'], 'error': str(e)}
        
        # Process scenes in parallel
        actual_workers = min(max_workers, len(scenes))
        print(f"[FL_SceneCut] Processing {len(scenes)} scenes with {actual_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            future_to_scene = {executor.submit(process_scene, scene): scene for scene in scenes}
            
            pbar = ProgressBar(len(scenes))
            completed = 0
            
            for future in as_completed(future_to_scene):
                scene = future_to_scene[future]
                try:
                    result = future.result()
                    if result['success']:
                        video_paths.append(result['filename'])
                        print(f"[FL_SceneCut] Created scene {result['idx']+1}/{num_scenes} containing {scene['frame_count']} frames (frames {scene['start']} through {scene['end']-1})")
                except Exception as e:
                    print(f"[FL_SceneCut] Exception processing scene {scene['idx']}: {str(e)}")
                
                completed += 1
                pbar.update_absolute(completed)
        
        print(f"[FL_SceneCut] Successfully created {len(video_paths)} video clips")
        return sorted(video_paths)