import torch
import numpy as np
import json
import cv2
from PIL import Image
import torchvision.transforms as transforms
import gc
from typing import Tuple, Optional, List

import comfy.model_management as mm


# ── Trajectory integration (inlined from cotracker_node/trajectory_integration.py) ──

def create_mask_from_tracks(forward_tracks, forward_visibility, frame_shape, radius=10, frame_idx=None):
    H, W = frame_shape

    if frame_idx is not None:
        mask = np.zeros((H, W), dtype=np.uint8)
        points = forward_tracks[frame_idx]
        visibility = forward_visibility[frame_idx]
        valid_points = points[visibility > 0]
        for point in valid_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(mask, (x, y), radius, 1, -1)
        return mask
    else:
        T = forward_tracks.shape[0]
        masks = np.zeros((T, H, W), dtype=np.uint8)
        for t in range(T):
            points = forward_tracks[t]
            visibility = forward_visibility[t]
            valid_points = points[visibility > 0]
            for point in valid_points:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(masks[t], (x, y), radius, 1, -1)
        return masks


def detect_empty_regions(forward_tracks, forward_visibility, frame_shape, frame_idx, radius=10, min_region_size=100):
    mask = create_mask_from_tracks(forward_tracks, forward_visibility, frame_shape, radius, frame_idx)
    empty_mask = 1 - mask
    contours, _ = cv2.findContours(empty_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    empty_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_region_size:
            x, y, w, h = cv2.boundingRect(contour)
            empty_regions.append((x, y, w, h))
    return empty_regions


def has_data_in_region(backward_tracks, backward_visibility, spatial_region, frame_idx, min_points=1):
    x, y, w, h = spatial_region
    points = backward_tracks[frame_idx]
    visibility = backward_visibility[frame_idx]
    valid_points = points[visibility > 0]
    if len(valid_points) == 0:
        return False
    in_region = ((valid_points[:, 0] >= x) & (valid_points[:, 0] < x + w) &
                 (valid_points[:, 1] >= y) & (valid_points[:, 1] < y + h))
    return np.sum(in_region) >= min_points


def extract_trajectory_with_indices(tracks, visibility, spatial_region, frame_idx):
    x, y, w, h = spatial_region
    points = tracks[frame_idx]
    vis = visibility[frame_idx]
    valid_mask = vis > 0
    in_region_mask = ((points[:, 0] >= x) & (points[:, 0] < x + w) &
                      (points[:, 1] >= y) & (points[:, 1] < y + h))
    selected_indices = np.where(valid_mask & in_region_mask)[0]
    if len(selected_indices) == 0:
        return np.empty((tracks.shape[0], 0, 2)), np.empty((tracks.shape[0], 0)), []
    extracted_tracks = tracks[:, selected_indices, :]
    extracted_visibility = visibility[:, selected_indices]
    return extracted_tracks, extracted_visibility, selected_indices.tolist()


def time_reverse(tracks, visibility):
    return np.flip(tracks, axis=0), np.flip(visibility, axis=0)


def integrate_tracking_results(forward_tracks, forward_visibility,
                               backward_tracks, backward_visibility,
                               frame_shape, radius=10, min_region_size=100,
                               min_points=1, output_dir=None):
    T = forward_tracks.shape[0]
    backward_tracks, backward_visibility = time_reverse(backward_tracks, backward_visibility)

    integrated_tracks_list = [forward_tracks]
    integrated_visibility_list = [forward_visibility]
    global_used_indices = set()

    for frame_idx in range(T):
        print(f"Processing frame {frame_idx}...")
        current_integrated_tracks = np.concatenate(integrated_tracks_list, axis=1)
        current_integrated_visibility = np.concatenate(integrated_visibility_list, axis=1)

        empty_regions = detect_empty_regions(
            current_integrated_tracks, current_integrated_visibility,
            frame_shape, frame_idx, radius, min_region_size
        )
        print(f"  Found {len(empty_regions)} empty regions")

        frame_new_tracks = []
        frame_new_visibility = []

        for region_idx, spatial_region in enumerate(empty_regions):
            available_indices = [i for i in range(backward_tracks.shape[1]) if i not in global_used_indices]
            if len(available_indices) == 0:
                print(f"  Region {region_idx}: No more available backward tracks")
                continue

            available_backward_tracks = backward_tracks[:, available_indices, :]
            available_backward_visibility = backward_visibility[:, available_indices]

            if has_data_in_region(available_backward_tracks, available_backward_visibility, spatial_region, frame_idx, min_points):
                backward_trajectory, backward_vis, local_extracted_indices = extract_trajectory_with_indices(
                    available_backward_tracks, available_backward_visibility, spatial_region, frame_idx
                )
                if backward_trajectory.shape[1] > 0:
                    global_extracted_indices = [available_indices[i] for i in local_extracted_indices]
                    print(f"  Region {region_idx}: Extracted {len(global_extracted_indices)} tracks: {global_extracted_indices}")
                    frame_new_tracks.append(backward_trajectory)
                    frame_new_visibility.append(backward_vis)
                    global_used_indices.update(global_extracted_indices)
                    print(f"  Total used indices so far: {len(global_used_indices)}")
            else:
                print(f"  Region {region_idx}: No backward data found")

        if frame_new_tracks:
            integrated_tracks_list.extend(frame_new_tracks)
            integrated_visibility_list.extend(frame_new_visibility)

    integrated_tracks = np.concatenate(integrated_tracks_list, axis=1)
    integrated_visibility = np.concatenate(integrated_visibility_list, axis=1)

    print(f"\nFinal summary:")
    print(f"Used backward track indices: {sorted(global_used_indices)}")
    print(f"Total backward tracks used: {len(global_used_indices)}")
    print(f"Original backward tracks: {backward_tracks.shape[1]}")
    print(f"Final integrated tracks shape: {integrated_tracks.shape}")

    return integrated_tracks, integrated_visibility


def trajectory_integration(forward_tracks, forward_visibility, backward_tracks, backward_visibility, frame_shape, grid_size):
    ft = forward_tracks.squeeze(0).cpu().numpy()
    fv = forward_visibility.squeeze(0).cpu().numpy()
    bt = backward_tracks.squeeze(0).cpu().numpy()
    bv = backward_visibility.squeeze(0).cpu().numpy()

    radius = min(frame_shape) / grid_size * 1.5
    radius = max(int(round(radius)), 3)
    min_region_size = radius ** 2

    t, v = integrate_tracking_results(ft, fv, bt, bv, frame_shape,
                                      radius=radius, min_region_size=min_region_size, min_points=1, output_dir=None)

    t = torch.from_numpy(t).float().unsqueeze(0)
    v = torch.from_numpy(v).float().unsqueeze(0)
    return t, v


# ── Main node ──

class FL_CoTracker:

    def __init__(self):
        self.device = mm.get_torch_device()
        self.offload_device = mm.unet_offload_device()
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tracking_points": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Enter x and y coordinates separated by a newline. This is optional — normally not needed, as points with large motion are selected automatically. \nExample:\n500,300\n200,250"
                }),
                "grid_size": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of divisions along both width and height to create a grid of tracking points."
                }),
                "max_num_of_points": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 10000,
                    "step": 1
                }),
            },
            "optional": {
                "tracking_mask": ("MASK", {"tooltip": "Mask for grid coordinates"}),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.90,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "min_distance": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "tooltip": "Minimum distance between tracking points"
                }),
                "force_offload": ("BOOLEAN", {"default": True}),
                "enable_backward": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("tracking_results", "image_with_results")
    FUNCTION = "track_points"
    CATEGORY = "🏵️Fill Nodes/Image"
    DESCRIPTION = "https://github.com/facebookresearch/co-tracker \nIf you get an OOM error, try lowering the `grid_size`."

    def load_model(self, model_type):
        try:
            if self.model is None:
                print(f"Loading CoTracker model: {model_type}")
                self.model = torch.hub.load("facebookresearch/co-tracker", model_type).to(self.device)
            self.model.to(self.device)
            self.model.eval()
            print("CoTracker model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load CoTracker model: {str(e)}")

    def parse_tracking_points(self, tracking_points_str):
        points = []
        lines = tracking_points_str.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and ',' in line:
                try:
                    x, y = line.split(',')
                    points.append([float(x.strip()), float(y.strip())])
                except ValueError:
                    print(f"parse_tracking_points : Invalid point format: {line}")
                    continue
        return np.array(points)

    def preprocess_images(self, images):
        if len(images.shape) == 4:
            images = images.permute(0, 3, 1, 2)
            images = images.unsqueeze(0)
        images = images.float()
        images = images * 255
        return images.to(self.device)

    def prepare_query_points(self, points, video_shape):
        query_points_tensor = []
        for x, y in points:
            query_points_tensor.append([0, x, y])
        query_points_tensor = torch.tensor(query_points_tensor, dtype=torch.float32)
        query_points_tensor = query_points_tensor[None].to(self.device)
        return query_points_tensor

    def track_points(self, images, tracking_points, grid_size, max_num_of_points, tracking_mask=None, confidence_threshold=0.5, min_distance=60, force_offload=True, enable_backward=False):
        self.load_model("cotracker3_online")

        points = self.parse_tracking_points(tracking_points)
        if len(points) == 0:
            print("Info : No valid points found in tracking_points")

        if tracking_mask is not None:
            print(f"{tracking_mask.shape=}")

        images_np = images.cpu().numpy()
        images_np = np.ascontiguousarray((images_np * 255).astype(np.uint8))

        video = self.preprocess_images(images)

        queries = self.prepare_query_points(points, video.shape)

        if video.shape[1] <= self.model.step:
            print(f"{video.shape[1]=}")
            raise ValueError(f"At least {self.model.step+1} frames are required to perform tracking.")

        results = []

        def _tracking(video, grid_size, queries, add_support_grid):
            with torch.no_grad():
                self.model(
                    video_chunk=video,
                    is_first_step=True,
                    grid_size=grid_size,
                    queries=queries,
                    add_support_grid=add_support_grid
                )
                for ind in range(0, video.shape[1] - self.model.step, self.model.step):
                    pred_tracks, pred_visibility = self.model(
                        video_chunk=video[:, ind : ind + self.model.step * 2],
                        is_first_step=False,
                        grid_size=grid_size,
                        queries=queries,
                        add_support_grid=add_support_grid
                    )
                return pred_tracks, pred_visibility

        if len(points) > 0:
            print(f"forward - queries")
            pred_tracks, pred_visibility = _tracking(video, 0, queries, True)
            results, images_np = self.format_results(pred_tracks, pred_visibility, None, confidence_threshold, points, max_num_of_points, 1, images_np)
            print(f"{len(results)=}")
            if len(results) >= max_num_of_points:
                return (results,)
            max_num_of_points -= len(results)
        else:
            results = []

        if grid_size > 0:
            print(f"forward - grid")
            pred_tracks, pred_visibility = _tracking(video, grid_size, None, False)
            if enable_backward:
                pred_tracks_b, pred_visibility_b = _tracking(video.flip(1), grid_size, None, False)
                _, _, _, H, W = video.shape
                pred_tracks, pred_visibility = trajectory_integration(pred_tracks, pred_visibility, pred_tracks_b, pred_visibility_b, (H, W), grid_size)
            results2, images_np = self.format_results(pred_tracks, pred_visibility, tracking_mask, confidence_threshold, points, max_num_of_points, min_distance, images_np, enable_backward)
            print(f"{len(results2)=}")
            results = results + results2

        images_with_markers = torch.from_numpy(images_np)
        images_with_markers = images_with_markers.float() / 255.0

        if force_offload:
            self.model.to(self.offload_device)
            mm.soft_empty_cache()
            gc.collect()

        return (results, images_with_markers)

    def select_diverse_points(self, motion_sorted_indices, tracks, visibility, max_points, min_distance):
        if len(motion_sorted_indices) == 0:
            return []

        selected_indices = []
        representative_positions = {}

        for point_idx in motion_sorted_indices:
            valid_frames = visibility[:, point_idx] == True
            if np.any(valid_frames):
                valid_positions = tracks[valid_frames, point_idx]
                representative_positions[point_idx] = np.mean(valid_positions, axis=0)
            else:
                representative_positions[point_idx] = np.mean(tracks[:, point_idx], axis=0)

        for candidate_idx in motion_sorted_indices:
            if len(selected_indices) >= max_points:
                break
            candidate_pos = representative_positions[candidate_idx]
            too_close = False
            for selected_idx in selected_indices:
                selected_pos = representative_positions[selected_idx]
                distance = np.linalg.norm(candidate_pos - selected_pos)
                if distance < min_distance:
                    too_close = True
                    break
            if not too_close:
                selected_indices.append(candidate_idx)

        return selected_indices

    def select_points(self, tracks, visibility, vis_threshold=0.5, max_points=9, min_distance=60):
        n_frames, n_points, _ = tracks.shape

        avg_visibility = np.mean(visibility, axis=0)
        valid_points = avg_visibility >= vis_threshold
        valid_indices = np.where(valid_points)[0]

        print(f"{len(valid_points)=}")
        print(f"{len(valid_indices)=}")

        if len(valid_indices) == 0:
            print("Warning: No points meet the confidence criteria")
            return []

        motion_magnitudes = []
        for point_idx in valid_indices:
            total_motion = 0.0
            valid_frame_count = 0
            for frame_idx in range(n_frames - 1):
                if (visibility[frame_idx, point_idx] == True and
                    visibility[frame_idx + 1, point_idx] == True):
                    pos1 = tracks[frame_idx, point_idx]
                    pos2 = tracks[frame_idx + 1, point_idx]
                    distance = np.linalg.norm(pos2 - pos1)
                    total_motion += distance
                    valid_frame_count += 1
            avg_motion = total_motion / max(valid_frame_count, 1)
            motion_magnitudes.append(avg_motion)

        motion_magnitudes = np.array(motion_magnitudes)

        selected_indices = []
        if False:
            selected_indices = valid_indices.tolist()
        else:
            motion_sorted_indices = valid_indices[np.argsort(motion_magnitudes)[::-1]]
            high_motion_indices = self.select_diverse_points(
                motion_sorted_indices, tracks, visibility, max_points=max_points-1, min_distance=min_distance
            )
            selected_indices.extend(high_motion_indices)
            if len(selected_indices) < max_points:
                remaining_indices = [idx for idx in motion_sorted_indices if idx not in selected_indices]
                if len(remaining_indices) > 0:
                    remaining_motions = [motion_magnitudes[np.where(valid_indices == idx)[0][0]]
                                        for idx in remaining_indices]
                    min_motion_idx = remaining_indices[np.argmin(remaining_motions)]
                    selected_indices.append(min_motion_idx)

        return selected_indices

    def format_results(self, tracks, visibility, mask, confidence_threshold, original_points, max_points, min_distance, images_np, enable_backward=False):
        tracks = tracks.squeeze(0).cpu().numpy()
        visibility = visibility.squeeze(0).cpu().numpy()

        if enable_backward:
            confidence_threshold = 0

        num_frames, num_points, _ = tracks.shape

        def filter_by_mask(trs, vis, mask):
            if mask is not None:
                mask = mask.cpu().numpy()
                while mask.ndim > 2 and mask.shape[0] == 1:
                    mask = mask[0]
                # Handle multi-channel masks (H, W, C) by taking first channel
                if mask.ndim == 3:
                    mask = mask[:, :, 0]

                initial_coords = trs[0]
                masked_indices = []
                for n in range(initial_coords.shape[0]):
                    x, y = initial_coords[n]
                    if (0 <= int(x) < mask.shape[1] and
                        0 <= int(y) < mask.shape[0] and
                        mask[int(y), int(x)] > 0):
                        masked_indices.append(n)

                if len(masked_indices) > 0:
                    filtered_tracks = trs[:, masked_indices]
                    filtered_visibility = vis[:, masked_indices]
                else:
                    filtered_tracks = np.empty((tracks.shape[0], 0, 2))
                    filtered_visibility = np.empty((visibility.shape[0], 0))
                return filtered_tracks, filtered_visibility
            else:
                return trs, vis

        tracks, visibility = filter_by_mask(tracks, visibility, mask)

        selected_indices = self.select_points(tracks, visibility, vis_threshold=confidence_threshold, max_points=max_points, min_distance=min_distance)

        marker_radius = 3
        marker_thickness = -1
        marker_color = (255, 0, 0)

        point_results = []
        for point_idx in selected_indices:
            point_track = []
            for frame_idx in range(num_frames):
                x, y = tracks[frame_idx, point_idx]
                vis = visibility[frame_idx, point_idx]

                if vis == True:
                    point_track.append({
                        "x": int(x),
                        "y": int(y),
                    })
                else:
                    if enable_backward:
                        point_track.append({
                            "x": -100,
                            "y": -100,
                        })
                        x = -100
                        y = -100
                    else:
                        if len(point_track) > 0:
                            last_point = point_track[-1].copy()
                            point_track.append(last_point)
                            x = last_point["x"]
                            y = last_point["y"]
                        else:
                            point_track.append({
                                "x": int(x),
                                "y": int(y),
                            })

                if frame_idx < images_np.shape[0]:
                    cv2.circle(images_np[frame_idx], (int(x), int(y)), marker_radius, marker_color, marker_thickness)

            point_results += [json.dumps(point_track)]

        return point_results, images_np
