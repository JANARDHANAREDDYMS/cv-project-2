import numpy as np
import cv2

def compute_depth_map(left_image_shape, matches, k=1):
    
    height, width = left_image_shape[:2]
    depth_map = np.zeros((height, width), dtype=np.uint8)

    z_values = []
    positions = []

    for (x1, y1), (x2, y2) in matches.items():
        disparity = abs(x1 - x2)
        if disparity == 0:
            continue  # Avoid division by zero
        z = k / disparity
        z_values.append(z)
        positions.append((int(y1), int(x1)))

    if len(z_values) == 0:
        return depth_map  # Return blank map if no valid matches

    z_values = np.array(z_values)
    z_min, z_max = np.min(z_values), np.max(z_values)

    if z_max == z_min:
        z_max += 1e-6  # Avoid division by zero

    for idx, z in enumerate(z_values):
        z_scaled = 255 - int(245 * ((z - z_min) / (z_max - z_min) + 0.5))
        z_scaled = np.clip(z_scaled, 10, 255)
        y, x = positions[idx]
        depth_map[y, x] = z_scaled

    return depth_map


def normalize_depth_map(depth_map):
   
    valid_depths = depth_map[depth_map > 0]
    if len(valid_depths) == 0:
        return np.zeros_like(depth_map, dtype=np.uint8)

    z_min, z_max = valid_depths.min(), valid_depths.max()
    if z_max == z_min:
        z_max += 1

    normalized = np.zeros_like(depth_map, dtype=np.uint8)
    scaled = 255 - ((245 * (depth_map - z_min) / (z_max - z_min)) + 0.5).astype(np.uint8)
    normalized[depth_map > 0] = scaled[depth_map > 0]

    return normalized


def visualize_depth_map(depth_map, window_name="Depth Map"):
    pass 