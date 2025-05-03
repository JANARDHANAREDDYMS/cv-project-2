import numpy as np
import cv2

def compute_depth_map(left_image_shape, matches, k=1):
    # Get the image dimensions 
    height, width = left_image_shape[:2]

    # Start with all zero depth map
    depth_map = np.zeros((height, width), dtype=np.uint8)

    # We will store each computed depth 'z' and the corresponding pixel position
    z_values = []
    positions = []

    # Loop through all the matching feature points
    for (x1, y1), (x2, y2) in matches.items():
        # Disparity is the difference in x-coordinates between the left and right points
        disparity = abs(x1 - x2)
        if disparity == 0:
            continue  # Skip if disparity is zero to avoid dividing by zero

        # Compute relative depth using the simple inverse formula z = k / disparity
        z = k / disparity
        z_values.append(z)
        positions.append((int(y1), int(x1)))  # Save (i, j) position for this depth

    # If no valid depth values were computed, just return the blank map
    if len(z_values) == 0:
        return depth_map

    # Convert depth values into a NumPy array to easily scale them
    z_values = np.array(z_values)
    z_min, z_max = np.min(z_values), np.max(z_values)

    # Prevent division by zero if all depths are the same
    if z_max == z_min:
        z_max += 1e-6

    # Loop through all computed z-values and map them to grayscale intensity
    for idx, z in enumerate(z_values):
        # Normalize the depth to the range 10–255 (darker = closer)
        z_scaled = 255 - int(245 * ((z - z_min) / (z_max - z_min) + 0.5))
        z_scaled = np.clip(z_scaled, 10, 255)  

        # Place the scaled depth value in the correct pixel location
        y, x = positions[idx]
        depth_map[y, x] = z_scaled

    return depth_map


def normalize_depth_map(depth_map):
    # Extract only the valid (not zero) depth values
    valid_depths = depth_map[depth_map > 0]
    if len(valid_depths) == 0:
        return np.zeros_like(depth_map, dtype=np.uint8) 

    # Find the min and max of the valid depth range
    z_min, z_max = valid_depths.min(), valid_depths.max()
    if z_max == z_min:
        z_max += 1  # Prevent divide-by-zero

    # Create a blank image to hold normalized depth values
    normalized = np.zeros_like(depth_map, dtype=np.uint8)

    # Scale all valid depth values to a 10–255 grayscale range
    scaled = 255 - ((245 * (depth_map - z_min) / (z_max - z_min)) + 0.5).astype(np.uint8)
    normalized[depth_map > 0] = scaled[depth_map > 0]

    return normalized
