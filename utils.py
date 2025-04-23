import cv2
import numpy as np

def draw_keypoints(image, keypoints, color=(0, 255, 0)):
    # This function draws small circles on the image at the given keypoints, useful to visually verify if corners or other points were detected correctly.

    img_with_kp = image.copy()
    
    # If it's a grayscale image, convert to color so we can draw in color.
    if len(img_with_kp.shape) == 2:
        img_with_kp = cv2.cvtColor(img_with_kp, cv2.COLOR_GRAY2BGR)

    # Draw a small circle at each keypoint (x, y).
    for x, y in keypoints:
        cv2.circle(img_with_kp, (int(x), int(y)), 3, color, -1)

    return img_with_kp


def draw_matches(img1, kp1, img2, kp2, matches):
    # This function creates a side-by-side visualization of matching keypoints between two images.

    # Create a blank canvas wide enough to place both images side by side.
    h = max(img1.shape[0], img2.shape[0])
    w = img1.shape[1] + img2.shape[1]

    # Convert grayscale images to color so we can draw colorful lines and points.
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    combined = np.zeros((h, w, 3), dtype=np.uint8)
    combined[:img1.shape[0], :img1.shape[1]] = img1
    combined[:img2.shape[0], img1.shape[1]:] = img2

    # Draw lines between matched keypoints
    for (x1, y1, x2, y2) in matches:
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2 + img1.shape[1]), int(y2))  # Adjust x2 for right image's offset

        cv2.circle(combined, pt1, 4, (0, 255, 0), -1)  # Green for left point
        cv2.circle(combined, pt2, 4, (0, 0, 255), -1)  # Red for right point
        cv2.line(combined, pt1, pt2, (255, 0, 0), 1)   # Blue line connecting them

    return combined


def save_image(path, image):
    # Just a simple wrapper to save an image using OpenCV
    cv2.imwrite(path, image)


def normalize_image(img):
    # Normalizes an image so that its values span the full range [0, 255]
    # This is helpful for display or saving when your image has float or narrow value ranges
    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def draw_corners(image, corners, save_path):
    # Draws a red '+' marker at each corner and saves the image to disk
    img_with_corners = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    for (x, y) in corners:
        cv2.drawMarker(img_with_corners, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
    cv2.imwrite(save_path, img_with_corners)


def display_and_save_depth_map(depth_map, filename="outputs/depth_map.png"):
    cv2.imwrite(filename, depth_map)


def save_corners_to_ascii(corners, filename):
    # Saves Harris corners to a .txt file in the format:
    # First line = number of corners
    # Next lines = (i, j) = (row, column) for each corner, sorted top-to-bottom and left-to-right

    corners = sorted(corners, key=lambda pt: (pt[0], pt[1]))  # Sort by y, then x 
    with open(filename, 'w') as f:
        f.write(f"{len(corners)}\n")
        for x, y in corners:
            f.write(f"{y} {x}\n") 


def save_matches_and_depth(matches, disparities, depth_map, filename):
    # Saves matching feature pairs and their depth/correlation info to a .txt file.
    # Format:
    # First line: number of matches
    # -Each line: i_left j_left i_right j_right correlation depth

    with open(filename, 'w') as f:
        f.write(f"{len(matches)}\n")
        for ((x1, y1), (x2, y2)) in matches.items():
            disparity = x1 - x2
            if disparity == 0:
                continue 
            z = 1 / disparity
            r = disparities.get((x1, y1), 0.0)
            z_scaled = int(depth_map[y1, x1]) 
            f.write(f"{y1} {x1} {y2} {x2} {r:.4f} {z_scaled}\n")