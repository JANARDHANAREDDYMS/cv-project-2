import cv2
import glob
import numpy as np

#I have separate python file for every feature step in our project

from harris import detect_harris_corners
from matcher import match_features
from depth import compute_depth_map, normalize_depth_map
from utils import draw_corners, display_and_save_depth_map, save_corners_to_ascii,save_matches_and_depth, draw_matches,save_image

#reading both the images
left_images = glob.glob("images/Moebius_left_corrected.*")
right_images = glob.glob("images/Moebius_right_corrected.*")


left_path = left_images[0]
right_path = right_images[0]

if left_path is None or right_path is None:
    raise ValueError("Images not found ")

left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

#Removing the noise
left_blur = cv2.GaussianBlur(left_img, (5, 5), 1)
right_blur = cv2.GaussianBlur(right_img, (5, 5), 1)

# find corners in both images using harris corner files
left_corners = detect_harris_corners(left_blur)
right_corners = detect_harris_corners(right_blur)


# now try to match the corners from left to right image using correlation
matches, disparities = match_features(left_img, right_img, left_corners, right_corners)


# show me that its done and save the depth map
print("done. depth map is ready .")


# ASCII outputs
save_corners_to_ascii(left_corners, "left_corners_Moebius.txt")
save_corners_to_ascii(right_corners, "right_corners_Moebius.txt")

# Corner visualization
draw_corners(left_img, left_corners, "left_corners_Moebius.bmp")
draw_corners(right_img, right_corners, "right_corners_Moebius.bmp")

# Depth map
normalized_depth = compute_depth_map(left_img.shape, matches)
display_and_save_depth_map(normalized_depth, filename="relative_depth_map_Moebius.png")
matched_vis = draw_matches(left_img, left_corners, right_img, right_corners, [(x1, y1, x2, y2) for (x1, y1), (x2, y2) in matches.items()])
save_image("matched_features_Moebius.png", matched_vis)
# Save relative depth map as .bmp
cv2.imwrite("depth_map_Moebius.bmp", normalized_depth)

# Save matches, correlation, and depth info
save_matches_and_depth(matches, disparities, normalized_depth, "matches_and_depth_Moebius.txt")