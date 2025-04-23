import cv2
import numpy as np

def match_features(left_img, right_img, left_keypoints, right_keypoints, window_size=7, row_tolerance=5):
    # This dictionary will store the best match in the right image for each keypoint in the left image.
    matches = {}

    # We will also keep track of the correlation value for each match.
    disparities = {}

    half_window = window_size // 2

    # Loop through each keypoint in the left image
    for (x1, y1) in left_keypoints:
        best_corr = -1     # Best correlation value found so far
        best_match = None  # Best matching point in the right image

        # Check if the patch around this keypoint would go out of bounds. Skip if it does.
        if (y1 - half_window < 0 or y1 + half_window >= left_img.shape[0] or
            x1 - half_window < 0 or x1 + half_window >= left_img.shape[1]):
            continue

        # Extract the patch centered at this keypoint from the left image
        patch_left = left_img[y1 - half_window:y1 + half_window + 1, x1 - half_window:x1 + half_window + 1]
        patch_left = patch_left.astype(np.float32)
        patch_left_mean = np.mean(patch_left)
        patch_left -= patch_left_mean  

        # Now look for the best matching patch in the right image
        for (x2, y2) in right_keypoints:
            # Only compare patches on approximately the same row to reduce false matches
            if abs(y2 - y1) > row_tolerance:
                continue

            # Again, check patch boundaries skip if going out of image bounds
            if (y2 - half_window < 0 or y2 + half_window >= right_img.shape[0] or
                x2 - half_window < 0 or x2 + half_window >= right_img.shape[1]):
                continue

            # Extract the matching patch from the right image
            patch_right = right_img[y2 - half_window:y2 + half_window + 1, x2 - half_window:x2 + half_window + 1]
            patch_right = patch_right.astype(np.float32)
            patch_right_mean = np.mean(patch_right)
            patch_right -= patch_right_mean  

            # Compute the normalized cross correlation between the two patches we have
            numerator = np.sum(patch_left * patch_right)
            denominator = np.sqrt(np.sum(patch_left ** 2) * np.sum(patch_right ** 2))

            if denominator == 0:
                continue 

            corr = numerator / denominator  

            # If this is the best correlation we have got, then store it
            if corr > best_corr:
                best_corr = corr
                best_match = (x2, y2)

        # Once we check all candidates, save the best match for this keypoint
        if best_match:
            matches[(x1, y1)] = best_match
            disparities[(x1, y1)] = best_corr 

    # Return both the matches and their correlation scores
    return matches, disparities