import cv2
import numpy as np

def detect_harris_corners(image, window_size=5, k=0.02, threshold_ratio=0.01):
    #Get gradients in x and y directions using Sobel filter
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # horizontal edges
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # vertical edges

    # We Compute products of derivatives 
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    #Rather than using a loop to move over picture with a window to sum the total, i am using gaussian blur to make the process faster. 
    #Apply Gaussian blur to these derivative products
    Sx2 = cv2.GaussianBlur(Ix2, (window_size, window_size), sigmaX=1)
    Sy2 = cv2.GaussianBlur(Iy2, (window_size, window_size), sigmaX=1)
    Sxy = cv2.GaussianBlur(Ixy, (window_size, window_size), sigmaX=1)

    # Compute Harris response R = det(M) - k * trace(M)^2
    h, w = image.shape
    R = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            A = Sx2[y, x]
            B = Sxy[y, x]
            C = Sy2[y, x]

            det = A * C - B ** 2
            trace = A + C
            R[y, x] = det - k * (trace ** 2)

    # Compute adaptive threshold based on max R
    threshold = threshold_ratio * np.max(R)

    # Non-maximum suppression to get only distinct corners, 
    corners = []
    offset = window_size // 2
    for y in range(offset, h - offset):
        for x in range(offset, w - offset):
            val = R[y, x]
            if val > threshold:
                local_patch = R[y - offset:y + offset + 1, x - offset:x + offset + 1]
                if val == np.max(local_patch):
                    corners.append((x, y))

    return corners