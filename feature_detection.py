import numpy as np
import cv2

from image_processing import *

# ================ Corner Detection ================

# Shi-tomasi method
def detect_corners_st(image, num_corners = 25, quality = 0.01, dist = 10, col_rgb = [255, 0, 0]):
    return image

# Harris method
def detect_corners_h(image, threshold = 0.01, blockSize = 2, ksize = 3, k = 0.04, col_rgb = [255, 0, 0]):
    return image

# ================ Edge Detection ================
# Uses the Canny method (manually)
def detect_edges_manual(image):
    return image

# Uses the Canny method in opencv
def detect_edges_auto(image):
    return image


# ================ Line Detection ================
# Uses the Hough Transform manually
def detect_lines_manual(image):
    return image

# Uses the Hough Transform in opencv
def detect_lines_auto(image):
    return image


# ================ Blob Detection ================
# Uses the MSERs. (Macimally Stable Extreme Regions)
def detect_blobs(image):
    return image
