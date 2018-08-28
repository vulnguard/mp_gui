import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt

# ================ Loading / Saving images ================
# Each of these should return the resepctive kernel 
def perwit_kx_kernal():
    pass

def perwit_ky_kernal():
    pass

def sobel_kx_kernal():
    pass

def sobel_ky_kernal():
    pass

def laplacian_kernal():
    pass

def gaussian_kernal():
    pass

# ================ Loading / Saving images ================

def load_image(image_path):
    # imports image path
    # returns what cv2.imread returns
    return np.zeros([100, 100, 3], dtype=np.uint8)


def save_image(image, file_path):
    # imports an image (as defined by opencv) and a path which to save it to
    # returns nothing
    pass

# ================ Accessors ================

def get_image_dimentions(image):
    # imports an image
    # returns rows, cols
    return 100, 100


# ================ Mutators - linear filtering ================
def apply_convolution(image, kernel):
    # Returns the comvoluted image
    pass

def general_gaussian_filter(image, sigma_x = 5, sigma_y = None, size_x = 5, size_y = 5):
    # Returns the blurred image
    pass

def apply_median_filter(image, kernel_size = 5):
    # Returns the blurred image
    pass

# ================ Mutators - Transforms ================
def crop_image(image, x, y, width, height):
    # Returns the cropped image
    pass

def rotate_image(image, angle_deg):
    # Returns the rotated image
    pass

def resize_image(image, x_scale_factor, y_scale_factor):
    # Returns the resized image
    pass

def apply_affine_transformation(image, starting_point, ending_point):
    # Returns the transformed image
    pass

def apply_affine_pixel_transform(image, alpha, beta):
    # Returns the pixel - transformed image (ignore the weird name)
    # (this is y =ax + b)
    pass

def normalize_histogram_bnw(image):
    # Returns the noramlized image
    pass

def normalize_histogram_bgr(image):
    # Returns the noramlized image
    pass

# ================ Mutators - Other ================
# Start pos, and end_pos and col are all tuples. (x, y) (r, g, b)
def draw_rect_on_image(image, start_pos, end_pos, thickness = 3, colour = (0, 0, 0)):
    # Returns a COPY of the image with the rect drawn on it
    pass

def inverse_bw(image):
    # Returns inversed image
    pass

# ================ Mutators - Morphological opperations ================
def morph_dilate(image, structuring_element = None, size =  5):
    # Returns dilated image
    pass

def morph_erode(image, structuring_element = None, size = 5):
    # Returns modified image
    pass

def morph_open(image, structuring_element = None, size= 5):
    # Returns modified image
    pass

def morph_close(image, structuring_element = None, size = 5):
    # Returns modified image
    pass

def morph_gradiant(image, structuring_element = None, size = 5):
    # Returns modified image
    pass

def morph_blackhat(image, structuring_element = None, size = 5):
    # Returns modified image
    pass

# ================ Format Converters ================
def rgb_to_bgr(image):
    # Returns modified image
    pass

def bgr_to_rgb(image):
    # Returns modified image
    pass

# @Note the following functions expect input image to be in bgr format.
def to_grayscale(image):
    # Returns modified image
    pass


def to_hsv(image):
    # Returns modified image
    pass


def to_luv(image):
    # Returns modified image
    pass


def to_lab(image):
    # Returns modified image
    pass


def to_yuv(image):
    # Returns modified image
    pass


