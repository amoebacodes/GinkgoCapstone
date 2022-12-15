import numpy as np
import glob
from PIL import Image
import cv2
import pandas as pd
from typing import Tuple

def crop_rotate_dir(input_dir = 'input',output_dir='output', angle = -1.5, left = 135, upper =85, right = 600, lower = 390):
    # read every image file from the input folder
    for filename in glob.glob(input_dir+'/*.jpg'):
        # print(filename)
        with Image.open(filename) as im:
            # (left, upper, right, lower) = (100, 60, 630, 400)
            rotated = im.rotate(angle, expand = 1)
            im_final = rotated.crop((left, upper, right, lower))            
            im_final.save(filename.replace(input_dir, output_dir))

"""
gaussian_kernel_size: greater = blurring in larger neighborhood
gaussian_sigma: greater sigma = more blurring
adp_th_block_size needs to be odd: greater = looking at local intensities in a larger neighborhood
adp_th_const is a constant that is subtracted from the weighted mean; greater = effectively more noise reduction
"""
def load_and_preprocess_img(filename: str, gaussian_kernel_size: Tuple = (3,3), gaussian_sigma: float = 1.0, adp_th_block_size: int = 5, adp_th_const: int = 4):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img_blurred = cv2.GaussianBlur(img, gaussian_kernel_size, gaussian_sigma)
    img_th = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adp_th_block_size, adp_th_const)
    return img_th

"""
exclude wells with black background
"""
def exclude_wells(col_idx, row_idx):
    if ((col_idx == 2 or col_idx == 3) and row_idx == 12) or ((row_idx ==0 or row_idx == 15) and (col_idx==11 or col_idx ==12)) or (row_idx == 2 and col_idx==20):
        return True
    return False
"""
input: img or preprocessed img
output: conceptually a matrix of 16 x 24, each entry is the isolated well image
"""
def isolate_each_well(img_th):
    to_return = []
    y = img_th.shape[0] / 16
    x = img_th.shape[1] / 24
    for row in np.arange(0, img_th.shape[0], y):
        col_out = []
        for col in np.arange(0, img_th.shape[1], x):
            col_out.append(img_th[round(row):round(row+y),round(col):round(col+x)])
        to_return.append(col_out)
    return np.array(to_return, dtype=object)

"""
If one and only one particle is detected, report 1
If no particle detected, report 0
else: report -1 to indicate ambiguity
"""
def particle_detection_prediction(well_img):
    nb_components = cv2.connectedComponentsWithStats(well_img, connectivity=8)
    if nb_components[0] - 1 == 1:
        pred = 1
    elif nb_components[0] - 1 == 0:
        pred = 0
    else:
        pred = -1
    return pred
