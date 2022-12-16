from PIL import Image
import numpy as np
from cmath import nan
from PIL import Image
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
import os, glob
from PIL import Image
import cv2
import pandas as pd
from typing import Tuple, List
import logging
import os
import argparse

# create logger
def config_logger() -> logging.Logger:
    logger = logging.getLogger('mainLogging')
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    return logger

logger = logging.getLogger('mainLogging')

def validate_image_path(image_path: str) -> None:
    if not os.path.isfile(image_path):
        raise FileNotFoundError('Image path is not a file.')

def validate_output_path(output_path) -> None:
    output_dir = '/'.join(output_path.split('/')[:-1])
    if not os.path.exists(output_dir):
        raise FileNotFoundError('Directory of the output path not found.')

def validate_algorithm_name(name) -> None:
    if  name != "adaptive_thresholding" and\
        name != "average_thresholding" and \
        name != "deep_learning" and\
        name != "deep_learning_aug":
        raise ValueError("Invalid algorithm name!")

def validate_cli(args: argparse.Namespace) -> None:
    # if show_heatmap is false, save heatmap, and thus output dir must be specified
    if not args.show_heatmap:
        if args.output_path == None:
            raise ValueError('When show_heatmap is set to false (default), user need to specify an output_dir' +
                'for saving the heatmap of the plate. If you wish not to save the heatmap, you can set show_heatmap' +
                'to true, and the heatmap will be a pop-up window.')
        else:
            validate_output_path(args.output_path)
    validate_image_path(args.image_path)
    validate_algorithm_name(args.algorithm_name)
    if args.label == None:
        logger.warning('We suggest giving your plate a name so the heatmap title can be more descriptive.')

def crop_rotate(filename: str, angle: float = -1.5, left: int = 135, 
                upper: int = 80, right: int = 600, lower: int = 390) -> Image.Image:
    '''
    crop and rotate operation for a single image from a filename
    '''
    with Image.open(filename) as img:
        # (left, upper, right, lower) = (100, 60, 630, 400)
        rotated = img.rotate(angle, expand=1)
    return rotated.crop((left, upper, right, lower))

def crop_rotate_from_arr(img: np.ndarray, angle: float = -1.5, left: int = 135, 
                        upper: int = 85, right: int = 600, lower: int = 390) -> Image.Image:
    '''
    crop and rotate operation for a single image from a 
    ndarray representation of the image; used if image registration is True
    '''
    im = Image.fromarray(img)
    rotated = im.rotate(angle, expand = 1)
    im_final = rotated.crop((left, upper, right, lower))            
    return im_final


def align_to_standard(img_path: str, src_path='src.jpeg') -> np.ndarray:
    """
    image registration
    returns the aligned image as a numpy array
    """
    src_color = cv2.imread(src_path)
    sbj_color = cv2.imread(img_path)

    src = cv2.cvtColor(src_color, cv2.COLOR_BGR2GRAY)
    sbj = cv2.cvtColor(sbj_color, cv2.COLOR_BGR2GRAY)
    
    orb_detector = cv2.ORB_create(2000)
    src_keypoints, src_descriptors = orb_detector.detectAndCompute(src, None)
    sbj_keypoints, sbj_descriptors = orb_detector.detectAndCompute(sbj,None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = list(matcher.match(sbj_descriptors, src_descriptors))
    
    matches.sort(key = lambda x: x.distance)
    matches = matches[:int(len(matches)*0.9)]
    # records matched keypoints' coordinates
    no_of_matches = len(matches)
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(no_of_matches):
        p1[i, :] = sbj_keypoints[matches[i].queryIdx].pt
        p2[i, :] = src_keypoints[matches[i].trainIdx].pt
    
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    transformed_img = cv2.warpPerspective(sbj_color,
                        homography, (src.shape[1], src.shape[0]))
    return transformed_img

def load_and_preprocess_img(img_arr: np.ndarray, gaussian_kernel_size: Tuple = (3,3), 
                            adapt=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh=cv2.THRESH_BINARY, 
                            gaussian_sigma: float = 1.0, adp_th_block_size: int = 5, 
                            adp_th_const: int = 4) -> np.ndarray:
    """
    Generate thresholded plate image (i.e. the heatmap)
    gaussian_kernel_size: greater = blurring in larger neighborhood
    gaussian_sigma: greater sigma = more blurring
    adp_th_block_size needs to be odd: greater = looking at local intensities in a larger neighborhood
    adp_th_const is a constant that is subtracted from the weighted mean; greater = effectively more noise reduction
    """
    img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img, gaussian_kernel_size, gaussian_sigma)
    img_th = cv2.adaptiveThreshold(img_blurred, 255, adapt, thresh, adp_th_block_size, adp_th_const)
    return img_th

def exclude_wells(col_idx: int, row_idx: int) -> bool:
    """
    exclude wells with black background (idexing from 0)
    output: True if we need to exclude
            False otherwise
    """
    # M3, M4, A12, A13, P12, P13, C21
    if ((col_idx == 2 or col_idx == 3) and row_idx == 12) or ((row_idx ==0 or row_idx == 15) and (col_idx==11 or col_idx ==12)) or (row_idx == 2 and col_idx==20):
        return True
    return False


def isolate_each_well(img_th: np.ndarray) -> np.ndarray:
    """
    input: img or preprocessed img
    output: conceptually a matrix of 16 x 24, each entry is the isolated well image
    """
    to_return = []
    y = img_th.shape[0] / 16
    x = img_th.shape[1] / 24
    for row in np.arange(0, img_th.shape[0], y):
        col_out = []
        for col in np.arange(0, img_th.shape[1], x):
            col_out.append(img_th[round(row):round(row + y), round(col):round(col + x)])
        to_return.append(col_out)
    return np.array(to_return, dtype=object)

def crop_well(well: np.ndarray, k=9) -> np.ndarray:
    """
    crop out the middle of a well; for adaptive thresholding
    output: cropped well as a ndarray
    """
    if well.shape[1] == 20:
        well = well[:, :-1]
    row_diff = (well.shape[0] - k) // 2
    well = well[row_diff:, :]
    well = well[:k, :]
    well = well[:, row_diff:]
    well = well[:, :k]
    return well

def coor_tuple_to_int(bead_coor: List[Tuple[int,int]]) -> List[int]:
    """
    transform (row, col) form well coordinate to int index 
    indexing from 0
    output: reformated coordinates, from 0 to 383
    """
    int_coor = []
    for c in bead_coor:
        int_coor.append((c[0]+1)*(c[1]+1)-1)
    return int_coor

def set_x_ticks(img_th: np.ndarray) -> List[float]:
    """
    set x ticks for the heatmap (thresholded image)
    output: positions of the x ticks
    """
    width = img_th.shape[1]
    sep = width/24
    x = [10]
    for i in range(23):
        x.append(x[-1] + sep)
    return x

def set_y_ticks(img_th: np.ndarray) -> List[float]:
    """
    set y ticks for the heatmap (thresholded image)
    output: positions of the y ticks
    """
    height = img_th.shape[0]
    sep = height/16
    y = [10]
    for i in range(15):
        y.append(y[-1] + sep)
    return y

letters_to_index = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "O": 14,
    "P": 15
}

index_to_letter = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P"
}


def letter2index_df(df: pd.DataFrame) -> None:
    """
    modify the dataframe loaded from 'xxx_labels.csv"
    split 'destination_well' and add column 'row_id' and 'col_id'
    """
    rows = []
    cols = []
    destination_wells = df['destination_well'].values
    for destination in destination_wells:
        row = letters_to_index[destination[0]]
        col = int(destination[1:]) - 1
        rows.append(row)
        cols.append(col)

    df['row_id'] = rows
    df['col_id'] = cols
