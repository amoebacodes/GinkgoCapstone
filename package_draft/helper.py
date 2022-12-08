from PIL import Image
import numpy as np


def crop_rotate(filename, angle=-1.5, left=135, upper=80, right=600, lower=390):
    '''crop and rotate operation for a single image'''
    with Image.open(filename) as img:
        # (left, upper, right, lower) = (100, 60, 630, 400)
        rotated = img.rotate(angle, expand=1)
    return rotated.crop((left, upper, right, lower))


def exclude_wells(col_idx, row_idx):
    """
    exclude wells with black background (idx start from 0)
    """
    if ((col_idx == 2 or col_idx == 3) and row_idx == 12) or (
            (row_idx == 0 or row_idx == 15) and (col_idx == 11 or col_idx == 12)) or (row_idx == 2 and col_idx == 20):
        return True
    return False


def isolate_each_well(img_th):
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


def letter2index_df(df):
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
