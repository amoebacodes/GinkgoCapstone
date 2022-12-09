import numpy as np
from helper import *

def AdaptiveThresholding(wells, *args, **kwargs):
    """
    Uses particle detection to predict whether or not bead is present.
    Returns accuracy and a list of unsure well locations for each img (if particle_detection_prediction returns -1)
        which means that more than one particle is detected in the well
    """
    bead_id, bead_coor = [], []
    
    for row in range(wells.shape[0]):
        for col in range(wells.shape[1]):
            if exclude_wells(col, row):
                continue
            
            # crop out a kxk square in the center of the well
            well = np.array(wells[row,col])
            well = crop_well(well)
            
            # pred = particle_detection_prediction(well) # comment this if using only num of val in thresholded img and uncomment the following line
            pred = len(np.unique(well)) - 1 

            if pred == 1:
                well_id = chr(ord('A') + row) + str(col + 1)
                bead_id.append(well_id)
                bead_coor.append((row, col))

    return bead_id, bead_coor, len(bead_id)
