import numpy as np
from helper import *
def AverageThresholding(wells, *args, **kwargs):
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
            well = np.array(wells[row,col])
            std = np.std(well)
            mean = np.mean(well)
            min = np.amin(well)

            pred = 1 if min < mean-4*std else 0

            if pred == 1:
                well_id = chr(ord('A') + row) + str(col + 1)
                bead_id.append(well_id)
                bead_coor.append((row, col))
                
    return bead_id, bead_coor, len(bead_id)