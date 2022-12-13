import numpy as np
from helper import *

def AdaptiveThresholding(wells: np.ndarray, *args, **kwargs) -> Tuple[List[str], List[Tuple[int,int]], int]:
    """
    After cropping out only the middle of a well from the thresholded image,
    if it contains two unique values, i.e. background and foreground, where the foreground is
    darker regions, which are assumed to be beads, then the prediction is 1, suggesting presence of beads;
    if it contains only one unique value, we assume it to be the background due to the size and color of 
    beads relative to the plate, then the prediction is 0, suggesting absence of beads.
    
    output: a list of well ids (e.g. A1) that has beads detected
            a list of well coordinates (0-indexed) with detected beads (tuple form)
            total number of wells with beads
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
