import numpy as np
from helper import *
from tqdm import tqdm

def AverageThresholding(wells: np.ndarray, *args, **kwargs) -> Tuple[List[str], List[Tuple[int,int]], int]:
    """
    In the grayscale image of a well, we predict that there is a bead if there is a lot of
    variations in intensity, and no beads if the variation is low.

    output: a list of well ids (e.g. A1) that has beads detected
            a list of well coordinates (0-indexed) with detected beads (tuple form)
            total number of wells with beads
    """
    bead_id, bead_coor = [], []
    
    for row in tqdm(range(wells.shape[0]),desc='processing rows'):
        for col in tqdm(range(wells.shape[1]),desc='processing columns for each row', leave=False):
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