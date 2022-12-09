# import algorithm function here
from DL_finder import DL_finder

from helper import crop_rotate, isolate_each_well, index_to_letter
import numpy as np

"""
BeadFinder implements a specified algorithm to identify beads from an image
"""
class BeadFinder():
    def __init__(self, image_path, algorithm_name, plate_name, image_registration, show_heatmap, output_dir):
        self.image_path = image_path
        self.algorithm_name = algorithm_name # algorithm_name name
        self.plate_name = plate_name
        self.image_registration = image_registration
        self.show_heatmap = show_heatmap
        self.output_dir = output_dir

        self._instantiate_algorithm()

    """
    Instantiate algorithm from algorithm name to prep for find_beads()
    """
    def _instantiate_algorithm(self):
        if self.algorithm_name == "adaptive_thresholding":
            pass
        elif self.algorithm_name == "daniel's":
            pass
        elif self.algorithm_name == "deep_learning" or self.algorithm_name == "deep_learning_aug":
            self.algorithm = DL_finder
        else:
            raise ValueError("Invalid algorithm name!")

    def get_algorithm_name(self):
        return self.algorithm_name

    def set_algorithm_to(self, algorithm_name):
        self.algorithm_name = algorithm_name
        self._instantiate_algorithm()

    """
    output: a list of well ids (e.g. A1) that has beads detected
            a list of well coordinates (0-indexed) with detected beads
            total number of wells with beads
    """
    def find_beads(self):
        # load image
        image_path = self.image_path
        img = crop_rotate(image_path)
        img = np.asarray(img)
        wells = isolate_each_well(img)  # a 16 x 24 numpy ndarray, each entry is the isolated well image

        beads_ids, beads_coors, count = self.algorithm(wells, self.algorithm_name)

        return beads_ids, beads_coors, count






