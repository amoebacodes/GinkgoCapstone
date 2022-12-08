# import algorithm function here
from DL_finder import DL_finder

from helper import crop_rotate, isolate_each_well, index_to_letter
import numpy as np


class BeadFinder():
    def __init__(self, image_path, algorithm, plate_name, image_registration, show_heatmap):
        self.image_path = image_path
        self.algorithm = algorithm
        self.plate_name = plate_name
        self.image_registration = image_registration
        self.show_heatmap = show_heatmap
        self.method = None

    def get_method(self):
        if self.algorithm == "melody's":
            pass
        if self.algorithm == "daniel's":
            pass
        if self.algorithm == "deep_learning" or self.algorithm == "deep_learning_aug":
            self.method = DL_finder
        else:
            print("No such algorithm!")

    def set_method_to(self, method):
        pass

    def find_beads(self):
        # load image
        image_path = self.image_path
        img = crop_rotate(image_path)
        img = np.asarray(img)
        wells = isolate_each_well(img)  # a 16 x 24 numpy ndarray, each entry is the isolated well image

        beads_ids, beads_coors, count = self.method(wells, self.algorithm)

        return beads_ids, beads_coors, count






