# import algorithm function here
from DL_finder import DL_finder
from AdaptiveThresholding import *
from AverageThresholding import *
from helper import *
import numpy as np

"""
BeadFinder implements a specified algorithm to identify beads from an image
"""
class BeadFinder():
    def __init__(self, image_path, algorithm_name, label, image_registration, show_heatmap, output_path):
        self.image_path = image_path
        self.algorithm_name = algorithm_name # algorithm_name name
        self.label = label
        self.image_registration = image_registration
        self.show_heatmap = show_heatmap
        self.output_path = output_path

        self._instantiate_algorithm()

    
    def _instantiate_algorithm(self):
        """
        Instantiate algorithm from algorithm name to prep for find_beads()
        """
        if self.algorithm_name == "adaptive_thresholding":
            self.algorithm = AdaptiveThresholding
        elif self.algorithm_name == "average_thresholding":
            self.algorithm = AverageThresholding
        elif self.algorithm_name == "deep_learning" or self.algorithm_name == "deep_learning_aug":
            self.algorithm = DL_finder
        else:
            raise ValueError("Invalid algorithm name!")
    
    def get_algorithm_name(self):
        # getter not supported on command line. for python-based dev purpose
        return self.algorithm_name

    def set_algorithm_to(self, algorithm_name):
        # setter not supported on command line. for python-based dev purpose
        self.algorithm_name = algorithm_name
        self._instantiate_algorithm()

    def _isolate_wells_for_algorithm(self, img, img_th):
        if self.algorithm_name== "deep_learning" or self.algorithm_name == "deep_learning_aug":
            wells = isolate_each_well(img)  # a 16 x 24 numpy ndarray, each entry is the isolated well image
        elif self.algorithm_name == "adaptive_thresholding":
            wells = isolate_each_well(img_th)
        else: # average_thresholding
            wells = isolate_each_well(np.asarray(Image.fromarray(img).convert('L')))
        return wells

    """
    output: a list of well ids (e.g. A1) that has beads detected
            a list of well coordinates (0-indexed) with detected beads
            total number of wells with beads
    """
    def find_beads(self):
        # load image and optional registration and crop and rotate
        image_path = self.image_path
        if self.image_registration:
            img = align_to_standard(image_path)
            img = crop_rotate_from_arr(img)
        else:
            img = crop_rotate(image_path)
        img = np.asarray(img)

        # get heatmap
        img_th = load_and_preprocess_img(img, gaussian_kernel_size=(3,3), thresh=cv2.THRESH_BINARY, adapt=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, adp_th_block_size=5, adp_th_const=4, gaussian_sigma=1.0)
        
        # isolate wells
        wells = self._isolate_wells_for_algorithm(img, img_th)
        
        beads_ids, beads_coors, count = self.algorithm(wells, self.algorithm_name)
        
        if self.label:
            img_th_title = f'Machine vision algorithm ({self.algorithm_name}) output: {self.label} \n {count}/384 or about {100 * count // 384}% wells have beads'
        plt.imshow(img_th)
        plt.title(img_th_title)
        plt.xticks(set_x_ticks(img_th), list(range(1,25)))
        plt.yticks(set_y_ticks(img_th), list(letters_to_index.keys()))
        if self.show_heatmap:
            plt.show()
        else:
            plt.savefig(f'{self.output_path}')

        return beads_ids, beads_coors, count






