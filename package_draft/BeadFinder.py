# import algorithm function here
from DL_finder import DL_finder
from AdaptiveThresholding import *
from AverageThresholding import *
from helper import *
import numpy as np
import logging
"""
BeadFinder implements a specified algorithm to identify beads from an image
"""
logger = logging.getLogger('mainLogging')

class BeadFinder():
    def __init__(self, image_path: str, algorithm_name: str, label: str, image_registration: bool, show_heatmap: bool, output_path: str):
        logger.info('Initializing...')
        self.image_path = image_path
        self.algorithm_name = algorithm_name # algorithm_name name
        self.label = label
        self.image_registration = image_registration
        self.show_heatmap = show_heatmap
        self.output_path = output_path

        self._instantiate_algorithm()

    
    def _instantiate_algorithm(self) -> None:
        """
        Instantiate algorithm from algorithm name to prep for find_beads()
        """
        logger.info(f'Getting ready to run {self.algorithm_name} method...')
        if self.algorithm_name == "adaptive_thresholding":
            self.algorithm = AdaptiveThresholding
        elif self.algorithm_name == "average_thresholding":
            self.algorithm = AverageThresholding
        elif self.algorithm_name == "deep_learning" or self.algorithm_name == "deep_learning_aug":
            self.algorithm = DL_finder
        else:
            raise ValueError("Invalid algorithm name!")
    
    def get_algorithm_name(self) -> str:
        """
        for python-based dev purpose or usage in python notebook, not supported on the command line
        gets the algorithm name
        """
        return self.algorithm_name

    def set_algorithm_to(self, algorithm_name: str) -> None:
        """
        for python-based dev purpose or usage in python notebook, not supported on command line 
        change the machine vision algorithm
        """
        self.algorithm_name = algorithm_name
        self._instantiate_algorithm()

    def _isolate_wells_for_algorithm(self, img: np.ndarray, img_th: np.ndarray) -> np.ndarray:
        """
        Depending on the algorithm specified, isolate wells from either:
            the RGB 3-channel plate image (deep learning)
            the thresholded plate image (adaptive thresholding)
            the grayscale plate image (average thresholding)
        output: a 16 x 24 ndarray, each entry is an isolated well-image
        """
        if self.algorithm_name== "deep_learning" or self.algorithm_name == "deep_learning_aug":
            wells = isolate_each_well(img)  # a 16 x 24 numpy ndarray, each entry is the isolated well image
        elif self.algorithm_name == "adaptive_thresholding":
            wells = isolate_each_well(img_th)
        else: # average_thresholding
            wells = isolate_each_well(np.asarray(Image.fromarray(img).convert('L')))
        return wells

    def find_beads(self) -> Tuple[List[str], List[int], int]:
        """
        print the following outputs and also show/save heatmap
        output: a list of well ids (e.g. A1) that has beads detected
                a list of well coordinates (0-indexed) with detected beads
                total number of wells with beads
        """
        # load image and optional registration and crop and rotate
        image_path = self.image_path
        if self.image_registration:
            logger.info('Performing image registration...')
            img = align_to_standard(image_path)
            img = crop_rotate_from_arr(img)
        else:
            img = crop_rotate(image_path)
        img = np.asarray(img)

        # get heatmap
        logger.info('Getting heatmap...')
        img_th = load_and_preprocess_img(img, gaussian_kernel_size=(3,3), thresh=cv2.THRESH_BINARY, adapt=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, adp_th_block_size=5, adp_th_const=4, gaussian_sigma=1.0)
        
        # isolate wells
        logger.info('Cropping out wells from the plate image...')
        wells = self._isolate_wells_for_algorithm(img, img_th)
        
        # get results
        logger.info('Finding beads...')
        beads_ids, beads_coors, count = self.algorithm(wells, self.algorithm_name)
        
        # convert bead coordinates to desired format
        beads_coors_to_int = coor_tuple_to_int(beads_coors)
        
        # show/save heatmap
        if self.label:
            img_th_title = f'Machine vision algorithm ({self.algorithm_name}) output: {self.label} \n {count}/384 or about {100 * count // 384}% wells have beads'
        else:
            img_th_title = f'Machine vision algorithm ({self.algorithm_name}) \n {count}/384 or about {100 * count // 384}% wells have beads'
        plt.imshow(img_th)
        plt.title(img_th_title)
        plt.xticks(set_x_ticks(img_th), list(range(1,25)))
        plt.yticks(set_y_ticks(img_th), list(letters_to_index.keys()))
        if self.show_heatmap:
            logger.info('Showing heatmap...Please close the pop-up window to proceed.')
            plt.show()
        else:
            logger.info('Saving heatmap...')
            plt.savefig(f'{self.output_path}')


        # print output
        print('Well_ids with beads: \n', beads_ids)
        print('They are at the following coordinates: \n', beads_coors_to_int)
        # print stats
        print(f"There are {count}/384 or about {100 * count // 384}% wells that have mag.beads")

        return beads_ids, beads_coors_to_int, count






