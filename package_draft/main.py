from argparse import ArgumentParser
from BeadFinder import BeadFinder
import os
import argparse
from helper import coor_tuple_to_int
import logging
import sys
# from typing import 
"""
Demo argument:
python3 main.py --algorithm_name=deep_learning --output_path=/Users/yiqingmelodywang/img.png --image_path='221019_122109_1043495.jpg' --image_registration=True --show_heatmap=True
"""
# create logger
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
        name != "deep_learning" or\
        name != "deep_learning_aug":
        raise ValueError("Invalid algorithm name!")

def validate(args: argparse.Namespace) -> None:
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

if __name__ == "__main__":
    # load in arguments from command line
    parser = ArgumentParser(description='meg.beads_detection')
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--algorithm_name', type=str, required=True)  # options: ..., deep_learning, deep_learning_aug
    parser.add_argument('--label', type=str, default=None)  
    parser.add_argument('--image_registration', type=bool, default=False) # default = False
    parser.add_argument('--show_heatmap', type=bool, default=False) # default = False
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()

    # log arguments
    logger.debug('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))

    # validate arguments
    try:
        validate(args)
        logger.info('Arguments validated!')
    except:
        logger.exception('Error encountered with arguments passed!')
        sys.exit()

    #instantiate bead finder for locating beads
    try:
        instance = BeadFinder(args.image_path, args.algorithm_name, args.label, args.image_registration, args.show_heatmap, args.output_path)
        beads_ids, beads_coors, count = instance.find_beads()
    except:
        logger.exception('Error encountered when looking for beads')
        sys.exit()