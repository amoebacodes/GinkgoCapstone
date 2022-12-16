from argparse import ArgumentParser
from BeadFinder import BeadFinder
from helper import *
import sys
# from typing import 
"""
Demo argument:
python3 main.py --algorithm_name=deep_learning --output_path=/Users/yiqingmelodywang/img.png --image_path='221019_122109_1043495.jpg' --image_registration=True --show_heatmap=True
"""
if __name__ == "__main__":
    # config logger
    logger = config_logger()
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
        validate_cli(args)
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