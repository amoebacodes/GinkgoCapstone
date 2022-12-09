from argparse import ArgumentParser
from BeadFinder import BeadFinder
import os
"""
Demo argument:
python main.py  --image_path '221019_122109_1043495.jpg' --algorithm 'deep_learning'
"""

def validate_paths(image_path, output_dir):
    if not os.path.isfile(image_path):
        raise FileNotFoundError('Image path is not a file.')
    if not os.path.isdir(output_dir):
        raise FileNotFoundError('Directory not found.')

def validate(args):
    # if show_heatmap is false, save heatmap, and thus output dir must be specified
    if not args.show_heatmap and args.output_dir == None:
        raise ValueError('When show_heatmap is set to false (default), user need to specify an output_dir\
                for saving the heatmap of the plate. If you wish not to save the heatmap, you can set show_heatmap\
                to true, and the heatmap will be a pop-up window.')
    validate_paths(args.image_path, args.output_dir)

def save_to_txt(beads_ids,beads_coors):
    # save a list of string well IDs that have mag.beads
    file_name_ids = "beads_ids_"+args.plate_name+".txt"
    with open(file_name_ids, "w") as output:
        output.write(str(beads_ids))
    print("saved well IDs that have mag.beads to", file_name_ids, "!")

    # save a list of 0-indexed row-wise/column-wise well coords that have mag.beads
    file_name_coords = "beads_coords_"+args.plate_name+".txt"
    with open(file_name_coords, "w") as output:
        output.write(str(beads_coors))
    print("saved well coords that have mag.beads to", file_name_coords, "!")

if __name__ == "__main__":
    # load in arguments from command line
    parser = ArgumentParser(description='meg.beads_detection')
    parser.add_argument('--image_path', type=str, default="221019_122109_1043495.jpg")
    parser.add_argument('--algorithm_name', type=str, default="deep_learning")  # options: ..., deep_learning, deep_learning_aug
    parser.add_argument('--plate_name', type=str, default="my_plate")  
    parser.add_argument('--image_registration', action='store_false') # default = False
    parser.add_argument('--show_heatmap', action='store_false') # default = False
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    # validate arguments
    validate(args)
    # show arguments
    print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))

    #instantiate bead finder for locating beads
    instance = BeadFinder(args.image_path, args.algorithm_name, args.plate_name, args.image_registration, args.show_heatmap, args.output_dir)
    beads_ids, beads_coors, count = instance.find_beads()

    # results
    # save_to_txt(beads_ids, beads_coors) #uncomment if decide to save well id and coor to txt
    print('Well_ids with beads: \n', beads_ids)
    print('They are at the following coordinates: \n', beads_coors)
    # print stats
    print(f"There are {count}/384 or about {100 * count // 384}% wells that have mag.beads")


"""
outputs already have:
a list of 0-indexed row-wise/column-wise well coords that have mag.beads
a list of string well IDs that have mag.beads
stats on the number/percentage wells that have mag.beads

still need:
heatmap
"""