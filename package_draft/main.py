from argparse import ArgumentParser
from BeadFinder import BeadFinder

"""
Demo argument:
python main.py  --image_path '221019_122109_1043495.jpg' --algorithm 'deep_learning'
"""

if __name__ == "__main__":
    parser = ArgumentParser(description='meg.beads_detection')
    parser.add_argument('--image_path', type=str, default='221019_122109_1043495.jpg')
    parser.add_argument('--algorithm', type=str, default="deep_learning")  # options: ..., deep_learning, deep_learning_aug
    parser.add_argument('--plate_name', type=str, default="my_plate")  # default = False
    parser.add_argument('--image_registration', action='store_true')
    parser.add_argument('--show_heatmap', action='store_true')
    args = parser.parse_args()
    print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))

    instance = BeadFinder(args.image_path, args.algorithm, args.plate_name, args.image_registration, args.show_heatmap)
    instance.get_method()
    beads_ids, beads_coors, count = instance.find_beads()

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

    # print stats
    print(f"There are {count}/{100 * count // 384}% wells that have mag.beads")


"""
outputs already have:
a list of 0-indexed row-wise/column-wise well coords that have mag.beads
a list of string well IDs that have mag.beads
stats on the number/percentage wells that have mag.beads

still need:
heatmap
"""