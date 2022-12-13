from PIL import Image
from DL_model import MLP
import torch
from torchvision import transforms
from helper import index_to_letter
from typing import List, Tuple
import numpy as np

# load model
# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
# elif torch.backends.mps.is_available():
#     device = torch.device('mps:0')
# else:
#     device = torch.device('cpu')
device = torch.device('cpu')
print("device: ", device)

model_path = r'saved_models/mlp.pth'
model2_path = r'saved_models/mlp_aug.pth'


def load_model(algorithm: str) -> MLP:
    """
    load MLP model depending on user specification (with or without augmentation)
    """
    model = MLP()
    if algorithm == "deep_learning":
        msg = "load DL detection model"
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    elif algorithm == "deep_learning_aug":
        msg = "load DL detection model with data augmentation "
        model.load_state_dict(torch.load(model2_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    print(msg)
    return model


def make_prediction(wells: np.ndarray, model: MLP) -> Tuple[List[str], List[Tuple[int,int]], int]:
    """
    output: a list of well ids (e.g. A1) that has beads detected
            a list of well coordinates (0-indexed) with detected beads (tuple form)
            total number of wells with beads
    """
    beads_ids = []
    beads_coors = []
    count = 0
    for row in range(wells.shape[0]):
        for col in range(wells.shape[1]):
            # make prediction for each single well
            well = wells[row, col][:19, :19, :]  # (19, 19, 3)
            img = Image.fromarray(well)  # PIL
            transform = transforms.PILToTensor()
            img = transform(img)
            input = img.type(torch.float32).to(device)  # (3, 19, 19)
            input = torch.unsqueeze(input, 0)  # (1, 3 ,19, 19)
            output = model(input)
            _, pred = torch.max(output.data, 1)
            pred = pred.item()
            if pred == 1:
                beads_coors.append((row, col))
                row_letter = index_to_letter[row]
                id = row_letter + str(col + 1)
                beads_ids.append(id)
                count += 1
    return beads_ids, beads_coors, count


def DL_finder(wells: np.ndarray, algorithm: str) -> Tuple[List[str], List[Tuple[int,int]], int]:
    """
    to be instantiated as a function during initialization of a BeadFinder instance
    and executes if BeadFinder.find_beads() is called
    output: a list of well ids (e.g. A1) that has beads detected
            a list of well coordinates (0-indexed) with detected beads (tuple form)
            total number of wells with beads
    """
    model = load_model(algorithm)
    with torch.no_grad():
        beads_ids, beads_coors, count = make_prediction(wells, model)
    return beads_ids, beads_coors, count
