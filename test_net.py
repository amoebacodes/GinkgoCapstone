import torch
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from argparse import ArgumentParser
from train_net import MLP, BuildDataset

device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

check_point = 100
path = os.path.join('train_models', str(check_point)+'.pth')


def main():
    print(device)

    test_data = BuildDataset(labels_csv='labels_val_221027.csv', img_dir="val_221027")
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # test
    model = MLP()
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].type(torch.float32).to(device), data[1].type(torch.int).to(device)
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the {len(test_data)} test images: {100 * correct // total} %')


if __name__ == "__main__":
    main()