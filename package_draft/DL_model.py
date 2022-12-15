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
import logging
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torchvision import transforms
import torch.nn.functional as F

transform = None


class BuildDataset(Dataset):
    """
    label_csv: a csv file that contains the well image index, source plate image information, and labels
    img_dir: a folder contains multiple well images
    """
    def __init__(self, labels_csv, img_dir, transform=transform, target_transform=None):
        self.img_labels = pd.read_csv(labels_csv)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{self.img_labels.iloc[idx,0]}.jpg')
        image = read_image(img_path)[:, :19, :19]  # fix the shape of well image
        label = self.img_labels.iloc[idx, 4]  # 4 is the col index of the labels
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label


class MLP(nn.Module):
    """
    try one layer MLP with 200 hidden nodes
    """
    def __init__(self, num_channel=3):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_channel*19*19, 200),
            nn.ReLU(),
            nn.Linear(200, 2)
        )

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        x = self.layers(x)
        return x


class CNN(nn.Module):
    """
    try a simple CNN with 2 convolutional layers and 3 linearmaps
    """
    def __init__(self, num_channel=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, kernel_size=2, padding=0)  # (6,18,18)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=2, padding=0)  # (16, 17, 17)
        self.fc1 = nn.Linear(16 * 17 * 17, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x