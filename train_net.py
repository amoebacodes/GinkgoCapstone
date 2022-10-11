import torch
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

"""
file/folder needed: small_train_labels.csv, small_train, small_val_labels.csv, small_val
"""


class BuildDataset(Dataset):
    """
    label_csv: a csv file that contains the well image index, source plate image information, and labels
    img_dir: a folder contains multiple well images
    """
    def __init__(self, labels_csv, img_dir, transform=None, target_transform=None):
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
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class MLP(nn.Module):
    """
    try one layer MLP with 200 hidden nodes
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1*19*19,200),
            nn.ReLU(),
            nn.Linear(200, 2)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


def main():
    device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
    '''check cuda availability: device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') '''
    print(device)

    train_data = BuildDataset(labels_csv="small_train_labels.csv", img_dir="small_train")
    test_data = BuildDataset(labels_csv="small_val_labels.csv", img_dir="small_val")

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    net = MLP()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.2)

    for epoch in range(200):  # loop over the dataset multiple times
        correct = 0
        total = 0
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].type(torch.float32).to(device), data[1].type(torch.int).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 3 == 2:  # print every 3 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 3:.3f} accuracy: {100 * correct // total} %')
                running_loss = 0.0
                total = 0.0
                correct = 0.0

    print('Finished Training')

    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].type(torch.float32).to(device), data[1].type(torch.int).to(device)
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 100 test images: {100 * correct // total} %')


if __name__ == "__main__":
    main()
