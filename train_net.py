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
    '''
    check cuda availability: 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    '''
    print(device)

    # Training parameters
    parser = ArgumentParser(description='MLP')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--num_iterations', type=int, default=5000)
    parser.add_argument('--check_point', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='train_models')
    # parser.add_argument('--conv_layers', type=int, default=1)
    # parser.add_argument('--filters', type=int, default=1)
    parser.add_argument('--title', type=str, default='experiment_221028')
    args = parser.parse_args()
    print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))

    train_data = BuildDataset(labels_csv='labels_train_221027.csv', img_dir="train_221027")
    test_data = BuildDataset(labels_csv='labels_val_221027.csv', img_dir="val3_221027")
    print("training size:", len(train_data), "test size:", len(test_data))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    num_batch = len(train_loader)
    print("number of batches:", num_batch)

    net = MLP()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)

    train_loss_log = []
    train_acc_log = []
    test_loss_log = []
    test_acc_log = []
    j = 0
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times
        print("epoch:", epoch + 1)
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
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if j % args.check_point == args.check_point - 1:  # print every 400 mini-batches
                path = os.path.join(args.save_path, str(j + 1)+'.pth')
                torch.save(net.state_dict(), path)

                acc = correct/total
                loss = running_loss/total
                train_loss_log.append(loss)
                train_acc_log.append(acc)

                # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 3:.3f} accuracy: {100 * correct // total} %')
                print(f'[iteration: {j + 1}] loss: {loss :.3f} accuracy: {100 * correct // total} %')
                running_loss = 0.0
                total = 0.0
                correct = 0.0

                # validation
                correct_ = 0.0
                loss_ = 0.0
                total_ = 0.0
                with torch.no_grad():
                    for data in test_loader:
                        inputs, labels = data[0].type(torch.float32).to(device), data[1].type(torch.int).to(device)
                        outputs = net(inputs)
                        loss_ += criterion(outputs, labels)
                        # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs.data, 1)
                        total_ += labels.size(0)
                        correct_ += (predicted == labels).sum().item()

                    acc_ = correct_ / total_
                    loss_ /= total_
                    test_loss_log.append(loss_.to('cpu'))
                    test_acc_log.append(acc_)

                # print(f'Accuracy of the network on the {len(test_data)} validation images: {100 * correct_ // total_} %')
                print(f'[validation] loss: {loss_ :.3f} accuracy: {100 * correct_ // total_} %')

            j += 1

            if j == args.num_iterations:
                break
        if j == args.num_iterations:
            break

    problem = args.title
    np.save("train_loss_" + problem, np.asarray(train_loss_log))
    np.save("train_acc_" + problem, np.asarray(train_acc_log))
    np.save("test_loss_" + problem, np.asarray(test_loss_log))
    np.save("test_acc_" + problem, np.asarray(test_acc_log))
    print(problem, "log is saved.")
    print('Finished Training')


if __name__ == "__main__":
    main()
