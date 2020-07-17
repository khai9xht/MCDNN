import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


class cnn10x10(nn.Module):
    def __init__(self):
        super(cnn10x10, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 2, padding=1)
        self.norm1 = nn.BatchNorm2d(num_features=20)
        self.relu = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(num_features=40)
        # original data
        self.fc1 = nn.Linear(180, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.pool2(x)
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(x.view(-1, 180))
        x = self.fc1(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x


class cnn12x12(nn.Module):
    def __init__(self):
        super(cnn12x12, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 2, padding=1)
        self.norm1 = nn.BatchNorm2d(num_features=20)
        self.relu = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(num_features=40)
        # original data
        self.fc1 = nn.Linear(360, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.pool2(x)
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(x.view(-1, 360))
        x = self.fc1(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x


class cnn14x14(nn.Module):
    def __init__(self):
        super(cnn14x14, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(num_features=20)
        self.relu = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, 4, padding=1)
        self.norm2 = nn.BatchNorm2d(num_features=40)
        # original data
        self.fc1 = nn.Linear(360, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.pool2(x)
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(x.view(-1, 360))
        x = self.fc1(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x


class cnn16x16(nn.Module):
    def __init__(self):
        super(cnn16x16, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(num_features=20)
        self.relu = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, 4, padding=1)
        self.norm2 = nn.BatchNorm2d(num_features=40)
        # original data
        self.fc1 = nn.Linear(360, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.pool2(x)
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(x.view(-1, 360))
        x = self.fc1(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x


class cnn18x18(nn.Module):
    def __init__(self):
        super(cnn18x18, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(num_features=20)
        self.relu = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, 4, padding=1)
        self.norm2 = nn.BatchNorm2d(num_features=40)
        # original data
        self.fc1 = nn.Linear(640, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.pool2(x)
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(x.view(-1, 640))
        x = self.fc1(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x


class cnn20x20(nn.Module):
    def __init__(self):
        super(cnn20x20, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 4, padding=1)
        self.norm1 = nn.BatchNorm2d(num_features=20)
        self.relu = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, 5, padding=1)
        self.norm2 = nn.BatchNorm2d(num_features=40)
        # original data
        self.fc1 = nn.Linear(360, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.pool2(x)
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(x.view(-1, 360))
        x = self.fc1(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x


class cnn28x28(nn.Module):
    def __init__(self):
        super(cnn28x28, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 4, padding=1)
        self.norm1 = nn.BatchNorm2d(num_features=20)
        self.relu = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, 5, padding=1)
        self.norm2 = nn.BatchNorm2d(num_features=40)
        self.pool3 = nn.MaxPool2d(3, 3)
        # original data
        self.fc1 = nn.Linear(360, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.pool2(x)
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.pool3(x)
        x = self.relu(x.view(-1, 360))
        x = self.fc1(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x
