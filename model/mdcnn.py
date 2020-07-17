import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import cv2


class mcdnn(nn.Module):
    def __init__(self):
        super(mcdnn, self).__init__()

        # 28 * 28
        self.n1c1 = nn.Conv2d(1, 20, kernel_size=4, padding=1)
        self.n1bn1 = nn.BatchNorm2d(20)
        self.n1c2 = nn.Conv2d(20, 40, kernel_size=5, padding=1)
        self.n1bn2 = nn.BatchNorm2d(40)
        self.n1fc1 = nn.Linear(360, 150)
        self.n1fc2 = nn.Linear(150, 10)

        # 20 * 20
        self.n2c1 = nn.Conv2d(1, 20, kernel_size=4, padding=1)
        self.n2bn1 = nn.BatchNorm2d(20)
        self.n2c2 = nn.Conv2d(20, 40, kernel_size=5, padding=1)
        self.n2bn2 = nn.BatchNorm2d(40)
        self.n2fc1 = nn.Linear(360, 150)
        self.n2fc2 = nn.Linear(150, 10)

        # 18 * 18
        self.n3c1 = nn.Conv2d(1, 20, kernel_size=3, padding=1)
        self.n3bn1 = nn.BatchNorm2d(20)
        self.n3c2 = nn.Conv2d(20, 40, kernel_size=4, padding=1)
        self.n3bn2 = nn.BatchNorm2d(40)
        self.n3fc1 = nn.Linear(640, 150)
        self.n3fc2 = nn.Linear(150, 10)

        # 16 * 16
        self.n4c1 = nn.Conv2d(1, 20, kernel_size=3, padding=1)
        self.n4bn1 = nn.BatchNorm2d(20)
        self.n4c2 = nn.Conv2d(20, 40, kernel_size=4, padding=1)
        self.n4bn2 = nn.BatchNorm2d(40)
        self.n4fc1 = nn.Linear(360, 150)
        self.n4fc2 = nn.Linear(150, 10)

        # 14 * 14
        self.n5c1 = nn.Conv2d(1, 20, kernel_size=3, padding=1)
        self.n5bn1 = nn.BatchNorm2d(20)
        self.n5c2 = nn.Conv2d(20, 40, kernel_size=4, padding=1)
        self.n5bn2 = nn.BatchNorm2d(40)
        self.n5fc1 = nn.Linear(360, 150)
        self.n5fc2 = nn.Linear(150, 10)

        # 12 * 12
        self.n6c1 = nn.Conv2d(1, 20, kernel_size=2, padding=1)
        self.n6bn1 = nn.BatchNorm2d(20)
        self.n6c2 = nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.n6bn2 = nn.BatchNorm2d(40)
        self.n6fc1 = nn.Linear(360, 150)
        self.n6fc2 = nn.Linear(150, 10)

        # 10 * 10
        self.n7c1 = nn.Conv2d(1, 20, kernel_size=2, padding=1)
        self.n7bn1 = nn.BatchNorm2d(20)
        self.n7c2 = nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.n7bn2 = nn.BatchNorm2d(40)
        self.n7fc1 = nn.Linear(160, 150)
        self.n7fc2 = nn.Linear(150, 10)

        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(3, 3)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        # 28 * 28
        n1_out = self.relu(self.pool2(self.n1bn1(self.n1c1(x))))
        n1_out = self.relu(self.pool3(self.n1bn2(self.n1c2(n1_out))))
        n1_out = n1_out.view(-1, 360)
        n1_out = self.relu(self.n1fc1(n1_out))
        n1_out = self.n1fc2(n1_out)

        # 20 * 20
        in_resized = []
        for el in x:
            temp = np.reshape(el.cpu().data.numpy(), (28, 28))
            in_resized.append(cv2.resize(temp, (20, 20)).astype(float)/255)
        in_resized = np.expand_dims(np.asarray(in_resized), axis=1)
        in_resized = Variable(torch.Tensor(in_resized)).cuda()

        n2_out = self.relu(self.pool2(self.n2bn1(self.n2c1(in_resized))))
        n2_out = self.relu(self.pool2(self.n2bn2(self.n2c2(n2_out))))
        n2_out = n2_out.view(-1, 360)
        n2_out = self.relu(self.n2fc1(n2_out))
        n2_out = self.n2fc2(n2_out)

        # 18 * 18
        in_resized = []
        for el in x:
            temp = np.reshape(el.cpu().data.numpy(), (28, 28))
            in_resized.append(cv2.resize(temp, (18, 18)).astype(float)/255)
        in_resized = np.expand_dims(np.asarray(in_resized), axis=1)
        in_resized = Variable(torch.Tensor(in_resized)).cuda()

        n3_out = self.relu(self.pool2(self.n3bn1(self.n3c1(in_resized))))
        n3_out = self.relu(self.pool2(self.n3bn2(self.n3c2(n3_out))))
        n3_out = n3_out.view(-1, 640)
        n3_out = self.relu(self.n3fc1(n3_out))
        n3_out = self.n3fc2(n3_out)

        # 16 * 16
        in_resized = []
        for el in x:
            temp = np.reshape(el.cpu().data.numpy(), (28, 28))
            in_resized.append(cv2.resize(temp, (16, 16)).astype(float)/255)
        in_resized = np.expand_dims(np.asarray(in_resized), axis=1)
        in_resized = Variable(torch.Tensor(in_resized)).cuda()

        n4_out = self.relu(self.pool2(self.n4bn1(self.n4c1(in_resized))))
        n4_out = self.relu(self.pool2(self.n4bn2(self.n4c2(n4_out))))
        n4_out = n4_out.view(-1, 360)
        n4_out = self.relu(self.n4fc1(n4_out))
        n4_out = self.n4fc2(n4_out)

        # 14 * 14
        in_resized = []
        for el in x:
            temp = np.reshape(el.cpu().data.numpy(), (28, 28))
            in_resized.append(cv2.resize(temp, (14, 14)).astype(float)/255)
        in_resized = np.expand_dims(np.asarray(in_resized), axis=1)
        in_resized = Variable(torch.Tensor(in_resized)).cuda()

        n5_out = self.relu(self.pool2(self.n5bn1(self.n5c1(in_resized))))
        n5_out = self.relu(self.pool2(self.n5bn2(self.n5c2(n5_out))))
        n5_out = n5_out.view(-1, 360)
        n5_out = self.relu(self.n5fc1(n5_out))
        n5_out = self.n5fc2(n5_out)

        # Scale to 12 * 12
        in_resized = []
        for el in x:
            temp = np.reshape(el.cpu().data.numpy(), (28, 28))
            in_resized.append(cv2.resize(temp, (12, 12)).astype(float)/255)
        in_resized = np.expand_dims(np.asarray(in_resized), axis=1)
        in_resized = Variable(torch.Tensor(in_resized)).cuda()

        n6_out = self.relu(self.pool2(self.n6bn1(self.n6c1(in_resized))))
        n6_out = self.relu(self.pool2(self.n6bn2(self.n6c2(n6_out))))
        n6_out = n6_out.view(-1, 360)
        n6_out = self.relu(self.n6fc1(n6_out))
        n6_out = self.n6fc2(n6_out)

        # Scale to 10 * 10
        in_resized = []
        for el in x:
            temp = np.reshape(el.cpu().data.numpy(), (28, 28))
            in_resized.append(cv2.resize(temp, (10, 10)).astype(float)/255)
        in_resized = np.expand_dims(np.asarray(in_resized), axis=1)
        in_resized = Variable(torch.Tensor(in_resized)).cuda()

        n7_out = self.relu(self.pool2(self.n7bn1(self.n7c1(in_resized))))
        n7_out = self.relu(self.pool2(self.n7bn2(self.n7c2(n7_out))))
        n7_out = n7_out.view(-1, 160)
        n7_out = self.relu(self.n7fc1(n7_out))
        n7_out = self.n7fc2(n7_out)
        return (n1_out + n2_out + n3_out + n4_out + n5_out + n6_out + n7_out)/7
