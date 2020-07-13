# TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # TODO: Define all the layers of this CNN, the only requirements are:
        # 1. This network takes in a square (same width and height), grayscale image as input
        # 2. It ends with a linear layer that represents the keypoints
        # it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 7, 3, 1)
        # torch.nn.init.xavier_uniform_(self.conv1.weight)
        # self.pool = nn.MaxPool2d(2, 2)
        #self.drop1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(32, 64, 5, 3, 0)
        # torch.nn.init.xavier_uniform_(self.conv2.weight)
        #self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(64, 128, 5, 3, 1)
        # torch.nn.init.xavier_uniform_(self.conv3.weight)
        #self.drop3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(128, 256, 3, 1, 0)
        # torch.nn.init.xavier_uniform_(self.conv4.weight)
        #self.drop4 = nn.Dropout(0.4)

        self.conv5 = nn.Conv2d(256, 512, 3, 1, 0)
        # torch.nn.init.xavier_uniform_(self.conv4.weight)
        #self.drop4 = nn.Dropout(0.4)

        self.conv6 = nn.Conv2d(512, 512, 1, 1, 0)
        # torch.nn.init.xavier_uniform_(self.conv4.weight)
        #self.drop4 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(4*4*512, 1024)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.dropfc1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(1024, 1024)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.dropfc2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(1024, 136)
        # torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        # TODO: Define the feedforward behavior of this model
        # x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = F.selu(self.conv3(x))
        x = F.selu(self.conv4(x))
        x = F.selu(self.conv5(x))
        x = F.selu(self.conv6(x))

        x = x.view(x.size(0), -1)

        x = self.dropfc1(F.selu(self.fc1(x)))
        x = self.dropfc2(F.selu(self.fc2(x)))
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
