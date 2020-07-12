import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        # shape of each input image is: (1, 28, 28)

        # 1 input image channel (grayscale), 10 output channels/feature maps
        # 3x3 square convolution kernel
        # output has dimensio (28/2)=14, (32, 14, 14)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)

        # TODO: Define the rest of the layers:
        # include another conv layer, maxpooling layers, and linear layers
        # also consider adding a dropout layer to avoid overfitting
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # after each conv layer, we have a MaxPool
        # 28/2=14
        # 14/2=7
        # 7/2=3.5 but  ceil_mode=True, so: 4
        self.fc1 = nn.Linear(128*28*28, 512)

        self.fc2 = nn.Linear(512, 136)

        self.drop = nn.Dropout(0.4)

    # TODO: define the feedforward behavior

    def forward(self, x):
        # one activated conv layer
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # x.size(0) is the number of samples in each batch and
        # -1 is for multipication of 3 all dimensions
        x = x.view(x.size(0), -1)

        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        # final output

        return x
