import torch.nn as nn
import torch.nn.functional as F


class SimpleConvNet(nn.Module):
    def __init__(self, num_classes, image_size):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.feature_size = (image_size - 5 + 1) // 2

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.feature_size = (self.feature_size - 3 + 1) // 2

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.feature_size = (self.feature_size + 1 - 3 + 1) // 2

        self.feature_size **= 2

        self.fc1 = nn.Linear(64 * self.feature_size, 250)
        self.fc2 = nn.Linear(250, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = x.view(-1, 64 * self.feature_size)
        x = F.dropout(F.relu(self.fc1(x)), training=self.training, p=0.4)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
