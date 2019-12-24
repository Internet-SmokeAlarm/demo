import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG11(nn.Module):

    def __init__(self, num_outputs=100):
        super(VGG11, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=1)

        self.fc1 = nn.Linear(512 * 1 * 1, num_outputs)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))

        x = self.pool(self.bn2(F.relu(self.conv2(x))))

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool(self.bn4(F.relu(self.conv4(x))))

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.pool(self.bn6(F.relu(self.conv6(x))))

        x = self.bn7(F.relu(self.conv7(x)))
        x = self.pool(self.bn8(F.relu(self.conv8(x))))

        x = self.avg_pool(x)

        x = x.view(-1, 512 * 1 * 1)

        x = self.fc1(x)

        return x
