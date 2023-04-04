import torch.nn as nn
import torch.nn.functional as F
# Creating a CNN class
class CNN(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(
            self,
            num_classes=2
    ):
        super().__init__()

        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=8, padding=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=8, padding=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=8, padding=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer4 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=8, padding=2)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=240, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

        self.flatten = nn.Flatten()

    # Progresses data across layers
    def forward(self, x, lens=None):
        x = F.leaky_relu(self.conv_layer1(x))
        x = self.max_pool1(x)

        x = F.leaky_relu(self.conv_layer2(x))
        x = self.max_pool2(x)

        x = F.leaky_relu(self.conv_layer3(x))
        x = self.max_pool3(x)

        x = F.leaky_relu(self.conv_layer4(x))
        x = self.max_pool4(x)

        x = self.flatten(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x)

        return x