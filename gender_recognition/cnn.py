import pdbr
import torch.nn as nn
import torch



# Creating a CNN class
class CNN(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(
            self,
            num_classes=2
    ):
        super().__init__()

        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=8, padding=2)
        self.activation1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=8, padding=2)
        self.activation2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=8, padding=2)
        self.activation3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer4 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=8, padding=2)
        self.activation4 = nn.ReLU()
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_features=240, out_features=32)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)
        self.relu1 = nn.ReLU()
        self.softmax = nn.Softmax()


    # Progresses data across layers
    def forward(self, x, lens=None):
        x = self.conv_layer1(x)
        x = self.activation1(x)
        x = self.max_pool1(x)

        x = self.conv_layer2(x)
        x = self.activation2(x)
        x = self.max_pool2(x)

        x = self.conv_layer3(x)
        x = self.activation3(x)
        x = self.max_pool3(x)

        x = self.conv_layer4(x)
        x = self.activation4(x)
        x = self.max_pool4(x)

        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.relu1(x)
        x = self.softmax(x)

        return x