import torch

from torch import nn

class FeMnistNetwork(nn.Module):
    def __init__(self) -> None:
        super(FeMnistNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear1 = nn.Linear(7*7*64, 2048)
        self.linear2 = nn.Linear(2048, 62)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu((self.linear1(x)))
        x = self.linear2(x)
        return x