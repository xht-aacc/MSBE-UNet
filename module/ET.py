import torch
import torch.nn as nn
import torch.nn.functional as F

class ET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ET, self).__init__()
        # Define layers for edge detection
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Forward pass
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = torch.sigmoid(self.conv3(x2))  # Using sigmoid activation for edge detection
        return x3 + x

if __name__ == '__main__':
    model = ET(1, 1)
    x = torch.randn(2, 1, 400, 400)
    y = model(x)
    print(y.shape)

