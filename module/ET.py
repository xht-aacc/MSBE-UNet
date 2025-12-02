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

# class EdgeDetector(nn.Module):
#     def __init__(self):
#         super(EdgeDetector, self).__init__()
#         # Define layers for edge detection
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
#         self.sobel_input = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
#         self.sobel_output = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
#
#         # Sobel kernel for edge detection
#         sobel_kernel_input = torch.FloatTensor([[1, 0, -1],
#                                                 [2, 0, -2],
#                                                 [1, 0, -1]])
#         sobel_kernel_input = sobel_kernel_input.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3) for Conv2d compatibility
#         self.sobel_input.weight = nn.Parameter(sobel_kernel_input)
#
#         sobel_kernel_output = torch.FloatTensor([[1, 2, 1],
#                                                  [0, 0, 0],
#                                                  [-1, -2, -1]])
#         sobel_kernel_output = sobel_kernel_output.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3) for Conv2d compatibility
#         self.sobel_output.weight = nn.Parameter(sobel_kernel_output)
#
#     def forward(self, x):
#         # Apply Sobel operator on input x
#         x_edge_input = F.conv2d(x, self.sobel_input.weight)
#
#         # Forward pass through convolutional layers
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = torch.sigmoid(self.conv3(x))  # Using sigmoid activation for edge detection
#
#         # Apply Sobel operator on output x
#         x_edge_output = F.conv2d(x, self.sobel_output.weight)
#
#         return x, x_edge_input, x_edge_output