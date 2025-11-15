import torch
import torch.nn as nn
import torch.nn.functional as F

class NNPilot(nn.Module):
    def __init__(self, num_classes=10):
        super(NNPilot, self).__init__()
        # Input: (B, 3, 32, 32) for RGB 32×32 images (like CIFAR-10)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # after 2 poolings (32→16→8)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Example usage
if __name__ == "__main__":
    model = NNPilot(num_classes=10)
    dummy_input = torch.randn(4, 3, 32, 32)  # batch of 4 images
    output = model(dummy_input)
    print(output.shape)  # torch.Size([4, 10])
