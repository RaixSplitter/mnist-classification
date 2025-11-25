import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # in_channels=1 for grayscale
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # halves H,W each time
        self.fc1 = nn.Linear(32 * 7 * 7, num_classes)  # 28 -> 14 -> 7 (after two pooling layers)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (B,16,64,64)
        x = self.pool(F.relu(self.conv2(x)))   # (B,32,32,32)
        x = x.view(x.size(0), -1)              # flatten
        x = self.fc1(x)                        # (B, num_classes)
        return x