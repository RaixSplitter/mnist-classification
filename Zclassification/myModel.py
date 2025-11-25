import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10, apply_softmax: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # in_channels=1 for grayscale
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # halves H,W each time
        self.fc1 = nn.Linear(32 * 32 * 32, num_classes)  # 128 -> 64 -> 32 (after two pooling layers)
        self.apply_softmax = apply_softmax

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (B,16,64,64)
        x = self.pool(F.relu(self.conv2(x)))   # (B,32,32,32)
        x = x.view(x.size(0), -1)              # flatten
        x = self.fc1(x)                        # (B, num_classes)
        if self.apply_softmax:
            x = F.softmax(x, dim=1)
        return x
    
class AdvancedCNN(nn.Module):
    def __init__(self, num_classes: int = 10, apply_softmax: bool = False):
        super().__init__()
        
        # Convolutional Block 1: 128x128 -> 64x64
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Convolutional Block 2: 64x64 -> 32x32
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Convolutional Block 3: 32x32 -> 16x16
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Convolutional Block 4: 16x16 -> 8x8
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.25)
        
        # Fully Connected Layers
        # After 4 pooling layers: 128 -> 64 -> 32 -> 16 -> 8
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout6 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        self.apply_softmax = apply_softmax
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        x = self.fc3(x)
        
        if self.apply_softmax:
            x = F.softmax(x, dim=1)
        
        return x