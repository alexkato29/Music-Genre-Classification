import torch.nn as nn
import torch.nn.functional as F

class AudioNet(nn.Module):
    def __init__(self, num_classes):
        super(AudioNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=128, stride=16, padding=64)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=64, stride=8, padding=32)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=32, stride=4, padding=16)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(128 * 20, 512)  # Adjusted based on new conv/pool layers
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x