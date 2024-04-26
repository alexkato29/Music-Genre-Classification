import torch.nn as nn
import torch.nn.functional as F

class AudioNet(nn.Module):
    def __init__(self, num_classes):
        super(AudioNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=64, stride=16)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=32, stride=8)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32 * 62, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.mean(dim=1)  # Assuming stereo to mono
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    @staticmethod
    def test_import():
        return 256
