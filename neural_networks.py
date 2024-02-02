import torch.nn as nn


class SingleHyperplaneNet(nn.Module):
    def __init__(self):
        super(SingleHyperplaneNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 2)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)  # Flatten the input
        x = self.fc1(x)
        return x


class ThreeHyperplanesNet(nn.Module):
    def __init__(self):
        super(ThreeHyperplanesNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, 2)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
