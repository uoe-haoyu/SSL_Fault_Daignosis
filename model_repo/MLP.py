import torch
from torch import nn
import os
from einops import rearrange
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, pretrain=None):
        super().__init__()
        self.name = os.path.basename(__file__).split('.')[0]
        self.fc1 = nn.Linear(4*32, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 19)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        feature = self.relu(self.fc1(input))

        output = self.fc2(feature)
        return output, feature
