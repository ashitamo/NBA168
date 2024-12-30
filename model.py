import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,input_size):
        super(Net, self).__init__()
        drop = 0.0
        self.L1 = nn.Sequential(
            nn.Linear(input_size, 796),
            nn.Dropout(drop),
            nn.ReLU(),
            nn.Linear(796, 256),
            nn.Dropout(drop),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(drop),
            nn.ReLU(),
        )
        self.L2 = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        return x