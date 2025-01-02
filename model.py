import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,input_size):
        super(Net, self).__init__()
        drop = 0.5
        self.inp = nn.Sequential(
            nn.Linear(input_size,128),
            nn.Dropout(drop),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.L1 = nn.Sequential(
            nn.Linear(128,64),
            nn.Dropout(drop),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.Dropout(drop),
            nn.BatchNorm1d(128),
        )
        self.out = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.inp(x)
        x = self.L1(x1)
        x = self.out(x+x1)
        return x
    
if __name__ == '__main__':
    model = Net(64)
    print("模型大小:", sum(p.numel() for p in model.parameters() if p.requires_grad))