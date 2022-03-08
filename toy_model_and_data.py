#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.distributed as dist

class ToyModel(nn.Module):

    def __init__(self):
        super(ToyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.layers(x)

class ToyData(Dataset):
    def __init__(self):
        self.data = [torch.tensor([float(x)]*2) for x in torch.randn(512)]
        self.y = [torch.randn(1)*0.5 + x[0]**2 for x in self.data]

    def __getitem__(self, idx):
        return self.data[idx], self.y[idx]

    def __len__(self):
        return len(self.data)

