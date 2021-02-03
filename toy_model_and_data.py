#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.distributed as dist

class ToyModel(nn.Module):

    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(2, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 1)

    def forward(self, x):
        # print(f"[{dist.get_rank()}]: {x}")
        # print(f"[{dist.get_rank()}]: {x.shape}")
        return self.net2(self.relu(self.net1(x)))

class ToyData(Dataset):
    def __init__(self):
        self.data = [[float(x)]*2 for x in range(10000)]
        self.y = [float(x % 10) for x in range(10000)]

    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx]), torch.tensor(self.y[idx])

    def __len__(self):
        return len(self.data)

