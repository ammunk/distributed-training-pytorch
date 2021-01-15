#!/usr/bin/env python3

import numpy as np

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class ToyModel(nn.Module):

    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 1)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

class ToyData(Dataset):
    def __init__(self):
        self.data = [[float(x)]*32 for x in range(10000)]
        self.y = [x % 10 for x in range(10000)]

    def __getitem__(self, idx):
        return np.asarray(self.data[idx]), self.y[idx]

    def __len__(self):
        return len(self.data)

distributed_toy_data_loader = Dataloader(ToyData(), batch_size=128,
                                         shuffle=False, num_workers=2,
                                         sampler=DistributedSampler(ToyData))
toy_data_loader = Dataloader(ToyData(), batch_size=128, shuffle=True,
                             num_workers=2)
