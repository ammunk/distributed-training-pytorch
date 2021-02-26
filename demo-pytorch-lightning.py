# https://pytorch-lightning.readthedocs.io/en/stable/new-project.html
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from toy_model_and_data import ToyData, ToyModel

def seed_all(seed):
    """Seed all devices deterministically off of seed and somewhat
    independently."""

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class LitToyModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = ToyModel()
        self._loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        x = self.model(x).squeeze()
        loss = self._loss(x, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        return optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--gpus', default=1, choices=list(range(1,9)), type=int)
    parser.add_argument('--num_nodes', default=1, type=int)
    parser.add_argument('--steps', default=50, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    args = parser.parse_args()
    seed_all(args.seed)

    train_dataloader = DataLoader(ToyData(), batch_size=2,
                                  num_workers=args.num_workers)
    model = LitToyModel()
    trainer = pl.Trainer(gpus=args.gpus, num_nodes=args.num_nodes,
                         max_steps=args.steps, accelerator='ddp', precision=16)
    trainer.fit(model, train_dataloader)
