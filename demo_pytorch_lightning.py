# https://pytorch-lightning.readthedocs.io/en/stable/new-project.html
import argparse
import random
import os

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch.distributed as dist

from toy_model_and_data import ToyData, ToyModel

class LitToyModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model_X = ToyModel()
        self.model_Y = ToyModel()
        self._loss = nn.MSELoss()

    def forward(self, x):
        return self.model_X(x), self.model_Y(x)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        output_X, output_Y = self(x)
        loss_X = self._loss(output_X, y)
        loss_Y = self._loss(output_Y, y)
        loss = loss_X + loss_Y
        self.log('loss/lossX', loss_X.detach())
        self.log('loss/lossY', loss_Y.detach())
        return loss

    def configure_optimizers(self):
        optimizer_X = torch.optim.Adam(self.model_X.parameters())
        optimizer_Y = torch.optim.Adam(self.model_Y.parameters())
        # lightning will loop through these and call step on each
        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        return [optimizer_X, optimizer_Y]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--gpus', default=1, type=int,
                        help="number of gpus per node")
    parser.add_argument('--nnodes', default=1, type=int)
    parser.add_argument('--steps', default=1000, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    args = parser.parse_args()

    wandb_logger = WandbLogger(project='lightning distributed tester', settings=wandb.Settings(start_method='thread'))

    train_dataloader = DataLoader(ToyData(), batch_size=args.batch_size,
                                  pin_memory=True,
                                  )
    model = LitToyModel()
    trainer = pl.Trainer(gpus=args.gpus, num_nodes=args.nnodes,
                         max_steps=args.steps, precision=32, accelerator='gpu',
                         log_every_n_steps=min(50, len(train_dataloader)/args.batch_size),
                         logger=wandb_logger,
                         strategy='ddp') # the backend used for ddp is specified via the PL_TORCH_DISTRIBUTED_BACKEND environment variable

    trainer.fit(model, train_dataloader)
