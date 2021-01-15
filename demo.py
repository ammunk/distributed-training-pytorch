import os
import socket

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

from argument_parser import get_args
from toy_model_and_data import ToyModel
from toy_model_and_data import distributed_toy_data_loader, toy_data_loader

def basic_demo(local_rank, local_size, dataloader):
    print(f"dist rank = {dist.get_rank()}, provided rank = {local_rank}, "
          + f"world_size = {dist.get_world_size()}, "
          + f"local_world_size = {local_size}")
    print(f"Available devices = {torch.cuda.device_count()}")

def training_demo(local_rank, local_size, dataloader):
    n = torch.cuda.device_count() // local_size
    device_ids = list(range(local_rank*n, (local_rank+1)*n))

    with torch.cuda.device(device_ids[0]):
        model = ToyModel().cuda()
        ddp_model = DDP(model, device_ids=device_ids)
        loss = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=1e-3)

        for idx, (data, target) in enumerate(toy_data_loader):
            data = data.cuda()
            target = target.cuda()
            output = ddp_model(data)
            l = loss(output, target)
            l.backward()
            optimizer.step()

def setup_and_run(local_rank, local_world_size, fn, dataloader):
    dist.init_process_group(backend='nccl')
    fn(local_rank, local_world_size, dataloader)
    dist.destroy_process_group()

if __name__ == '__main__':
    args = get_args()
    if args.demo == 'basic':
        fn = basic_demo
    elif args.demo == 'training_demo':
        fn = training_demo
    else:
        raise ValueError("not a valid demo")

    if args.dataloader == 'distributed':
        dataloader = distributed_toy_data_loader
    else:
        dataloader = toy_data_loader

    setup_and_run(args.local_rank, args.local_world_size, fn, dataloader)
