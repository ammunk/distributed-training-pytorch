import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.optim as optim

from argument_parser import get_args
from toy_model_and_data import ToyModel, ToyData

def basic_demo(local_rank, local_world_size, dataloader):
    print(f"dist rank = {dist.get_rank()}, provided rank = {local_rank}, "
          + f"world_size = {dist.get_world_size()}, "
          + f"local_world_size = {local_world_size}")
    print(f"Available devices = {torch.cuda.device_count()}")

def training_demo(local_rank, local_world_size, dataloader):
    n = torch.cuda.device_count() // local_world_size
    device_ids = list(range(local_rank*n, (local_rank+1)*n))
    GLOBAL_WORLD_SIZE = int(os.getenv('WORLD_SIZE', None))

    # setup groups
    all_ranks = torch.arange(GLOBAL_WORLD_SIZE)
    split_training = local_world_size > 1
    if split_training:
        X_ranks, Y_ranks = [ranks.tolist() for ranks in all_ranks.chunk(2)]
    else:
        X_ranks = all_ranks.tolist()
        Y_ranks = all_ranks.tolist()
    # create group X
    grp_X = dist.new_group(X_ranks)
    # create group Y
    grp_Y = dist.new_group(Y_ranks)
    # create group Z
    grp_Z = dist.new_group(all_ranks)

    with torch.cuda.device(device_ids[0]):
        print(f"[{dist.get_rank()}]: Training")
        model_X = ToyModel().cuda()
        model_Y = ToyModel().cuda()
        model_Z = ToyModel().cuda()
        if dist.get_rank() in X_ranks:
            ddp_model_X = DDP(model_X, device_ids=device_ids,
                              process_group=grp_X)
            optimizer_X = optim.SGD(ddp_model_X.parameters(), lr=1e-3)
        if dist.get_rank() in Y_ranks:
            ddp_model_Y = DDP(model_Y, device_ids=device_ids,
                              process_group=grp_Y)
            optimizer_Y = optim.SGD(ddp_model_Y.parameters(), lr=1e-3)
        ddp_model_Z = DDP(model_Z, device_ids=device_ids, process_group=grp_Z)
        loss = nn.MSELoss()
        optimizer_Z = optim.SGD(ddp_model_Z.parameters(), lr=1e-3)

        for idx, (data, target) in enumerate(dataloader):
            data = data.cuda()
            target = target.cuda()
            output = 0
            if dist.get_rank() in X_ranks:
                output += ddp_model_X(data).squeeze()
            elif dist.get_rank() in Y_ranks:
                output += ddp_model_Y(data).squeeze()
            output += ddp_model_Z(data).squeeze()
            l = loss(output, target)
            l.backward()
            if dist.get_rank() in X_ranks:
                optimizer_X.step()
            if dist.get_rank() in Y_ranks:
                optimizer_Y.step()
            optimizer_Z.step()
            if idx > 100:
                break
    print(f"[{dist.get_rank()}]: Finished")

def setup_and_run(local_rank, local_world_size, fn, args):
    dist.init_process_group(backend='nccl', init_method='env://')
    print(f"[RANK {dist.get_rank()}]: STARTING RUN")

    if dist.get_rank() == 0:
        print(f"world_size = {dist.get_world_size()}")
        print(f"local_world_size = {local_world_size}")
        print(f"Available devices = {torch.cuda.device_count()}")

    if args.dataloader == 'distributed':
        # Will partition the data so that each process works on their own
        # partition
        dataloader = DataLoader(ToyData(), batch_size=2, shuffle=False,
                                num_workers=2,
                                sampler=DistributedSampler(ToyData(),
                                                           shuffle=True))
    elif args.dataloader == 'standard':
        # Data will not be partiions. Each process draws data from the entire
        # dataset. As a consequence, setting shuffle=false leads to each process
        # working on the same minibatch.
        dataloader = DataLoader(ToyData(), batch_size=2, shuffle=False,
                                num_workers=2)

    fn(local_rank, local_world_size, dataloader)
    dist.destroy_process_group()

if __name__ == '__main__':
    args = get_args()
    if args.demo == 'basic':
        fn = basic_demo
    elif args.demo == 'training':
        fn = training_demo
    else:
        raise ValueError("not a valid demo")

    setup_and_run(args.local_rank, args.local_world_size, fn, args)
