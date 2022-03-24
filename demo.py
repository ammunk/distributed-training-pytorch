import os
import datetime
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.optim as optim
import multiprocessing
import socket
import wandb
from tqdm import tqdm
import torch.multiprocessing
from torch.distributed.elastic.multiprocessing.errors import record

from argument_parser import get_args
from toy_model_and_data import ToyModel, ToyData

def setup_distributed(config):

    # initialize models
    model_X = ToyModel()
    model_Y = ToyModel()

    if config.torchrun:
        # nccl is faster and is included in the pre-built binaries with CUDA
        dist.init_process_group(backend=config.backend)
        try:
            global_world_size = int(os.getenv('WORLD_SIZE', None))
            local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', None))
        except Exception as e:
            raise RuntimeError("WORLD_SIZE environment variable is required and "
                                + "must be an interger.")
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        tasks_per_node = int(os.environ.get("TASKS_PER_NODE"))
        local_rank = int(os.environ.get("SLURM_LOCALID"))
        if config.use_node_rank:
            global_rank = int(os.environ.get("NODE_RANK"))*tasks_per_node + local_rank
        else:
            global_rank = int(os.environ.get("SLURM_PROCID"))

        if local_rank is None:
            local_rank = int(os.getenv("PBS_LOCALID", None))
        if local_rank is None:
            raise ValueError("Either PBS_LOCALID or SLURM_LOCALID must be set to specify this process' rank")

        world_size = int(os.getenv('WORLD_SIZE', None))

        addr = os.getenv("MASTER_ADDR", None)
        port = os.getenv("MASTER_PORT", None)
        if addr is None or port is None:
            raise ValueError("MASTER_ADDR and MASTER_PORT must be propvided with the env:// init_method")
        dist.init_process_group(backend=config.backend, init_method=f'tcp://{addr}:{port}', rank=global_rank, world_size=world_size)

    if dist.get_rank() == 0:
        print(f"[Process {dist.get_rank()}] World_size: {dist.get_world_size()}")
    print(f"[Process {dist.get_rank()}] Hello from {socket.gethostname()}")

    if local_rank == 0:
        print(f"[Process {dist.get_rank()}] Available devices on machine: {torch.cuda.device_count()}")

    # ensure each worker are seeded differently
    base_seed = config.seed
    config.seed += dist.get_rank()
    print(f"[Process {dist.get_rank()}] Base seed: {base_seed} and  worker seed: {config.seed}")
    print(f"[Process {dist.get_rank()}] Using torchrun: {config.torchrun}")
    print(f"[Process {dist.get_rank()}] Using backend: {config.backend}")

    # calling .cuda() will push model onto device local_rank
    torch.cuda.set_device(local_rank)

    # push models to devices and create DDP models
    model_X = model_X.cuda()
    model_X = DDP(model_X, device_ids=[local_rank])
    model_Y= model_Y.cuda()
    model_Y = DDP(model_Y, device_ids=[local_rank],)
    return model_X, model_Y

def training_demo(model_X, model_Y, dataloader, sampler):
    if dist.get_rank() == 0:
       run = wandb.init(project='distributed tester', settings=wandb.Settings(start_method='thread'))

    optimizer_X = optim.Adam(model_X.parameters(), lr=1e-3)
    optimizer_Y = optim.Adam(model_Y.parameters(), lr=1e-3)

    loss = nn.MSELoss()
    training = True
    max_iterations = 1000
    iteration = 0

    iteration_pbar = tqdm(range(int(max_iterations)), desc=f"Iteration",
                          disable=dist.get_rank()!=0)

    epoch = 0
    while training:
        if sampler is not None:
            # see https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
            sampler.set_epoch(epoch)
        for _, (data, target) in enumerate(dataloader):
            optimizer_X.zero_grad(set_to_none=True)
            optimizer_Y.zero_grad(set_to_none=True)
            data = data.cuda()
            target = target.cuda()
            output_X = model_X(data)
            output_Y = model_Y(data)
            l_X = loss(output_X, target)
            l_X.backward()
            l_Y = loss(output_Y, target)
            l_Y.backward()
            optimizer_X.step()
            optimizer_Y.step()
            if dist.get_rank() == 0:
                wandb.log({'loss/lossX': l_X.detach()}, commit=False)
                wandb.log({'loss/lossY': l_Y.detach()})

            iteration_pbar.update(1)
            iteration += 1

            if iteration == max_iterations:
                training = False
                break
        epoch += 1
    if dist.get_rank() == 0:
        iteration_pbar.close()
    print(f"[Process {dist.get_rank()}] Finished")

def get_dataloader(config, dataset):

    if config.dataloader == 'distributed':
        # Will partition the data so that each process works on their own
        # partition
        sampler = DistributedSampler(dataset, shuffle=True)
        return DataLoader(dataset, batch_size=256, shuffle=False,
                          pin_memory=True,
                          num_workers=config.num_workers,
                          sampler=sampler), sampler
    elif config.dataloader == 'standard':
        # Data will not be partiions. Each process draws data from the entire
        # dataset. As a consequence, setting shuffle=false leads to each process
        # working on the same minibatch.
        return DataLoader(dataset, pin_memory=True, batch_size=256,
                          shuffle=False, num_workers=config.num_workers), None

@record
def main():
    config = get_args()

    if config.dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'

    if config.backend == 'nccl' and config.num_workers > 0:
        # see
        # https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
        torch.multiprocessing.set_start_method('forkserver') # or spawn
        # If there are shared tensors in the dataset we set the following as per
        # https://github.com/pytorch/pytorch/issues/11201 and
        # https://github.com/pytorch/pytorch/issues/11899
        torch.multiprocessing.set_sharing_strategy('file_system')

    model_X, model_Y = setup_distributed(config)

    dataloader, sampler = get_dataloader(config, ToyData())

    training_demo(model_X, model_Y, dataloader, sampler)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
