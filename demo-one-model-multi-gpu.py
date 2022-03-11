import os
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

from argument_parser import get_args
from toy_model_and_data import ToyData

class MultiGPUModel(nn.Module):

    def __init__(self, dev0, dev1):
        self.dev0 = dev0
        self.dev1 = dev1
        self.layers0 = nn.Sequential(
            nn.Linear(2, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
        )
        self.layers1 = nn.Sequential(
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1),
        )
        self.layers0.to(self.dev0)
        self.layers1.to(self.dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.layer0(x).to(self.dev1)
        return self.layer1(x)


def setup_distributed(config):
    # nccl is faster and is included in the pre-built binaries with CUDA
    dist.init_process_group(backend=config.backend)
    try:
        global_world_size = int(os.getenv('WORLD_SIZE', None))
        local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', None))
    except Exception as e:
        raise RuntimeError("WORLD_SIZE environment variable is required and "
                            + "must be an interger.")
    if dist.get_rank() == 0:
        print(f"world_size: {dist.get_world_size()}")
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        print(f"[Process {dist.get_rank()}] Available devices on machine: {torch.cuda.device_count()}")

    print(f"[Process {dist.get_rank()}] Hello from {socket.gethostname()}")

    # ensure each worker are seeded differently
    base_seed = config.seed
    config.seed += dist.get_rank()
    print(f"[Process {dist.get_rank()}] Base seed: {base_seed} and  worker seed: {config.seed}")

    dev0 = (dist.get_rank()*2) % local_world_size
    dev1 = (dist.get_rank()*2 + 1 ) % local_world_size

    # initialize models
    model = MultiGPUModel(dev0, dev1)

    # NOTE: device_ids MUST be None when splitting across multiple gpus
    # https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    model = DDP(model, device_ids=None)
    return model

def training_demo(model, dataloader, sampler):
    if dist.get_rank() == 0:
        run = wandb.init(project='distributed tester', settings=wandb.Settings(start_method='thread'))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
            optimizer.zero_grad(set_to_none=True)
            target = target.to(model.dev1)
            output = model(data)
            loss = loss(output, target)
            loss.backward()
            optimizer.step()
            if dist.get_rank() == 0:
                wandb.log({'loss/loss': loss.detach()})

            iteration_pbar.update(1)
            iteration += 1

            if iteration == max_iterations:
                training = False
                break
        epoch += 1
    iteration_pbar.close()
    dist.barrier()
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

if __name__ == '__main__':
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

    dist.destroy_process_group()
