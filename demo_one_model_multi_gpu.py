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
        super().__init__()
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
        x = self.layers0(x).to(self.dev1)
        return self.layers1(x)


def setup_distributed(config):

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
        local_world_size = int(os.environ.get("TASKS_PER_NODE"))
        local_rank = int(os.environ.get("SLURM_LOCALID"))
        if config.use_node_rank:
            global_rank = int(os.environ.get("NODE_RANK"))*local_world_size + local_rank
        else:
            global_rank = int(os.environ.get("SLURM_PROCID"))

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

    # assume one task per node consuming all gpus
    num_gpus_per_task = torch.cuda.device_count()
    assert num_gpus_per_task == 2
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
        run = wandb.init(project='distributed tester',
                         group='multi-gpu-per-node',
                         settings=wandb.Settings(start_method='thread'))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = nn.MSELoss()
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
            target = target.to(model.module.dev1)
            output = model(data)
            loss = loss_fn(output, target)
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

    model = setup_distributed(config)

    dataloader, sampler = get_dataloader(config, ToyData())

    training_demo(model, dataloader, sampler)

    dist.destroy_process_group()
