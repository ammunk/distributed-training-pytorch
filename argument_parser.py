#!/usr/bin/env python3

import argparse
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataloader', choices=['distributed','standard'],
                        type=str, default='distributed')
    parser.add_argument('--backend', choices=['nccl','mpi', 'gloo'],
                        type=str, default='nccl')
    parser.add_argument("--torchrun", action="store_true",
                        help="Specify we are using torchrun to distribute jobs")

    parser.add_argument("--use_node_rank", action="store_true",
                        help="Specify whether to use the NODE_RANK environment variable to determine global rank of each process")

    parser.add_argument('--seed', default=random.randint(0,2**32-1), type=int)

    parser.add_argument('--num_workers', default=0, type=int)

    parser.add_argument("--dry_run", action="store_true",
                        help="Dry run (do not log to wandb)")

    # distributed settings
    args = parser.parse_args()

    return args
