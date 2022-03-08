#!/usr/bin/env python3

import argparse
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataloader', choices=['distributed','standard'],
                        type=str)
    parser.add_argument('--backend', choices=['nccl','mpi'],
                        type=str)

    parser.add_argument('--seed', default=random.randint(0,2**32-1), type=int)

    parser.add_argument("--dry_run", action="store_true",
                        help="Dry run (do not log to wandb)")

    # distributed settings
    args = parser.parse_args()

    return args
