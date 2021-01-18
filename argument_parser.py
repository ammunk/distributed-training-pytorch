#!/usr/bin/env python3

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', choices=['basic', 'training'], type=str)
    parser.add_argument('--dataloader', choices=['distributed','standard'],
                        type=str)

    # distributed settings

    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--local_world_size', type=int, help="Should be equal "
                        + "to the number of processes per node. Typically this "
                        + "would be equal to the number of GPUs on the node")

    args = parser.parse_args()

    return args
