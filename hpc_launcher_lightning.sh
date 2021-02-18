#!/bin/bash
## for torch distributed launch
## **all the same, except  node_rank = 1**
nnodes=$1               # total number of nodes used in this computation
nproc_per_node=2        # number of processes (models) per node
SOURCE_DIR="$(pwd)"
cd "${SLURM_TMPDIR}"
cp ${SOURCE_DIR}/* .

echo "LAUNCHING PYTHON SCRIPT on $(hostname)"
demo-pytorch-lightning.py \
    --gpus ${nproc_per_node}
    --num_nodes ${nnodes}
    --num_workers 2

nvidia-smi topo -m
