#!/bin/bash
## for torch distributed launch
## **all the same, except  node_rank = 1**
nnodes=$1               # total number of nodes used in this computation
node_rank=$2            # current node rank, 0-indexed
nproc_per_node=2        # number of processes (models) per node
master_addr=$3          # hostname for the master node
TMPDIR=$4
port=8888               #
SOURCE_DIR="$(pwd)"
mkdir -p $TMPDIR
cd "$TMPDIR"
cp ${SOURCE_DIR}/* .

# echo "LAUNCHING PYTHON SCRIPT on $(hostname)"
# python -m torch.distributed.launch \
#     --nproc_per_node ${nproc_per_node} \
#     --nnodes ${nnodes} \
#     --node_rank ${node_rank} \
#     --master_addr ${master_addr} \
#     --master_port ${port} \
#     demo.py \
#     --local_world_size ${nproc_per_node} \
#     --demo basic \
#     --dataloader standard

ls /scratch-ssd
cd $SOURCE_DIR
rm -rf /scratch-ssd/amunk_*
echo "DONE CLEANING"
