WANDB_CREDENTIALS_PATH=~/wandb_credentials.txt
export WANDB_API_KEY=$(cat $WANDB_CREDENTIALS_PATH)
export OMP_NUM_THREADS=1
export WORLD_SIZE=$SLURM_NTASKS
export TASKS_PER_NODE=1 # used internally to specify global_rank

source ../virtual_env/bin/activate

# reserve hostname as the main interactive node
nodes_list=(`scontrol show hostname $SLURM_NODELIST`)
num_nodes=${#nodes_list[@]}
echo "[$(hostname)]: Allocated nodes: ${nodes_list[@]}"
hostname="$(hostname | cut -d '.' -f 1)"
master_node=${nodes_list[0]}
# Job will be allocated on nodes in the same order as "nodes". The master node
# also coincides with the salloc landing node. Therefore is we use all nodes
# allocated to salloc we can use hostname as the master address. If using only a
# subset of the allocated node be careful and ensure that the master address
# (rank 0) lives at master address.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=2345

# assumes gpus are allocated using gres so that each task on the same node sees
# ALL gpus allocated per node
num_gpus_per_node=$(srun -w"${master_node}" -n1 -N1 --mem=1M -c1 bash -c 'echo ${CUDA_VISIBLE_DEVICES}' | awk -F ',' "{ print NF }")

# manually specify possible nodes
valid_nodes=$(printf ",%s" "${nodes_list[@]}")
valid_nodes="${valid_nodes:1}"
num_valid_nodes=$num_nodes

echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "Valid nodes: ${valid_nodes}"
echo "Num valid nodes: ${num_valid_nodes}"
echo "Master node: ${master_node}"
echo "Gpus per node: ${num_gpus_per_node}"


#################### GENERAL NOTES IF USING LIGHTNING WITH SLURM ####################

# PyTorch Lightning will query the environment to figure out if it is running inside a SLURM batch job
# If it is, it expects the user to have requested one task per GPU.
# If you do not ask for 1 task per GPU, and you do not run your script with "srun", your job will fail!

#################### NCCL ####################

# DOES NOT WORK WITH WHEEL torch-1.10.0+computecanada - very likely to crash

##############################################

# export NCCL_BLOCKING_WAIT=1 # Pytorch Lightning uses the NCCL backend for
                            # inter-GPU communication by default. Set this
                            # variable to avoid timeout errors. (CAN CAUSE LARGE
                            # OVERHEAD)
export NCCL_ASYNC_ERROR_HANDLING=1
echo "Running job with the NCCL backend"
export PL_TORCH_DISTRIBUTED_BACKEND=nccl
srun -w"${valid_nodes}" -N${num_valid_nodes} -n${WORLD_SIZE} \
    -c${SLURM_CPUS_PER_TASK} -o demo_nccl_lightning_output.out -D"$(dirname "$(pwd)")" \
    python demo_pytorch_lightning.py --gpus=${TASKS_PER_NODE} --nnodes=${num_valid_nodes}

#################### GLOO ####################

echo "Running job with the GLOO backend"
export PL_TORCH_DISTRIBUTED_BACKEND=gloo
srun -w"${valid_nodes}" -N${num_valid_nodes} -n${WORLD_SIZE} --gres=gpu:${num_gpus_per_node} \
    -c${SLURM_CPUS_PER_TASK} -o demo_gloo_lightning_output.out -D"$(dirname "$(pwd)")" \
    python demo_pytorch_lightning.py --gpus=${TASKS_PER_NODE} --nnodes=${num_valid_nodes}
