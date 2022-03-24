WANDB_CREDENTIALS_PATH=~WANDB_CREDENTIALS_PATH=~/wandb_credentials.txt
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

# For the difference between different backends see
# https://pytorch.org/docs/stable/distributed.html?highlight=init_process_group#torch.distributed.init_process_group

#################### NCCL ####################

# DOES NOT WORK WITH WHEEL torch-1.10.0+computecanada - very likely to crash

##############################################

export NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.
echo "Running job with the GLOO backend"
srun -w"${valid_nodes}" -N${num_valid_nodes} -n${WORLD_SIZE} \
    -c${SLURM_CPUS_PER_TASK} -o demo_nccl_multi_gpu_model_output.out -D"$(dirname "$(pwd)")" \
    python demo_one_model_multi_gpu.py --help
