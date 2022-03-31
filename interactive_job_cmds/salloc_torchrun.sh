WANDB_CREDENTIALS_PATH=~/wandb_credentials.txt
export WANDB_API_KEY=$(cat $WANDB_CREDENTIALS_PATH)
export OMP_NUM_THREADS=1
export WORLD_SIZE=$SLURM_NTASKS
export TASKS_PER_NODE=$(( SLURM_NTASKS / SLURM_NNODES )) # used internally to specify global_rank

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
echo "Tasks per node: ${TASKS_PER_NODE}"

echo "Running distributed jobs by looping over individual nodes"
for i in `seq 0 $(( num_valid_nodes - 1 ))`;
do
    node=${nodes_list[i]}
    echo "Running on node ${node} with rank $i"
    export NODE_RANK=${i} # used internally to specify global_rank
    srun -w"$node" -N1 -n${TASKS_PER_NODE} -o "demo_individual_output.out" -D"$(dirname "$(pwd)")" \
      python demo.py --backend=nccl --use_node_rank &
    echo "Process id: $!"
done
wait

# For the difference between different backends see
# https://pytorch.org/docs/stable/distributed.html?highlight=init_process_group#torch.distributed.init_process_group

#################### NCCL ####################

# DOES NOT WORK WITH WHEEL torch-1.10.0+computecanada - very likely to crash

##############################################

echo "Running job with the NCCL backend and torchrun"
export NCCL_ASYNC_ERROR_HANDLING=1
srun -w"${valid_nodes}" -N${num_valid_nodes} -n${num_valid_nodes} \
    -c${SLURM_CPUS_PER_TASK} -o demo_nccl_output.out -D"$(dirname "$(pwd)")" \
    torchrun --nnodes=${num_valid_nodes} --nproc_per_node=${TASKS_PER_NODE} \
    --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    demo.py --backend=nccl --torchrun

#################### MPI ####################

# To use MPI as the pytorch distributed backend you need to use a pytorch
# version that is build from source on a host with MPI installed.
# On computecanada this requires using their pre-built pytorch wheels
#
# AS PER COMPUTECANADA SUPPORT: THIS CURRENTLY DOESN NOT WORK ON CEDAR DUE TO
# THE INTERCONNECTION BEING Intel OmniPath - should work on GRAHAM which uses
# Infiniband.

##############################################

# echo "Running job with the MPI backend"
# srun -w"${valid_nodes}" -N${num_valid_nodes} -n${WORLD_SIZE} --mpi=pmi2 \
#     -c${SLURM_CPUS_PER_TASK} -o demo_mpi_output.out -D"$(dirname "$(pwd)")" \
#     python demo.py --backend=mpi
# wait

echo "Running MPI job with mpiexec"
mpiexec -n ${WORLD_SIZE} \
    -wdir "$(dirname "$(pwd)")" \
    python demo_assume_started_with_mpiexec.py --backend=nccl
wait

#################### GLOO ####################

echo "Running job with the GLOO backend"
srun -w"${valid_nodes}" -N${num_valid_nodes} -n${WORLD_SIZE} -o "demo_gloo_output.out" -D"$(dirname "$(pwd)")" python demo.py --backend=gloo
