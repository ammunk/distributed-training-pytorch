#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=tester_distributed
#SBATCH --partition=plai

export PLAI_TMPDIR="/scratch-ssd/amunk_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

if [ ! -d virtual_env ]; then
    # setup virtual environment
    mkdir virtual_env
    python3 -m venv virtual_env
    source virtual_env/bin/activate

    pip install --upgrade pip
    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
else
    source virtual_env/bin/activate
fi

var=(`scontrol show hostname $SLURM_NODELIST`)
node_sz=${#var[@]}
nvidia-smi topo -m
for i in `seq 0 $(echo $node_sz -1 | bc)`;
do
    echo "launching ${i} job on ${var[i]} with master address ${var[0]}"
    srun -w ${var[$i]} -N1 -n1 bash hpc_launcher_launcher.sh ${node_sz} ${i} \
        ${var[0]} $PLAI_TMPDIR &
done

wait
