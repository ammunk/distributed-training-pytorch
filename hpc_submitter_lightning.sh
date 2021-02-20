#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:10:00
#SBATCH --account=rrg-kevinlb
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:p100:2
#SBATCH --job-name=tester_distributed

var=(`scontrol show hostname $SLURM_NODELIST`)
node_sz=${#var[@]}

# load the necessary modules, depend on your hpc env
module load python/3.8.2

if [ ! -d virtual_env ]; then
    # setup virtual environment
    mkdir virtual_env
    python3 -m venv virtual_env
    source virtual_env/bin/activate

    pip install --upgrade pip
    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 \
        torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html \
	    pytorch-lightning
else
    source virtual_env/bin/activate
fi

srun bash hpc_launcher_lightning.sh ${node_sz}
