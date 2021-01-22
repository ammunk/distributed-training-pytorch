#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=00:10:00
#SBATCH --account=rrg-kevinlb
#SBATCH --mem=5G
#SBATCH --cpus-per-task=2
#SBATCH --gpus=p100:2
#SBATCH --job-name=tester_distributed

ORIG_DIR="$(pwd)"

cd "$SLURM_TMPDIR"
cp $ORIG_DIR/* .

var=(`scontrol show hostname $SLURM_NODELIST`)
node_sz=${#var[@]}

# load the necessary modules, depend on your hpc env
module load python/3.8.2

if [ ! -d virtual_env ]; then
    # setup virtual environment
    mkdir virtual_env && virtualenv --no-download virtual_env && source virtual_env/bin/activate

    pip install --no-index --upgrade pip
    pip install --no-index -r requirements.txt
else
    source virtual_env/bin/activate
fi

echo ${SLURM_JOB_NUM_NODES}
echo ${SLURM_NODEID}
echo $SLURM_NTASKS
echo $SLURM_JOB_NODELIST

for i in `seq 0 $(echo $node_sz -1 | bc)`;
do
    echo "launching ${i} job on ${var[i]} with master address ${var[0]}"
    srun -w ${var[$i]} -N1 -n1 -l bash -c 'echo "$(hostname) ${SLURM_NODEID}"' # hpc_launcher.sh # ${node_sz} ${i} ${var[0]} &
done

wait
