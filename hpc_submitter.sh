#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=1:00:00
#SBATCH --account=rrg-kevinlb
#SBATCH --mem=5G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --job-name=tester_distributed

ORIG_DIR="$(pwd)"

cd "$SLURM_TMPDIR"
cp $ORIG_DIR/* .

var=(`scontrol show hostname $SLURM_NODELIST`)
node_sz=${#var[@]}

# load the necessary modules, depend on your hpc env
module load python/3.8.2

mkdir virtual_env && virtualenv --no-download virtual_env && source virtual_env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

for i in `seq 0 $(echo $node_sz -1 | bc)`;
do
    echo "launching ${i} job on ${var[i]} with master address ${var[0]}"
    srun -w ${var[$i]} -N 1 -n 1 -c 2 sh hpc_launcher.sh ${node_sz} ${i} ${var[0]} &
done

wait
