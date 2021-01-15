#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --account=rrg-kevinlb
#SBATCH --exclusive
#SBATCH --job-name=tester_distributed

var=(`scontrol show hostname $SLURM_NODELIST`)
node_sz=${#var[@]}
echo $var
echo "$node_sz"
echo `seq 0 $(echo $node_sz -1 | bc)`

# load the necessary modules, depend on your hpc env
module load python/3.8.2
pipenv install

for i in `seq 0 $(echo $node_sz -1 | bc)`;
do
    echo "launching ${i} job on ${var[i]} with master address ${var[0]}"
    srun -w ${var[$i]} -N 1 -n 1 -c 24 launch.sh ${node_sz} ${i} ${var[0]} &
done

wait
