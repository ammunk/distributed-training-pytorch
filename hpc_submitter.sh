#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --account=rrg-kevinlb
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:p100:2
#SBATCH --job-name=tester_distributed

ORIG_DIR="$(pwd)"

var=(`scontrol show hostname $SLURM_NODELIST`)
node_sz=${#var[@]}

# load the necessary modules, depend on your hpc env
module load python/3.8.2

echo "MOVING FILES AND INSTALLING VIRTUAL ENV"
cd "$SLURM_TMPDIR"
cp ${SUBMIT_DIR}/* .

# setup virtual environment
mkdir virtual_env && virtualenv --no-download virtual_env \
    && source virtual_env/bin/activate
pip install --upgrade pip
pip install pipenv
# we skip locking as it takes quite some time and is redundant
# note that we use the Pipfile and not the Pipfile.lock here -
# this is because compute canada's wheels may not include the specific
# versions specified in the Pipfile.lock file. The Pipfile is a bit less
# picky and so allows the packages to be installed. Although this could mean
# slightly inconsistencies in the various versions of the packages.
time pipenv install --skip-lock

for i in `seq 0 $(echo $node_sz -1 | bc)`;
do
    echo "launching ${i} job on ${var[i]} with master address ${var[0]}"
    srun -w ${var[$i]} -N1 -n1 ./hpc_launcher.sh ${node_sz} ${i} ${var[0]} $ORIG_DIR &
done

wait
