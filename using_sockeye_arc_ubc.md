# Sockeye on Arc (UBC)

## Load modules
```bash
module load Software_Collection/2021
module load gcc/9.4.0 cuda/11.3.1
module load openmpi/4.1.1-cuda11-3
```
##  Make sure to install mpi4py with openmpi loaded (must have access to the mpi library)
```bash
pip install mpi4py # + other packages such as torch, numpy, etc
```
## Setup hpc script (example)

```bash
#PBS -l walltime=0:20:00,select=2:ncpus=4:ngpus=2:mem=10gb:mpiprocs=2
#PBS -N [NAME]
#PBS -A st-fwood-1-gpu
#PBS -o output.txt
#PBS -e error.txt

cd /scratch/st-fwood-1/amunk/distributed-training-pytorch
export TASKS_PER_NODE=2 # make sure this matches 'mpiprocs' specified at the top

module load openmpi
module load Software_Collection/2021
module load gcc/9.4.0 cuda/11.3.1

# Count the number of processes across all nodes.
np=$(wc -l < $PBS_NODEFILE)

source virtual_env/bin/activate

mpiexec -np $np python demo_assume_started_with_mpiexec.py

```

Where the `-l` flag specifies requested resources with fairly self-explanatory
arguments. Note that the requested resources are **per node** and the number of
nodes is specified using the `select` argument.
 
