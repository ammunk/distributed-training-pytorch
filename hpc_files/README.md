# SLURM hpc scripts

The approach taken here rely on **bash** as opposed to **python**, and the hpc
scripts serve one of three (overlapping) purposes:

- [Multi-node distributed GPU training](#multi-node-distributed-gpu-training) of
  [PyTorch](https://pytorch.org/) models
- [Singuarity](https://sylabs.io/guides/3.7/user-guide.pdf) or virtual
  environment based projects
- [Weights and Biases sweeper](https://docs.wandb.ai/sweeps) jobs (great for
  hyperparameter searches)
  
The scripts are designed in order to make the transfer of a locally working
application to the hpc clusters as easy and painless as possible.
  
There are two types of scripts, which differ by how dependencies are managed for
your application: 

- **Singularity containers**
- **python virtual environments** 

The Singularity approach offers much greater flexibility where all dependencies
are specified in a "Singularity file", whereas the python virtual environment
approach (obviously) must be a python application.

Depending on whether you use Singularity or a python virtual environment they
each pose slightly different constraints on how experiments run once a job has
been submitted. These constraints are minimal so that you do not have to give up
e.g. Singularity's flexibility yet ensures the script can make some assumptions
about how to run your experiments. The details on this can be found in the
[Singularity readme] or the [virtual environment readme].
  
To use these scripts, simply copy them into your **[appropriately
structured](#project-structure)** project. The scripts are written to be
(almost) project agnostic, which effectively means that the scripts will:

- Automatically set up the experiments which prevents mixing different projects.

- Ensure the correct relationship between requested number of *gpus*, *nodes*,
and *cpus per gpu* depending on the type of distributed job. 

- [Manage the transfer](#copying-datasets-and-other-files-to-slurm_tmpdir) of
data and directories to and from local nodes for faster read/write operations -
i.e. via the `SLURM_TMPDIR` environment variable.

The created folders and their location are easily accessible as [environment
variables](#environment-variables). One thing to pay attention to is that
Singularity based jobs needs additional folders compared to the virtual
environment based jobs. For details see the [created folders](#created-folders).

#### Important:

The scripts rely on the `SCRATCH` environment variable. If `SCRATCH` is not set
by default add

``` bash
export SCRATCH=[path to scratch]
```
to your `~/.bashrc`.

Additionally, you will notice references to the `SLURM_TMPDIR`. This variable
points to a temporary directory created for each job pointing to a job-specific
directory on each local node. If the job is allocated multiple nodes the
temporary directory is unique on each node. Some clusters will have these
automatically set. However, if this is not the case make sure to set this up
yourself.

## Submitting jobs

To submit jobs call one of two submitter jobs. Which one depends on whether your
application uses Singularity or a virtual environment. Note that the job
submitter file by default assumes you use a virtual environment. To specify a
Singularity based job, use the `-s, --singularity-container` option.

The scripts distinguish between two types of jobs, and how to specify the
experiment's configurations depend on which type:

- Array jobs for hyperparameter searches using `wandb` sweeps - see [integration
  with Weights and Biases](#integration-with-weight-and-biases) for more
  details.
- Single jobs which supports multi-node distributed gpu applications
  - The experiments configurations are specified using the
    [experiment_configurations.txt] file. It's format differs slightly depending
    on whether you use Singularity or a virtual environment. For details, see
    the [Singularity readme] or the [virtual environment readme].

The options that control the job submissions are:

``` text
-a, --account                 Account to use on cedar (def-fwood, rrg-kevinlb).
                                Ignored on the PLAI cluster. Default: rrg-kevinlb
-g, --gpus                    Number of gpus per node. Default: 1
-c, --cpus                    Number of cpus per node: Default: 2
-j, --job-type                Type of job to run, one of
                                (standard, sweep, distributed).
                                Default: standard
-W, --which-distributed       Kind of distributed gpu application backend used
                                (lightning, torchrun). Must be provided if using
                                "--job-type distributed"
-t, --time                    Requested runtime. Format: dd-HH:MM:SS.
                                Default: 00-01:00:00
-m, --mem                     Amount of memory per node. E.g. 10G or 10M.
                                Default: 10G
-G, --gpu-type                Type of gpu to use (p100, p100l, v100l). Ignored on
                                the PLAI cluster. Default: v100l
-e, --exp-name                Name of the experiment. Used to created convenient
                                folders in ${SCRATCH}/${project_name} and to name
                                the generated output files. Default: "" (empty)
-n, --num_nodes               Number of nodes. Default: 1
-d, --data                    Whitespace separated list of paths to directories or
                                files to transfer to ${SLURM_TMPDIR}. These paths
                                MUST be relative to ${SCRATCH}/${project_name}
-s, --singularity-container   Path to singularity container. If specified the
                                job is submitted as a Singularity based job
-w, --workdir                 Path to a mounted working directory in the
                                Singularity container
-C, --configs                 Path to file specifying the experiment
                                configuration. Default: experiment_configurations.txt

-h, --help                    Show this message
```
#### Example

Assume we have a project with the [appropriate structure](#project-structure).


To submit a job first `cd [path to project]/hpc_files`, and then

``` bash
bash job_submitter.sh \
  --gpus 2 \
  --cpus 2 \
  --exp-name testing \
  --num-nodes 2
```

### Integration with Weights and Biases

To use the [Weight and Biases](https://wandb.ai/) sweeps, you need to first
install `wandb` into your Singularity container or virtual environment,

``` python
pip install wandb
```

To use `wandb` requires a user login. Either do `wandb login`, where `wandb`
will prompt for a username and password, or set the `WANDB_API_KEY` environment
variable to the api key provided by weight and biases after you sign up.

The scripts found here take the latter approach by searching for your api key in
`~/wandb_credentials.txt`. As long as you copy your api key into
`~/wandb_credentials.txt` your applications can log experiment progress using
`wandb`.

#### Sweeper jobs

When you submit a `wandb` sweep array job, you only need to specify the sweep
id. That is, first initiate the sweep (either locally or on your favorite
cluster),

``` bash
wandb sweep sweeper.yml
```

This will create a pending sweep on `wandb`'s servers. Then in
`project root/hpc_files` do

``` bash
bash job_submitter.sh --job_type sweep [other options]
```

The script will then prompt for the sweep id and the number of sweeps.

The provided [sweeper.yml] file can serve as a template, but should be
modified to your specific sweep. Think of the [sweeper.yml] file as the
sweep's equivalent of the more general
[experiment_configurations.txt] file.

### How to specify experiment configurations:

- For sweep jobs edit `sweeper.yml`.
- Otherwise edit [experiment_configurations.txt]. See the [Singularity readme]
  or [virtual environment readme] for the format.

## Copying datasets and other files to `SLURM_TMPDIR`

To copy data to the local nodes when submitting a job, simply use the `-d,
--data` option. To transfer multiple files and directories specify these using a
whitespace separated list of **relative** paths.

The main purpose of this functionality is to copy large amounts of data, which
typically is stored on `SCRTACH`. Therefore, the paths are going to be
relative to `${SCRATCH}/${project_name}`. The script will then create a tarball
using `tar` and transfer the files and directories to `SLURM_TMPDIR`. You can
then access the files and directories at `SLURM_TMPDIR` using the same paths
used when using the `-d, --data` option.

If a tarball already exists, no new tarball is created. If you want to update
the tarball you should delete the old one first.

### Example

Assume you work on a project named `project_root`, and on `${SCRATCH}` you have,

``` text
${SCRATCH}
├── project_root
│   └── datasets
│       ├──dataset1
│       ├──dataset2
│       └──dataset3
.
. other files on scratch
.
```

If you want to move the entire directory `datasets` to `${SLURM_TMPDIR}`, you
would do

``` bash
bash job_submitter.sh \
  --data datasets
```

This would then lead to the following structure on `${SLURM_TMPDIR}`

``` text
${SLURM_TMPDIR}
├── datasets
│   ├── dataset1
│   ├── dataset2
│   └── dataset3
```

If instead you want to move only `dataset1` and `dataset2`, you would do

``` bash
bash job_submitter.sh \
  --data datasets/dataset1 datasets/dataset2
```
This would then lead to the following structure
``` text
${SLURM_TMPDIR}
├── datasets
│   ├── dataset1
│   └── dataset2
```

In your specific experiment, you would have an option to specify the location of
a dataset (using e.g. python's `argparse`). You could then configure your
program to look for `dataset1` by running

``` bash
python my_program.py --data_dir=${SLURM_TMPDIR}/datasets/dataset1 [other arguments]
```

## Multi-node distributed gpu training

The scripts have been tested with two different ways to do multi-node
distributed gpu training with [PyTorch]('https://pytorch.org/'),

- Using PyTorch's [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)


The practical difference in terms of submitting a job is what each approach
considers a task. The hpc scripts found in this repo will make sure to submit a
job with the appropriate relationship between gpus, nodes, and cpus.

In terms of writing the application code, Lightning removes a lot of the
distributed training setup and does this for you. It also offers multiple
optimization tricks that have been found to improve training of neural network
based models. The downside is that Lightning is (slightly) more rigid in terms
of managing the gpus across the distributed processes. Using PyTorch's
`torchrun` offers full flexibility, but requires manually setting up the
distributed training.

To get comfortable with these different approaches and play around with them
check out my [distributed training
repository](https://github.com/ammunk/distributed-training-pytorch) which also
uses the hpc scripts found here.

### Lightning

Lightning is built on PyTorch and requires your code to be written using a
certain structure. It has a lot of functionality, but it attempts to streamline
the training process to be agnostic to any particular neural network training
program. Lightning includes loads of functionalities, but fundamentally you can
think of Lightning as doing the training loop for you. You only have to write
the training step, which is then called by Lightning.

The benefit of the design of Lightning is that Lightning manages distributing
your code across multiple gpus without you having to really change your code.

### `torchrun`

If you use the `torchrun` approach you achieve full flexibility in how to manage
the gpus for each process. Under the hood `torchrun` spawns subprocesses, and
requires you to specify which machine the is the "master" machine as well as
which port these processes use to communicate to each other.

If you use a virtual environment for you application, the hpc scripts provided
in this repository handles this for you. However, if you use Singularity you
have to manage this yourself: either as a command passed to the Singularity
container or build the Singularity container to take care of this.

[`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) comes with the
installation of PyTorch, and should be executed on **each node** using the
following execution pattern,

``` bash
torchrun --nproc_per_node 2 --nnodes 2 \
    --rdzv_id=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.1:2345 \
    --max_restarts=3 \
    YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other arguments of your training script)
```

See the
[virtual_env_hpc_files/distributed_scripts/torchrun_launcher.sh](virtual_env_hpc_files/distributed_scripts/torchrun_launcher.sh)
file for how this is handled if you use a virtual environment approach.

## Environment variables

The scripts sets various environment variables. These are used
internally, and can be used downstream within a program.

Some are automatically inferred from the name of the project folder, while other
should be manually (optional) specified. The variables are then available within
your program using e.g. python's `os` package:

``` python
import os
source_dir = os.environ['source_dir']
```

### Automatically assigned

- `source_dir`: absolute path to the root of the project.
- `project_name`: set to be the name of the project folder.
- `scratch_dir=${SCRATCH}/${project_name}`: path to a folder created on
  `SCRATCH`. This folder is project specific and is created using
  `project_name`. No need to worry about having multiple different project
  overwrite one another.
  - This path should be considered the "root" location of the project to store
    large files - e.g. model checkpoints etc.
  - Since this is on `SCRATCH` read/write operation may be **slow**. Try
    using `path_to_local_node_storage=${SLURM_TMPDIR}` instead.

### Manually (optional) assigned
- `exp_name`: a name which describe the current experiment belonging to the
  overarching project (`project_name`)
  - For instance, the project could be "gan_training". An experiment could then
    be `exp_name=celebA` for training a GAN using the [celebA
    dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Created folders

The scripts will automatically create the following directories. Your experiment
can easily access these using the created [environment
variables](#environment-variables). They are only created if they do not already
exist.

- `${SCRATCH}/${project_name}`: if you have a dataset on scratch, you should
  create this directory yourself and put whatever data you need for your jobs
  here.
- `${scratch_dir}/hpc_outputs`: location of yours jobs' output files.
- `${scratch_dir}/exp_name/checkpoints`: a directory meant to store checkpoints
  and other files created as your experiment runs.

## Project structure

Regardless of whether your project uses Singularity or virtual environments the
scripts assumes a certain structure

``` text
your_project_name
├── hpc_files
│   ├── experiment_configurations.txt
│   ├── job_submitter.sh
│   ├── plai_cleanups
│   │   ├── plai_cleanup.sh
│   │   └── submit_plai_cleanup
│   ├── README.md
│   ├── singularity_hpc_files
│   │   ├── distributed_dispatcher.sh
│   │   ├── README.md
│   │   └── standard_job.sh
│   ├── sweeper.yml
│   └── virtual_env_hpc_files
│       ├── distributed_dispatcher.sh
│       ├── distributed_scripts
│       │   ├── lightning_launcher.sh
│       │   └── script_launcher.sh
│       ├── README.md
│       └── standard_job.sh
├── Pipfile
├── requirements.txt
├── singularity_container.sif
├── Singularity.bld
│ 
.
. other project source files
.
```

## EXAMPLE: `SLURM_TMPDIR` on PLAI's cluster

The `SLURM_TMPDIR` not provided on the PLAI cluster. This is why the scripts
will check if you submit your job on the PLAI cluster and set this for you -
`SLURM_TMPDIR=/scratch-ssd/${USER}`.

The scripts will then create the temporary directory for each job on each node.
Upon completion of the job the directory will be deleted. Note, however, that
should the job end prematurely due to hitting the time limit or the job simply
crashes, the cleanup will not happen.

### Keeping PLAI local storage clean

To keep the local storages clean on the PLAI cluster, consider running the
[cleanup script](plai_cleanups/submit_plai_cleanup). This script submits a
job to each machine on the plai cluster and removes all directories and files
found in `/scratch-ssd` that matches the pattern `${USER}*`.

## Additional resources

- PyTorch [distributed communication package](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) 
  - [Elastic launch](https://pytorch.org/docs/stable/elastic/run.html)
  - [PyTorch distributed tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html?highlight=distributed)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster.html)
- Difference between using `--gres` (e.g. `--gres:gpu:2`) and `--gpus-per-task`:
  (https://stackoverflow.com/questions/67091056/gpu-allocation-in-slurm-gres-vs-gpus-per-task-and-mpirun-vs-srun)
  - Particularly be careful with `--gpu-per-task`
  

[sweeper.yml]: sweeper.yml
[Singularity readme]: singularity_hpc_files/README.md
[virtual environment readme]: virtual_env_hpc_files/README.md
[experiment_configurations.txt]: experiment_configurations.txt
