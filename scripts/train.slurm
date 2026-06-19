#!/bin/bash

#SBATCH --job-name=pretrain_fr
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96           # number of cores per tasks
#SBATCH --gres=gpu:4                 # reserve number of GPUs per node
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4        # QoS
#SBATCH --time=100:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --constraint=h100
#SBATCH --account=ilr@h100           # account to use, with xyz the 3 letter code of your project

set -x -e

echo "START TIME: $(date)"

# set up environment
module purge
module load arch/h100
# module load git-lfs
# module load unrar
module load anaconda-py3/2024.06
# module load cuda/12.8.0
module load cuda/12.4.1
# env
conda activate bert24

# debug
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# wandb
export WANDB_MODE=offline

# Print some useful information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"

# Validate script arguments
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_yaml> [additional composer args]"
  exit 1
fi

CONFIG="$1"
shift  # Remove the config file from $@

# Derive distributed parameters. composer's launcher does NOT auto-pick up SLURM
# env vars, so without these each node forms its own NCCL world of 4 GPUs and
# the nodes never talk → "8/16-GPU" runs actually scale only inside one node.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29500}

# NOTE: $SLURM_NODEID is escaped (\$) so it's expanded by the per-task bash, NOT the
# job-script bash (where it's always 0). Without the escape, every node launches with
# --node_rank 0 → NCCL hangs forever at bootstrap.
srun bash -c "composer \
    --world_size $((SLURM_NNODES * 4)) \
    --node_rank \$SLURM_NODEID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    -n 4 \
    main.py $CONFIG $*"

echo "END TIME: $(date)"
