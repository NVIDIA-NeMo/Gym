#!/bin/bash
# 
# Example run:
# NUM_NODES=4 \
# SBATCH_ACCOUNT=my-slurm-account \
# SBATCH_PARTITION=batch \
# SBATCH_GRES=gpu:4 \
# CONTAINER=/path/to/vllm/container \
# MOUNTS=/shared/fs:/shared/fs \
# bash scripts/sbatch_interactive.sh
# 

set -euo pipefail

# Input arguments and validation
NUM_NODES=$NUM_NODES
CONTAINER=$CONTAINER
MOUNTS=$MOUNTS

sbatch \
    --nodes=$NUM_NODES \
    --time=04:00:00 \
    --job-name=$USER-dev \
    --exclusive \
    scripts/sbatch_base.sh
