#!/bin/bash
# 
# Example run:
# NUM_NODES=4 \
# SBATCH_ACCOUNT=my-slurm-account \
# SBATCH_PARTITION=batch \
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
    --gres=gpu:4 \
    --time=04:00:00 \
    --job-name=$USER-dev \
    --exclusive \
    --segment=$NUM_NODES \
    scripts/sbatch_base.sh
