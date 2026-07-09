#!/bin/bash

set -euo pipefail

# Get the Ray head node IP
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
RAY_HEAD_NODE_IP=${nodes_array[0]}:6379
echo "Ray head node IP address: $RAY_HEAD_NODE_IP"

# Start Ray cluster using symmetric_run.py on all nodes.
# Symmetric run will automatically start Ray on all nodes and run the script ONLY the head node.
# Use the '--' separator to separate Ray arguments and the entrypoint command.
# The --min-nodes argument ensures all nodes join before running the script.
ENTRYPOINT_WITH_RAY_WRAPPER=$(cat <<EOF
ray symmetric-run \
    --address $RAY_HEAD_NODE_IP \
    --min-nodes $SLURM_JOB_NUM_NODES \
    --num-cpus=$SLURM_CPUS_PER_TASK \
    --num-gpus=$SLURM_GPUS_PER_TASK \
    -- \
    bash $@
EOF
)

# All nodes (including head and workers) will execute this block.
# The command after '--' will only run on the head node
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" bash $ENTRYPOINT_WITH_RAY_WRAPPER
