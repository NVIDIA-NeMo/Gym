#!/bin/bash
# One log file for both stdout and stderr
#SBATCH --output=%j-%x.log
### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1

set -euo pipefail

# Get the Ray head node IP
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
RAY_HEAD_NODE_IP=${nodes_array[0]}:6379
echo "Ray head node IP address: $RAY_HEAD_NODE_IP"

echo "Running from $SLURM_SUBMIT_DIR"

# Start Ray cluster using symmetric_run.py on all nodes.
# Symmetric run will automatically start Ray on all nodes and run the script ONLY the head node.
# The '--' separator is used to separate Ray arguments and the entrypoint command.
# The --min-nodes argument ensures all nodes join before running the script.
ENTRYPOINT_WITH_RAY_WRAPPER=$(cat <<EOF
source .venv/bin/activate
uv sync

ray symmetric-run \
    --address $RAY_HEAD_NODE_IP \
    --min-nodes $SLURM_JOB_NUM_NODES \
    --num-cpus=${SLURM_CPUS_PER_TASK:-$SLURM_CPUS_ON_NODE} \
    --num-gpus=${SLURM_GPUS_PER_TASK:-$SLURM_GPUS_ON_NODE} \
    -- \
    bash $@
EOF
)

# All nodes (including head and workers) will execute this block.
# The command after '--' above will only run on the head node
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" bash $ENTRYPOINT_WITH_RAY_WRAPPER
