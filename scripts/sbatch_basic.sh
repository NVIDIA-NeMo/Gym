#!/bin/bash
# One log file for both stdout and stderr
#SBATCH --output=slurm-%j-%x.log
### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1

set -euo pipefail

# Get the Ray head node IP
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node_hostname=${nodes_array[0]}
head_node_ip=$(getent hosts $head_node_hostname | awk '{print $1}')
RAY_HEAD_NODE_IP=$head_node_ip:6379
echo "Ray head node IP address: $RAY_HEAD_NODE_IP"

# Start Ray cluster using symmetric_run.py on all nodes.
# Symmetric run will automatically start Ray on all nodes and run the script ONLY the head node.
# The '--' separator is used to separate Ray arguments and the entrypoint command.
# The --min-nodes argument ensures all nodes join before running the script.

# All nodes (including head and workers) will execute this block.
# The command after '--' will only run on the head node
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES \
    echo "Running from $SLURM_SUBMIT_DIR" && source .venv/bin/activate \
    && ray symmetric-run \
    --address $RAY_HEAD_NODE_IP \
    --min-nodes $SLURM_JOB_NUM_NODES \
    --num-cpus=${SLURM_CPUS_PER_TASK:-$SLURM_CPUS_ON_NODE} \
    --num-gpus=${SLURM_GPUS_PER_TASK:-$SLURM_GPUS_ON_NODE} \
    -- \
    bash $@
