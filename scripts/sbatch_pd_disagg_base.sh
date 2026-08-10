#!/bin/bash
# One log file for both stdout and stderr.
#SBATCH --output=slurm-logs/%j-%x.log
# One task per allocated node: rank 0 is prefill/router and rank 1 is decode.
#SBATCH --ntasks-per-node=1

set -euo pipefail

CONTAINER=${CONTAINER:?Set CONTAINER to the container image}
MOUNTS=${MOUNTS:?Set MOUNTS to the required container mounts}

if (( $# == 0 )); then
    echo "Usage: $0 <command> [args...]" >&2
    exit 2
fi

if [[ "${SLURM_JOB_NUM_NODES:-}" != 2 ]]; then
    echo "This P/D disaggregation base script requires exactly two nodes." >&2
    exit 2
fi

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
prefill_host=$(getent hosts "${nodes[0]}" | awk 'NR == 1 {print $1}')
decode_host=$(getent hosts "${nodes[1]}" | awk 'NR == 1 {print $1}')
if [[ -z "$prefill_host" || -z "$decode_host" ]]; then
    echo "Could not resolve allocated prefill/decode node addresses." >&2
    exit 1
fi

# Unlike sbatch_base.sh, this deliberately does not start Ray. Each task runs
# its node-local vLLM server directly, allowing Slurm to keep the prefill and
# decode GPUs physically disjoint.
PD_PREFILL_HOST=$prefill_host \
PD_DECODE_HOST=$decode_host \
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --container-image="$CONTAINER" \
    --container-name=container-on-node \
    --container-mounts="$MOUNTS" \
    --container-workdir="$SLURM_SUBMIT_DIR" \
    --no-container-mount-home \
    bash -lc '
        set -euo pipefail
        cd "$SLURM_SUBMIT_DIR"
        exec "$@"
    ' bash "$@"
