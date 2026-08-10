#!/bin/bash
# One log file for both stdout and stderr.
#SBATCH --output=slurm-logs/%j-%x.log
# One task per allocated node: ranks 0-1 are prefill DP ranks, 2-3 decode DP.
#SBATCH --ntasks-per-node=1

set -euo pipefail

CONTAINER=${CONTAINER:?Set CONTAINER to the container image}
MOUNTS=${MOUNTS:?Set MOUNTS to the required container mounts}

if (( $# == 0 )); then
    echo "Usage: $0 <command> [args...]" >&2
    exit 2
fi

if [[ "${SLURM_JOB_NUM_NODES:-}" != 4 ]]; then
    echo "This DP=2 prefill / DP=2 decode base script requires exactly four nodes." >&2
    exit 2
fi

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
prefill_host_0=$(getent hosts "${nodes[0]}" | awk 'NR == 1 {print $1}')
prefill_host_1=$(getent hosts "${nodes[1]}" | awk 'NR == 1 {print $1}')
decode_host_0=$(getent hosts "${nodes[2]}" | awk 'NR == 1 {print $1}')
decode_host_1=$(getent hosts "${nodes[3]}" | awk 'NR == 1 {print $1}')
if [[ -z "$prefill_host_0" || -z "$prefill_host_1" ||
      -z "$decode_host_0" || -z "$decode_host_1" ]]; then
    echo "Could not resolve allocated prefill/decode node addresses." >&2
    exit 1
fi

# Unlike sbatch_base.sh, this deliberately does not start Ray. Each task runs
# its node-local vLLM server directly, allowing Slurm to keep the prefill and
# decode GPUs physically disjoint.
PD_PREFILL_HOSTS="$prefill_host_0 $prefill_host_1" \
PD_DECODE_HOSTS="$decode_host_0 $decode_host_1" \
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 \
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
