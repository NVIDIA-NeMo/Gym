#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Reuse RL's ray.sub and SingleController entrypoint; no pipeline repository needed.
set -euo pipefail

dry_run=0
if [[ "${1:-}" == --dry-run ]]; then dry_run=1; shift; fi
: "${NEMO_RL_ROOT:?Set the RL checkout on the shared filesystem}"
: "${CONFIG_PATH:?Set the prepared config.yaml path}"
: "${MODEL_PATH:?Set the reference checkpoint HF directory}"
: "${CONTAINER:?Set a container compatible with this RL checkout}"
: "${SLURM_ACCOUNT:?Set an authorized account}"
: "${SLURM_PARTITION:?Set the target partition}"
: "${SHARED_ROOT:?Set the shared filesystem root containing assets, source and prepared output}"

gym_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=$(cd -- "$(dirname -- "$CONFIG_PATH")" && pwd)
for path in "$NEMO_RL_ROOT" "$gym_root" "$output_dir" "$MODEL_PATH"; do
    case "$path/" in
        "${SHARED_ROOT%/}/"*) ;;
        *) printf 'Path is outside the shared container mount: %s\n' "$path" >&2; exit 1 ;;
    esac
done
export BASE_LOG_DIR="${BASE_LOG_DIR:-$output_dir/slurm}"
export GPUS_PER_NODE=4
export USE_CUSTOM_VLLM="${USE_CUSTOM_VLLM:-0}"
export SETUP_COMMAND="${SETUP_COMMAND:-}"
export MOUNTS="${SHARED_ROOT}:${SHARED_ROOT},${NEMO_RL_ROOT}/nemo_rl:/opt/nemo-rl/nemo_rl,${NEMO_RL_ROOT}/examples:/opt/nemo-rl/examples,${gym_root}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
export OPENSANDBOX_DOMAIN=k8s-cellrout-cellrout-101e0e7b17-b0ecd469dafdeefb.elb.us-east-2.amazonaws.com
export OPENSANDBOX_PROTOCOL=https
export NEMO_GYM_VENV_DIR="${NEMO_GYM_VENV_DIR:-$output_dir/gym_venvs}"
# Preserve the reference recipe's worker environment, without writing credentials.
export OMP_NUM_THREADS=16
export NRL_WG_USE_RAY_REF=1
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800
export VLLM_USE_FLASHINFER_MOE_FP8=1
export VLLM_FLASHINFER_MOE_BACKEND=latency

driver=(uv run --no-sync /opt/nemo-rl/examples/run_grpo_single_controller.py --config "$CONFIG_PATH" "$@")
printf -v driver_command '%q ' "${driver[@]}"
export COMMAND="cd /opt/nemo-rl && exec $driver_command"
submit=(sbatch --nodes=16 --exclusive --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION"
    --time="${WALLTIME:-4:00:00}" --job-name="${EXP_NAME:-swe-single-agent-turn}")
if [[ -n "${SLURM_QOS:-}" ]]; then submit+=(--qos="$SLURM_QOS"); fi
submit+=("$NEMO_RL_ROOT/ray.sub")
if (( dry_run )); then
    printf 'Driver: %s\n' "$COMMAND"
    printf 'Submit: '; printf '%q ' "${submit[@]}"; printf '\n'
else
    mkdir -p -- "$BASE_LOG_DIR"
    "${submit[@]}"
fi
