#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

dry_run=0
if [[ "${1:-}" == --dry-run ]]; then dry_run=1; shift; fi
if (( $# )); then printf 'Unexpected arguments; prepare a new matched config instead.\n' >&2; exit 2; fi
: "${NEMO_RL_ROOT:?Set the reviewed RL checkout}"
: "${CONFIG_PATH:?Set the prepared config.yaml}"
: "${CONTAINER:?Set the built SquashFS file for the prepared hardware profile}"
: "${CONTAINER_PROVENANCE:?Set the container build provenance JSON}"
: "${SHARED_ROOT:?Set the shared filesystem root containing the prepared assets}"
: "${SLURM_ACCOUNT:?Set a currently granted Slurm account}"
: "${SLURM_PARTITION:?Set the checked partition for the prepared hardware profile}"
: "${OPENSANDBOX_DOMAIN:?Set the required sandbox router}"
: "${OPENSANDBOX_API_KEY:?Inject the per-user sandbox key into the environment}"

gym_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=$(cd -- "$(dirname -- "$CONFIG_PATH")" && pwd)
for path in "$NEMO_RL_ROOT" "$gym_root" "$output_dir" "$CONTAINER" "$CONTAINER_PROVENANCE"; do
    case "$path" in
        "${SHARED_ROOT%/}/"*) ;;
        *) printf 'Path is outside SHARED_ROOT: %s\n' "$path" >&2; exit 1 ;;
    esac
done
gpus_per_node=$(uv run --project "$gym_root" --no-sync python "$gym_root/scripts/check_swe_training_recipe.py" \
    --prepared "$output_dir" --rl-root "$NEMO_RL_ROOT" --gym-root "$gym_root" \
    --container "$CONTAINER" --container-provenance "$CONTAINER_PROVENANCE" --shared-root "$SHARED_ROOT" \
    --print-gpus-per-node)

export GPUS_PER_NODE="$gpus_per_node"
export BASE_LOG_DIR="$output_dir/slurm"
# The image contains the selected source and dependencies. Only experiment assets are mounted.
export MOUNTS="${SHARED_ROOT}:${SHARED_ROOT}"
export USE_CUSTOM_VLLM=0
export SETUP_COMMAND=""
export OMP_NUM_THREADS=16
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800
image_gym=/opt/nemo-rl/3rdparty/Gym-workspace/Gym
check=(uv run --no-sync python "$image_gym/scripts/check_swe_training_recipe.py"
    --prepared "$output_dir" --rl-root /opt/nemo-rl --gym-root "$image_gym" --runtime)
driver=(uv run --no-sync examples/run_grpo_single_controller.py --config "$CONFIG_PATH")
printf -v check_command '%q ' "${check[@]}"
printf -v driver_command '%q ' "${driver[@]}"
export COMMAND="cd /opt/nemo-rl && $check_command && exec $driver_command"
submit=(sbatch --nodes=1 --exclusive --gres="gpu:$GPUS_PER_NODE" --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION"
    --time="${WALLTIME:-4:00:00}" --job-name="${EXP_NAME:-swe-env-server}" --dependency=
    --cpus-per-task=32 --mem=128G)
if [[ -n "${SLURM_QOS:-}" ]]; then submit+=(--qos="$SLURM_QOS"); fi
submit+=("$NEMO_RL_ROOT/ray.sub")
printf 'Driver: %s\n' "$COMMAND"
printf 'Submit: '; printf '%q ' "${submit[@]}"; printf '\n'
if (( ! dry_run )); then
    mkdir -p -- "$BASE_LOG_DIR"
    "${submit[@]}"
fi
