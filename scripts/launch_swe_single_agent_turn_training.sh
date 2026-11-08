#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

dry_run=0
source_mount=0
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --source-mount) source_mount=1 ;;
        *) printf 'Unexpected argument: %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done
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
source_args=()
if (( source_mount )); then source_args+=(--source-mount); fi
gpus_per_node=$(uv run --project "$gym_root" --no-sync python "$gym_root/scripts/check_swe_training_recipe.py" \
    --prepared "$output_dir" --rl-root "$NEMO_RL_ROOT" --gym-root "$gym_root" \
    --container "$CONTAINER" --container-provenance "$CONTAINER_PROVENANCE" --shared-root "$SHARED_ROOT" \
    --print-gpus-per-node ${source_args[@]+"${source_args[@]}"})

export GPUS_PER_NODE="$gpus_per_node"
export BASE_LOG_DIR="$output_dir/slurm"
# Baked source remains the default. Mount at the same absolute editable-install path.
export MOUNTS="${SHARED_ROOT}:${SHARED_ROOT}"
if (( source_mount )); then
    case "$NEMO_RL_ROOT$SHARED_ROOT" in
        *[,:]*|*[[:space:]]*) printf 'Source mount paths cannot contain whitespace, commas or colons.\n' >&2; exit 2 ;;
    esac
    if [[ "$gym_root" != "$NEMO_RL_ROOT/3rdparty/Gym-workspace/Gym" ]]; then
        printf 'Mount the complete reviewed RL snapshot with its nested Gym checkout.\n' >&2; exit 2
    fi
    if [[ -n "${NRL_FORCE_REBUILD_VENVS:-}${NRL_IGNORE_VERSION_MISMATCH:-}${NEMO_RL_PY_EXECUTABLES_SYSTEM:-}" ]]; then
        printf 'Source mounting requires the image environments without rebuild or interpreter overrides.\n' >&2; exit 2
    fi
    export MOUNTS="$MOUNTS,$NEMO_RL_ROOT:/opt/nemo-rl:ro"
    export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
    export NEMO_RL_VENV_DIR=/opt/ray_venvs
    export NEMO_GYM_VENV_DIR=/opt/gym_venvs
    export UV_NO_SYNC=1
    export PYTHONDONTWRITEBYTECODE=1
    export HF_HOME="$output_dir/runtime-cache/huggingface"
    export HF_DATASETS_CACHE="$output_dir/runtime-cache/datasets"
fi
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
