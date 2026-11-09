#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

dry_run=0
source_mount=0
reuse_swe_rebench_venv=0
gym_root_override=
excluded_nodes=
wandb_project=
wandb_entity=
wandb_name=
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --source-mount) source_mount=1 ;;
        --reuse-swe-rebench-venv) reuse_swe_rebench_venv=1 ;;
        --exclude) excluded_nodes=${2:?Pass the excluded node list}; shift ;;
        --walltime) WALLTIME=${2:?Pass the Slurm time limit}; shift ;;
        --wandb-project) wandb_project=${2:?Pass the W&B project}; shift ;;
        --wandb-name) wandb_name=${2:?Pass the W&B display name}; shift ;;
        --wandb-entity) wandb_entity=${2:?Pass the W&B entity}; shift ;;
        --gym-root) gym_root_override=${2:?Pass the source Gym root}; shift ;;
        *) printf 'Unexpected argument: %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done
: "${NEMO_RL_ROOT:?Set the reviewed RL checkout}"
: "${CONFIG_PATH:?Set the prepared config.yaml}"
: "${CONTAINER:?Set the built SquashFS file for the prepared hardware profile}"
: "${SHARED_ROOT:?Set the shared filesystem root containing the prepared assets}"
: "${SLURM_ACCOUNT:?Set a currently granted Slurm account}"
: "${SLURM_PARTITION:?Set the checked partition for the prepared hardware profile}"
: "${OPENSANDBOX_DOMAIN:?Set the required sandbox router}"
: "${OPENSANDBOX_API_KEY:?Inject the per-user sandbox key into the environment}"

tools_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gym_root=$(cd -- "${gym_root_override:-$tools_dir/..}" && pwd)
output_dir=$(cd -- "$(dirname -- "$CONFIG_PATH")" && pwd)
# Read allocation settings only when the caller has not already supplied them.
# Diagnostics are separate tools, never a gate in the experimental launch path.
if [[ -z "${GPUS_PER_NODE:-}" ]]; then
    GPUS_PER_NODE=$(uv run --project "$gym_root" --no-sync python -c \
        'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["cluster"]["gpus_per_node"])' "$CONFIG_PATH")
fi
export GPUS_PER_NODE
export BASE_LOG_DIR="$output_dir/slurm"
# Baked source remains the default. Mount at the same absolute editable-install path.
export MOUNTS="${SHARED_ROOT}:${SHARED_ROOT}"
if (( source_mount )); then
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
if (( reuse_swe_rebench_venv )); then
    # These two resource servers use the same dependencies. Reuse the compiled
    # environment in the disposable container; the mounted source stays read-only.
    export SETUP_COMMAND='if [[ ! -e /opt/gym_venvs/resources_servers/swe_next/.venv ]]; then
mkdir -p /opt/gym_venvs/resources_servers/swe_next
ln -s /opt/gym_venvs/resources_servers/swe_rebench/.venv /opt/gym_venvs/resources_servers/swe_next/.venv
fi'
fi
export OMP_NUM_THREADS=16
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800
driver=(uv run --no-sync examples/run_grpo_single_controller.py --config "$CONFIG_PATH")
if [[ -n "$wandb_project" ]]; then
    driver+=("++logger.wandb_enabled=true" "++logger.wandb.project=$wandb_project"
        "++logger.wandb.mode=online"
        "++logger.wandb.id=null" "++logger.wandb.resume=null" "++logger.wandb.log_nemo_gym_full_result_tables=false")
    if [[ -n "$wandb_entity" ]]; then driver+=("++logger.wandb.entity=$wandb_entity"); fi
fi
if [[ -n "$wandb_name" ]]; then driver+=("++logger.wandb.name=$wandb_name"); fi
printf -v driver_command '%q ' "${driver[@]}"
export COMMAND="cd /opt/nemo-rl && exec $driver_command"
# Exclusivity alone does not reserve node memory; checkpoint staging needs more than a fixed 128 GiB cap.
submit=(sbatch --nodes=1 --exclusive --gres="gpu:$GPUS_PER_NODE" --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION"
    --time="${WALLTIME:-4:00:00}" --job-name="${EXP_NAME:-swe-env-server}" --dependency=
    --cpus-per-task=32 --mem=0 --chdir="$output_dir"
    --output="$BASE_LOG_DIR/slurm-%j.out" --error="$BASE_LOG_DIR/slurm-%j.err")
if [[ -n "$excluded_nodes" ]]; then submit+=(--exclude="$excluded_nodes"); fi
if [[ -n "${SLURM_QOS:-}" ]]; then submit+=(--qos="$SLURM_QOS"); fi
submit+=("$NEMO_RL_ROOT/ray.sub")
printf 'Driver: %s\n' "$COMMAND"
printf 'Submit: '; printf '%q ' "${submit[@]}"; printf '\n'
if (( ! dry_run )); then
    mkdir -p -- "$BASE_LOG_DIR"
    "${submit[@]}"
fi
