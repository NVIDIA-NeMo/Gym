#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fixed Qwen3.8-Flash-Next judge for visual_agent runs of other policies (e.g.
# launch_nemotron_3.5_super.sh): a vLLM-only job, one vision-enabled TP4 replica per node behind
# vllm-router, that writes its router URL to ENDPOINT_FILE once it answers and removes it on
# exit. Policy jobs read that file through vllm_model `endpoint_file` (external_judge_override.yaml).
#
# Inputs: JUDGE_MODEL is the Qwen3.8-Flash-Next checkpoint, CONTAINER a vLLM + Gym image
# (.sqsh), EXTRA_MOUNTS the container mounts for the filesystems holding them (e.g. /data:/data).
# Slurm account, partition and QoS come from the usual SBATCH_* environment variables.
#
#   bash benchmarks/visual_agent/launch_judge_qwen3.8.sh               # 2 nodes, 4 h, + 1 singleton retry
#   NUM_NODES=1 RETRIES=0 bash benchmarks/visual_agent/launch_judge_qwen3.8.sh
#   DRY_RUN=1 ...                                                       # print, do not submit
#
# The job serves until its wall clock ends; scancel it when the policy runs are done.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_root"

JUDGE_MODEL=${JUDGE_MODEL:?Set JUDGE_MODEL to the Qwen3.8-Flash-Next checkpoint directory}
[[ -f "$JUDGE_MODEL/config.json" ]] || { echo "No config.json under $JUDGE_MODEL" >&2; exit 1; }
VLLM_CONFIG=${VLLM_CONFIG:-benchmarks/nemotron_3.5_super/vllm_configs/qwen_3.8_flash_next_vision.sh}
CONTAINER=${CONTAINER:?Set CONTAINER to a vLLM + Gym image (.sqsh) that serves Qwen3.8-Flash-Next}
[[ -f "$CONTAINER" ]] || { echo "Missing container: $CONTAINER" >&2; exit 1; }
CONTAINER=$(realpath "$CONTAINER")
# Must match `model` in external_judge_override.yaml.
MODEL_NAME=qwen3.8-flash-next
ENDPOINT_FILE=${ENDPOINT_FILE:-$repo_root/results/visual-agent-judge/qwen3.8-flash-next.endpoint}

# Account, partition and QoS come from the caller's SBATCH_* environment.
export SBATCH_GRES=${SBATCH_GRES:-gpu:4}
export SBATCH_TIME=${SBATCH_TIME:-04:00:00}
unset SBATCH_RESERVATION SBATCH_NODELIST
NUM_NODES=${NUM_NODES:-2}
SLURM_COMMENT=${SLURM_COMMENT:-}

submit() {
    local dependency=${1:-}
    echo "Launching Qwen3.8-Flash-Next judge ($NUM_NODES aggregated replicas, publishes $ENDPOINT_FILE${dependency:+, dependency=$dependency})"
    MODEL="$JUDGE_MODEL" \
    MODEL_NAME="$MODEL_NAME" \
    VLLM_CONFIG="$VLLM_CONFIG" \
    VLLM_MODE=aggregated \
    NUM_NODES="$NUM_NODES" \
    EXPERIMENT_NAME=visual-agent-judge-$MODEL_NAME \
    SLURM_COMMENT="$SLURM_COMMENT" \
    SBATCH_DEPENDENCY="$dependency" \
    CONTAINER="$CONTAINER" \
    MOUNTS="${EXTRA_MOUNTS:+$EXTRA_MOUNTS,}$repo_root:/opt/Gym" \
    ENDPOINT_FILE="$ENDPOINT_FILE" \
    bash benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh
}

mkdir -p slurm-logs
RETRIES=${RETRIES:-1}
submit singleton
for (( r = 0; r < RETRIES; r++ )); do
    submit singleton
done
