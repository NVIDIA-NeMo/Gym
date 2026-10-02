#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Visual agent benchmark (MiMo-V2.6 §4.2.3) with a Nemotron 3.5 Super VL/omni checkpoint as the
# policy, graded by the fixed Qwen3.8-Flash-Next judge of launch_judge_qwen3.8.sh so its scores
# compare with the Qwen3.8-Flash-Next runs. With the default 2 + 2 nodes both jobs fit the
# same four nodes.
#
# Inputs: MODEL (or the first argument) is the checkpoint directory, CONTAINER a vLLM + Gym image
# (.sqsh), EXTRA_MOUNTS the container mounts for the filesystems holding them (e.g. /data:/data).
# Slurm account, partition and QoS come from the usual SBATCH_* environment variables.
#
#   bash benchmarks/visual_agent/launch_judge_qwen3.8.sh                    # first: the judge
#   bash benchmarks/visual_agent/launch_nemotron_3.5_super.sh /path/to/hf_ckpt   # full run: 200 tasks x 4
#   SMOKE=1 bash benchmarks/visual_agent/launch_nemotron_3.5_super.sh       # 12 rollouts = 3 tasks x 4, 2 h
#   bash benchmarks/visual_agent/launch_nemotron_3.5_super.sh /path/to/ckpt ++limit=8
#   DRY_RUN=1 ...                                                            # print, do not submit
#
# If the judge has not published its endpoint yet, the policy job waits for the judge job to
# start (after:<id>); both load weights in parallel and the first grading call comes after the
# first finished rollout.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_root"

MODEL_CKPT_PATH=${MODEL:-}
if (( $# > 0 )) && [[ "$1" != --* && "$1" != *=* ]]; then
    MODEL_CKPT_PATH=$1
    shift
fi
GYM_ARGS=("$@")
[[ -n "$MODEL_CKPT_PATH" ]] || { echo "Pass the checkpoint directory as the first argument or set MODEL" >&2; exit 1; }
[[ -f "$MODEL_CKPT_PATH/config.json" ]] || { echo "No config.json under $MODEL_CKPT_PATH" >&2; exit 1; }

VLLM_CONFIG=${VLLM_CONFIG:-benchmarks/nemotron_3.5_super/vllm_configs/nemotron_3.5_super_vision.sh}
# vLLM >= 0.29 with Gym and vllm-router; it serves NemotronH_Omni_Reasoning_V3 as a multimodal model.
CONTAINER=${CONTAINER:?Set CONTAINER to a vLLM + Gym image (.sqsh) that serves the checkpoint}
[[ -f "$CONTAINER" ]] || { echo "Missing container: $CONTAINER" >&2; exit 1; }
CONTAINER=$(realpath "$CONTAINER")
JUDGE_ENDPOINT_FILE=${JUDGE_ENDPOINT_FILE:-$repo_root/results/visual-agent-judge/qwen3.8-flash-next.endpoint}

# Account, partition and QoS come from the caller's SBATCH_* environment.
export SBATCH_GRES=${SBATCH_GRES:-gpu:4}
export SBATCH_TIME=${SBATCH_TIME:-04:00:00}
unset SBATCH_RESERVATION SBATCH_NODELIST
NUM_NODES=${NUM_NODES:-2}
# Rollouts in flight; each holds a policy sandbox and, while grading, a grader sandbox and a
# judge session on the judge job.
PARALLEL=${PARALLEL:-128}
SLURM_COMMENT=${SLURM_COMMENT:-}
# results/<prefix>/ caches the prepared inputs, so a new task-set version needs a new prefix
# (the 2026-09-24 v1 run is results/visual-agent-qwen3.8-flash-next/).
EXPERIMENT_PREFIX=${EXPERIMENT_PREFIX:-visual-agent-v2b-nemotron-3.5-super}
MODEL_SLUG=nemotron-3.5-super

judge_dependency=""
if [[ ! -s "$JUDGE_ENDPOINT_FILE" && ${DRY_RUN:-0} != 1 ]]; then
    judge_job=$(squeue -h -t PENDING,RUNNING -u "$USER" -n "gym-visual-agent-judge-qwen3.8-flash-next-$USER" -o %i | sort -n | head -1)
    if [[ -z "$judge_job" ]]; then
        echo "No judge endpoint at $JUDGE_ENDPOINT_FILE and no judge job queued; run launch_judge_qwen3.8.sh first" >&2
        exit 1
    fi
    judge_dependency=after:$judge_job
fi

run_eval() {
    local dependency=${1:-}
    [[ -n "$judge_dependency" ]] && dependency=${dependency:+$dependency,}$judge_dependency
    local ckpt_path=${MODEL_CKPT_PATH%/}
    local sanitized_model_name
    sanitized_model_name=$(printf '%s\n' "$ckpt_path" | rev | cut -d/ -f1-3 | rev | tr '/' '_')
    local experiment_name=$EXPERIMENT_PREFIX/opencode_visual_agent/$sanitized_model_name
    echo "Launching visual_agent for $MODEL_CKPT_PATH ($NUM_NODES aggregated replicas, $PARALLEL parallel${dependency:+, dependency=$dependency})"
    MODEL="$MODEL_CKPT_PATH" \
    MODEL_NAME="$MODEL_SLUG" \
    VLLM_CONFIG="$VLLM_CONFIG" \
    VLLM_MODE=aggregated \
    NUM_NODES="$NUM_NODES" \
    EXPERIMENT_NAME="$experiment_name" \
    ROLLOUTS_FPATH="results/$experiment_name.jsonl" \
    SLURM_COMMENT="$SLURM_COMMENT" \
    SBATCH_DEPENDENCY="$dependency" \
    CONTAINER="$CONTAINER" \
    MOUNTS="${EXTRA_MOUNTS:+$EXTRA_MOUNTS,}$repo_root:/opt/Gym" \
    EVAL_SETUP_SCRIPT=benchmarks/visual_agent/eval_setup.sh \
    bash benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh \
        --config responses_api_models/vllm_model/configs/vllm_model.yaml \
        --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
        --config benchmarks/visual_agent/opencode.yaml \
        --config benchmarks/visual_agent/nemotron_3.5_super_override.yaml \
        --config benchmarks/visual_agent/external_judge_override.yaml \
        ++judge_model.responses_api_models.vllm_model.endpoint_file="$JUDGE_ENDPOINT_FILE" \
        ++num_samples_in_parallel="$PARALLEL" \
        ++resume_from_cache=true \
        "${GYM_ARGS[@]}"
}

mkdir -p slurm-logs
if [[ ${SMOKE:-0} == 1 ]]; then
    export SBATCH_TIME=${SMOKE_TIME:-02:00:00}
    # ++limit counts rows after the 4x repetition, so keep it a multiple of 4 (see the Qwen launcher).
    SMOKE_LIMIT=${SMOKE_LIMIT:-12}
    (( SMOKE_LIMIT % 4 == 0 )) || { echo "SMOKE_LIMIT must be a multiple of 4 (num_repeats)" >&2; exit 1; }
    EXPERIMENT_PREFIX=$EXPERIMENT_PREFIX-smoke$SMOKE_LIMIT
    GYM_ARGS+=(++limit="$SMOKE_LIMIT")
    run_eval
    exit 0
fi

# The batch partition caps jobs at 4 h; chained singleton submissions resume the same rollouts.
RETRIES=${RETRIES:-1}
run_eval singleton
for (( r = 0; r < RETRIES; r++ )); do
    run_eval singleton
done
