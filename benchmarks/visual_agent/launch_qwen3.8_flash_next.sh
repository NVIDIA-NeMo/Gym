#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Visual agent benchmark (MiMo-V2.6 §4.2.3) with Qwen3.8-Flash-Next as the policy AND the agentic
# judge, on 4 nodes.
#
# Serving: vision-enabled preset (vllm_configs/qwen_3.8_flash_next_vision.sh, no --language-model-only)
# as 4 independent TP4 replicas behind vllm-router (VLLM_MODE=aggregated): no P/D transfer of
# multimodal prompts. Gym runs on the second node (see sbatch_external_vllm.sh).
#
# Inputs: MODEL (or the first argument) is the checkpoint directory, CONTAINER a vLLM + Gym image
# (.sqsh), EXTRA_MOUNTS the container mounts for the filesystems holding them (e.g. /data:/data).
# Slurm account, partition and QoS come from the usual SBATCH_* environment variables.
#
#   bash benchmarks/visual_agent/launch_qwen3.8_flash_next.sh                 # full run: 200 tasks x 4
#   SMOKE=1 bash benchmarks/visual_agent/launch_qwen3.8_flash_next.sh         # 12 rollouts = 3 tasks x 4, 2 h
#   SMOKE=1 SMOKE_LIMIT=32 ...                                                 # 8 tasks x 4
#   RETRIES=2 ...                                                              # chained singleton resumes
#   DRY_RUN=1 ...                                                              # print, do not submit
#   bash benchmarks/visual_agent/launch_qwen3.8_flash_next.sh /path/to/ckpt ++limit=10   # extra Gym overrides
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

# Benchmark config; benchmarks/visual_agent/calibration.yaml grades prepared artifacts instead (judge_calibration.py).
BENCHMARK_CONFIG=${BENCHMARK_CONFIG:-benchmarks/visual_agent/opencode.yaml}
VLLM_CONFIG=${VLLM_CONFIG:-benchmarks/nemotron_3.5_super/vllm_configs/qwen_3.8_flash_next_vision.sh}
# A Gym build of the vllm/vllm-openai:qwen38-flash-next image (PyPI vLLM lacks Qwen4Exp).
CONTAINER=${CONTAINER:?Set CONTAINER to a vLLM + Gym image (.sqsh) that serves Qwen3.8-Flash-Next}
[[ -f "$CONTAINER" ]] || { echo "Missing container: $CONTAINER" >&2; exit 1; }
CONTAINER=$(realpath "$CONTAINER")

# Account, partition and QoS come from the caller's SBATCH_* environment.
export SBATCH_GRES=${SBATCH_GRES:-gpu:4}
export SBATCH_TIME=${SBATCH_TIME:-04:00:00}
unset SBATCH_RESERVATION SBATCH_NODELIST
NUM_NODES=${NUM_NODES:-4}
# Rollouts in flight. Each holds one policy sandbox, and while grading also a grader sandbox and a
# judge session. Screenshot-heavy sessions grow to ~100K tokens, so a TP4 replica (~4.3M KV
# tokens) keeps ~40 live sessions cached. At 400 in flight (2026-09-24) KV hit 100%, the
# prefix-cache hit rate fell from 68% to 22% and generation dropped to ~1.2K tok/s per replica.
PARALLEL=${PARALLEL:-128}
SLURM_COMMENT=${SLURM_COMMENT:-}
# results/<prefix>/ caches the prepared inputs, so a new task-set version needs a new prefix
# (the 2026-09-24 v1 run is results/visual-agent-qwen3.8-flash-next/).
EXPERIMENT_PREFIX=${EXPERIMENT_PREFIX:-visual-agent-v2b-qwen3.8-flash-next}
MODEL_SLUG=qwen3.8-flash-next

run_eval() {
    local dependency=${1:-}
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
        --config "$BENCHMARK_CONFIG" \
        --config benchmarks/visual_agent/qwen3.8_flash_next_override.yaml \
        ++num_samples_in_parallel="$PARALLEL" \
        ++resume_from_cache=true \
        "${GYM_ARGS[@]}"
}

mkdir -p slurm-logs
if [[ ${SMOKE:-0} == 1 ]]; then
    export SBATCH_TIME=${SMOKE_TIME:-02:00:00}
    # ++limit counts rows after the 4x repetition (task rows are consecutive), so keep it a
    # multiple of 4: a partial group never fills its groupwise cohort and waits out the timeout.
    SMOKE_LIMIT=${SMOKE_LIMIT:-12}
    (( SMOKE_LIMIT % 4 == 0 )) || { echo "SMOKE_LIMIT must be a multiple of 4 (num_repeats)" >&2; exit 1; }
    # A separate prefix per limit: results/<prefix>/ caches materialized inputs and rollouts.
    EXPERIMENT_PREFIX=$EXPERIMENT_PREFIX-smoke$SMOKE_LIMIT
    GYM_ARGS+=(++limit="$SMOKE_LIMIT")
    run_eval
    exit 0
fi

# The batch partition caps jobs at 4 h. Chained singleton submissions resume the same rollouts
# file (++resume_from_cache) if the first window does not finish every rollout.
RETRIES=${RETRIES:-1}
run_eval singleton
for (( r = 0; r < RETRIES; r++ )); do
    run_eval singleton
done
