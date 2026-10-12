#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Submit Apex Agents profiling against GLM-5.2 BF16. One vLLM endpoint uses
# one Ray data-parallel rank per allocated node; each node also runs one local
# Gym worker, and the input is sharded across those workers.
#
# Usage:
#   ./scripts/launch_apex_agents_glm52_bf16_unified.sh <tag> [nodes] [global_concurrency] [num_repeats]

set -Eeuo pipefail

readonly APEX_GYM_DIR="/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/artij/ApexGym"
readonly BATCH_SCRIPT="${APEX_GYM_DIR}/scripts/run_apex_agents_glm52_bf16_unified.sbatch"

TAG=${1:?usage: $0 <tag> [nodes] [global_concurrency] [num_repeats]}
NODES=${2:-16}
CONCURRENCY=${3:-120}
NUM_REPEATS=${4:-1}

[[ "${TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "ERROR: tag may contain only letters, numbers, dots, underscores, and dashes" >&2
    exit 64
}
[[ "${NODES}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: nodes must be a positive integer" >&2; exit 64; }
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: global_concurrency must be a positive integer" >&2
    exit 64
}
[[ "${NUM_REPEATS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: num_repeats must be a positive integer" >&2
    exit 64
}
(( NODES == 4 || NODES == 8 || NODES == 16 )) || {
    echo "ERROR: GLM-5.2 BF16 supports 4 nodes (DP=4), 8 nodes (DP=8), or 16 nodes (DP=16); got ${NODES}" >&2
    exit 64
}

export APEX_GYM_DIR
# This is a model-specific launcher: never inherit another launcher's profile.
export PROFILE=${APEX_GYM_DIR}/scripts/profiles/apex-glm52-bf16.env
export GLM52_NUM_NODES=${NODES}
# Default: CUDA graphs without Inductor (GLM52_CUDAGRAPH_MODE=PIECEWISE in the
# serve script). Eager execution avoided the Blackwell TorchInductor autotuning
# failure, but decodes at ~4.8 tok/s per stream and turned most rollouts into
# 12600 s timeouts; PIECEWISE graphs skip Inductor entirely and measured ~20 tok/s
# per stream on the full 452 x 3 benchmark. Set GLM52_ENFORCE_EAGER=true to fall
# back to the eager path.
export GLM52_ENFORCE_EAGER=${GLM52_ENFORCE_EAGER:-false}
export GLM52_CUDAGRAPH_MODE=${GLM52_CUDAGRAPH_MODE:-PIECEWISE}
export DATASET=${DATASET:-${APEX_GYM_DIR}/benchmarks/apex_agents/data/apex_agents_benchmark.jsonl}
export GYM_CONFIG=${GYM_CONFIG:-${APEX_GYM_DIR}/env.yaml}
export CONCURRENCY NUM_REPEATS
export LIMIT=${LIMIT:-}
export SPLIT=${SPLIT:-benchmark}
export MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-32768}
export APEX_AGENT_TIMEOUT=${APEX_AGENT_TIMEOUT:-12600}
export NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS:-3}
export MAX_ROTATIONS=${MAX_ROTATIONS:-8}

[[ "${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: NEMO_GYM_MAX_ROLLOUT_ATTEMPTS must be a positive integer" >&2
    exit 64
}

timestamp=$(date +%Y%m%d_%H%M%S)
# Model-specific override names prevent stale generic Kimi variables from
# redirecting GLM output into an unrelated resumable run.
export RUN_DIR=${GLM52_RUN_DIR_OVERRIDE:-${APEX_GYM_DIR}/results/apex_glm52_bf16_unified/${TAG}_${timestamp}}
export OUTPUT_JSONL=${GLM52_OUTPUT_JSONL_OVERRIDE:-${RUN_DIR}/rollouts.jsonl}

# Avoid inheriting serialized state or a dependency from another evaluation.
unset NEMO_GYM_CONFIG_DICT NEMO_GYM_CONFIG_PATH SBATCH_DEPENDENCY

mkdir -p "${RUN_DIR}/logs" "${APEX_GYM_DIR}/results/slurm-logs"

# Rack locality: --segment=<nodes per replica> makes Slurm place the replica's
# node group inside one NVL72 block. --switches=1 alone is a soft preference that
# Slurm drops after its wait budget, and every rack-straddling replica we ran died
# at its first decode step (sample_tokens RPC timeout, engine dead) while rack-local
# replicas stayed healthy. This launcher runs one Ray-DP replica across all nodes,
# so the segment defaults to NODES; SBATCH_SEGMENT=none disables it.
SBATCH_SEGMENT=${SBATCH_SEGMENT:-${NODES}}
[[ "${SBATCH_SEGMENT}" == "none" || "${SBATCH_SEGMENT}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: SBATCH_SEGMENT must be a positive integer or 'none'" >&2
    exit 64
}
placement_args=(--switches=1)
if [[ "${SBATCH_SEGMENT}" != "none" ]]; then
    placement_args+=(--segment="${SBATCH_SEGMENT}")
fi

bash "${BATCH_SCRIPT}" --validate
if [[ "${VALIDATE_ONLY:-false}" == "true" ]]; then
    echo "Validation-only mode passed; no Slurm job was submitted."
    echo "Topology: one GLM-5.2 BF16 endpoint with Ray DP=${NODES} x TP=4 across ${NODES} nodes"
    echo "Collectors: up to $((NODES < CONCURRENCY ? NODES : CONCURRENCY)) node-local Gym workers"
    echo "Global concurrency: ${CONCURRENCY}; limit=${LIMIT:-all}"
    echo "Local recovery policy: per-task timeout=${APEX_AGENT_TIMEOUT}s; max attempts=${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS}"
    exit 0
fi

job_id=$(
    sbatch --parsable \
        --nodes="${NODES}" \
        --ntasks="${NODES}" \
        --account="${SBATCH_ACCOUNT:-nemotron_n4_post}" \
        --partition="${SBATCH_PARTITION:-batch}" \
        --qos="${SBATCH_QOS:-normal}" \
        --time="${WALLTIME:-04:00:00}" \
        "${placement_args[@]}" \
        --output="${RUN_DIR}/logs/%j_rollout.out" \
        --error="${RUN_DIR}/logs/%j_rollout.err" \
        --export=ALL \
        "${BATCH_SCRIPT}"
)
job_id=${job_id%%;*}
printf '%s\n' "${job_id}" > "${RUN_DIR}/JOBID"

echo "Submitted Apex/GLM-5.2 BF16 unified job: ${job_id}"
echo "Topology: one GLM-5.2 BF16 endpoint with Ray DP=${NODES} x TP=4 across ${NODES} nodes"
echo "Collectors: up to $((NODES < CONCURRENCY ? NODES : CONCURRENCY)) node-local Gym workers"
echo "Global concurrency: ${CONCURRENCY}; limit=${LIMIT:-all}"
echo "Local recovery policy: per-task timeout=${APEX_AGENT_TIMEOUT}s; max attempts=${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS}"
echo "Run directory: ${RUN_DIR}"
echo "Output: ${OUTPUT_JSONL}"
echo "Monitor: squeue -j ${job_id}"
echo "Log: tail -F ${RUN_DIR}/logs/${job_id}_rollout.out"
