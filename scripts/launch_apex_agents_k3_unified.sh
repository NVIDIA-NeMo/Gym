#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Submit the unified Apex Agents + Kimi-K3 harness.
#
# Usage:
#   ./scripts/launch_apex_agents_k3_unified.sh <tag> [nodes] [global_concurrency] [num_repeats]
#
# The default production topology follows Serge's Kimi setup: 16 nodes form
# four rack-aware TP16 replicas behind a least-connections proxy. The batch
# script also starts one node-local Gym/Apptainer worker per allocated node and
# splits the global concurrency across those workers.

set -Eeuo pipefail

readonly APEX_GYM_DIR="/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/artij/ApexGym"
readonly BATCH_SCRIPT="${APEX_GYM_DIR}/scripts/run_apex_agents_k3_unified.sbatch"

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
(( NODES % 4 == 0 )) || { echo "ERROR: nodes must be a multiple of four for Kimi-K3" >&2; exit 64; }

export APEX_GYM_DIR
export PROFILE=${PROFILE:-${APEX_GYM_DIR}/scripts/profiles/apex-kimi-k3.env}
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
export RUN_DIR=${RUN_DIR_OVERRIDE:-${APEX_GYM_DIR}/results/apex_k3_unified/${TAG}_${timestamp}}
export OUTPUT_JSONL=${OUTPUT_JSONL_OVERRIDE:-${RUN_DIR}/rollouts.jsonl}

# Do not inherit serialized Gym state or an unrelated dependency from an
# interactive shell that previously launched another evaluation.
unset NEMO_GYM_CONFIG_DICT NEMO_GYM_CONFIG_PATH SBATCH_DEPENDENCY

mkdir -p "${RUN_DIR}/logs" "${APEX_GYM_DIR}/results/slurm-logs"

bash "${BATCH_SCRIPT}" --validate
if [[ "${VALIDATE_ONLY:-false}" == "true" ]]; then
    echo "Validation-only mode passed; no Slurm job was submitted."
    echo "Topology: $((NODES / 4)) rack-aware Kimi replica(s) x 4 nodes behind least-connections, up to $((NODES < CONCURRENCY ? NODES : CONCURRENCY)) node-local Gym workers"
    echo "Global concurrency: ${CONCURRENCY}; limit=${LIMIT:-all}"
    echo "Local recovery policy: per-task timeout=${APEX_AGENT_TIMEOUT}s; max attempts=${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS}"
    exit 0
fi

job_id=$(
    sbatch --parsable \
        --nodes="${NODES}" \
        --account="${SBATCH_ACCOUNT:-nemotron_n4_post}" \
        --partition="${SBATCH_PARTITION:-batch}" \
        --time="${WALLTIME:-04:00:00}" \
        --output="${RUN_DIR}/logs/%j_rollout.out" \
        --error="${RUN_DIR}/logs/%j_rollout.err" \
        --export=ALL \
        "${BATCH_SCRIPT}"
)
job_id=${job_id%%;*}
printf '%s\n' "${job_id}" >"${RUN_DIR}/JOBID"

echo "Submitted Apex/K3 unified job: ${job_id}"
echo "Topology: $((NODES / 4)) rack-aware Kimi replica(s) x 4 nodes behind least-connections, up to $((NODES < CONCURRENCY ? NODES : CONCURRENCY)) node-local Gym workers"
echo "Global concurrency: ${CONCURRENCY}; limit=${LIMIT:-all}"
echo "Local recovery policy: per-task timeout=${APEX_AGENT_TIMEOUT}s; max attempts=${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS}"
echo "Run directory: ${RUN_DIR}"
echo "Output: ${OUTPUT_JSONL}"
echo "Monitor: squeue -j ${job_id}"
echo "Log: tail -F ${RUN_DIR}/logs/${job_id}_rollout.out"
