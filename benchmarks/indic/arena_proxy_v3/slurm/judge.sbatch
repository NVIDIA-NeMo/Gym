#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set +x
set -euo pipefail
: "${GYM_REPO:?Submit using slurm/submit.sh, or export GYM_REPO}"
source "$GYM_REPO/benchmarks/indic/arena_proxy_v3/slurm/common.sh"
set_run_paths
[[ -s "$GENERATIONS" && -s "$MATERIALIZED_INPUTS" ]] || die "Generation rollouts or materialized inputs are missing."
check_judge_key_file
# Read only on the CPU node: no credential in sbatch --export arguments or job metadata.
ARENA_JUDGE_API_KEY=$(< "$JUDGE_KEY_FILE")
[[ -n "$ARENA_JUDGE_API_KEY" && ! "$ARENA_JUDGE_API_KEY" =~ [[:space:]] ]] || die "JUDGE_KEY_FILE must contain one nonempty token with no embedded whitespace."
export ARENA_JUDGE_API_KEY
export ARENA_JUDGE_BASE_URL="${ARENA_JUDGE_BASE_URL:-https://inference-api.nvidia.com/v1}"
export ARENA_JUDGE_MODEL="${ARENA_JUDGE_MODEL:-gcp/google/gemini-3.8-flash}"
HEAD_PORT="${HEAD_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 30000))}"

"$GYM_BIN" eval reverify --config "$BENCHMARK_DIR/judge.yaml" \
    --inputs "$MATERIALIZED_INPUTS" --rollouts "$GENERATIONS" \
    --output "$JUDGMENTS" --resume --concurrency "${JUDGE_CONCURRENCY:-12}" \
    ++head_server.host=127.0.0.1 "++head_server.port=$HEAD_PORT"
unset ARENA_JUDGE_API_KEY
SUMMARY_PYTHON_BIN="${SUMMARY_PYTHON_BIN:-$GYM_REPO/resources_servers/arena_judge/.venv/bin/python}"
[[ -x "$SUMMARY_PYTHON_BIN" ]] || die "Set SUMMARY_PYTHON_BIN to the arena_judge environment's Python (it needs the scoring dependencies)."
"$SUMMARY_PYTHON_BIN" -m benchmarks.indic.arena_proxy_v3.summarize "$JUDGMENTS" \
    --inputs "$INPUT_JSONL" --output "$RUN_DIR/metrics.json" --languages "${languages[@]}"
