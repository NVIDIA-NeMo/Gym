#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set +x
set -euo pipefail
GYM_REPO="${GYM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
source "$GYM_REPO/benchmarks/indic/arena_proxy_v3/slurm/common.sh"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT to an account you can use}"
: "${MODEL_PATH:?Set MODEL_PATH to the checkpoint directory}"
check_judge_key_file
MODEL_PATH=$(realpath "$MODEL_PATH")
JUDGE_KEY_FILE=$(realpath "$JUDGE_KEY_FILE")
IMAGE="${IMAGE:-/lustre/fsw/portfolios/llmservice/users/kanishks/images/vllm-openai-v0.26.0.sqsh}"
[[ -r "$MODEL_PATH/config.json" ]] || die "Checkpoint config.json is missing or unreadable."
[[ -r "$IMAGE" ]] || die "Set IMAGE to a readable local vLLM .sqsh file."
check_generation_budget
command -v sbatch >/dev/null || die "sbatch is unavailable; run this helper on a SLURM login node."
RUN_DIR=$(realpath -m "${RUN_DIR:-$GYM_REPO/results/arena_proxy_v3/$MODEL_KEY-$(date -u +%Y%m%dT%H%M%S)}")
[[ ! -e "$RUN_DIR/inputs" ]] || die "RUN_DIR already has prepared inputs. Choose a new RUN_DIR, or resubmit generate.sbatch/judge.sbatch directly to resume."
mkdir -p "$RUN_DIR/logs"
prepare_args=(--output-dir "$RUN_DIR/inputs" --languages "${languages[@]}")
[[ -z "${SOURCE_DIR:-}" ]] || prepare_args+=(--source-dir "$SOURCE_DIR")
[[ -z "${LIMIT:-}" ]] || prepare_args+=(--limit "$LIMIT")
"$PYTHON_BIN" "$BENCHMARK_DIR/prepare.py" "${prepare_args[@]}"
set_run_paths
export MODEL_PATH JUDGE_KEY_FILE IMAGE RUN_DIR
# Do not include Hub or inference tokens in SLURM's exported submission environment.
unset ARENA_JUDGE_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN
generation_job=$(sbatch --parsable --account="$SLURM_ACCOUNT" --partition="${GPU_PARTITION:-interactive}" \
    --job-name="arena-gen-$MODEL_KEY" --chdir="$GYM_REPO" --output="$RUN_DIR/logs/generate-%j.log" \
    --export=ALL "$BENCHMARK_DIR/slurm/generate.sbatch")
generation_job=${generation_job%%;*}
[[ "$generation_job" =~ ^[0-9]+$ ]] || die "sbatch returned an invalid generation job ID."
printf 'Generation job: %s\nRun directory: %s\n' "$generation_job" "$RUN_DIR"
judge_job=$(sbatch --parsable --account="$SLURM_ACCOUNT" --partition="${CPU_PARTITION:-cpu_interactive}" \
    --job-name="arena-judge-$MODEL_KEY" --dependency="afterok:$generation_job" \
    --chdir="$GYM_REPO" --output="$RUN_DIR/logs/judge-%j.log" --export=ALL "$BENCHMARK_DIR/slurm/judge.sbatch")
judge_job=${judge_job%%;*}
printf 'Judge job: %s\nMetrics: %s/metrics.json\n' "$judge_job" "$RUN_DIR"
