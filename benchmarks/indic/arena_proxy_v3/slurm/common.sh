#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sourced by the submission helper and both batch scripts.
set +x
set -euo pipefail
umask 077

die() { printf '%s\n' "$*" >&2; exit 1; }

: "${GYM_REPO:?Set GYM_REPO to the Indic_Gym checkout}"
: "${MODEL_KEY:?Set MODEL_KEY to a short model name}"
[[ "$MODEL_KEY" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "MODEL_KEY must contain only letters, numbers, _, . and -."
GYM_REPO=$(realpath "$GYM_REPO")
cd "$GYM_REPO"
BENCHMARK_DIR="$GYM_REPO/benchmarks/indic/arena_proxy_v3"
GYM_BIN="${GYM_BIN:-$GYM_REPO/.venv/bin/gym}"
PYTHON_BIN="${PYTHON_BIN:-$GYM_REPO/.venv/bin/python}"
[[ -x "$GYM_BIN" && -x "$PYTHON_BIN" ]] || die "Install Gym in .venv, or set GYM_BIN and PYTHON_BIN to its executable paths."
LANGUAGES="${LANGUAGES:-bn gu hi kn ml mr ne or pa ta te ur}"
[[ "$LANGUAGES" != *,* ]] || die "LANGUAGES must be space-separated, not comma-separated."
read -r -a languages <<< "$LANGUAGES"
((${#languages[@]})) || die "LANGUAGES cannot be empty."
declare -A seen_languages=()
for language in "${languages[@]}"; do
    case "$language" in en|bn|gu|hi|kn|ml|mr|ne|or|pa|ta|te|ur) ;; *) die "Unsupported language: $language" ;; esac
    [[ -z "${seen_languages[$language]:-}" ]] || die "Duplicate language: $language"
    seen_languages[$language]=1
done
export GYM_REPO MODEL_KEY LANGUAGES GYM_BIN PYTHON_BIN

check_judge_key_file() {
    : "${JUDGE_KEY_FILE:?Set JUDGE_KEY_FILE to your NVIDIA inference API key file}"
    [[ -f "$JUDGE_KEY_FILE" && -r "$JUDGE_KEY_FILE" && -O "$JUDGE_KEY_FILE" ]] || die "JUDGE_KEY_FILE must be a readable file owned by you."
    local mode
    mode=$(stat -Lc '%a' "$JUDGE_KEY_FILE")
    (( (8#$mode & 077) == 0 )) || die "JUDGE_KEY_FILE must have no group or other permissions; run chmod 600 on it."
}

check_generation_budget() {
    CONTEXT="${CONTEXT:-163840}"
    [[ "$CONTEXT" =~ ^[1-9][0-9]*$ ]] || die "CONTEXT must be a positive integer."
    MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-120000}"
    [[ "$MAX_OUTPUT_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "MAX_OUTPUT_TOKENS must be a positive integer."
    export CONTEXT MAX_OUTPUT_TOKENS
}

set_run_paths() {
    : "${RUN_DIR:?Set RUN_DIR to the prepared run directory}"
    RUN_DIR=$(realpath -m "$RUN_DIR")
    INPUT_JSONL="$RUN_DIR/inputs/arena_proxy_v3_benchmark.jsonl"
    GENERATIONS="$RUN_DIR/generations/rollouts.jsonl"
    MATERIALIZED_INPUTS="$RUN_DIR/generations/rollouts_materialized_inputs.jsonl"
    JUDGMENTS="$RUN_DIR/judgments/rollouts.jsonl"
    [[ -s "$INPUT_JSONL" ]] || die "Prepared input missing: $INPUT_JSONL. Use slurm/submit.sh first."
    mkdir -p "$RUN_DIR/logs" "$RUN_DIR/generations" "$RUN_DIR/judgments"
    export RUN_DIR
}
