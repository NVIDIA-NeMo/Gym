#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Evaluate ONE policy model on BIRD dev (or, with --split train, BIRD train) via the standard, existing birdbench pipeline
# (`gym eval prepare`/`gym eval run --benchmark birdbench`, as documented in this directory's own
# README) -- no custom collection code (for training data with logprobs see collect.py). Verification
# runs through resources_servers/bird_sql/eval_utils.py exactly as it always has, so this
# benefits from the GIL-contention-safe concurrency and `conn.close()` fixes made there
# automatically, with zero changes needed here.
#
# `gym eval prepare` only needs to run once (the prepared jsonl doesn't depend on which model
# you're evaluating) -- skipped automatically on subsequent calls if it already exists.
#
# Usage:
#   ./benchmarks/birdbench/eval.sh --model <name> [options...]
#
# Options (all optional except --model):
#   --output_dir DIR         Default: benchmarks/birdbench/results/<sanitized model name>
#                              (benchmarks/birdbench/data/ is reserved for input/prepared data)
#   --max-output-tokens N     Default: 16000
#   --temperature T            Default: 1.0
#   --num-repeats N              Default: 1 = exactly one rollout per task (1534 rollouts). The dataset itself is
#                                  never repeated (config.yaml's num_repeats is 1); this flag is the only repeat
#                                  control, so N > 1 gives 1534 x N rollouts.
#   --model-url URL                Overrides env.yaml's policy_base_url for this run
#   --model-api-key KEY               Overrides env.yaml's policy_api_key for this run
#   --model-type NAME                   Default: vllm_model (loads the responses_api_models
#                                         config that defines `policy_model`, referenced by
#                                         birdbench_bird_sql_simple_agent's model_server field --
#                                         required, same as `gym env start --model-type ...`)
#   --dialect sqlite|spark                Default: sqlite. The SQL dialect the model writes. Selects the prepared
#                                           file (data/birdbench_<dialect>_benchmark.jsonl), the prompt
#                                           (prompts/default.yaml for sqlite, prompts/<dialect>.yaml otherwise), what
#                                           `prepare` renders, and the engine the server runs the model's SQL on.
#                                           `spark` needs a JDK (JAVA_HOME or `java` on PATH) and pyspark + sqlglot in
#                                           resources_servers/bird_sql/.venv -- see the "SQL dialects" section of
#                                           README.md. By default the output dir gets a `_<dialect>` suffix for
#                                           non-sqlite dialects.
#   --split dev|train                      Default: dev. Which BIRD split's questions and databases to use. `train` is the
#                                            ReViSQL-verified "uberset" of BIRD train (~9 GB download on first use; see
#                                            README.md, "Splits") and is meant for generating training data. Files get a
#                                            `train` infix (data/birdbench_train_<dialect>.jsonl) and the output dir a
#                                            `_train` suffix. Not gym's own `--split` (that one is fixed to `benchmark`).
#   --concurrency N                        Default: 8. Caps concurrent in-flight requests
#                                            (nemo_gym's own retry on 429/502/503/504/520 has no
#                                            upper bound and no backoff growth -- see
#                                            NeMoGymAsyncOpenAI._request_with_retry -- so with
#                                            unbounded concurrency (the `gym eval run` default)
#                                            a whole benchmark's worth of requests fire at once
#                                            and can create a self-sustaining 429 storm against a
#                                            model's RPM cap. Lower this further for models
#                                            already reporting "Remaining: 0" in their 429 body.
# --split benchmark is hardcoded below (not exposed as a flag) -- required by `gym eval run`
# whenever it's not started with --no-serve; "benchmark" matches config.yaml's dataset `type:
# benchmark` declaration for birdbench (nemo_gym/rollout_collection.py's `split` field only
# accepts "train"/"validation"/"benchmark" -- birdbench's own dataset is always "benchmark").
#
# Any other flag (e.g. --limit, --top-p) is passed straight through to `gym eval run`.
#
# Example:
#   ./benchmarks/birdbench/eval.sh --model nvidia/zai-org/glm-5.2 \
#       --max-output-tokens 16000 --temperature 1.0 --output_dir results/glm-5.2

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# `gym eval run` starts each server through a shell wrapper and, on shutdown, SIGINTs then SIGKILLs only that
# wrapper (nemo_gym/cli/env.py shutdown()). The real server process survives, orphaned (parent PID 1) -- for
# bird_sql together with its ~2 GB Spark JVM. Reap those orphans (children first) whenever this script exits,
# including on failure or Ctrl-C. Only processes that are orphaned AND run from this repo's server venvs are
# touched, so servers of another, still-running `gym` (whose wrapper is alive) are left alone. Assumes
# orphans are re-parented to PID 1 (macOS, most Linux; not under a non-init subreaper).
reap_orphaned_servers() {
    local root pid child
    root="$(pwd)"
    for pid in $(ps -axo pid=,ppid=,command= | awk -v root="${root}/" '$2 == 1 && index($0, root) && $0 ~ /\/\.venv\/bin\/python app\.py/ {print $1}'); do
        for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
            kill "${child}" 2>/dev/null || true
        done
        kill "${pid}" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "${pid}" 2>/dev/null || break
            sleep 1
        done
        kill -9 "${pid}" 2>/dev/null || true
    done
    return 0
}
trap reap_orphaned_servers EXIT

MODEL=""
OUTPUT_DIR=""
MAX_OUTPUT_TOKENS="16000"
TEMPERATURE="1.0"
NUM_REPEATS="1"
MODEL_URL=""
MODEL_API_KEY=""
MODEL_TYPE="vllm_model"
CONCURRENCY="8"
DIALECT="sqlite"
SPLIT="dev"
# (Empty arrays are expanded as ${ARR[@]+"${ARR[@]}"} below: plain "${ARR[@]}" errors under `set -u` on bash < 4.4,
# e.g. macOS's stock bash 3.2, when the array is empty.)
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --output_dir|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max-output-tokens) MAX_OUTPUT_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --num-repeats) NUM_REPEATS="$2"; shift 2 ;;
        --model-url) MODEL_URL="$2"; shift 2 ;;
        --model-api-key) MODEL_API_KEY="$2"; shift 2 ;;
        --model-type) MODEL_TYPE="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --dialect) DIALECT="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${MODEL}" ]]; then
    echo "Usage: $0 --model <policy_model_name> [--output_dir DIR] [--max-output-tokens N] [--temperature T] [--num-repeats K] [--model-url URL] [--model-api-key KEY]" >&2
    exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    SAFE_MODEL_NAME="$(echo "${MODEL}" | tr '/: ' '___')"
    OUTPUT_DIR="benchmarks/birdbench/results/${SAFE_MODEL_NAME}"
    if [[ "${DIALECT}" != "sqlite" ]]; then
        OUTPUT_DIR="${OUTPUT_DIR}_${DIALECT}"
    fi
    if [[ "${SPLIT}" != "dev" ]]; then
        OUTPUT_DIR="${OUTPUT_DIR}_${SPLIT}"
    fi
fi
# dev + sqlite is config.yaml's own default, so it needs no overrides. Any other dialect or split overrides, for both
# `gym eval prepare` and `gym eval run`: what `prepare` renders, the engine the server runs, and the dataset
# (Hydra cannot override one list item, so the whole `datasets` list is replaced; `gym eval prepare` also requires
# prepare()'s output path to equal the dataset's jsonl_fpath). num_repeats is 1: the dataset is never repeated.
# (File naming mirrors prepare.output_fpath_for.)
case "${SPLIT}" in
    dev) PREPARED_JSONL="benchmarks/birdbench/data/birdbench_${DIALECT}_benchmark.jsonl"; DATASET_NAME="birdbench" ;;
    train) PREPARED_JSONL="benchmarks/birdbench/data/birdbench_train_${DIALECT}.jsonl"; DATASET_NAME="birdbench_train" ;;
    *)
        echo "Unsupported --split '${SPLIT}'; expected dev or train." >&2
        exit 1
        ;;
esac
PROMPT_CONFIG="benchmarks/birdbench/prompts/default.yaml"
if [[ "${DIALECT}" != "sqlite" ]]; then
    PROMPT_CONFIG="benchmarks/birdbench/prompts/${DIALECT}.yaml"
fi
DIALECT_ARGS=()
case "${DIALECT}" in
    sqlite) ;;
    spark)
        if ! "${JAVA_HOME:+${JAVA_HOME}/bin/}java" -version >/dev/null 2>&1; then
            echo "--dialect spark needs a JDK: set JAVA_HOME or put java on PATH (see README.md, 'SQL dialects')." >&2
            exit 1
        fi
        ;;
    *)
        echo "Unsupported --dialect '${DIALECT}'; expected sqlite or spark." >&2
        exit 1
        ;;
esac
if [[ "${DIALECT}" != "sqlite" || "${SPLIT}" != "dev" ]]; then
    DIALECT_ARGS=(
        "+prepare_script_args.dialect=${DIALECT}"
        "+prepare_script_args.split=${SPLIT}"
        "+birdbench_bird_sql_resources_server.resources_servers.bird_sql.dialect=${DIALECT}"
        "+birdbench_bird_sql_resources_server.resources_servers.bird_sql.split=${SPLIT}"
        "++birdbench_bird_sql_simple_agent.responses_api_agents.simple_agent.datasets=[{name:${DATASET_NAME},type:benchmark,jsonl_fpath:${PREPARED_JSONL},prompt_config:${PROMPT_CONFIG},prepare_script:benchmarks/birdbench/prepare.py,num_repeats:1}]"
    )
    if [[ "${SPLIT}" != "dev" ]]; then
        # Train has its own db_ids: keep its Spark warehouse apart from dev's.
        DIALECT_ARGS+=("+birdbench_bird_sql_resources_server.resources_servers.bird_sql.spark_warehouse_dir=resources_servers/bird_sql/.bird_sql/spark_warehouse_${SPLIT}")
    fi
fi
mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${PREPARED_JSONL}" ]]; then
    echo "Preparing BIRD ${SPLIT} data for ${DIALECT} (one-time, shared across every model) -> ${PREPARED_JSONL} ..."
    gym eval prepare --benchmark birdbench ${DIALECT_ARGS[@]+"${DIALECT_ARGS[@]}"}
fi

MODEL_ARGS=(--model "${MODEL}")
if [[ -n "${MODEL_URL}" ]]; then
    MODEL_ARGS+=(--model-url "${MODEL_URL}")
fi
if [[ -n "${MODEL_API_KEY}" ]]; then
    MODEL_ARGS+=(--model-api-key "${MODEL_API_KEY}")
fi

# A failed LLM request deliberately aborts the run (gym's default; do NOT pass +route_failures_to_sidecar): no
# rollouts_aggregate_metrics.json is written, rather than a score over a silently smaller dev set. Rollouts that
# completed stay in rollouts.jsonl, so the run can be resumed.
echo "=== Evaluating model=${MODEL} on BIRD ${SPLIT} (${DIALECT}) -> ${OUTPUT_DIR} ==="
EVAL_STATUS=0
gym eval run \
    --benchmark birdbench \
    --model-type "${MODEL_TYPE}" \
    --split benchmark \
    --output "${OUTPUT_DIR}/rollouts.jsonl" \
    --temperature "${TEMPERATURE}" \
    --max-output-tokens "${MAX_OUTPUT_TOKENS}" \
    --num-repeats "${NUM_REPEATS}" \
    --concurrency "${CONCURRENCY}" \
    "${MODEL_ARGS[@]}" \
    ${DIALECT_ARGS[@]+"${DIALECT_ARGS[@]}"} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} || EVAL_STATUS=$?

if [[ "${EVAL_STATUS}" -ne 0 ]]; then
    echo "=== Run FAILED (exit ${EVAL_STATUS}): no results are reported. Rollouts completed before the failure are kept in" \
        "${OUTPUT_DIR}/rollouts.jsonl and are PARTIAL; do not score them. ===" >&2
    exit "${EVAL_STATUS}"
fi

# Print the failure breakdown that bird_sql wrote into rollouts_aggregate_metrics.json next to pass@1.
python3 - "${OUTPUT_DIR}/rollouts_aggregate_metrics.json" <<'PY' || true
import json, sys

try:
    entries = json.load(open(sys.argv[1]))
except (OSError, ValueError) as e:
    sys.exit(f"[summary] no aggregate metrics to summarize: {e}")
names = ["no_sql", "sql-timeout", "gold-sql-timeout", "pred-sql-timeout",
         "sql_errors", "gold-sql-error", "pred-sql-error", "unclean"]
for entry in entries:
    am = entry.get("agent_metrics", {})
    total = am.get("rollouts_count")
    if not total:
        continue
    print(f"=== BIRD failure breakdown over {total} rollouts ===")
    for name in names:
        count = am.get(f"{name}_count")
        if count is not None:
            print(f"  {name:<18} {count:>6}  ({100 * count / total:5.2f}%)")
PY
