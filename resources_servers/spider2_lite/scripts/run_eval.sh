#!/usr/bin/env bash
# Run the Spider 2.0-Lite benchmark against any OpenAI-compatible model
# endpoint (local vLLM, hosted inference service, etc.).
#
# Prerequisites:
#   - Gym installed into Gym/.venv (uv venv --python 3.13.14 .venv && uv sync).
#   - BigQuery modes: `gcloud auth application-default login` first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: run_eval.sh [options]

Model:
  --model-url URL                OpenAI-compatible base URL. Required.
                                 Falls back to the POLICY_BASE_URL env var.
  --model NAME                   Model name. Required.
                                 Falls back to the POLICY_MODEL_NAME env var.

The model server API key is read only from the POLICY_API_KEY env var and
forwarded to `gym eval run` when non-empty. Export it (or source a file that
sets it) so the key never appears on the command line.

Dataset:
  --engine {sqlite,bigquery,all} Default sqlite (135 tasks, no credentials).
                                 `bigquery` and `all` need gcloud ADC.
  --smoke                        Run the committed 5-row example set for --engine.
                                 `--engine all` concatenates both engines at runtime.
  --input PATH                   Use any JSONL instead. Exclusive with --smoke.

Run control:
  --purge-cache                  Wipe all cached artifacts (git clone, SQLite
                                 download, sidecars) and rebuild from scratch.
  --num-repeats N                Rollouts per task. Default 5.
  --concurrency N                Benchmark tasks evaluated in parallel. Capped
                                 by the resource server's semaphore (32 SQLite,
                                 8 BigQuery). Default 32.
  --limit N                      Process at most N tasks from the dataset.
                                 Useful for ad-hoc subsets beyond --smoke.
  --output PATH                  Default results/benchmarks/spider2_lite_<model>_<mode>.jsonl
                                 (model slug: slashes replaced with _).
  --resume                       Skip tasks already present in --output,
                                 continuing an interrupted run.
  -h, --help                     Show this message.
EOF
}

# Default to the credential-free, zero-cost engine. BigQuery and "all" are
# opt-in because they require gcloud ADC and bill against a GCP project.
ENGINE="sqlite"
SMOKE=false
INPUT_PATH=""
PURGE_CACHE=false
NUM_REPEATS=5
CONCURRENCY=32
LIMIT=""
OUTPUT=""
RESUME=false
MODEL_URL_FLAG=""
MODEL_NAME_FLAG=""

while (( $# > 0 )); do
  case "$1" in
    --model-url) MODEL_URL_FLAG="$2"; shift 2 ;;
    --model) MODEL_NAME_FLAG="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --smoke) SMOKE=true; shift ;;
    --input) INPUT_PATH="$2"; shift 2 ;;
    --purge-cache) PURGE_CACHE=true; shift ;;
    --num-repeats) NUM_REPEATS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Validate --engine.
case "${ENGINE}" in
  sqlite|bigquery|all) ;;
  *)
    echo "--engine must be one of: sqlite, bigquery, all (got '${ENGINE}')" >&2
    exit 2
    ;;
esac

# Reject conflicting dataset selectors rather than silently preferring one.
if [[ -n "${INPUT_PATH}" ]] && ${SMOKE}; then
  echo "--input and --smoke are mutually exclusive" >&2
  exit 2
fi

# Pick the dataset: --input, --smoke, or the --engine default (in that
# precedence order, enforced by the mutual exclusion check above).
#   MODE        label used for output path and log tagging
#   SOURCE_REL  JSONL path (relative to GYM_ROOT, or absolute for --input)
#
if [[ -n "${INPUT_PATH}" ]]; then
  MODE="input"
  SOURCE_REL="${INPUT_PATH}"
elif ${SMOKE}; then
  MODE="${ENGINE}_smoke"
  case "${ENGINE}" in
    sqlite)   SOURCE_REL="resources_servers/spider2_lite/data/example.jsonl" ;;
    bigquery) SOURCE_REL="resources_servers/spider2_lite/data/bigquery_example.jsonl" ;;
    all)      SOURCE_REL="" ;;  # built at runtime via concat below
  esac
else
  MODE="${ENGINE}"
  SOURCE_REL="benchmarks/spider2_lite/data/spider2_lite_benchmark.jsonl"
fi

if [[ -n "${SOURCE_REL}" ]]; then
  SOURCE_ABS="${SOURCE_REL}"
  case "${SOURCE_ABS}" in /*) ;; *) SOURCE_ABS="${GYM_ROOT}/${SOURCE_REL}" ;; esac
else
  SOURCE_ABS=""
fi

# BigQuery modes need application-default credentials.
if [[ "${ENGINE}" == "bigquery" || "${ENGINE}" == "all" ]]; then
  cred_path="${HOME}/.config/gcloud/application_default_credentials.json"
  if [[ ! -f "${cred_path}" ]]; then
    echo "BigQuery mode requires ${cred_path}." >&2
    echo "Run: gcloud auth application-default login" >&2
    exit 1
  fi
fi

# URL / name fall back to the Gym env vars POLICY_BASE_URL / POLICY_MODEL_NAME.
# The API key reads POLICY_API_KEY.
MODEL_URL="${MODEL_URL_FLAG:-${POLICY_BASE_URL:-}}"
MODEL_NAME="${MODEL_NAME_FLAG:-${POLICY_MODEL_NAME:-}}"
MODEL_API_KEY="${POLICY_API_KEY:-}"

missing=()
if [[ -z "${MODEL_URL}" ]]; then
  missing+=("--model-url")
fi
if [[ -z "${MODEL_NAME}" ]]; then
  missing+=("--model")
fi
if (( ${#missing[@]} > 0 )); then
  echo "Missing required: ${missing[*]}" >&2
  exit 2
fi

# Output path. --output overrides the per-mode default. Default includes the
# model slug so repeated runs against different models do not clobber each
# other.
MODEL_SLUG="${MODEL_NAME//\//_}"
if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="results/benchmarks/spider2_lite_${MODEL_SLUG}_${MODE}.jsonl"
fi

GYM_BIN="${GYM_ROOT}/.venv/bin/gym"
if [[ ! -x "${GYM_BIN}" ]]; then
  echo "Gym venv missing: ${GYM_BIN}" >&2
  echo "Create it: (cd ${GYM_ROOT} && uv venv --python 3.13.14 .venv && uv sync)" >&2
  exit 1
fi

# Run from GYM_ROOT so Hydra config_paths and all relative dataset paths resolve.
cd "${GYM_ROOT}"
mkdir -p "$(dirname "${OUTPUT}")"

# Two prepare paths. Both replace the inherited datasets list wholesale with
# a single entry so Hydra's merge stays well-defined:
#   * Default (--engine X): point at spider2_lite_benchmark.jsonl and run
#     prepare_script to generate the sidecar and copy it into place.
#   * Smoke / input: point at a hand-maintained snapshot under
#     resources_servers/spider2_lite/data/ (or a user-supplied PATH) and set
#     ++use_cached_prepared_benchmarks=true so Gym reads the file as-is.

DATASETS_KEY="++spider2_lite_benchmark_simple_agent.responses_api_agents.spider2_lite_agent.datasets"

HYDRA_OVERRIDES=(
  "++prepare_script_args={engine:'${ENGINE}',purge_cache:${PURGE_CACHE}}"
)

if [[ "${MODE}" == "input" || "${MODE}" == *_smoke ]]; then
  SOURCE_COPY="results/benchmarks/spider2_lite_${MODE}_source.jsonl"
  mkdir -p "$(dirname "${SOURCE_COPY}")"
  if [[ "${MODE}" == "all_smoke" ]]; then
    # Concatenate the two committed smoke files into SOURCE_COPY.
    sqlite_src="${GYM_ROOT}/resources_servers/spider2_lite/data/example.jsonl"
    bigquery_src="${GYM_ROOT}/resources_servers/spider2_lite/data/bigquery_example.jsonl"
    for f in "${sqlite_src}" "${bigquery_src}"; do
      if [[ ! -f "${f}" ]]; then
        echo "Committed smoke file missing: ${f}" >&2
        exit 1
      fi
    done
    cat "${sqlite_src}" "${bigquery_src}" > "${SOURCE_COPY}"
    SOURCE_REL="(example.jsonl + bigquery_example.jsonl)"
  else
    if [[ ! -f "${SOURCE_ABS}" ]]; then
      echo "Dataset not found: ${SOURCE_ABS}" >&2
      exit 1
    fi
    cp "${SOURCE_ABS}" "${SOURCE_COPY}"
  fi
  # Hydra parses the datasets=[{...}] value as one token, so the inner
  # list literal has to stay on one line, but we hoist it into a variable so
  # the HYDRA_OVERRIDES array stays readable.
  dataset_entry="{name:spider2_lite,type:benchmark,jsonl_fpath:${SOURCE_COPY}"
  dataset_entry+=",prompt_config:null,prepare_script:benchmarks/spider2_lite/prepare.py"
  dataset_entry+=",num_repeats:${NUM_REPEATS}}"
  HYDRA_OVERRIDES+=(
    "++use_cached_prepared_benchmarks=true"
    "${DATASETS_KEY}=[${dataset_entry}]"
  )
else
  # Replace the inherited datasets list with a single benchmark entry whose
  # prepare_script copies the chosen engine sidecar into jsonl_fpath on each
  # invocation.
  dataset_entry="{name:spider2_lite,type:benchmark,jsonl_fpath:${SOURCE_REL}"
  dataset_entry+=",prompt_config:null,prepare_script:benchmarks/spider2_lite/prepare.py"
  dataset_entry+=",num_repeats:${NUM_REPEATS}}"
  HYDRA_OVERRIDES+=(
    "${DATASETS_KEY}=[${dataset_entry}]"
  )
fi

cat <<EOF
Spider 2.0-Lite run:
  mode:        ${MODE}
  engine:      ${ENGINE}
  dataset:     ${SOURCE_REL}
  model:       ${MODEL_NAME}
  model URL:   ${MODEL_URL}
  repeats:     ${NUM_REPEATS}
  concurrency: ${CONCURRENCY}
  purge cache: ${PURGE_CACHE}
  output:      ${OUTPUT}
EOF

EXTRA_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi
if ${RESUME}; then
  EXTRA_ARGS+=(--resume)
fi
if [[ -n "${MODEL_API_KEY}" ]]; then
  EXTRA_ARGS+=(--model-api-key "${MODEL_API_KEY}")
fi

# Non-smoke modes (--engine X without --smoke, no --input) run the benchmark's
# prepare_script to materialize the dataset at jsonl_fpath before the eval reads
# it. The smoke and input modes already have the file in place and set
# ++use_cached_prepared_benchmarks=true, so prepare is a no-op there.
if [[ "${MODE}" != "input" && "${MODE}" != *_smoke ]]; then
  "${GYM_BIN}" eval prepare \
    --benchmark spider2_lite \
    "${HYDRA_OVERRIDES[@]}"
fi

exec "${GYM_BIN}" eval run \
  --model-type vllm_model \
  --benchmark spider2_lite \
  --output "${OUTPUT}" \
  --split benchmark \
  --model-url "${MODEL_URL}" \
  --model "${MODEL_NAME}" \
  --num-repeats "${NUM_REPEATS}" \
  --concurrency "${CONCURRENCY}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  "${HYDRA_OVERRIDES[@]}"
