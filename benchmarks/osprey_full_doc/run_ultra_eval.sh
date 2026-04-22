#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BENCHMARK_JSONL="${BENCHMARK_JSONL:-${SCRIPT_DIR}/data/osprey_full_doc_benchmark.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/osprey_full_doc_ultra_eval}"
NG_BIN_DIR="${NG_BIN_DIR:-${REPO_ROOT}/.venv/bin}"
HEAD_SERVER_HOST="${HEAD_SERVER_HOST:-127.0.0.1}"
HEAD_SERVER_PORT="${HEAD_SERVER_PORT:-11012}"
POLICY_BASE_URL="${POLICY_BASE_URL:-http://127.0.0.1:9010/v1}"
POLICY_API_KEY="${POLICY_API_KEY:-dummy}"
POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-model}"
NUM_SAMPLES_IN_PARALLEL="${NUM_SAMPLES_IN_PARALLEL:-4}"
LIMIT="${LIMIT:-}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"

resolve_bin() {
    local binary_name="$1"
    local candidate=""

    if candidate="$(command -v "${binary_name}" 2>/dev/null)"; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    candidate="${REPO_ROOT}/.venv/bin/${binary_name}"
    if [[ -x "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    candidate="${NG_BIN_DIR}/${binary_name}"
    if [[ -x "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    echo "Could not find ${binary_name} on PATH or in a known .venv location." >&2
    exit 1
}

require_pinned_sampling() {
    python3 - "${BENCHMARK_JSONL}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    row = json.loads(next(f))

params = row["responses_create_params"]
expected = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_output_tokens": 32000,
}
for key, value in expected.items():
    if params.get(key) != value:
        raise SystemExit(
            f"Prepared benchmark row is missing the pinned {key} contract: "
            f"expected {value!r}, found {params.get(key)!r}"
        )
PY
}

wait_for_port() {
    local host="$1"
    local port="$2"
    python3 - "${host}" "${port}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1.0)
    raise SystemExit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
}

if [[ ! -f "${BENCHMARK_JSONL}" ]]; then
    echo "Missing prepared benchmark JSONL: ${BENCHMARK_JSONL}" >&2
    echo "Run ng_prepare_benchmark or benchmarks/osprey_full_doc/prepare.py first." >&2
    exit 1
fi

require_pinned_sampling

NG_RUN_BIN="$(resolve_bin ng_run)"
NG_COLLECT_ROLLOUTS_BIN="$(resolve_bin ng_collect_rollouts)"

mkdir -p "${OUTPUT_DIR}"

RUN_LOG="${OUTPUT_DIR}/ng_run.log"
ROLLOUTS_JSONL="${OUTPUT_DIR}/osprey_full_doc_rollouts.jsonl"
AGGREGATE_JSON="${OUTPUT_DIR}/osprey_full_doc_rollouts_aggregate_metrics.json"
SUMMARY_JSON="${OUTPUT_DIR}/osprey_full_doc_rollouts_summary.json"
REPORT_MD="${OUTPUT_DIR}/osprey_full_doc_rollouts_report.md"
collect_args=(
    ++head_server.host="${HEAD_SERVER_HOST}"
    ++head_server.port="${HEAD_SERVER_PORT}"
    +agent_name=osprey_full_doc_benchmark_agent
    +input_jsonl_fpath="${BENCHMARK_JSONL}"
    +output_jsonl_fpath="${ROLLOUTS_JSONL}"
    +num_samples_in_parallel="${NUM_SAMPLES_IN_PARALLEL}"
)
if [[ -n "${LIMIT}" ]]; then
    collect_args+=("+limit=${LIMIT}")
fi

env UV_CACHE_DIR="${UV_CACHE_DIR}" XDG_CACHE_HOME="${XDG_CACHE_HOME}" UV_LINK_MODE="${UV_LINK_MODE}" \
    "${NG_RUN_BIN}" \
    "+config_paths=[benchmarks/osprey_full_doc/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    ++head_server.host="${HEAD_SERVER_HOST}" \
    ++head_server.port="${HEAD_SERVER_PORT}" \
    +policy_base_url="${POLICY_BASE_URL}" \
    +policy_api_key="${POLICY_API_KEY}" \
    +policy_model_name="${POLICY_MODEL_NAME}" \
    >"${RUN_LOG}" 2>&1 &
NG_RUN_PID=$!

cleanup() {
    if kill -0 "${NG_RUN_PID}" 2>/dev/null; then
        kill "${NG_RUN_PID}" 2>/dev/null || true
        wait "${NG_RUN_PID}" || true
    fi
}
trap cleanup EXIT

for _ in $(seq 1 90); do
    if wait_for_port "${HEAD_SERVER_HOST}" "${HEAD_SERVER_PORT}"; then
        break
    fi
    sleep 2
done

if ! wait_for_port "${HEAD_SERVER_HOST}" "${HEAD_SERVER_PORT}"; then
    echo "Timed out waiting for ng_run on ${HEAD_SERVER_HOST}:${HEAD_SERVER_PORT}" >&2
    tail -n 50 "${RUN_LOG}" >&2 || true
    exit 1
fi

env UV_CACHE_DIR="${UV_CACHE_DIR}" XDG_CACHE_HOME="${XDG_CACHE_HOME}" UV_LINK_MODE="${UV_LINK_MODE}" \
    "${NG_COLLECT_ROLLOUTS_BIN}" \
    "${collect_args[@]}"

if [[ -f "${AGGREGATE_JSON}" ]]; then
    python3 "${SCRIPT_DIR}/eval_rollouts.py" \
        --aggregate-json "${AGGREGATE_JSON}" \
        --rollouts-jsonl "${ROLLOUTS_JSONL}" \
        --expected-input-jsonl "${BENCHMARK_JSONL}" \
        --summary-json "${SUMMARY_JSON}" \
        --report-md "${REPORT_MD}"
else
    python3 "${SCRIPT_DIR}/eval_rollouts.py" \
        --rollouts-jsonl "${ROLLOUTS_JSONL}" \
        --expected-input-jsonl "${BENCHMARK_JSONL}" \
        --summary-json "${SUMMARY_JSON}" \
        --report-md "${REPORT_MD}"
fi

echo "Wrote rollout artifacts to ${OUTPUT_DIR}"
