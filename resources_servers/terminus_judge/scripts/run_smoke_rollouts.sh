#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERMINUS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
GYM_DIR="$(cd "${TERMINUS_DIR}/../.." && pwd)"

DATASET_NAME="${DATASET_NAME:-openthoughts_agent_v1_sft}"
DATA_DIR="${DATA_DIR:-${TERMINUS_DIR}/data/${DATASET_NAME}}"
SMOKE_DIR="${SMOKE_DIR:-${DATA_DIR}/smoke}"
VALIDATION_PATH="${VALIDATION_PATH:-${DATA_DIR}/validation.jsonl}"
SMOKE_INPUT_PATH="${SMOKE_INPUT_PATH:-${SMOKE_DIR}/smoke_input.jsonl}"
SMOKE_OUTPUT_PATH="${SMOKE_OUTPUT_PATH:-${SMOKE_DIR}/smoke_rollouts.jsonl}"

SMOKE_LIMIT="${SMOKE_LIMIT:-5}"
AGENT_NAME="${AGENT_NAME:-terminus_judge_simple_agent}"
NUM_SAMPLES_IN_PARALLEL="${NUM_SAMPLES_IN_PARALLEL:-4}"

CONFIG_PATHS="${CONFIG_PATHS:-resources_servers/terminus_judge/configs/terminus_judge_simple.yaml,responses_api_models/openai_model/configs/openai_model.yaml}"
POLICY_BASE_URL="${POLICY_BASE_URL:-}"
POLICY_API_KEY="${POLICY_API_KEY:-}"
POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-}"

NG_RUN_BIN="${NG_RUN_BIN:-ng_run}"
NG_COLLECT_BIN="${NG_COLLECT_BIN:-ng_collect_rollouts}"
NG_STATUS_BIN="${NG_STATUS_BIN:-ng_status}"
WAIT_RETRIES="${WAIT_RETRIES:-180}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-5}"
STATUS_TIMEOUT_SEC="${STATUS_TIMEOUT_SEC:-8}"

RUN_LOG="${SMOKE_DIR}/smoke_ng_run.log"
COLLECT_LOG="${SMOKE_DIR}/smoke_ng_collect.log"

mkdir -p "${SMOKE_DIR}"

if [[ ! -f "${VALIDATION_PATH}" ]]; then
  echo "Missing validation input: ${VALIDATION_PATH}" >&2
  exit 1
fi

if ! command -v "${NG_RUN_BIN}" >/dev/null 2>&1; then
  echo "ng_run not found on PATH (or NG_RUN_BIN is invalid)." >&2
  exit 1
fi
if ! command -v "${NG_COLLECT_BIN}" >/dev/null 2>&1; then
  echo "ng_collect_rollouts not found on PATH (or NG_COLLECT_BIN is invalid)." >&2
  exit 1
fi
if ! command -v "${NG_STATUS_BIN}" >/dev/null 2>&1; then
  echo "ng_status not found on PATH (or NG_STATUS_BIN is invalid)." >&2
  exit 1
fi

python - "${VALIDATION_PATH}" "${SMOKE_INPUT_PATH}" "${SMOKE_LIMIT}" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
limit = int(sys.argv[3])
written = 0

with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        json.loads(line)
        f_out.write(line + "\n")
        written += 1
        if written >= limit:
            break

if written == 0:
    raise SystemExit(f"No rows copied from {src}")

print(f"Wrote {written} smoke rows to {dst}")
PY

wait_for_server() {
  local expected_count="$1"
  local log_file="$2"
  local pid="$3"
  for _ in $(seq 1 "${WAIT_RETRIES}"); do
    if grep -q "All ${expected_count} / ${expected_count} servers ready!" "${log_file}" 2>/dev/null; then
      return 0
    fi
    if grep -q "finished unexpectedly" "${log_file}" 2>/dev/null; then
      return 1
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 1
    fi
    local status_output
    status_output="$(timeout "${STATUS_TIMEOUT_SEC}" "${NG_STATUS_BIN}" 2>/dev/null || true)"
    if printf '%s\n' "${status_output}" | grep -q "${expected_count} servers found (${expected_count} healthy, 0 unhealthy)"; then
      return 0
    fi
    sleep "${WAIT_INTERVAL_SEC}"
  done
  return 1
}

cleanup() {
  if [[ -n "${NG_PID:-}" ]]; then
    kill "${NG_PID}" 2>/dev/null || true
    wait "${NG_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

rm -f "${SMOKE_OUTPUT_PATH}" "${RUN_LOG}" "${COLLECT_LOG}"

cd "${GYM_DIR}"
"${NG_RUN_BIN}" \
  "+config_paths=[${CONFIG_PATHS}]" \
  "+policy_base_url=${POLICY_BASE_URL}" \
  "+policy_api_key=${POLICY_API_KEY}" \
  "+policy_model_name=${POLICY_MODEL_NAME}" \
  >"${RUN_LOG}" 2>&1 &
NG_PID=$!

if ! wait_for_server 3 "${RUN_LOG}" "${NG_PID}"; then
  echo "ng_run did not become ready. See ${RUN_LOG}" >&2
  timeout "${STATUS_TIMEOUT_SEC}" "${NG_STATUS_BIN}" >&2 || true
  tail -n 200 "${RUN_LOG}" >&2 || true
  exit 1
fi

"${NG_COLLECT_BIN}" \
  "+agent_name=${AGENT_NAME}" \
  "+input_jsonl_fpath=${SMOKE_INPUT_PATH}" \
  "+output_jsonl_fpath=${SMOKE_OUTPUT_PATH}" \
  "+num_samples_in_parallel=${NUM_SAMPLES_IN_PARALLEL}" \
  >"${COLLECT_LOG}" 2>&1

echo "Smoke rollouts complete: ${SMOKE_OUTPUT_PATH}"
echo "ng_run log: ${RUN_LOG}"
echo "ng_collect_rollouts log: ${COLLECT_LOG}"
