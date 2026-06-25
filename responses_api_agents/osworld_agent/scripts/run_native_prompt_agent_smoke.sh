#!/usr/bin/env bash
# Run a small OSWorld smoke using upstream mm_agents.agent.PromptAgent.
#
# Defaults are intentionally conservative:
#   - LIMIT=1 keeps the smoke short.
#   - num_samples_in_parallel=1 avoids launching multiple VMs at once.
#   - START_NG_RUN=1 starts a matching agent/model server config.
#   - ng_status is used to wait for server readiness before collecting.
#
# Examples:
#   bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh
#   RUNNER_NAME=prompt_agent_computer_13 LIMIT=null bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh
#   START_NG_RUN=0 bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh
#   DRY_RUN=1 bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="${GYM_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${GYM_ROOT}"

if [[ -f ".colossus-runtime.env" ]]; then
    # shellcheck disable=SC1091
    source ".colossus-runtime.env"
fi

AGENT_NAME="${AGENT_NAME:-osworld_simple_agent}"
RUNNER_NAME="${RUNNER_NAME:-prompt_agent}"
INPUT_JSONL="${INPUT_JSONL:-responses_api_agents/osworld_agent/data/example.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-results/osworld_native_prompt_agent_smoke.jsonl}"
NUM_REPEATS="${NUM_REPEATS:-1}"
LIMIT="${LIMIT:-1}"
NUM_SAMPLES_IN_PARALLEL="${NUM_SAMPLES_IN_PARALLEL:-1}"
START_NG_RUN="${START_NG_RUN:-1}"
DRY_RUN="${DRY_RUN:-0}"
NG_RUN_BIN="${NG_RUN_BIN:-ng_run}"
NG_COLLECT_BIN="${NG_COLLECT_BIN:-ng_collect_rollouts}"
NG_STATUS_BIN="${NG_STATUS_BIN:-ng_status}"
NG_RUN_WAIT_RETRIES="${NG_RUN_WAIT_RETRIES:-60}"
NG_RUN_WAIT_INTERVAL_SECONDS="${NG_RUN_WAIT_INTERVAL_SECONDS:-3}"
# ng_status lists the registered Gym servers, not the head server. This
# smoke starts osworld_simple_agent and policy_model, so the default is 2.
EXPECTED_SERVERS="${EXPECTED_SERVERS:-2}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-1.0}"
CONFIG_PATHS="${CONFIG_PATHS:-responses_api_agents/osworld_agent/configs/osworld_agent.yaml,responses_api_agents/osworld_agent/configs/osworld_agent_native_prompt_agent.yaml,responses_api_models/openai_model/configs/openai_model.yaml}"
POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-}"

mkdir -p "$(dirname "${OUTPUT_JSONL}")"

echo "=== OSWorld native PromptAgent smoke ==="
echo "root: ${GYM_ROOT}"
echo "runner: ${RUNNER_NAME}"
echo "input: ${INPUT_JSONL}"
echo "output: ${OUTPUT_JSONL}"
echo "limit: ${LIMIT}"
if [[ -n "${POLICY_MODEL_NAME}" ]]; then
    echo "policy model override: ${POLICY_MODEL_NAME}"
fi
echo "ng_run bin: ${NG_RUN_BIN}"
echo "ng_collect bin: ${NG_COLLECT_BIN}"
echo "ng_status bin: ${NG_STATUS_BIN}"
echo

ng_run_cmd=(
    "${NG_RUN_BIN}"
    "+config_paths=[${CONFIG_PATHS}]"
    "++osworld_simple_agent.responses_api_agents.osworld_agent.runner_name=${RUNNER_NAME}"
)

if [[ -n "${POLICY_MODEL_NAME}" ]]; then
    ng_run_cmd+=("++policy_model_name=${POLICY_MODEL_NAME}")
fi

collect_cmd=(
    "${NG_COLLECT_BIN}"
    "+agent_name=${AGENT_NAME}"
    "+input_jsonl_fpath=${INPUT_JSONL}"
    "+output_jsonl_fpath=${OUTPUT_JSONL}"
    "+num_repeats=${NUM_REPEATS}"
    "+num_samples_in_parallel=${NUM_SAMPLES_IN_PARALLEL}"
    "+responses_create_params={max_output_tokens: ${MAX_OUTPUT_TOKENS}, temperature: ${TEMPERATURE}}"
)

if [[ "${LIMIT}" != "null" ]]; then
    collect_cmd+=("+limit=${LIMIT}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "--- ng_run command ---"
    printf ' %q' "${ng_run_cmd[@]}"
    echo
    echo
    echo "--- ng_collect_rollouts command ---"
    printf ' %q' "${collect_cmd[@]}"
    echo
    exit 0
fi

wait_for_servers_ready() {
    local expected_line="${EXPECTED_SERVERS} servers found (${EXPECTED_SERVERS} healthy, 0 unhealthy)"
    local status_output=""

    echo "waiting for ${EXPECTED_SERVERS} healthy servers via ${NG_STATUS_BIN}"
    for attempt in $(seq 1 "${NG_RUN_WAIT_RETRIES}"); do
        if [[ -n "${ng_run_pid}" ]] && ! kill -0 "${ng_run_pid}" >/dev/null 2>&1; then
            echo "ng_run exited before servers became ready"
            return 1
        fi

        status_output="$("${NG_STATUS_BIN}" 2>&1 || true)"
        if grep -Fq "${expected_line}" <<< "${status_output}"; then
            echo "servers ready: ${expected_line}"
            return 0
        fi

        if [[ "${attempt}" == "1" || $((attempt % 5)) -eq 0 ]]; then
            echo "server readiness attempt ${attempt}/${NG_RUN_WAIT_RETRIES}"
            sed 's/^/  /' <<< "${status_output}"
        fi
        sleep "${NG_RUN_WAIT_INTERVAL_SECONDS}"
    done

    echo "servers did not become ready after ${NG_RUN_WAIT_RETRIES} attempts"
    sed 's/^/  /' <<< "${status_output}"
    return 1
}

ng_run_pid=""
cleanup() {
    if [[ -n "${ng_run_pid}" ]]; then
        echo
        echo "Stopping ng_run pid ${ng_run_pid}"
        kill "${ng_run_pid}" >/dev/null 2>&1 || true
        wait "${ng_run_pid}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "${START_NG_RUN}" == "1" ]]; then
    echo "--- starting ng_run ---"
    "${ng_run_cmd[@]}" &
    ng_run_pid="$!"
    echo "ng_run pid: ${ng_run_pid}"
    wait_for_servers_ready
else
    echo "--- using existing ng_run ---"
    echo "Expected running config:"
    echo "  config_paths=[${CONFIG_PATHS}]"
    echo "  runner_name=${RUNNER_NAME}"
fi

echo
echo "--- collecting rollouts ---"
printf ' %q' "${collect_cmd[@]}"
echo
"${collect_cmd[@]}"

echo
echo "--- smoke result summary ---"
python - <<'PY' "${OUTPUT_JSONL}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"missing output file: {path}")

for idx, line in enumerate(path.read_text().splitlines(), start=1):
    row = json.loads(line)
    meta = row.get("verifier_metadata") or {}
    print(
        json.dumps(
            {
                "row": idx,
                "task_id": meta.get("task_id"),
                "reward": row.get("reward"),
                "mask_sample": row.get("mask_sample"),
                "score": meta.get("osworld_score"),
                "finished": meta.get("osworld_finished"),
                "steps": len(meta.get("osworld_steps") or []),
                "error": meta.get("osworld_error"),
            },
            ensure_ascii=False,
        )
    )
PY
