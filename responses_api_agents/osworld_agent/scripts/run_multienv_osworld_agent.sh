#!/usr/bin/env bash
# Run OSWorld through NeMo Gym with multiple DesktopEnv instances in parallel.
#
# This is the Gym-facing equivalent of OSWorld's run_multienv_*.py scripts:
# NUM_ENVS maps to ng_collect_rollouts +num_samples_in_parallel, while ng_run
# hosts the OSWorld agent server and policy model server.
#
# Example:
#   RUNNER_NAME=prompt_agent \
#   POLICY_MODEL_NAME=nvidia/minimaxai/minimax-m3 \
#   NUM_ENVS=4 LIMIT=4 \
#   bash responses_api_agents/osworld_agent/scripts/run_multienv_osworld_agent.sh

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
RUN_TAG="${RUN_TAG:-osworld-multienv-${RUNNER_NAME}-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-results/${RUN_TAG}}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${RUN_DIR}/rollouts.jsonl}"
NUM_REPEATS="${NUM_REPEATS:-1}"
LIMIT="${LIMIT:-4}"
NUM_ENVS="${NUM_ENVS:-4}"
NUM_SAMPLES_IN_PARALLEL="${NUM_SAMPLES_IN_PARALLEL:-${NUM_ENVS}}"
START_NG_RUN="${START_NG_RUN:-1}"
DRY_RUN="${DRY_RUN:-0}"
RECORD_VIDEO="${RECORD_VIDEO:-1}"
VIDEO_SAMPLE_PER="${VIDEO_SAMPLE_PER:-100}"
VIDEO_SAMPLE_COUNT="${VIDEO_SAMPLE_COUNT:-4}"
VIDEO_SAMPLE_SEED="${VIDEO_SAMPLE_SEED:-${RUN_TAG}}"
VIDEO_SAMPLE_TASK_IDS_FILE="${VIDEO_SAMPLE_TASK_IDS_FILE:-${RUN_DIR}/video_task_ids.txt}"
NG_RUN_BIN="${NG_RUN_BIN:-ng_run}"
NG_COLLECT_BIN="${NG_COLLECT_BIN:-ng_collect_rollouts}"
NG_STATUS_BIN="${NG_STATUS_BIN:-ng_status}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NG_RUN_WAIT_RETRIES="${NG_RUN_WAIT_RETRIES:-80}"
NG_RUN_WAIT_INTERVAL_SECONDS="${NG_RUN_WAIT_INTERVAL_SECONDS:-3}"
EXPECTED_SERVERS="${EXPECTED_SERVERS:-2}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-1.0}"
CONFIG_PATHS="${CONFIG_PATHS:-responses_api_agents/osworld_agent/configs/osworld_agent.yaml,responses_api_agents/osworld_agent/configs/osworld_agent_native_prompt_agent.yaml,responses_api_models/openai_model/configs/openai_model.yaml}"
POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-}"
MAX_STEPS="${MAX_STEPS:-}"

mkdir -p "${RUN_DIR}" "$(dirname "${OUTPUT_JSONL}")"

if [[ "${RECORD_VIDEO}" == "1" || "${RECORD_VIDEO}" == "sample" ]]; then
    export OSWORLD_RECORD_VIDEO_DIR="${OSWORLD_RECORD_VIDEO_DIR:-${RUN_DIR}/videos}"
    mkdir -p "${OSWORLD_RECORD_VIDEO_DIR}"
fi

if [[ "${RECORD_VIDEO}" == "sample" ]]; then
    export OSWORLD_RECORD_VIDEO_TASK_IDS_FILE="${VIDEO_SAMPLE_TASK_IDS_FILE}"
    "${PYTHON_BIN}" - <<'PY' "${INPUT_JSONL}" "${LIMIT}" "${VIDEO_SAMPLE_PER}" "${VIDEO_SAMPLE_COUNT}" "${VIDEO_SAMPLE_SEED}" "${VIDEO_SAMPLE_TASK_IDS_FILE}" "${RUN_DIR}/video_sample_manifest.json"
import json
import random
import sys
from pathlib import Path

input_jsonl, limit_s, per_s, count_s, seed, out_txt, out_json = sys.argv[1:]
per = int(per_s)
count = int(count_s)
if per <= 0:
    raise SystemExit("VIDEO_SAMPLE_PER must be > 0")
if count < 0:
    raise SystemExit("VIDEO_SAMPLE_COUNT must be >= 0")

rows = []
with open(input_jsonl, encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            rows.append(json.loads(line))

if limit_s != "null":
    rows = rows[: int(limit_s)]

rng = random.Random(seed)
selected = []
blocks = []
for start in range(0, len(rows), per):
    block = rows[start : start + per]
    indexed_ids = []
    for offset, row in enumerate(block):
        metadata = row.get("verifier_metadata") or {}
        task = metadata.get("osworld_task") or {}
        task_id = metadata.get("task_id") or task.get("id") or task.get("task_id")
        if task_id:
            indexed_ids.append((start + offset, str(task_id)))
    picked = rng.sample(indexed_ids, min(count, len(indexed_ids))) if indexed_ids else []
    picked.sort(key=lambda item: item[0])
    selected.extend(task_id for _idx, task_id in picked)
    blocks.append(
        {
            "start_row": start + 1,
            "end_row": start + len(block),
            "eligible": len(indexed_ids),
            "selected": [task_id for _idx, task_id in picked],
        }
    )

out_txt_path = Path(out_txt)
out_txt_path.parent.mkdir(parents=True, exist_ok=True)
out_txt_path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")

Path(out_json).write_text(
    json.dumps(
        {
            "input_jsonl": input_jsonl,
            "limit": None if limit_s == "null" else int(limit_s),
            "sample_per": per,
            "sample_count": count,
            "seed": seed,
            "total_rows": len(rows),
            "selected_count": len(selected),
            "task_ids_file": str(out_txt_path),
            "blocks": blocks,
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)
print(f"video sample task ids: {out_txt_path} ({len(selected)} selected)")
PY
fi

cat > "${RUN_DIR}/run.env" <<EOF
RUN_TAG=${RUN_TAG}
RUN_DIR=${RUN_DIR}
INPUT_JSONL=${INPUT_JSONL}
OUTPUT_JSONL=${OUTPUT_JSONL}
RUNNER_NAME=${RUNNER_NAME}
POLICY_MODEL_NAME=${POLICY_MODEL_NAME}
LIMIT=${LIMIT}
NUM_ENVS=${NUM_ENVS}
NUM_SAMPLES_IN_PARALLEL=${NUM_SAMPLES_IN_PARALLEL}
NUM_REPEATS=${NUM_REPEATS}
MAX_STEPS=${MAX_STEPS}
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS}
TEMPERATURE=${TEMPERATURE}
RECORD_VIDEO=${RECORD_VIDEO}
OSWORLD_RECORD_VIDEO_DIR=${OSWORLD_RECORD_VIDEO_DIR:-}
VIDEO_SAMPLE_PER=${VIDEO_SAMPLE_PER}
VIDEO_SAMPLE_COUNT=${VIDEO_SAMPLE_COUNT}
VIDEO_SAMPLE_SEED=${VIDEO_SAMPLE_SEED}
OSWORLD_RECORD_VIDEO_TASK_IDS_FILE=${OSWORLD_RECORD_VIDEO_TASK_IDS_FILE:-}
EOF

echo "=== OSWorld multienv agent run ==="
echo "root:        ${GYM_ROOT}"
echo "run tag:     ${RUN_TAG}"
echo "run dir:     ${RUN_DIR}"
echo "runner:      ${RUNNER_NAME}"
echo "input:       ${INPUT_JSONL}"
echo "output:      ${OUTPUT_JSONL}"
echo "limit:       ${LIMIT}"
echo "num envs:    ${NUM_ENVS}"
echo "parallel:    ${NUM_SAMPLES_IN_PARALLEL}"
if [[ -n "${MAX_STEPS}" ]]; then
    echo "max steps:   ${MAX_STEPS}"
fi
if [[ -n "${POLICY_MODEL_NAME}" ]]; then
    echo "model:       ${POLICY_MODEL_NAME}"
fi
if [[ "${RECORD_VIDEO}" == "1" || "${RECORD_VIDEO}" == "sample" ]]; then
    echo "video dir:   ${OSWORLD_RECORD_VIDEO_DIR}"
fi
if [[ "${RECORD_VIDEO}" == "sample" ]]; then
    echo "video sample:${OSWORLD_RECORD_VIDEO_TASK_IDS_FILE} (${VIDEO_SAMPLE_COUNT}/${VIDEO_SAMPLE_PER}, seed=${VIDEO_SAMPLE_SEED})"
fi
echo

ng_run_cmd=(
    "${NG_RUN_BIN}"
    "+config_paths=[${CONFIG_PATHS}]"
    "++osworld_simple_agent.responses_api_agents.osworld_agent.runner_name=${RUNNER_NAME}"
)

if [[ -n "${POLICY_MODEL_NAME}" ]]; then
    ng_run_cmd+=("++policy_model_name=${POLICY_MODEL_NAME}")
fi

if [[ -n "${MAX_STEPS}" ]]; then
    ng_run_cmd+=("++osworld_simple_agent.responses_api_agents.osworld_agent.max_steps=${MAX_STEPS}")
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

echo "--- ng_run command ---"
printf ' %q' "${ng_run_cmd[@]}"
echo
echo
echo "--- ng_collect_rollouts command ---"
printf ' %q' "${collect_cmd[@]}"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

wait_for_servers_ready() {
    local expected_line="${EXPECTED_SERVERS} servers found (${EXPECTED_SERVERS} healthy, 0 unhealthy)"
    local status_output=""

    echo
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

        if [[ "${attempt}" == "1" || $((attempt % 10)) -eq 0 ]]; then
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
    echo
    echo "--- starting ng_run ---"
    "${ng_run_cmd[@]}" &
    ng_run_pid="$!"
    echo "ng_run pid: ${ng_run_pid}"
    wait_for_servers_ready
else
    echo
    echo "--- using existing ng_run ---"
fi

echo
echo "--- collecting rollouts ---"
"${collect_cmd[@]}"

echo
echo "--- multienv result summary ---"
"${PYTHON_BIN}" - <<'PY' "${OUTPUT_JSONL}"
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

echo
echo "Run dir: ${RUN_DIR}"
if [[ "${RECORD_VIDEO}" == "1" || "${RECORD_VIDEO}" == "sample" ]]; then
    echo "Videos:  ${OSWORLD_RECORD_VIDEO_DIR}"
fi
if [[ "${RECORD_VIDEO}" == "sample" ]]; then
    echo "Video sample task ids: ${OSWORLD_RECORD_VIDEO_TASK_IDS_FILE}"
fi
