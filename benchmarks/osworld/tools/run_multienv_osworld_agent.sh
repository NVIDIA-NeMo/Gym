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
#   bash benchmarks/osworld/tools/run_multienv_osworld_agent.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="${GYM_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${GYM_ROOT}"
GYM_ROOT="$(pwd -P)"

# ng_run launches component servers from non-interactive Ray subprocesses.
# Those processes do not necessarily inherit shell startup files, so a user
# install such as ~/.local/bin/uv can exist while the servers still fail with
# `uv: command not found`. Resolve it once here and export its directory.
requested_uv_bin="${UV_BIN:-}"
resolved_uv_bin=""
if [[ -n "${requested_uv_bin}" ]]; then
    if [[ ! -x "${requested_uv_bin}" ]]; then
        echo "UV_BIN is not executable: ${requested_uv_bin}" >&2
        exit 2
    fi
    resolved_uv_bin="$(cd "$(dirname "${requested_uv_bin}")" && pwd)/$(basename "${requested_uv_bin}")"
elif command -v uv >/dev/null 2>&1; then
    resolved_uv_bin="$(command -v uv)"
else
    for candidate in "${HOME}/.local/bin/uv" "${HOME}/.cargo/bin/uv"; do
        if [[ -x "${candidate}" ]]; then
            resolved_uv_bin="${candidate}"
            break
        fi
    done
fi
if [[ -z "${resolved_uv_bin}" ]]; then
    echo "uv is required but was not found; install it or set UV_BIN" >&2
    exit 2
fi
UV_BIN="${resolved_uv_bin}"
export UV_BIN
export PATH="$(dirname "${UV_BIN}"):${PATH}"

gym_absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "${GYM_ROOT}" "$1" ;;
    esac
}

AGENT_NAME="${AGENT_NAME:-osworld_simple_agent}"
RUNNER_NAME="${RUNNER_NAME:-prompt_agent}"
INPUT_JSONL="${INPUT_JSONL:-benchmarks/osworld/data/example.jsonl}"
RUN_TAG="${RUN_TAG:-osworld-multienv-${RUNNER_NAME}-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-results/${RUN_TAG}}"
RUN_DIR="$(gym_absolute_path "${RUN_DIR}")"
export RUN_DIR
OUTPUT_JSONL="${OUTPUT_JSONL:-${RUN_DIR}/rollouts.jsonl}"
OUTPUT_JSONL="$(gym_absolute_path "${OUTPUT_JSONL}")"
NUM_REPEATS="${NUM_REPEATS:-1}"
LIMIT="${LIMIT:-4}"
EXPECTED_INPUT_ROWS="${EXPECTED_INPUT_ROWS:-}"
NUM_ENVS="${NUM_ENVS:-4}"
NUM_SAMPLES_IN_PARALLEL="${NUM_SAMPLES_IN_PARALLEL:-${NUM_ENVS}}"
RESUME_FROM_CACHE="${RESUME_FROM_CACHE:-0}"
START_NG_RUN="${START_NG_RUN:-1}"
DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
RECORD_VIDEO="${RECORD_VIDEO:-1}"
TASK_ARTIFACTS="${TASK_ARTIFACTS:-1}"
FULL_MODEL_IO="${FULL_MODEL_IO:-0}"
OSWORLD_ENABLE_PROXY="${OSWORLD_ENABLE_PROXY:-0}"
PROXY_CONFIG_FILE="${PROXY_CONFIG_FILE:-}"
TASK_ARTIFACT_ROOT="${TASK_ARTIFACT_ROOT:-}"
VIDEO_SAMPLE_PER="${VIDEO_SAMPLE_PER:-100}"
VIDEO_SAMPLE_COUNT="${VIDEO_SAMPLE_COUNT:-4}"
VIDEO_SAMPLE_SEED="${VIDEO_SAMPLE_SEED:-${RUN_TAG}}"
VIDEO_SAMPLE_TASK_IDS_FILE="${VIDEO_SAMPLE_TASK_IDS_FILE:-}"
NG_RUN_BIN="${NG_RUN_BIN:-ng_run}"
NG_COLLECT_BIN="${NG_COLLECT_BIN:-ng_collect_rollouts}"
NG_STATUS_BIN="${NG_STATUS_BIN:-ng_status}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NG_RUN_WAIT_RETRIES="${NG_RUN_WAIT_RETRIES:-80}"
NG_RUN_WAIT_INTERVAL_SECONDS="${NG_RUN_WAIT_INTERVAL_SECONDS:-3}"
EXPECTED_SERVERS="${EXPECTED_SERVERS:-2}"
NEMO_GYM_HEAD_HOST="${NEMO_GYM_HEAD_HOST:-127.0.0.1}"
NEMO_GYM_HEAD_PORT="${NEMO_GYM_HEAD_PORT:-11000}"
if [[ -z "${NEMO_GYM_HEAD_HOST}" ]]; then
    echo "NEMO_GYM_HEAD_HOST must not be empty" >&2
    exit 2
fi
if [[ ! "${NEMO_GYM_HEAD_PORT}" =~ ^[1-9][0-9]*$ ]] || (( NEMO_GYM_HEAD_PORT > 65535 )); then
    echo "NEMO_GYM_HEAD_PORT must be an integer between 1 and 65535" >&2
    exit 2
fi
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-1.0}"
CONFIG_PATHS="${CONFIG_PATHS:-responses_api_agents/osworld_agent/configs/osworld_agent.yaml,benchmarks/osworld/configs/osworld_agent_native_prompt_agent.yaml,responses_api_models/openai_model/configs/openai_model.yaml}"
OSWORLD_EXECUTION_BACKEND="${OSWORLD_EXECUTION_BACKEND:-osworld_provider}"
OSWORLD_SANDBOX_VM_PATH="${OSWORLD_SANDBOX_VM_PATH:-}"
POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-}"
MAX_STEPS="${MAX_STEPS:-}"
TASK_PARITY_REFERENCE_INPUT="${TASK_PARITY_REFERENCE_INPUT:-}"
TASK_PARITY_IDS_FILE="${TASK_PARITY_IDS_FILE:-}"
TASK_PARITY_REPORT="${TASK_PARITY_REPORT:-}"
RUN_ATTEMPT_ID="${RUN_ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}}"
if [[ ! "${RUN_ATTEMPT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "RUN_ATTEMPT_ID contains unsupported characters: ${RUN_ATTEMPT_ID}" >&2
    exit 2
fi
SERVER_VENV_ROOT="${SERVER_VENV_ROOT:-}"
if [[ -n "${SERVER_VENV_ROOT}" ]]; then
    SERVER_VENV_ROOT="$(gym_absolute_path "${SERVER_VENV_ROOT}")"
fi

case "${OSWORLD_EXECUTION_BACKEND}" in
    osworld_provider)
        ;;
    gym_sandbox)
        if [[ -z "${OSWORLD_SANDBOX_VM_PATH}" ]]; then
            echo "OSWORLD_SANDBOX_VM_PATH is required for OSWORLD_EXECUTION_BACKEND=gym_sandbox" >&2
            exit 2
        fi
        OSWORLD_SANDBOX_VM_PATH="$(gym_absolute_path "${OSWORLD_SANDBOX_VM_PATH}")"
        if [[ ! -r "${OSWORLD_SANDBOX_VM_PATH}" || ! -f "${OSWORLD_SANDBOX_VM_PATH}" ]]; then
            echo "OSWorld Sandbox qcow2 is not a readable file: ${OSWORLD_SANDBOX_VM_PATH}" >&2
            exit 2
        fi
        if [[ ! -c /dev/kvm || ! -r /dev/kvm || ! -w /dev/kvm ]]; then
            echo "OSWorld Gym Sandbox requires readable/writable /dev/kvm" >&2
            exit 2
        fi
        sandbox_config="benchmarks/osworld/configs/osworld_sandbox.yaml"
        case ",${CONFIG_PATHS}," in
            *",${sandbox_config},"*) ;;
            *) CONFIG_PATHS="${CONFIG_PATHS},${sandbox_config}" ;;
        esac
        ;;
    *)
        echo "OSWORLD_EXECUTION_BACKEND must be osworld_provider or gym_sandbox, got: ${OSWORLD_EXECUTION_BACKEND}" >&2
        exit 2
        ;;
esac
export OSWORLD_EXECUTION_BACKEND OSWORLD_SANDBOX_VM_PATH

# A fresh invocation owns a fresh run directory. An explicit cache resume is
# the only supported exception: it must target a terminal, failed attempt with
# both files that Gym needs to verify its cache. Resume control records live in
# a new subdirectory so prior manifests and terminal markers remain untouched.
output_filename="$(basename "${OUTPUT_JSONL}")"
output_stem="${output_filename%.*}"
MATERIALIZED_INPUT_JSONL="$(dirname "${OUTPUT_JSONL}")/${output_stem}_materialized_inputs.jsonl"
if [[ -e "${RUN_DIR}" ]]; then
    if [[ "${RESUME_FROM_CACHE}" != "1" ]]; then
        echo "immutable run already exists: ${RUN_DIR}" >&2
        exit 2
    fi
    if [[ ! -d "${RUN_DIR}" || -L "${RUN_DIR}" ]]; then
        echo "resume target must be a real directory, not a file or symlink: ${RUN_DIR}" >&2
        exit 2
    fi
    if [[ ! -f "${OUTPUT_JSONL}" || ! -f "${MATERIALIZED_INPUT_JSONL}" ]]; then
        echo "resume requires existing output and materialized input JSONL files" >&2
        echo "  output:       ${OUTPUT_JSONL}" >&2
        echo "  materialized: ${MATERIALIZED_INPUT_JSONL}" >&2
        exit 2
    fi
    if [[ ! -f "${RUN_DIR}/finished_at.txt" || ! -f "${RUN_DIR}/exit_code.txt" ]]; then
        echo "resume target has no complete terminal lifecycle: ${RUN_DIR}" >&2
        exit 2
    fi
    previous_exit_code="$(<"${RUN_DIR}/exit_code.txt")"
    if [[ ! "${previous_exit_code}" =~ ^[0-9]+$ ]]; then
        echo "resume target has an invalid exit code: ${previous_exit_code}" >&2
        exit 2
    fi
    completed_attempt=0
    if [[ "${previous_exit_code}" == "0" ]]; then
        completed_attempt=1
    fi
    for previous_attempt in "${RUN_DIR}"/resume-attempts/*; do
        [[ -d "${previous_attempt}" ]] || continue
        if [[ ! -f "${previous_attempt}/finished_at.txt" || ! -f "${previous_attempt}/exit_code.txt" ]]; then
            echo "resume target contains a non-terminal attempt: ${previous_attempt}" >&2
            exit 2
        fi
        previous_exit_code="$(<"${previous_attempt}/exit_code.txt")"
        if [[ ! "${previous_exit_code}" =~ ^[0-9]+$ ]]; then
            echo "resume attempt has an invalid exit code: ${previous_attempt}" >&2
            exit 2
        fi
        if [[ "${previous_exit_code}" == "0" ]]; then
            completed_attempt=1
        fi
    done
    if [[ "${completed_attempt}" == "1" ]]; then
        echo "refusing to resume a successfully completed run: ${RUN_DIR}" >&2
        exit 2
    fi
    mkdir -p "${RUN_DIR}/resume-attempts"
    RUN_LIFECYCLE_DIR="${RUN_DIR}/resume-attempts/${RUN_ATTEMPT_ID}"
    if ! mkdir "${RUN_LIFECYCLE_DIR}"; then
        echo "resume attempt already exists: ${RUN_LIFECYCLE_DIR}" >&2
        exit 2
    fi
else
    if [[ "${RESUME_FROM_CACHE}" == "1" ]]; then
        echo "RESUME_FROM_CACHE=1 requires an existing run directory: ${RUN_DIR}" >&2
        exit 2
    fi
    mkdir -p "$(dirname "${RUN_DIR}")"
    if ! mkdir "${RUN_DIR}"; then
        echo "failed to create immutable run directory: ${RUN_DIR}" >&2
        exit 2
    fi
    RUN_LIFECYCLE_DIR="${RUN_DIR}"
fi
export RUN_LIFECYCLE_DIR

ng_run_pid=""
finalize_run() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ -n "${ng_run_pid}" ]]; then
        echo
        echo "Stopping ng_run pid ${ng_run_pid}"
        kill "${ng_run_pid}" >/dev/null 2>&1 || true
        wait "${ng_run_pid}" >/dev/null 2>&1 || true
    fi
    date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_LIFECYCLE_DIR}/finished_at.txt" || true
    printf '%s\n' "${rc}" > "${RUN_LIFECYCLE_DIR}/exit_code.txt" || true
    exit "${rc}"
}
trap finalize_run EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_LIFECYCLE_DIR}/started_at.txt"
printf '%s\n' "$$" > "${RUN_LIFECYCLE_DIR}/launcher.pid"

case "${OSWORLD_ENABLE_PROXY}" in
    0|1) ;;
    *)
        echo "OSWORLD_ENABLE_PROXY must be exactly 0 or 1, got: ${OSWORLD_ENABLE_PROXY}" >&2
        exit 2
        ;;
esac

PROXY_CONFIG_CONFIGURED=0
PROXY_CONFIG_SHA256=""
PROXY_CONFIG_ENTRY_COUNT=0
if [[ -n "${PROXY_CONFIG_FILE}" ]]; then
    PROXY_CONFIG_FILE="$(gym_absolute_path "${PROXY_CONFIG_FILE}")"
    PROXY_CONFIG_CONFIGURED=1
fi
if [[ "${OSWORLD_ENABLE_PROXY}" == "1" ]]; then
    if [[ -z "${PROXY_CONFIG_FILE}" ]]; then
        echo "PROXY_CONFIG_FILE is required when OSWORLD_ENABLE_PROXY=1" >&2
        exit 2
    fi
    proxy_inspection="$("${PYTHON_BIN}" "${GYM_ROOT}/responses_api_agents/osworld_agent/proxy.py" "${PROXY_CONFIG_FILE}")" || exit $?
    IFS=$'\t' read -r PROXY_CONFIG_SHA256 PROXY_CONFIG_ENTRY_COUNT PROXY_CONFIG_FILE <<< "${proxy_inspection}"
fi
export OSWORLD_ENABLE_PROXY PROXY_CONFIG_FILE

TASK_ARTIFACT_ROOT="${TASK_ARTIFACT_ROOT:-${RUN_LIFECYCLE_DIR}/task-artifacts}"
VIDEO_SAMPLE_TASK_IDS_FILE="${VIDEO_SAMPLE_TASK_IDS_FILE:-${RUN_LIFECYCLE_DIR}/video_task_ids.txt}"
TASK_PARITY_REPORT="${TASK_PARITY_REPORT:-${RUN_LIFECYCLE_DIR}/task-input-parity.json}"

# Ray appends a long session/sockets suffix to its temp root. Linux limits
# AF_UNIX socket paths to 107 bytes, so a descriptive absolute run directory
# can otherwise fail before either Gym server starts. Keep an explicit short,
# per-run temp root and replace only caller-provided values that cannot fit.
RAY_TMPDIR_REQUESTED="${RAY_TMPDIR:-}"
ray_run_key="$(printf '%s' "${RUN_TAG}-${RUN_ATTEMPT_ID}" | sha256sum | cut -c1-12)"
short_ray_tmpdir="/tmp/ngray-${ray_run_key}"
if [[ -n "${RAY_TMPDIR_REQUESTED}" ]]; then
    ray_socket_probe="${RAY_TMPDIR_REQUESTED%/}/ray/session_2099-12-31_23-59-59_999999_999999/sockets/plasma_store"
    if (( ${#ray_socket_probe} > 107 )); then
        echo "WARNING: RAY_TMPDIR is too long for Ray AF_UNIX sockets; using ${short_ray_tmpdir} instead of ${RAY_TMPDIR_REQUESTED}" >&2
        RAY_TMPDIR="${short_ray_tmpdir}"
    else
        RAY_TMPDIR="${RAY_TMPDIR_REQUESTED}"
    fi
else
    RAY_TMPDIR="${short_ray_tmpdir}"
fi
export RAY_TMPDIR

mkdir -p "$(dirname "${OUTPUT_JSONL}")" "${RAY_TMPDIR}"
if [[ -n "${SERVER_VENV_ROOT}" ]]; then
    mkdir -p "${SERVER_VENV_ROOT}"
fi

if [[ -n "${TASK_PARITY_REFERENCE_INPUT}" ]]; then
    parity_cmd=(
        "${PYTHON_BIN}"
        "${SCRIPT_DIR}/check_task_input_parity.py"
        "${TASK_PARITY_REFERENCE_INPUT}"
        "${INPUT_JSONL}"
        "--output"
        "${TASK_PARITY_REPORT}"
    )
    if [[ -n "${TASK_PARITY_IDS_FILE}" ]]; then
        parity_cmd+=("--task-ids" "${TASK_PARITY_IDS_FILE}")
    fi
    echo "--- OSWorld task-definition parity preflight ---"
    "${parity_cmd[@]}"
fi

if [[ "${FULL_MODEL_IO}" == "1" ]]; then
    OSWORLD_MODEL_IO_LOG="$(gym_absolute_path "${OSWORLD_MODEL_IO_LOG:-${RUN_LIFECYCLE_DIR}/model-io-agent.jsonl}")"
    NEMO_GYM_VLLM_TRANSPORT_LOG="$(gym_absolute_path "${NEMO_GYM_VLLM_TRANSPORT_LOG:-${OSWORLD_TRANSPORT_IO_LOG:-${RUN_LIFECYCLE_DIR}/model-io-transport.jsonl}}")"
    # Backward-compatible launcher variable; vllm_model consumes only the
    # provider-neutral NEMO_GYM_VLLM_TRANSPORT_LOG name.
    OSWORLD_TRANSPORT_IO_LOG="${NEMO_GYM_VLLM_TRANSPORT_LOG}"
    OSWORLD_VM_EXEC_LOG="$(gym_absolute_path "${OSWORLD_VM_EXEC_LOG:-${RUN_LIFECYCLE_DIR}/vm-exec.jsonl}")"
    export OSWORLD_MODEL_IO_LOG OSWORLD_TRANSPORT_IO_LOG NEMO_GYM_VLLM_TRANSPORT_LOG OSWORLD_VM_EXEC_LOG
    mkdir -p \
        "$(dirname "${OSWORLD_MODEL_IO_LOG}")" \
        "$(dirname "${NEMO_GYM_VLLM_TRANSPORT_LOG}")" \
        "$(dirname "${OSWORLD_VM_EXEC_LOG}")"
fi

if [[ "${TASK_ARTIFACTS}" == "1" ]]; then
    export OSWORLD_TASK_ARTIFACT_ROOT="${OSWORLD_TASK_ARTIFACT_ROOT:-${TASK_ARTIFACT_ROOT}}"
    mkdir -p "${OSWORLD_TASK_ARTIFACT_ROOT}"
else
    unset OSWORLD_TASK_ARTIFACT_ROOT
fi

if [[ "${RECORD_VIDEO}" == "1" || "${RECORD_VIDEO}" == "sample" ]]; then
    export OSWORLD_RECORD_VIDEO_DIR="${OSWORLD_RECORD_VIDEO_DIR:-${RUN_LIFECYCLE_DIR}/videos}"
    mkdir -p "${OSWORLD_RECORD_VIDEO_DIR}"
fi

if [[ "${RECORD_VIDEO}" == "sample" ]]; then
    export OSWORLD_RECORD_VIDEO_TASK_IDS_FILE="${VIDEO_SAMPLE_TASK_IDS_FILE}"
    "${PYTHON_BIN}" - <<'PY' "${INPUT_JSONL}" "${LIMIT}" "${VIDEO_SAMPLE_PER}" "${VIDEO_SAMPLE_COUNT}" "${VIDEO_SAMPLE_SEED}" "${VIDEO_SAMPLE_TASK_IDS_FILE}" "${RUN_LIFECYCLE_DIR}/video_sample_manifest.json"
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

cat > "${RUN_LIFECYCLE_DIR}/run.env" <<EOF
RUN_TAG=${RUN_TAG}
RUN_DIR=${RUN_DIR}
RUN_ATTEMPT_ID=${RUN_ATTEMPT_ID}
RUN_LIFECYCLE_DIR=${RUN_LIFECYCLE_DIR}
INPUT_JSONL=${INPUT_JSONL}
OUTPUT_JSONL=${OUTPUT_JSONL}
MATERIALIZED_INPUT_JSONL=${MATERIALIZED_INPUT_JSONL}
RUNNER_NAME=${RUNNER_NAME}
POLICY_MODEL_NAME=${POLICY_MODEL_NAME}
LIMIT=${LIMIT}
NUM_ENVS=${NUM_ENVS}
NUM_SAMPLES_IN_PARALLEL=${NUM_SAMPLES_IN_PARALLEL}
RESUME_FROM_CACHE=${RESUME_FROM_CACHE}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY}
NUM_REPEATS=${NUM_REPEATS}
MAX_STEPS=${MAX_STEPS}
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS}
TEMPERATURE=${TEMPERATURE}
OSWORLD_EXECUTION_BACKEND=${OSWORLD_EXECUTION_BACKEND}
OSWORLD_SANDBOX_VM_PATH=${OSWORLD_SANDBOX_VM_PATH}
RECORD_VIDEO=${RECORD_VIDEO}
TASK_ARTIFACTS=${TASK_ARTIFACTS}
FULL_MODEL_IO=${FULL_MODEL_IO}
OSWORLD_ENABLE_PROXY=${OSWORLD_ENABLE_PROXY}
PROXY_CONFIG_FILE=${PROXY_CONFIG_FILE}
PROXY_CONFIG_CONFIGURED=${PROXY_CONFIG_CONFIGURED}
PROXY_CONFIG_SHA256=${PROXY_CONFIG_SHA256}
PROXY_CONFIG_ENTRY_COUNT=${PROXY_CONFIG_ENTRY_COUNT}
OSWORLD_TASK_ARTIFACT_ROOT=${OSWORLD_TASK_ARTIFACT_ROOT:-}
OSWORLD_RECORD_VIDEO_DIR=${OSWORLD_RECORD_VIDEO_DIR:-}
OSWORLD_MODEL_IO_LOG=${OSWORLD_MODEL_IO_LOG:-}
OSWORLD_TRANSPORT_IO_LOG=${OSWORLD_TRANSPORT_IO_LOG:-}
NEMO_GYM_VLLM_TRANSPORT_LOG=${NEMO_GYM_VLLM_TRANSPORT_LOG:-}
OSWORLD_VM_EXEC_LOG=${OSWORLD_VM_EXEC_LOG:-}
TASK_PARITY_REFERENCE_INPUT=${TASK_PARITY_REFERENCE_INPUT}
TASK_PARITY_IDS_FILE=${TASK_PARITY_IDS_FILE}
TASK_PARITY_REPORT=${TASK_PARITY_REPORT}
RAY_TMPDIR=${RAY_TMPDIR}
RAY_TMPDIR_REQUESTED=${RAY_TMPDIR_REQUESTED}
SERVER_VENV_ROOT=${SERVER_VENV_ROOT}
UV_BIN=${UV_BIN}
NEMO_GYM_HEAD_HOST=${NEMO_GYM_HEAD_HOST}
NEMO_GYM_HEAD_PORT=${NEMO_GYM_HEAD_PORT}
VIDEO_SAMPLE_PER=${VIDEO_SAMPLE_PER}
VIDEO_SAMPLE_COUNT=${VIDEO_SAMPLE_COUNT}
VIDEO_SAMPLE_SEED=${VIDEO_SAMPLE_SEED}
OSWORLD_RECORD_VIDEO_TASK_IDS_FILE=${OSWORLD_RECORD_VIDEO_TASK_IDS_FILE:-}
EOF

echo "=== OSWorld multienv agent run ==="
echo "root:        ${GYM_ROOT}"
echo "run tag:     ${RUN_TAG}"
echo "run dir:     ${RUN_DIR}"
echo "attempt:     ${RUN_ATTEMPT_ID}"
echo "lifecycle:   ${RUN_LIFECYCLE_DIR}"
echo "runner:      ${RUNNER_NAME}"
echo "input:       ${INPUT_JSONL}"
echo "output:      ${OUTPUT_JSONL}"
echo "limit:       ${LIMIT}"
echo "num envs:    ${NUM_ENVS}"
echo "parallel:    ${NUM_SAMPLES_IN_PARALLEL}"
echo "env backend: ${OSWORLD_EXECUTION_BACKEND}"
if [[ "${OSWORLD_EXECUTION_BACKEND}" == "gym_sandbox" ]]; then
    echo "VM base:     ${OSWORLD_SANDBOX_VM_PATH}"
fi
echo "proxy:      enabled=${OSWORLD_ENABLE_PROXY} configured=${PROXY_CONFIG_CONFIGURED} entries=${PROXY_CONFIG_ENTRY_COUNT}"
echo "resume:      ${RESUME_FROM_CACHE}"
echo "ray tmp:     ${RAY_TMPDIR}"
if [[ -n "${SERVER_VENV_ROOT}" ]]; then
    echo "server venv: ${SERVER_VENV_ROOT}"
fi
echo "uv:          ${UV_BIN}"
echo "head server: ${NEMO_GYM_HEAD_HOST}:${NEMO_GYM_HEAD_PORT}"
if [[ -n "${MAX_STEPS}" ]]; then
    echo "max steps:   ${MAX_STEPS}"
fi
if [[ -n "${POLICY_MODEL_NAME}" ]]; then
    echo "model:       ${POLICY_MODEL_NAME}"
fi
if [[ "${RECORD_VIDEO}" == "1" || "${RECORD_VIDEO}" == "sample" ]]; then
    echo "video dir:   ${OSWORLD_RECORD_VIDEO_DIR}"
fi
if [[ "${TASK_ARTIFACTS}" == "1" ]]; then
    echo "task logs:   ${OSWORLD_TASK_ARTIFACT_ROOT}"
fi
if [[ "${FULL_MODEL_IO}" == "1" ]]; then
    echo "agent I/O:   ${OSWORLD_MODEL_IO_LOG}"
    echo "transport:   ${NEMO_GYM_VLLM_TRANSPORT_LOG}"
    echo "VM exec:     ${OSWORLD_VM_EXEC_LOG}"
fi
if [[ -n "${TASK_PARITY_REFERENCE_INPUT}" ]]; then
    echo "parity ref:  ${TASK_PARITY_REFERENCE_INPUT}"
    echo "parity report: ${TASK_PARITY_REPORT}"
fi
if [[ "${RECORD_VIDEO}" == "sample" ]]; then
    echo "video sample:${OSWORLD_RECORD_VIDEO_TASK_IDS_FILE} (${VIDEO_SAMPLE_COUNT}/${VIDEO_SAMPLE_PER}, seed=${VIDEO_SAMPLE_SEED})"
fi
echo

ng_run_cmd=(
    "${NG_RUN_BIN}"
    "+config_paths=[${CONFIG_PATHS}]"
    "++head_server.host=${NEMO_GYM_HEAD_HOST}"
    "++head_server.port=${NEMO_GYM_HEAD_PORT}"
    "++osworld_simple_agent.responses_api_agents.osworld_agent.runner_name=${RUNNER_NAME}"
)

if [[ "${OSWORLD_ENABLE_PROXY}" == "1" ]]; then
    proxy_hydra_value=true
else
    proxy_hydra_value=false
fi
ng_run_cmd+=("++osworld_simple_agent.responses_api_agents.osworld_agent.enable_proxy=${proxy_hydra_value}")
if [[ -n "${PROXY_CONFIG_FILE}" ]]; then
    ng_run_cmd+=("++osworld_simple_agent.responses_api_agents.osworld_agent.proxy_config_file=${PROXY_CONFIG_FILE}")
fi

if [[ -n "${POLICY_MODEL_NAME}" ]]; then
    ng_run_cmd+=("++policy_model_name=${POLICY_MODEL_NAME}")
fi

if [[ -n "${MAX_STEPS}" ]]; then
    ng_run_cmd+=("++osworld_simple_agent.responses_api_agents.osworld_agent.max_steps=${MAX_STEPS}")
fi

if [[ -n "${SERVER_VENV_ROOT}" ]]; then
    ng_run_cmd+=("++uv_venv_dir=${SERVER_VENV_ROOT}")
fi

collect_cmd=(
    "${NG_COLLECT_BIN}"
    "++head_server.host=${NEMO_GYM_HEAD_HOST}"
    "++head_server.port=${NEMO_GYM_HEAD_PORT}"
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

if [[ "${RESUME_FROM_CACHE}" == "1" ]]; then
    collect_cmd+=("+resume_from_cache=true")
fi

{
    echo "--- ng_run command ---"
    printf ' %q' "${ng_run_cmd[@]}"
    echo
    echo
    echo "--- ng_collect_rollouts command ---"
    printf ' %q' "${collect_cmd[@]}"
    echo
} > "${RUN_LIFECYCLE_DIR}/resolved-command.log"
cat "${RUN_LIFECYCLE_DIR}/resolved-command.log"

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

preflight_cmd=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/preflight_osworld_run.py"
    "--config-paths"
    "${CONFIG_PATHS}"
    "--input-jsonl"
    "${INPUT_JSONL}"
    "--runner-name"
    "${RUNNER_NAME}"
)
if [[ -n "${EXPECTED_INPUT_ROWS}" ]]; then
    preflight_cmd+=("--expected-rows" "${EXPECTED_INPUT_ROWS}")
fi

echo
echo "--- OSWorld runtime preflight ---"
"${preflight_cmd[@]}"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
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

        status_output="$("${NG_STATUS_BIN}" "++head_server.host=${NEMO_GYM_HEAD_HOST}" "++head_server.port=${NEMO_GYM_HEAD_PORT}" 2>&1 || true)"
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
echo "Lifecycle: ${RUN_LIFECYCLE_DIR}"
if [[ "${TASK_ARTIFACTS}" == "1" ]]; then
    echo "Task artifacts: ${OSWORLD_TASK_ARTIFACT_ROOT}"
fi
if [[ "${RECORD_VIDEO}" == "1" || "${RECORD_VIDEO}" == "sample" ]]; then
    echo "Videos:  ${OSWORLD_RECORD_VIDEO_DIR}"
fi
if [[ "${RECORD_VIDEO}" == "sample" ]]; then
    echo "Video sample task ids: ${OSWORLD_RECORD_VIDEO_TASK_IDS_FILE}"
fi
