#!/usr/bin/env bash
set -euo pipefail

# Submit and validate the dedicated GLM-5.2 service used by GDPVal rollout
# batches. Dataset-specific values come from a profile in datasets/.

umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STAGING_ROOT="${GDPVAL_HSG_STAGING_ROOT:-${SCRATCH:-$HOME}/gdpval-rollouts}"
LAUNCHER="${GDPVAL_GLM52_LAUNCHER:-${SCRIPT_DIR}/run_parallel_fp8-glm52-vllm.sh}"
EXPECTED_LAUNCHER_SHA256="${GDPVAL_EXPECTED_GLM52_LAUNCHER_SHA256:-93c48116c9cef175a5ec6ddb511f9c5f5edbeede4a83e2e7d4c9b1e37fe2cd26}"
MODEL_PATH="${GDPVAL_MODEL_PATH:-}"
VLLM_IMAGE="${GDPVAL_VLLM_IMAGE:-}"
MODEL_BASE_NAME="${GDPVAL_MODEL_BASE_NAME:-GLM-52-fp8-afterquery-20260721}"
EXPECTED_MODEL_NAME="${MODEL_BASE_NAME}-0"
ACCOUNT="${GDPVAL_SLURM_ACCOUNT:-}"
PARTITION="${GDPVAL_SLURM_PARTITION:-batch}"
QOS="${GDPVAL_SLURM_QOS:-normal}"
TIME_LIMIT="${GDPVAL_SERVER_TIME_LIMIT:-04:00:00}"
STATE_FILE="${GDPVAL_SERVER_STATE_FILE:-${STAGING_ROOT}/model_server.current}"
CURRENT_ENDPOINT_ENV="${GDPVAL_CURRENT_ENDPOINT_ENV:-${STAGING_ROOT}/endpoint.current.env}"
SUBMISSION_LOCK_DIR="${GDPVAL_SERVER_SUBMISSION_LOCK_DIR:-${STAGING_ROOT}/.submit_afterquery_glm52.lock}"

# Site-specific inputs have no default on purpose: a wrong-but-plausible default
# silently points the run at somebody else's model, image or Slurm account.
_missing=()
[[ -n "${ACCOUNT}" ]]    || _missing+=("GDPVAL_SLURM_ACCOUNT")
[[ -n "${MODEL_PATH}" ]] || _missing+=("GDPVAL_MODEL_PATH")
[[ -n "${VLLM_IMAGE}" ]] || _missing+=("GDPVAL_VLLM_IMAGE")
if (( ${#_missing[@]} )); then
    echo "ERROR: required setting(s) not set: ${_missing[*]}" >&2
    echo "       Copy benchmarks/gdpval/hsg/cluster.env.example, fill it in, and source it." >&2
    exit 2
fi
VALIDATOR="${REPO_ROOT}/benchmarks/gdpval/validate_gdpval_batch.py"
# Dataset-scoped: a profile (benchmarks/gdpval/datasets/*.env) points these at
# another GDPVal-format batch. Overrides may be empty when a dataset ships no
# prompt-backed reference repairs.
INPUT_JSONL="${GDPVAL_INPUT_JSONL:-${STAGING_ROOT}/input/afterquery_gdpval_subset_100_20260721.jsonl}"
OVERRIDES_JSON="${GDPVAL_REFERENCE_OVERRIDES-${REPO_ROOT}/benchmarks/gdpval/data/afterquery_gdpval_subset_100_20260721.reference_overrides.json}"
SOURCE_SHA256="${GDPVAL_EXPECTED_SOURCE_SHA256:-104e8c3bccc20420c7f225cd1b7f7822335764d018f665700042b341355c4f34}"
EXPECTED_COUNT="${GDPVAL_EXPECTED_COUNT:-100}"
REFERENCE_MODE="${GDPVAL_REFERENCE_MODE:-https}"
TASK_ID_PATTERN="${GDPVAL_TASK_ID_PATTERN-}"
if [[ -z "${TASK_ID_PATTERN}" ]]; then
  TASK_ID_PATTERN='AQ-\d{5}$'
fi

usage() {
  cat <<'EOF'
Usage: submit_gdpval_glm52.sh ACTION

Actions:
  preflight  Validate paths, hashes, account settings, and scheduler state.
  submit     Submit one 16-node / 64-GPU GLM-5.2 service job.
  status     Show the captured job state and server-info availability.
  wait       Wait for readiness, validate /models, and write endpoint.env.

By default, submit refuses to queue while no batch nodes are allocatable. Set
GDPVAL_ALLOW_QUEUE_DURING_MAINTENANCE=1 only when intentionally queueing early.

The wait action allows 24 hours in the scheduler queue by default, then starts
a separate 4-hour readiness timeout once the job reaches RUNNING. Override the
limits with GDPVAL_SERVER_QUEUE_TIMEOUT and GDPVAL_SERVER_READY_TIMEOUT (seconds).
EOF
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: required file not found: ${path}" >&2
    return 1
  fi
}

file_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

preflight() {
  require_file "${LAUNCHER}"
  require_file "${VLLM_IMAGE}"
  require_file "${INPUT_JSONL}"
  if [[ -n "${OVERRIDES_JSON}" ]]; then
    require_file "${OVERRIDES_JSON}"
  fi
  if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "ERROR: model path not found: ${MODEL_PATH}" >&2
    return 1
  fi
  command -v sbatch >/dev/null
  command -v squeue >/dev/null
  command -v sinfo >/dev/null
  command -v curl >/dev/null
  command -v sha256sum >/dev/null

  local launcher_sha input_sha
  launcher_sha="$(file_sha256 "${LAUNCHER}")"
  input_sha="$(file_sha256 "${INPUT_JSONL}")"
  if [[ "${launcher_sha}" != "${EXPECTED_LAUNCHER_SHA256}" ]]; then
    echo "ERROR: launcher hash mismatch (set GDPVAL_EXPECTED_GLM52_LAUNCHER_SHA256 if you edited it): ${launcher_sha}" >&2
    return 1
  fi
  if grep -q 'HF_TOKEN' "${LAUNCHER}"; then
    echo "ERROR: launcher must not hardcode HF_TOKEN" >&2
    return 1
  fi
  if [[ "${input_sha}" != "${SOURCE_SHA256}" ]]; then
    echo "ERROR: input hash mismatch: ${input_sha}" >&2
    return 1
  fi
  bash -n "${LAUNCHER}"

  local allocatable_nodes
  allocatable_nodes="$(sinfo -h -p "${PARTITION}" -t idle,alloc,mix -o '%D' | awk '{sum += $1} END {print sum + 0}')"
  if (( allocatable_nodes == 0 )) && [[ "${GDPVAL_ALLOW_QUEUE_DURING_MAINTENANCE:-0}" != "1" ]]; then
    echo "ERROR: partition ${PARTITION} has no allocatable nodes; maintenance is still active" >&2
    return 1
  fi

  echo "Server preflight passed"
  echo "  expected model: ${EXPECTED_MODEL_NAME}"
  echo "  account/qos: ${ACCOUNT}/${QOS}"
  echo "  partition nodes currently allocatable: ${allocatable_nodes}"
  echo "  shape: 2 nodes, 8 GPUs, TP8, DP1 (fp8)"
}

write_state() {
  local server_root="$1"
  local job_id="$2"
  local server_info="${server_root}/server_info/${EXPECTED_MODEL_NAME}.env"
  local tmp_state="${STATE_FILE}.tmp.$$"
  mkdir -p "$(dirname "${STATE_FILE}")"
  {
    echo "server_root=${server_root}"
    echo "job_id=${job_id}"
    echo "expected_model_name=${EXPECTED_MODEL_NAME}"
    echo "server_info=${server_info}"
    echo "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"${tmp_state}"
  chmod 600 "${tmp_state}"
  mv "${tmp_state}" "${STATE_FILE}"
}

load_state() {
  require_file "${STATE_FILE}"
  SERVER_ROOT=""
  JOB_ID=""
  STATE_EXPECTED_MODEL=""
  SERVER_INFO=""
  STATE_SUBMITTED_AT_UTC=""
  while IFS='=' read -r key value; do
    case "${key}" in
      server_root) SERVER_ROOT="${value}" ;;
      job_id) JOB_ID="${value}" ;;
      expected_model_name) STATE_EXPECTED_MODEL="${value}" ;;
      server_info) SERVER_INFO="${value}" ;;
      submitted_at_utc) STATE_SUBMITTED_AT_UTC="${value}" ;;
    esac
  done <"${STATE_FILE}"
  if [[ -z "${SERVER_ROOT}" || ! "${JOB_ID}" =~ ^[0-9]+$ || -z "${SERVER_INFO}" || -z "${STATE_SUBMITTED_AT_UTC}" ]]; then
    echo "ERROR: malformed server state: ${STATE_FILE}" >&2
    return 1
  fi
  if [[ "${STATE_EXPECTED_MODEL}" != "${EXPECTED_MODEL_NAME}" ]]; then
    echo "ERROR: state expects ${STATE_EXPECTED_MODEL}, launcher expects ${EXPECTED_MODEL_NAME}" >&2
    return 1
  fi
}

SUBMISSION_LOCK_HELD=0

acquire_submission_lock() {
  mkdir -p "$(dirname "${SUBMISSION_LOCK_DIR}")"
  if ! mkdir "${SUBMISSION_LOCK_DIR}" 2>/dev/null; then
    echo "ERROR: another GLM-5.2 submission holds the lock: ${SUBMISSION_LOCK_DIR}" >&2
    return 1
  fi
  SUBMISSION_LOCK_HELD=1
}

release_submission_lock() {
  if [[ "${SUBMISSION_LOCK_HELD}" == "1" ]]; then
    rmdir "${SUBMISSION_LOCK_DIR}" 2>/dev/null || true
    SUBMISSION_LOCK_HELD=0
  fi
}

invalidate_current_endpoint() {
  if [[ -e "${CURRENT_ENDPOINT_ENV}" || -L "${CURRENT_ENDPOINT_ENV}" ]]; then
    local stale_endpoint
    stale_endpoint="${CURRENT_ENDPOINT_ENV}.stale.$(date -u +%Y%m%dT%H%M%SZ).$$"
    mv "${CURRENT_ENDPOINT_ENV}" "${stale_endpoint}"
    chmod 600 "${stale_endpoint}"
    echo "Archived stale endpoint environment: ${stale_endpoint}"
  fi
}

submit_server() {
  acquire_submission_lock
  trap release_submission_lock EXIT
  preflight
  if [[ -e "${STATE_FILE}" ]]; then
    echo "ERROR: state file already exists; inspect it before another submission: ${STATE_FILE}" >&2
    return 1
  fi
  invalidate_current_endpoint
  local timestamp server_root submission_output submission_log job_id
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  server_root="${STAGING_ROOT}/model_servers/${MODEL_BASE_NAME}_${timestamp}"
  if [[ -e "${server_root}" ]]; then
    echo "ERROR: server output path already exists: ${server_root}" >&2
    return 1
  fi
  mkdir -p "${server_root}"

  submission_log="${server_root}/submission.stdout"
  submission_output="$(
    env -i \
      HOME="${HOME}" \
      USER="${USER}" \
      LOGNAME="${LOGNAME:-${USER}}" \
      PATH="${PATH}" \
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      SLURM_CONF="${SLURM_CONF:-}" \
      LANG="${LANG:-C.UTF-8}" \
      bash "${LAUNCHER}" \
      --vllm-image "${VLLM_IMAGE}" \
      --model-path "${MODEL_PATH}" \
      --model-name "${MODEL_BASE_NAME}" \
      --num-instances 1 \
      --output-dir "${server_root}" \
      --num-nodes 2 \
      --account "${ACCOUNT}" \
      --partition "${PARTITION}" \
      --qos "${QOS}" \
      --time-limit "${TIME_LIMIT}" \
      --tp-size 8 \
      --pp-size 1 \
      --dp-size 1 \
      --dp-size-local 1 \
      --gpu-memory-utilization 0.90 \
      --max-model-len 262144
  )"
  printf '%s\n' "${submission_output}" >"${submission_log}"
  chmod 600 "${submission_log}"
  job_id="$(printf '%s\n' "${submission_output}" | awk '/Submitted batch job [0-9]+/{print $4}' | tail -1)"
  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse the submitted Slurm job ID; inspect ${submission_log}" >&2
    return 1
  fi
  write_state "${server_root}" "${job_id}"
  release_submission_lock
  trap - EXIT
  echo "Submitted GLM-5.2 service job ${job_id}"
  echo "  server root: ${server_root}"
  echo "  state: ${STATE_FILE}"
}

job_state() {
  local state
  state="$(squeue -h -j "${JOB_ID}" -o '%T' 2>/dev/null | head -1 || true)"
  if [[ -z "${state}" ]]; then
    state="$(sacct -n -X -j "${JOB_ID}" -o State | awk 'NF {print $1; exit}')"
  fi
  printf '%s' "${state:-UNKNOWN}"
}

status_server() {
  load_state
  echo "Job ${JOB_ID}: $(job_state)"
  echo "Expected model: ${EXPECTED_MODEL_NAME}"
  if [[ -f "${SERVER_INFO}" ]]; then
    echo "Server info: ready at ${SERVER_INFO}"
  else
    echo "Server info: not written yet"
  fi
}

read_server_info() {
  INFO_MODEL_NAME=""
  INFO_SERVER_URL=""
  while IFS='=' read -r key value; do
    case "${key}" in
      MODEL_NAME) INFO_MODEL_NAME="${value}" ;;
      SERVER_URL) INFO_SERVER_URL="${value}" ;;
    esac
  done <"${SERVER_INFO}"
  if [[ "${INFO_MODEL_NAME}" != "${EXPECTED_MODEL_NAME}" ]]; then
    echo "ERROR: server-info model mismatch: ${INFO_MODEL_NAME}" >&2
    return 1
  fi
  if [[ ! "${INFO_SERVER_URL}" =~ ^http://[A-Za-z0-9._:-]+/v1$ ]]; then
    echo "ERROR: invalid server URL in ${SERVER_INFO}" >&2
    return 1
  fi
}

write_endpoint_env() {
  local endpoint_env="${SERVER_ROOT}/endpoint.env"
  local tmp_env="${endpoint_env}.tmp.$$"
  local current_env="${CURRENT_ENDPOINT_ENV}"
  local tmp_current="${current_env}.tmp.$$"
  local created_at_utc
  created_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    printf 'export POLICY_MODEL_NAME=%q\n' "${EXPECTED_MODEL_NAME}"
    printf 'export POLICY_BASE_URL=%q\n' "${INFO_SERVER_URL}"
    printf 'export POLICY_API_KEY=%q\n' "dummy"
    printf 'export GDPVAL_SERVER_JOB_ID=%q\n' "${JOB_ID}"
    printf 'export GDPVAL_SERVER_ROOT=%q\n' "${SERVER_ROOT}"
    printf 'export GDPVAL_SERVER_SUBMITTED_AT_UTC=%q\n' "${STATE_SUBMITTED_AT_UTC}"
    printf 'export GDPVAL_ENDPOINT_CREATED_AT_UTC=%q\n' "${created_at_utc}"
  } >"${tmp_env}"
  chmod 600 "${tmp_env}"
  mv "${tmp_env}" "${endpoint_env}"
  cp "${endpoint_env}" "${tmp_current}"
  chmod 600 "${tmp_current}"
  mv "${tmp_current}" "${current_env}"
  echo "Validated endpoint environment: ${endpoint_env}"
  echo "Current endpoint environment: ${current_env}"
  echo "Endpoint provenance: job ${JOB_ID}, server root ${SERVER_ROOT}, created ${created_at_utc}"
}

wait_for_server() {
  load_state
  local queue_timeout="${GDPVAL_SERVER_QUEUE_TIMEOUT:-86400}"
  local ready_timeout="${GDPVAL_SERVER_READY_TIMEOUT:-14400}"
  local interval="${GDPVAL_SERVER_POLL_INTERVAL:-15}"
  if [[ ! "${queue_timeout}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GDPVAL_SERVER_QUEUE_TIMEOUT must be a positive integer: ${queue_timeout}" >&2
    return 1
  fi
  if [[ ! "${ready_timeout}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GDPVAL_SERVER_READY_TIMEOUT must be a positive integer: ${ready_timeout}" >&2
    return 1
  fi
  if [[ ! "${interval}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GDPVAL_SERVER_POLL_INTERVAL must be a positive integer: ${interval}" >&2
    return 1
  fi
  local queue_deadline=$((SECONDS + queue_timeout))
  local ready_deadline=0
  local wait_phase="queue"
  while [[ ! -f "${SERVER_INFO}" ]]; do
    local state
    state="$(job_state)"
    case "${state}" in
      FAILED*|CANCELLED*|TIMEOUT*|NODE_FAIL*|OUT_OF_MEMORY*|PREEMPTED*|BOOT_FAIL*|DEADLINE*|REVOKED*|COMPLETED*|COMPLETING*)
        echo "ERROR: server job ${JOB_ID} reached terminal state ${state}" >&2
        return 1
        ;;
    esac
    if [[ "${wait_phase}" == "queue" ]]; then
      if [[ "${state}" == "RUNNING" ]]; then
        wait_phase="readiness"
        ready_deadline=$((SECONDS + ready_timeout))
        echo "GLM server job ${JOB_ID} is RUNNING; starting ${ready_timeout}s readiness timeout"
      elif (( SECONDS >= queue_deadline )); then
        echo "ERROR: timed out after ${queue_timeout}s waiting for job ${JOB_ID} to start; job state ${state}" >&2
        return 1
      fi
    elif (( SECONDS >= ready_deadline )); then
      echo "ERROR: timed out after ${ready_timeout}s waiting for ${SERVER_INFO}; job state ${state}" >&2
      return 1
    fi
    echo "Waiting for GLM server (${wait_phase}); job ${JOB_ID} is ${state}"
    sleep "${interval}"
  done

  if [[ "$(job_state)" != "RUNNING" ]]; then
    echo "ERROR: server-info exists but job ${JOB_ID} is not RUNNING" >&2
    return 1
  fi
  read_server_info
  # Dataset-scoped: an empty overrides path must be omitted entirely (argparse
  # would turn "" into Path(".")), and the row count / id pattern / locator mode
  # all come from the active profile rather than AfterQuery's values.
  local endpoint_args=(
    --input "${INPUT_JSONL}"
    --expected-count "${EXPECTED_COUNT}"
    --expected-sha256 "${SOURCE_SHA256}"
    --task-id-pattern "${TASK_ID_PATTERN}"
    --reference-mode "${REFERENCE_MODE}"
    --check-model-endpoint
    --model-base-url "${INFO_SERVER_URL}"
    --model-name "${EXPECTED_MODEL_NAME}"
  )
  if [[ -n "${OVERRIDES_JSON}" ]]; then
    endpoint_args+=(--reference-overrides "${OVERRIDES_JSON}")
  fi
  POLICY_API_KEY=dummy "${REPO_ROOT}/.venv/bin/python" "${VALIDATOR}" "${endpoint_args[@]}"
  write_endpoint_env
}

case "${1:-preflight}" in
  preflight) preflight ;;
  submit) submit_server ;;
  status) status_server ;;
  wait) wait_for_server ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
