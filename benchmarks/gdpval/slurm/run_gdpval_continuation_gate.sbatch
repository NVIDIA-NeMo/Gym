#!/usr/bin/env bash
#SBATCH --job-name=gdpval-cont-gate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00

set -euo pipefail
umask 077

STAGING_ROOT="${GDPVAL_SLURM_STAGING_ROOT:-${SCRATCH:-$HOME}/gdpval-rollouts}"
CHECKOUT="${GDPVAL_SLURM_CHECKOUT:-${STAGING_ROOT}/checkout}"
RUN_ROOT="${GDPVAL_RUN_ROOT:-${STAGING_ROOT}/GLM-52-fp8_afterquery-subset100_20260721}"
PARENT_SERVER_JOB_ID="${GDPVAL_PARENT_SERVER_JOB_ID:?GDPVAL_PARENT_SERVER_JOB_ID is required}"
PARENT_CLIENT_JOB_ID="${GDPVAL_PARENT_CLIENT_JOB_ID:?GDPVAL_PARENT_CLIENT_JOB_ID is required}"
CONTINUATION_STATE="${GDPVAL_CONTINUATION_STATE_FILE:-${STAGING_ROOT}/continuation.current}"
SERVER_STATE="${GDPVAL_SERVER_STATE_FILE:-${STAGING_ROOT}/model_server.current}"
CLIENT_STATE="${GDPVAL_CLIENT_STATE_FILE:-${STAGING_ROOT}/client.current}"
SUBMITTER="${CHECKOUT}/benchmarks/gdpval/slurm/submit_gdpval_glm52.sh"
MONITOR_BATCH="${CHECKOUT}/benchmarks/gdpval/slurm/run_gdpval_continuation_monitor.sbatch"
VALIDATOR="${CHECKOUT}/benchmarks/gdpval/validate_gdpval_batch.py"
RESUME_VALIDATOR="${CHECKOUT}/benchmarks/gdpval/slurm/validate_gdpval_resume_state.py"
PYTHON_BIN="${CHECKOUT}/.venv/bin/python"
LAUNCH_INPUT="${RUN_ROOT}/input/afterquery_gdpval_subset_100_20260721.launch.jsonl"
REFERENCE_OVERRIDES="${CHECKOUT}/benchmarks/gdpval/data/afterquery_gdpval_subset_100_20260721.reference_overrides.json"
RESULTS_DIR="${RUN_ROOT}/results"
FULL_OUTPUT="${RESULTS_DIR}/rollouts.jsonl"
FULL_MATERIALIZED="${RESULTS_DIR}/rollouts_materialized_inputs.jsonl"
FULL_FAILURES="${RESULTS_DIR}/rollouts_failures.jsonl"
DELIVERABLES_DIR="${RUN_ROOT}/deliverables"
SMOKE_TASK_ID="${GDPVAL_SMOKE_TASK_ID:-AQ-09135}"
SMOKE_OUTPUT="${RESULTS_DIR}/smoke_${SMOKE_TASK_ID}.jsonl"
SMOKE_MATERIALIZED="${RESULTS_DIR}/smoke_${SMOKE_TASK_ID}_materialized_inputs.jsonl"
SMOKE_FAILURES="${RESULTS_DIR}/smoke_${SMOKE_TASK_ID}_failures.jsonl"
SMOKE_DELIVERABLES="${RUN_ROOT}/smoke_deliverables"
EXPECTED_MODEL_NAME="${GDPVAL_EXPECTED_MODEL_NAME:-GLM-52-bf16-afterquery-20260721-0}"
EXPECTED_LAUNCH_SHA256="${GDPVAL_EXPECTED_LAUNCH_SHA256:-9e8d2d182a142e44f60fabb0dff1eaf9f7bd9f7e5c8d23fc2dc5879373d04bad}"
LOCK_DIR="${GDPVAL_CONTINUATION_LOCK_DIR:-${STAGING_ROOT}/.gdpval_continuation_gate.lock}"
LOCK_HELD=0

cleanup() {
  if [[ "${LOCK_HELD}" == "1" ]]; then
    rmdir "${LOCK_DIR}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || {
    echo "ERROR: required file not found: ${path}" >&2
    return 1
  }
}

job_state() {
  local job_id="$1"
  local state
  state="$(squeue -h -j "${job_id}" -o '%T' | head -1)"
  if [[ -z "${state}" ]]; then
    state="$(sacct -n -X -j "${job_id}" -o State | awk 'NF {print $1; exit}')"
  fi
  printf '%s' "${state:-UNKNOWN}"
}

require_terminal_job() {
  local job_id="$1"
  local label="$2"
  local state
  state="$(job_state "${job_id}")"
  case "${state}" in
    COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|NODE_FAIL*|OUT_OF_MEMORY*|PREEMPTED*|BOOT_FAIL*|DEADLINE*|REVOKED*)
      echo "${label} job ${job_id} is terminal: ${state}"
      ;;
    *)
      echo "ERROR: ${label} job ${job_id} is not terminal: ${state}" >&2
      return 1
      ;;
  esac
}

state_value() {
  local path="$1"
  local wanted="$2"
  awk -F= -v wanted="${wanted}" '$1 == wanted {sub(/^[^=]*=/, ""); print; exit}' "${path}"
}

write_continuation_state() {
  local status="$1"
  local action="${2:-}"
  local new_server_job_id="${3:-}"
  local monitor_job_id="${4:-}"
  local tmp="${CONTINUATION_STATE}.tmp.$$"
  {
    echo "status=${status}"
    echo "parent_server_job_id=${PARENT_SERVER_JOB_ID}"
    echo "parent_client_job_id=${PARENT_CLIENT_JOB_ID}"
    echo "gate_job_id=${SLURM_JOB_ID:-manual}"
    [[ -n "${action}" ]] && echo "action=${action}"
    [[ -n "${new_server_job_id}" ]] && echo "new_server_job_id=${new_server_job_id}"
    [[ -n "${monitor_job_id}" ]] && echo "monitor_job_id=${monitor_job_id}"
    echo "updated_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"${tmp}"
  chmod 600 "${tmp}"
  mv "${tmp}" "${CONTINUATION_STATE}"
}

line_count() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    wc -l <"${path}" | tr -d ' '
  else
    printf '0'
  fi
}

validate_full() {
  "${PYTHON_BIN}" "${VALIDATOR}" \
    --input "${LAUNCH_INPUT}" \
    --expected-count 100 \
    --expected-sha256 "${EXPECTED_LAUNCH_SHA256}" \
    --reference-overrides "${REFERENCE_OVERRIDES}" \
    --rollouts "${FULL_OUTPUT}" \
    --deliverables-dir "${DELIVERABLES_DIR}" \
    --expected-response-model "${EXPECTED_MODEL_NAME}"
}

validate_smoke() {
  "${PYTHON_BIN}" "${VALIDATOR}" \
    --input "${LAUNCH_INPUT}" \
    --expected-count 100 \
    --expected-sha256 "${EXPECTED_LAUNCH_SHA256}" \
    --reference-overrides "${REFERENCE_OVERRIDES}" \
    --smoke-task-id "${SMOKE_TASK_ID}" \
    --rollouts "${SMOKE_OUTPUT}" \
    --deliverables-dir "${SMOKE_DELIVERABLES}" \
    --require-deliverable \
    --expected-response-model "${EXPECTED_MODEL_NAME}"
}

validate_partial_full_state() {
  "${PYTHON_BIN}" "${RESUME_VALIDATOR}" \
    --launch-input "${LAUNCH_INPUT}" \
    --rollouts "${FULL_OUTPUT}" \
    --materialized-inputs "${FULL_MATERIALIZED}" \
    --failures "${FULL_FAILURES}" \
    --run-config "${RUN_ROOT}/run_config.sha256" \
    --expected-launch-sha256 "${EXPECTED_LAUNCH_SHA256}" \
    --expected-model "${EXPECTED_MODEL_NAME}" \
    --deliverables-dir "${DELIVERABLES_DIR}"
}

directory_has_entries() {
  local path="$1"
  [[ -d "${path}" && -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

archive_empty_lock() {
  local path="$1"
  local archive_dir="$2"
  if [[ ! -e "${path}" ]]; then
    return
  fi
  if [[ ! -d "${path}" ]] || directory_has_entries "${path}"; then
    echo "ERROR: refusing to archive nonempty or non-directory lock: ${path}" >&2
    return 1
  fi
  mv "${path}" "${archive_dir}/$(basename "${path}")"
  echo "Archived stale lock: ${path}"
}

archive_unfinished_smoke() {
  local timestamp archive_dir
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  archive_dir="${RUN_ROOT}/continuation_archive/${timestamp}_after_${PARENT_SERVER_JOB_ID}"
  mkdir -p "${archive_dir}/results" "${archive_dir}/logs"
  chmod 700 "${RUN_ROOT}/continuation_archive" "${archive_dir}" "${archive_dir}/results" "${archive_dir}/logs"

  local unexpected
  unexpected="$(
    find "${RESULTS_DIR}" -mindepth 1 -maxdepth 1 \
      ! -name "smoke_${SMOKE_TASK_ID}.jsonl" \
      ! -name "smoke_${SMOKE_TASK_ID}_materialized_inputs.jsonl" \
      ! -name "smoke_${SMOKE_TASK_ID}_failures.jsonl" \
      -print -quit
  )"
  if [[ -n "${unexpected}" ]]; then
    echo "ERROR: unfinished smoke has unexpected result state: ${unexpected}" >&2
    return 1
  fi

  local path
  for path in "${SMOKE_OUTPUT}" "${SMOKE_MATERIALIZED}" "${SMOKE_FAILURES}"; do
    if [[ -e "${path}" ]]; then
      mv "${path}" "${archive_dir}/results/"
    fi
  done
  if [[ -d "${SMOKE_DELIVERABLES}" ]]; then
    mv "${SMOKE_DELIVERABLES}" "${archive_dir}/smoke_deliverables"
  fi
  mkdir -p "${SMOKE_DELIVERABLES}"
  chmod 700 "${SMOKE_DELIVERABLES}"

  if [[ -f "${RUN_ROOT}/logs/gym_env_start.log" ]]; then
    mv "${RUN_ROOT}/logs/gym_env_start.log" "${archive_dir}/logs/"
  fi
  if [[ -f "${RUN_ROOT}/run_manifest.txt" ]]; then
    mv "${RUN_ROOT}/run_manifest.txt" "${archive_dir}/"
  fi
  archive_empty_lock "${RUN_ROOT}/.launch.lock" "${archive_dir}"
  echo "Archived unfinished smoke state: ${archive_dir}"
}

archive_server_state() {
  require_file "${SERVER_STATE}"
  local captured_job_id terminal_state
  captured_job_id="$(state_value "${SERVER_STATE}" job_id)"
  if [[ "${captured_job_id}" != "${PARENT_SERVER_JOB_ID}" ]]; then
    echo "ERROR: current server state points to ${captured_job_id}, expected ${PARENT_SERVER_JOB_ID}" >&2
    return 1
  fi
  terminal_state="$(job_state "${PARENT_SERVER_JOB_ID}")"
  local archived="${SERVER_STATE}.terminal.${PARENT_SERVER_JOB_ID}.$(date -u +%Y%m%dT%H%M%SZ)"
  mv "${SERVER_STATE}" "${archived}"
  chmod 600 "${archived}"
  echo "Archived terminal server state (${terminal_state}): ${archived}"
}

verify_captured_state_identity() {
  require_file "${SERVER_STATE}"
  require_file "${CLIENT_STATE}"
  require_file "${CONTINUATION_STATE}"
  local current_server current_client captured_gate captured_server captured_client
  current_server="$(state_value "${SERVER_STATE}" job_id)"
  current_client="$(state_value "${CLIENT_STATE}" job_id)"
  captured_gate="$(state_value "${CONTINUATION_STATE}" gate_job_id)"
  captured_server="$(state_value "${CONTINUATION_STATE}" parent_server_job_id)"
  captured_client="$(state_value "${CONTINUATION_STATE}" parent_client_job_id)"
  if [[ "${current_server}" != "${PARENT_SERVER_JOB_ID}" || "${captured_server}" != "${PARENT_SERVER_JOB_ID}" ]]; then
    echo "ERROR: captured server identity changed; refusing continuation" >&2
    return 1
  fi
  if [[ "${current_client}" != "${PARENT_CLIENT_JOB_ID}" || "${captured_client}" != "${PARENT_CLIENT_JOB_ID}" ]]; then
    echo "ERROR: captured client identity changed; refusing continuation" >&2
    return 1
  fi
  if [[ -n "${SLURM_JOB_ID:-}" && "${captured_gate}" != "${SLURM_JOB_ID}" ]]; then
    echo "ERROR: continuation state belongs to gate ${captured_gate}, not ${SLURM_JOB_ID}" >&2
    return 1
  fi
}

for path in "${SUBMITTER}" "${MONITOR_BATCH}" "${VALIDATOR}" "${RESUME_VALIDATOR}" "${PYTHON_BIN}" "${LAUNCH_INPUT}" "${REFERENCE_OVERRIDES}"; do
  require_file "${path}"
done
for command_name in sbatch squeue sacct awk sha256sum; do
  command -v "${command_name}" >/dev/null
done
for job_id in "${PARENT_SERVER_JOB_ID}" "${PARENT_CLIENT_JOB_ID}"; do
  [[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: invalid parent job ID: ${job_id}" >&2
    exit 2
  }
done

mkdir -p "${STAGING_ROOT}/continuation_logs"
chmod 700 "${STAGING_ROOT}/continuation_logs"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "ERROR: another continuation gate owns ${LOCK_DIR}" >&2
  exit 2
fi
LOCK_HELD=1

require_terminal_job "${PARENT_SERVER_JOB_ID}" "server"
require_terminal_job "${PARENT_CLIENT_JOB_ID}" "client"
verify_captured_state_identity

if [[ "$(line_count "${FULL_OUTPUT}")" == "100" ]] && validate_full; then
  echo "All 100 rollouts are valid; no continuation allocation is needed"
  write_continuation_state complete_no_continuation
  exit 0
fi

ACTION=""
if [[ -f "${FULL_OUTPUT}" || -f "${FULL_MATERIALIZED}" ]]; then
  if [[ ! -f "${FULL_OUTPUT}" || ! -f "${FULL_MATERIALIZED}" ]]; then
    echo "ERROR: full continuation requires both ${FULL_OUTPUT} and ${FULL_MATERIALIZED}" >&2
    exit 1
  fi
  validate_partial_full_state
  ACTION=resume
  echo "Detected partial full collection: $(line_count "${FULL_OUTPUT}")/100 rows; scheduling resume"
elif [[ "$(line_count "${SMOKE_OUTPUT}")" == "1" ]] && validate_smoke; then
  ACTION=resume
  echo "Smoke is valid and full collection has not started; scheduling full collection"
else
  if directory_has_entries "${DELIVERABLES_DIR}"; then
    echo "ERROR: unfinished smoke has an unexpected nonempty full deliverables cache: ${DELIVERABLES_DIR}" >&2
    exit 1
  fi
  archive_unfinished_smoke
  ACTION=run
  echo "Smoke did not finish; scheduling one clean smoke retry"
fi

mkdir -p "${RUN_ROOT}/continuation_archive"
chmod 700 "${RUN_ROOT}/continuation_archive"
archive_empty_lock "${RUN_ROOT}/.launch.lock" "${RUN_ROOT}/continuation_archive"
archive_empty_lock "${STAGING_ROOT}/.monitor_gdpval_pipeline.lock" "${RUN_ROOT}/continuation_archive"
archive_server_state

GDPVAL_SLURM_STAGING_ROOT="${STAGING_ROOT}" "${SUBMITTER}" submit
NEW_SERVER_JOB_ID="$(state_value "${SERVER_STATE}" job_id)"
[[ "${NEW_SERVER_JOB_ID}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: submitter wrote an invalid new server job ID: ${NEW_SERVER_JOB_ID}" >&2
  exit 1
}
write_continuation_state server_queued "${ACTION}" "${NEW_SERVER_JOB_ID}"

MONITOR_SUBMISSION="$(
  sbatch --parsable \
    --dependency="after:${NEW_SERVER_JOB_ID}+1" \
    --export="USER=${USER},HOME=${HOME},LOGNAME=${LOGNAME:-${USER}},PATH=${PATH},LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-},SLURM_CONF=${SLURM_CONF:-},GDPVAL_ACTION=${ACTION},GDPVAL_SLURM_STAGING_ROOT=${STAGING_ROOT}" \
    "${MONITOR_BATCH}"
)"
MONITOR_JOB_ID="${MONITOR_SUBMISSION%%;*}"
[[ "${MONITOR_JOB_ID}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: could not parse continuation monitor job ID" >&2
  exit 1
}

write_continuation_state continuation_queued "${ACTION}" "${NEW_SERVER_JOB_ID}" "${MONITOR_JOB_ID}"
echo "Queued continuation server ${NEW_SERVER_JOB_ID}"
echo "Queued endpoint monitor ${MONITOR_JOB_ID} with action ${ACTION}"
