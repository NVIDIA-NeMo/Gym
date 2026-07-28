#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGING_ROOT="${GDPVAL_SLURM_STAGING_ROOT:-${SCRATCH:-$HOME}/gdpval-rollouts}"
SERVER_STATE="${GDPVAL_SERVER_STATE_FILE:-${STAGING_ROOT}/model_server.current}"
CLIENT_STATE="${GDPVAL_CLIENT_STATE_FILE:-${STAGING_ROOT}/client.current}"
CONTINUATION_STATE="${GDPVAL_CONTINUATION_STATE_FILE:-${STAGING_ROOT}/continuation.current}"
GATE_BATCH="${SCRIPT_DIR}/run_gdpval_continuation_gate.sbatch"
LOG_DIR="${STAGING_ROOT}/continuation_logs"
SUBMISSION_LOCK_DIR="${GDPVAL_CONTINUATION_SUBMISSION_LOCK_DIR:-${STAGING_ROOT}/.submit_gdpval_continuation.lock}"
SUBMISSION_LOCK_HELD=0

release_submission_lock() {
  if [[ "${SUBMISSION_LOCK_HELD}" == "1" ]]; then
    rmdir "${SUBMISSION_LOCK_DIR}" 2>/dev/null || true
    SUBMISSION_LOCK_HELD=0
  fi
}

usage() {
  cat <<'EOF'
Usage: submit_gdpval_continuation.sh ACTION [SERVER_JOB_ID CLIENT_JOB_ID]

Actions:
  preflight  Validate the current parent jobs and continuation scripts.
  submit     Queue one conditional continuation gate after both parent jobs.
  status     Show the gate and any continuation server/monitor/client jobs.

The gate validates all 100 outputs before requesting more GPUs. If the current
run is complete it exits without a new allocation. Otherwise it resumes a
partial full collection, starts full collection after a valid smoke, or
recoverably archives and retries an unfinished smoke.
EOF
}

state_value() {
  local path="$1"
  local wanted="$2"
  awk -F= -v wanted="${wanted}" '$1 == wanted {sub(/^[^=]*=/, ""); print; exit}' "${path}"
}

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

resolve_parent_jobs() {
  PARENT_SERVER_JOB_ID="${2:-${GDPVAL_PARENT_SERVER_JOB_ID:-}}"
  PARENT_CLIENT_JOB_ID="${3:-${GDPVAL_PARENT_CLIENT_JOB_ID:-}}"
  if [[ -z "${PARENT_SERVER_JOB_ID}" ]]; then
    require_file "${SERVER_STATE}"
    PARENT_SERVER_JOB_ID="$(state_value "${SERVER_STATE}" job_id)"
  fi
  if [[ -z "${PARENT_CLIENT_JOB_ID}" ]]; then
    require_file "${CLIENT_STATE}"
    PARENT_CLIENT_JOB_ID="$(state_value "${CLIENT_STATE}" job_id)"
  fi
  for job_id in "${PARENT_SERVER_JOB_ID}" "${PARENT_CLIENT_JOB_ID}"; do
    [[ "${job_id}" =~ ^[0-9]+$ ]] || {
      echo "ERROR: invalid parent job ID: ${job_id}" >&2
      return 1
    }
  done
}

preflight() {
  resolve_parent_jobs "$@"
  require_file "${GATE_BATCH}"
  require_file "${SCRIPT_DIR}/run_gdpval_continuation_monitor.sbatch"
  require_file "${SCRIPT_DIR}/submit_gdpval_glm52.sh"
  command -v sbatch >/dev/null
  command -v scontrol >/dev/null
  command -v squeue >/dev/null
  command -v sacct >/dev/null
  bash -n "${GATE_BATCH}"
  bash -n "${SCRIPT_DIR}/run_gdpval_continuation_monitor.sbatch"

  if [[ -e "${CONTINUATION_STATE}" ]]; then
    local prior_gate prior_state
    prior_gate="$(state_value "${CONTINUATION_STATE}" gate_job_id)"
    prior_state=""
    if [[ "${prior_gate}" =~ ^[0-9]+$ ]]; then
      prior_state="$(job_state "${prior_gate}")"
    fi
    echo "ERROR: continuation state already exists (gate ${prior_gate:-unknown}, ${prior_state:-unknown}): ${CONTINUATION_STATE}" >&2
    return 1
  fi

  echo "Continuation preflight passed"
  echo "  parent server: ${PARENT_SERVER_JOB_ID} ($(job_state "${PARENT_SERVER_JOB_ID}"))"
  echo "  parent client: ${PARENT_CLIENT_JOB_ID} ($(job_state "${PARENT_CLIENT_JOB_ID}"))"
  echo "  dependency: afterany:${PARENT_SERVER_JOB_ID}:${PARENT_CLIENT_JOB_ID}"
}

submit_gate() {
  if ! mkdir "${SUBMISSION_LOCK_DIR}" 2>/dev/null; then
    echo "ERROR: another continuation submission owns ${SUBMISSION_LOCK_DIR}" >&2
    return 1
  fi
  SUBMISSION_LOCK_HELD=1
  trap release_submission_lock EXIT
  preflight "$@"
  mkdir -p "${LOG_DIR}"
  chmod 700 "${LOG_DIR}"
  local submission gate_job_id tmp_state
  submission="$(
    sbatch --parsable \
      --hold \
      --dependency="afterany:${PARENT_SERVER_JOB_ID}:${PARENT_CLIENT_JOB_ID}" \
      --export="USER=${USER},HOME=${HOME},LOGNAME=${LOGNAME:-${USER}},PATH=${PATH},LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-},SLURM_CONF=${SLURM_CONF:-},GDPVAL_SLURM_STAGING_ROOT=${STAGING_ROOT},GDPVAL_PARENT_SERVER_JOB_ID=${PARENT_SERVER_JOB_ID},GDPVAL_PARENT_CLIENT_JOB_ID=${PARENT_CLIENT_JOB_ID},GDPVAL_CONTINUATION_STATE_FILE=${CONTINUATION_STATE}" \
      "${GATE_BATCH}"
  )"
  gate_job_id="${submission%%;*}"
  [[ "${gate_job_id}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse continuation gate job ID" >&2
    return 1
  }

  tmp_state="${CONTINUATION_STATE}.tmp.$$"
  {
    echo "status=gate_queued"
    echo "parent_server_job_id=${PARENT_SERVER_JOB_ID}"
    echo "parent_client_job_id=${PARENT_CLIENT_JOB_ID}"
    echo "gate_job_id=${gate_job_id}"
    echo "dependency=afterany:${PARENT_SERVER_JOB_ID}:${PARENT_CLIENT_JOB_ID}"
    echo "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"${tmp_state}"
  chmod 600 "${tmp_state}"
  mv "${tmp_state}" "${CONTINUATION_STATE}"
  scontrol release "${gate_job_id}"
  release_submission_lock
  trap - EXIT

  echo "Queued conditional continuation gate ${gate_job_id}"
  echo "  dependency: afterany:${PARENT_SERVER_JOB_ID}:${PARENT_CLIENT_JOB_ID}"
  echo "  state: ${CONTINUATION_STATE}"
}

show_job() {
  local label="$1"
  local job_id="$2"
  if [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "${label} ${job_id}: $(job_state "${job_id}")"
  fi
}

status() {
  require_file "${CONTINUATION_STATE}"
  local status_value parent_server parent_client gate new_server monitor
  status_value="$(state_value "${CONTINUATION_STATE}" status)"
  parent_server="$(state_value "${CONTINUATION_STATE}" parent_server_job_id)"
  parent_client="$(state_value "${CONTINUATION_STATE}" parent_client_job_id)"
  gate="$(state_value "${CONTINUATION_STATE}" gate_job_id)"
  new_server="$(state_value "${CONTINUATION_STATE}" new_server_job_id)"
  monitor="$(state_value "${CONTINUATION_STATE}" monitor_job_id)"
  echo "Continuation status: ${status_value:-unknown}"
  show_job "Parent server" "${parent_server}"
  show_job "Parent client" "${parent_client}"
  show_job "Gate" "${gate}"
  show_job "New server" "${new_server}"
  show_job "Endpoint monitor" "${monitor}"
  if [[ -f "${CLIENT_STATE}" ]]; then
    show_job "Current client" "$(state_value "${CLIENT_STATE}" job_id)"
  fi
}

case "${1:-status}" in
  preflight) preflight "$@" ;;
  submit) submit_gate "$@" ;;
  status) status ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
