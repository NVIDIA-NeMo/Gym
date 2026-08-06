#!/usr/bin/env bash
set -euo pipefail

# Wait for the staged model endpoint, then submit exactly one rollout client.
# This runs on the Slurm login node; the client itself runs on cpu/cpu-normal.

umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGING_ROOT="${GDPVAL_SLURM_STAGING_ROOT:-${SCRATCH:-$HOME}/gdpval-rollouts}"
# The monitor drives `<wrapper> wait`, which validates /v1/models against the
# dataset's expected model name, so it must see the dataset profile itself --
# forwarding it to the client alone is not enough.
if [[ -n "${GDPVAL_ENV_FILE:-}" ]]; then
  if [[ ! -r "${GDPVAL_ENV_FILE}" ]]; then
    echo "ERROR: GDPVAL_ENV_FILE is not readable: ${GDPVAL_ENV_FILE}" >&2
    exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "${GDPVAL_ENV_FILE}"
  set +a
fi
PRIVATE_ENV="${GDPVAL_PRIVATE_ENV_FILE:-${STAGING_ROOT}/afterquery.private.env}"
ENDPOINT_ENV="${GDPVAL_ENDPOINT_ENV_FILE:-${STAGING_ROOT}/endpoint.current.env}"
CLIENT_SCRIPT="${SCRIPT_DIR}/run_gdpval_client.sbatch"
SERVER_WRAPPER="${SCRIPT_DIR}/submit_model_server.sh"
CLIENT_STATE="${GDPVAL_CLIENT_STATE_FILE:-${STAGING_ROOT}/client.current}"
LOCK_DIR="${GDPVAL_PIPELINE_LOCK_DIR:-${STAGING_ROOT}/.monitor_gdpval_pipeline.lock}"
ACTION="${GDPVAL_ACTION:-run}"
TAVILY_WAIT_TIMEOUT="${GDPVAL_TAVILY_WAIT_TIMEOUT:-1800}"
POLL_INTERVAL="${GDPVAL_PIPELINE_POLL_INTERVAL:-15}"
LOCK_HELD=0
SERVER_WAIT_PID=""

cleanup() {
  if [[ -n "${SERVER_WAIT_PID}" ]] && kill -0 "${SERVER_WAIT_PID}" 2>/dev/null; then
    kill "${SERVER_WAIT_PID}" 2>/dev/null || true
    wait "${SERVER_WAIT_PID}" 2>/dev/null || true
  fi
  if [[ "${LOCK_HELD}" == "1" ]]; then
    rmdir "${LOCK_DIR}" 2>/dev/null || true
  fi
}

handle_signal() {
  local signal="$1"
  cleanup
  trap - EXIT
  if [[ "${signal}" == "INT" ]]; then
    exit 130
  fi
  exit 143
}

trap cleanup EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

case "${ACTION}" in
  run|full|resume) ;;
  *) echo "ERROR: GDPVAL_ACTION must be run, full, or resume (got ${ACTION})" >&2; exit 2 ;;
esac
for value in "${TAVILY_WAIT_TIMEOUT}" "${POLL_INTERVAL}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: wait settings must be positive integers" >&2; exit 2; }
done

mkdir -p "${STAGING_ROOT}/client_logs"
chmod 700 "${STAGING_ROOT}/client_logs"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "ERROR: another pipeline monitor owns ${LOCK_DIR}" >&2
  exit 2
fi
LOCK_HELD=1

[[ -x "${SERVER_WRAPPER}" ]] || { echo "ERROR: missing server wrapper: ${SERVER_WRAPPER}" >&2; exit 1; }
[[ -r "${CLIENT_SCRIPT}" ]] || { echo "ERROR: missing client script: ${CLIENT_SCRIPT}" >&2; exit 1; }
[[ -f "${PRIVATE_ENV}" ]] || { echo "ERROR: missing private environment: ${PRIVATE_ENV}" >&2; exit 1; }

"${SERVER_WRAPPER}" wait &
SERVER_WAIT_PID=$!
wait "${SERVER_WAIT_PID}"
SERVER_WAIT_PID=""
[[ -f "${ENDPOINT_ENV}" ]] || { echo "ERROR: endpoint wait returned without ${ENDPOINT_ENV}" >&2; exit 1; }

deadline=$((SECONDS + TAVILY_WAIT_TIMEOUT))
while true; do
  if grep -q '^export GDPVAL_ALLOW_NO_TAVILY=1$' "${PRIVATE_ENV}"; then
    echo "Private environment explicitly enables offline Tavily mode"
    break
  fi
  if grep -Eq '^export TAVILY_API_KEY=.+$' "${PRIVATE_ENV}" \
    && ! grep -q 'REPLACE_WITH_PRIVATE_TAVILY_KEY' "${PRIVATE_ENV}"; then
    echo "Private Tavily configuration is present"
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "ERROR: timed out waiting for Tavily configuration in ${PRIVATE_ENV}" >&2
    exit 1
  fi
  echo "Endpoint is ready; waiting for private Tavily configuration"
  sleep "${POLL_INTERVAL}"
done

if [[ -e "${CLIENT_STATE}" ]]; then
  prior_job="$(grep '^job_id=' "${CLIENT_STATE}" | cut -d= -f2 || true)"
  prior_state=""
  if [[ "${prior_job}" =~ ^[0-9]+$ ]]; then
    # squeue exits nonzero once a job leaves the queue ("Invalid job id
    # specified"), which under `set -euo pipefail` aborted this monitor before
    # it could archive a stale client state. Treat a purged job as "no live
    # client", but still fail loudly when squeue itself is unusable: a
    # controller outage must not be misread as "safe to submit another client".
    if ! squeue_out="$(squeue -h -j "${prior_job}" -o '%T' 2>&1)"; then
      if ! grep -qi 'invalid job id' <<<"${squeue_out}"; then
        echo "ERROR: squeue failed for prior client job ${prior_job}: ${squeue_out}" >&2
        exit 1
      fi
      squeue_out=""
    fi
    prior_state="$(head -1 <<<"${squeue_out}")"
  fi
  if [[ -n "${prior_state}" ]]; then
    echo "ERROR: rollout client job ${prior_job} is already ${prior_state}" >&2
    exit 1
  fi
  stale_state="${CLIENT_STATE}.stale.$(date -u +%Y%m%dT%H%M%SZ).$$"
  mv "${CLIENT_STATE}" "${stale_state}"
  chmod 600 "${stale_state}"
fi

# sbatch --export is an allowlist: GDPVAL_PRIVATE_ENV_FILE selects which private env
# (and therefore which dataset profile) the client sources, so a Mercor run does
# not silently inherit AfterQuery's.
submission="$(
  sbatch --parsable \
    --export="USER=${USER},HOME=${HOME},LOGNAME=${LOGNAME:-${USER}},PATH=${PATH},LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-},SLURM_CONF=${SLURM_CONF:-},GDPVAL_ACTION=${ACTION},GDPVAL_PRIVATE_ENV_FILE=${PRIVATE_ENV},GDPVAL_ENV_FILE=${GDPVAL_ENV_FILE:-}" \
    "${CLIENT_SCRIPT}"
)"
job_id="${submission%%;*}"
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: could not parse rollout client job ID from sbatch" >&2
  exit 1
fi

tmp_state="${CLIENT_STATE}.tmp.$$"
{
  echo "job_id=${job_id}"
  echo "action=${ACTION}"
  echo "server_endpoint_env=${ENDPOINT_ENV}"
  echo "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${tmp_state}"
chmod 600 "${tmp_state}"
mv "${tmp_state}" "${CLIENT_STATE}"

echo "Submitted rollout client job ${job_id} with action ${ACTION}"
echo "Client state: ${CLIENT_STATE}"
