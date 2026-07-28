#!/usr/bin/env bash
#SBATCH --job-name=gdpval-glm52-client
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=180G
#SBATCH --time=03:45:00

set -euo pipefail
umask 077

STAGING_ROOT="${GDPVAL_SLURM_STAGING_ROOT:-${SCRATCH:-$HOME}/gdpval-rollouts}"
CHECKOUT="${GDPVAL_SLURM_CHECKOUT:-${STAGING_ROOT}/checkout}"
PRIVATE_ENV="${GDPVAL_PRIVATE_ENV_FILE:-${STAGING_ROOT}/afterquery.private.env}"
ENDPOINT_ENV="${GDPVAL_ENDPOINT_ENV_FILE:-${STAGING_ROOT}/endpoint.current.env}"
BOOTSTRAP="${CHECKOUT}/benchmarks/gdpval/slurm/bootstrap_gdpval_runtime.sh"
RUNNER="${CHECKOUT}/benchmarks/gdpval/run_gdpval_rollouts.sh"
ACTION="${GDPVAL_ACTION:-run}"

validate_action() {
  case "$1" in
    run|full|resume) ;;
    *)
      echo "ERROR: GDPVAL_ACTION must be run, full, or resume (got $1)" >&2
      return 1
      ;;
  esac
}

# Fail invalid sbatch --export values before touching private launch state.
validate_action "${ACTION}"

require_private_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: required private environment file not found: ${path}" >&2
    return 1
  fi
  local mode
  mode="$(stat -c '%a' "${path}")"
  if [[ "${mode}" != "600" && "${mode}" != "400" ]]; then
    echo "ERROR: private environment file must have mode 600 or 400: ${path} (${mode})" >&2
    return 1
  fi
}

require_private_file "${PRIVATE_ENV}"
require_private_file "${ENDPOINT_ENV}"
[[ -x "${BOOTSTRAP}" ]] || { echo "ERROR: bootstrap script not executable: ${BOOTSTRAP}" >&2; exit 1; }
[[ -x "${RUNNER}" ]] || { echo "ERROR: rollout runner not executable: ${RUNNER}" >&2; exit 1; }

set -a
# The private file supplies Tavily and run settings. The validated endpoint file
# is loaded second so stale placeholder endpoint values cannot win.
# shellcheck disable=SC1090
source "${PRIVATE_ENV}"
# shellcheck disable=SC1090
source "${ENDPOINT_ENV}"
set +a
unset GDPVAL_ENV_FILE

ACTION="${GDPVAL_ACTION:-run}"
validate_action "${ACTION}"

: "${POLICY_MODEL_NAME:?validated endpoint model is unset}"
: "${POLICY_BASE_URL:?validated endpoint URL is unset}"
: "${POLICY_API_KEY:?validated endpoint API key is unset}"
: "${GDPVAL_SERVER_JOB_ID:?validated endpoint server job is unset}"
: "${GDPVAL_SERVER_ROOT:?validated endpoint server root is unset}"
: "${GDPVAL_ENDPOINT_CREATED_AT_UTC:?validated endpoint creation time is unset}"
: "${GDPVAL_APPTAINER_BIN:?GDPVAL_APPTAINER_BIN must name the private Apptainer executable}"
if [[ "${GDPVAL_ALLOW_NO_TAVILY:-0}" != "1" ]]; then
  : "${TAVILY_API_KEY:?TAVILY_API_KEY must be set in the private environment file}"
  if [[ "${TAVILY_API_KEY}" == REPLACE_WITH_* ]]; then
    echo "ERROR: TAVILY_API_KEY still contains the template placeholder" >&2
    exit 1
  fi
fi
if [[ "${GDPVAL_APPTAINER_BIN}" != /* ]]; then
  echo "ERROR: GDPVAL_APPTAINER_BIN must be absolute (got ${GDPVAL_APPTAINER_BIN})" >&2
  exit 1
fi
if [[ ! -x "${GDPVAL_APPTAINER_BIN}" ]]; then
  echo "ERROR: GDPVAL_APPTAINER_BIN is not executable: ${GDPVAL_APPTAINER_BIN}" >&2
  exit 1
fi
if [[ ! "${GDPVAL_SERVER_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: validated endpoint has an invalid server job ID: ${GDPVAL_SERVER_JOB_ID}" >&2
  exit 1
fi
case "${GDPVAL_SERVER_ROOT}" in
  "${STAGING_ROOT}/model_servers/"*) ;;
  *)
    echo "ERROR: validated endpoint server root is outside the staging tree: ${GDPVAL_SERVER_ROOT}" >&2
    exit 1
    ;;
esac
if [[ ! -d "${GDPVAL_SERVER_ROOT}" ]]; then
  echo "ERROR: validated endpoint server root does not exist: ${GDPVAL_SERVER_ROOT}" >&2
  exit 1
fi
command -v squeue >/dev/null
# squeue exits nonzero once the server job leaves the queue, which under
# `set -euo pipefail` killed this script before the explicit check below could
# report why. Failing closed here is correct: any unreadable state is treated as
# "not RUNNING" so the clear error is printed instead of a silent bash abort.
SERVER_STATE="$(squeue -h -j "${GDPVAL_SERVER_JOB_ID}" -o '%T' 2>/dev/null | head -1 || true)"
if [[ "${SERVER_STATE}" != "RUNNING" ]]; then
  echo "ERROR: endpoint server job ${GDPVAL_SERVER_JOB_ID} is not RUNNING (state ${SERVER_STATE:-UNKNOWN})" >&2
  exit 1
fi
export PATH="$(dirname "${GDPVAL_APPTAINER_BIN}"):${PATH}"
# Ray on a Lustre path is slow to start and can exceed the 107-byte AF_UNIX
# socket path limit; /tmp is node-local.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp}"

cd "${CHECKOUT}"
"${BOOTSTRAP}" validate
"${RUNNER}" "${ACTION}"
