#!/usr/bin/env bash
set -euo pipefail

# Poll Slurm for a specific user/account association and stop once it appears.
# This monitor is intentionally passive: it never submits or cancels jobs.

umask 077

STAGING_ROOT="${GDPVAL_SLURM_STAGING_ROOT:-${SCRATCH:-$HOME}/gdpval-rollouts}"
TARGET_USER="${GDPVAL_ACCOUNT_TARGET_USER:-${USER:-$(id -un)}}"
TARGET_ACCOUNT="${GDPVAL_SLURM_ACCOUNT:-}"
TARGET_CLUSTER="${GDPVAL_ACCOUNT_TARGET_CLUSTER:-}"
REQUIRED_QOS="${GDPVAL_ACCOUNT_REQUIRED_QOS:-normal}"
POLL_INTERVAL="${GDPVAL_ACCOUNT_POLL_INTERVAL:-600}"
QUERY_TIMEOUT="${GDPVAL_ACCOUNT_QUERY_TIMEOUT:-30}"
STATE_FILE="${GDPVAL_ACCOUNT_STATE_FILE:-${STAGING_ROOT}/slurm_account_onboarding.state}"
LOCK_FILE="${GDPVAL_ACCOUNT_LOCK_FILE:-${STAGING_ROOT}/.monitor_n4_onboarding.lock}"
PID_FILE="${GDPVAL_ACCOUNT_PID_FILE:-${STAGING_ROOT}/monitor_n4_onboarding.pid}"
ACTION="${1:-monitor}"
PID_WRITTEN=0

usage() {
  cat <<'EOF'
Usage: monitor_n4_onboarding.sh [monitor|once]

Actions:
  monitor  Poll until the Slurm association appears (default).
  once     Check once and exit 0 if approved or 3 if still pending.

Environment:
  GDPVAL_ACCOUNT_TARGET_USER      Slurm username (default: $USER)
  GDPVAL_SLURM_ACCOUNT       Slurm account to wait for (required)
  GDPVAL_ACCOUNT_TARGET_CLUSTER   Restrict to one Slurm cluster (default: any)
  GDPVAL_ACCOUNT_REQUIRED_QOS     Required QOS token (default: normal)
  GDPVAL_ACCOUNT_POLL_INTERVAL    Poll interval in seconds (default: 600)
  GDPVAL_ACCOUNT_QUERY_TIMEOUT    sacctmgr timeout in seconds (default: 30)
  GDPVAL_ACCOUNT_STATE_FILE       Atomic status marker path
  GDPVAL_ACCOUNT_LOCK_FILE        Single-monitor flock path
  GDPVAL_ACCOUNT_PID_FILE         Active monitor PID path
EOF
}

cleanup() {
  if [[ "${PID_WRITTEN}" == "1" ]]; then
    rm -f "${PID_FILE}"
  fi
}

handle_signal() {
  cleanup
  trap - EXIT
  exit 143
}

write_state() {
  local status="$1"
  local checked_at_utc="$2"
  local association="${3:-}"
  local tmp_state="${STATE_FILE}.tmp.$$"

  {
    printf 'status=%s\n' "${status}"
    printf 'checked_at_utc=%s\n' "${checked_at_utc}"
    printf 'cluster=%s\n' "${TARGET_CLUSTER}"
    printf 'user=%s\n' "${TARGET_USER}"
    printf 'account=%s\n' "${TARGET_ACCOUNT}"
    printf 'required_qos=%s\n' "${REQUIRED_QOS}"
    if [[ -n "${association}" ]]; then
      printf 'association=%s\n' "${association}"
    fi
  } >"${tmp_state}"
  chmod 600 "${tmp_state}"
  mv "${tmp_state}" "${STATE_FILE}"
}

find_association() {
  local output rc
  if output="$(
    timeout "${QUERY_TIMEOUT}s" sacctmgr -r -nP show associations \
      "Accounts=${TARGET_ACCOUNT}" \
      "Clusters=${TARGET_CLUSTER}" \
      "Users=${TARGET_USER}" \
      Format=Cluster,Account,User,Partition,QOS 2>&1
  )"; then
    :
  else
    rc=$?
    printf 'ERROR: sacctmgr association query failed (rc=%s): %s\n' \
      "${rc}" "${output}" >&2
    return "${rc}"
  fi

  printf '%s\n' "${output}" |
    awk -F'|' \
      -v target_cluster="${TARGET_CLUSTER}" \
      -v target_account="${TARGET_ACCOUNT}" \
      -v target_user="${TARGET_USER}" \
      -v required_qos="${REQUIRED_QOS}" '
      {
        for (i = 1; i <= NF; i++) {
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
        }
        qos_ok = 0
        qos_count = split($5, qos_tokens, ",")
        for (i = 1; i <= qos_count; i++) {
          if (qos_tokens[i] == required_qos) {
            qos_ok = 1
          }
        }
        if ($1 == target_cluster &&
            $2 == target_account &&
            $3 == target_user &&
            ($4 == "" || $4 == "batch") &&
            qos_ok) {
          print
          exit
        }
      }
    '
}

case "${ACTION}" in
  monitor|once) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "ERROR: action must be monitor or once (got ${ACTION})" >&2
    usage >&2
    exit 2
    ;;
esac

[[ "${TARGET_USER}" =~ ^[A-Za-z0-9_.-]+$ ]] ||
  { echo "ERROR: invalid Slurm username: ${TARGET_USER}" >&2; exit 2; }
[[ "${TARGET_ACCOUNT}" =~ ^[A-Za-z0-9_.-]+$ ]] ||
  { echo "ERROR: invalid Slurm account: ${TARGET_ACCOUNT}" >&2; exit 2; }
[[ "${TARGET_CLUSTER}" =~ ^[A-Za-z0-9_.-]+$ ]] ||
  { echo "ERROR: invalid Slurm cluster: ${TARGET_CLUSTER}" >&2; exit 2; }
[[ "${REQUIRED_QOS}" =~ ^[A-Za-z0-9_.-]+$ ]] ||
  { echo "ERROR: invalid required QOS: ${REQUIRED_QOS}" >&2; exit 2; }
[[ "${POLL_INTERVAL}" =~ ^[1-9][0-9]*$ ]] ||
  { echo "ERROR: GDPVAL_ACCOUNT_POLL_INTERVAL must be a positive integer" >&2; exit 2; }
[[ "${QUERY_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] ||
  { echo "ERROR: GDPVAL_ACCOUNT_QUERY_TIMEOUT must be a positive integer" >&2; exit 2; }

command -v sacctmgr >/dev/null
command -v timeout >/dev/null
command -v flock >/dev/null
mkdir -p "${STAGING_ROOT}" "$(dirname "${STATE_FILE}")"

if [[ "${ACTION}" == "monitor" ]]; then
  exec 9>>"${LOCK_FILE}"
  chmod 600 "${LOCK_FILE}"
  if ! flock -n 9; then
    echo "Onboarding monitor is already running"
    exit 0
  fi
  printf '%s\n' "$$" >"${PID_FILE}"
  chmod 600 "${PID_FILE}"
  PID_WRITTEN=1
  trap cleanup EXIT
  trap handle_signal INT TERM
fi

while true; do
  checked_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if association="$(find_association)"; then
    :
  else
    rc=$?
    write_state error "${checked_at_utc}"
    printf '%s status=error query_rc=%s user=%s account=%s; retrying\n' \
      "${checked_at_utc}" "${rc}" "${TARGET_USER}" "${TARGET_ACCOUNT}"
    if [[ "${ACTION}" == "once" ]]; then
      exit 4
    fi
    sleep "${POLL_INTERVAL}"
    continue
  fi

  if [[ -n "${association}" ]]; then
    write_state approved "${checked_at_utc}" "${association}"
    printf '%s status=approved user=%s account=%s\n' \
      "${checked_at_utc}" "${TARGET_USER}" "${TARGET_ACCOUNT}"
    exit 0
  fi

  write_state pending "${checked_at_utc}"
  printf '%s status=pending user=%s account=%s\n' \
    "${checked_at_utc}" "${TARGET_USER}" "${TARGET_ACCOUNT}"

  if [[ "${ACTION}" == "once" ]]; then
    exit 3
  fi
  sleep "${POLL_INTERVAL}"
done
