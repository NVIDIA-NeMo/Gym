#!/usr/bin/env bash
#SBATCH --job-name=gdpval-cont-monitor
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00

set -euo pipefail
umask 077

STAGING_ROOT="${GDPVAL_SLURM_STAGING_ROOT:-${SCRATCH:-$HOME}/gdpval-rollouts}"
CHECKOUT="${GDPVAL_SLURM_CHECKOUT:-${STAGING_ROOT}/checkout}"
MONITOR="${CHECKOUT}/benchmarks/gdpval/slurm/monitor_gdpval_pipeline.sh"
ACTION="${GDPVAL_ACTION:-resume}"

case "${ACTION}" in
  run|full|resume) ;;
  *)
    echo "ERROR: GDPVAL_ACTION must be run, full, or resume (got ${ACTION})" >&2
    exit 2
    ;;
esac

[[ -x "${MONITOR}" ]] || {
  echo "ERROR: pipeline monitor is not executable: ${MONITOR}" >&2
  exit 1
}

mkdir -p "${STAGING_ROOT}/continuation_logs"
chmod 700 "${STAGING_ROOT}/continuation_logs"

export GDPVAL_ACTION="${ACTION}"
export GDPVAL_SLURM_STAGING_ROOT="${STAGING_ROOT}"
export GDPVAL_SERVER_READY_TIMEOUT="${GDPVAL_SERVER_READY_TIMEOUT:-3300}"
exec "${MONITOR}"
