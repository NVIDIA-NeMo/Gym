#!/bin/bash
# 02 - Deal a materialized sweep into N sibling sweep directories, one per Slurm job.
#
# Only needed to go past one job's node ceiling (~16: --segment needs a topology-contiguous
# allocation and an NVL72 rack is 18 nodes). Each shard directory is a complete SWEEP_DIR, so
# 03_run_single.sh runs against one unmodified.
#
# USAGE
#   SWEEP_DIR=<sweep> NUM_SHARDS=16 bash $R/scripts/02_shard.sh
#
# REQUIRED
#   SWEEP_DIR     the OUT_DIR/<nickname> directory 01 wrote
#   NUM_SHARDS    how many shards to deal into
#
# OPTIONAL
#   SHARDS_DIR    where shard_NNN/ go                     (default: SWEEP_DIR/shards)
#
# Safe to re-run with a different NUM_SHARDS. Rollouts already collected are folded back into the
# parent and the parent is snapshotted to snapshots/<UTC>/ before any shard directory is touched,
# then carried into the new layout. Rows are dealt round-robin, so every shard carries every
# environment and none inherits a whole slow one.
set -euo pipefail

SWEEP_DIR=${SWEEP_DIR:?set SWEEP_DIR to the <out-dir>/<nickname> directory 01_materialize.sh wrote}
NUM_SHARDS=${NUM_SHARDS:-16}
SHARDS_DIR=${SHARDS_DIR:-$SWEEP_DIR/shards}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "$REPO_ROOT"

# nemo_gym needs its dependencies importable. Inside the eval container that is automatic; on a
# login node it is not, and the failure is otherwise a bare ModuleNotFoundError from deep inside
# the CLI. GYM_SITE_PACKAGES points PYTHONPATH at a venv's site-packages if you are not in one.
if ! python -c "import orjson, nemo_gym" >/dev/null 2>&1; then
    if [[ -n "${GYM_SITE_PACKAGES:-}" ]]; then
        export PYTHONPATH="$REPO_ROOT:$GYM_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
    fi
    if ! python -c "import orjson, nemo_gym" >/dev/null 2>&1; then
        echo "ERROR: cannot import nemo_gym and its deps." >&2
        echo "       Run inside the eval container, activate the Gym venv, or set" >&2
        echo "       GYM_SITE_PACKAGES=<venv>/lib/python3.*/site-packages" >&2
        exit 2
    fi
fi

python -m nemo_gym.sweep shard "$SWEEP_DIR" --num-shards "$NUM_SHARDS" --out-dir "$SHARDS_DIR"
echo
echo "Each shard directory is a valid SWEEP_DIR. Run them with:"
echo "  SWEEP_DIR=$SHARDS_DIR/shard_000 bash .../scripts/03_run_single.sh"
echo "or all of them, with resubmission of any that die:"
echo "  SWEEP_DIR=$SWEEP_DIR NUM_SHARDS=$NUM_SHARDS bash .../scripts/03_run_sharded.sh"
