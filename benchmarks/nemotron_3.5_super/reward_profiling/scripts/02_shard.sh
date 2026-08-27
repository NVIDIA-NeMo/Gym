#!/bin/bash
# Optional: deal a materialized sweep into N sibling sweep directories, one per job.
#
# Only needed to run wider than one job can go. --segment needs a topology-contiguous allocation
# and an NVL72 rack is 18 nodes, so a single job caps out around P4D12; N jobs over disjoint slices
# is how you reach 256. Skip this and run 03_run.sh directly for a single-job sweep.
#
#   SWEEP_DIR=<outputs/sweeps>/<nickname> NUM_SHARDS=16 bash .../scripts/02_shard.sh
#
# Re-running with a different NUM_SHARDS reshards. Rollouts already collected are carried into
# whichever shard now owns each row, so a reshard resumes rather than recollecting -- but merge the
# old shards back first (04_merge_shards.sh), because the carry reads the parent, not the shards.
set -euo pipefail

SWEEP_DIR=${SWEEP_DIR:?set SWEEP_DIR to the <out-dir>/<nickname> directory 01_prepare_sweep.sh wrote}
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
echo "  SWEEP_DIR=$SHARDS_DIR/shard_000 bash .../scripts/03_run.sh"
echo "or all of them, with resubmission of any that die:"
echo "  SWEEP_DIR=$SWEEP_DIR NUM_SHARDS=$NUM_SHARDS bash .../scripts/03_run_sharded.sh"
