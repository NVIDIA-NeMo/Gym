#!/bin/bash
# Unshard: concatenate every shard's rollouts back into the parent sweep.
#
#   SWEEP_DIR=<outputs/sweeps>/<nickname> bash .../scripts/04_merge_shards.sh
#
# Deduplicates on (_ng_task_index, _ng_rollout_index), the same key Gym resumes on, so a shard
# rerun after a walltime kill cannot double-count. Run this before resharding to a different
# NUM_SHARDS: 02_shard.sh carries work from the parent, so anything left only in the old shards
# would be recollected.
#
# 03_run_sharded.sh already does this at the end; use this directly when shards were launched
# individually, or to merge partial results mid-run.
set -euo pipefail

SWEEP_DIR=${SWEEP_DIR:?set SWEEP_DIR to the <out-dir>/<nickname> directory}
SHARDS_DIR=${SHARDS_DIR:-$SWEEP_DIR/shards}
OUTPUT=${OUTPUT:-$SWEEP_DIR/rollouts.jsonl}
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

python -m nemo_gym.sweep merge "$SHARDS_DIR" --output "$OUTPUT"
