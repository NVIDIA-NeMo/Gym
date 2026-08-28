#!/bin/bash
# 04 - Unshard: concatenate every shard's rollouts back into the parent sweep.
#
# Run after individually-launched shards, or before resharding. 03_run_sharded.sh does it for you.
#
# USAGE
#   SWEEP_DIR=<sweep> bash $R/scripts/04_merge_shards.sh
#
# REQUIRED
#   SWEEP_DIR     the sweep whose shards/ should be merged
#
# OPTIONAL
#   SHARDS_DIR    where shard_NNN/ live                   (default: SWEEP_DIR/shards)
#   OUTPUT        merged rollouts path                    (default: SWEEP_DIR/rollouts.jsonl)
#
# Deduplicates on (_ng_task_index, _ng_rollout_index) -- the same key Gym resumes on -- so a shard
# that was rerun cannot double-count. Safe on a partial sweep.
set -euo pipefail

SWEEP_DIR=${SWEEP_DIR:?set SWEEP_DIR to the <out-dir>/<nickname> directory}
SHARDS_DIR=${SHARDS_DIR:-$SWEEP_DIR/shards}
OUTPUT=${OUTPUT:-$SWEEP_DIR/rollouts.jsonl}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "$REPO_ROOT"

# nemo_gym needs its dependencies importable. Automatic inside the eval container, not on a login
# node, where the failure is otherwise a bare ModuleNotFoundError from deep in the CLI.
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
