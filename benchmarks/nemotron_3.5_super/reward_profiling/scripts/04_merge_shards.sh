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
#   GYM_SITE_PACKAGES a venv's site-packages, if orjson/yaml/pydantic are not importable
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
# The sweep package lives beside these scripts, not in nemo_gym.
RP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"

# `infra` needs orjson/yaml/pydantic importable. It no longer imports nemo_gym -- the package is
# self-contained under reward_profiling/infra -- but the deps still have to be there, and on a login
# node the failure is otherwise a bare ModuleNotFoundError from deep in the CLI.
if ! PYTHONPATH="$RP_DIR${PYTHONPATH:+:$PYTHONPATH}" python -c "import orjson, infra" >/dev/null 2>&1; then
    if [[ -n "${GYM_SITE_PACKAGES:-}" ]]; then
        export PYTHONPATH="$REPO_ROOT:$GYM_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
    fi
    if ! PYTHONPATH="$RP_DIR${PYTHONPATH:+:$PYTHONPATH}" python -c "import orjson, infra" >/dev/null 2>&1; then
        echo "ERROR: cannot import infra and its deps (orjson, yaml, pydantic)." >&2
        echo "       Run inside the eval container, activate the Gym venv, or set" >&2
        echo "       GYM_SITE_PACKAGES=<venv>/lib/python3.*/site-packages" >&2
        exit 2
    fi
fi

PYTHONPATH="$RP_DIR${PYTHONPATH:+:$PYTHONPATH}" python -m infra merge "$SHARDS_DIR" --output "$OUTPUT"
